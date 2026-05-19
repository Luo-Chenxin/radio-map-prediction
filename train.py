import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import lr_scheduler

from src.utils.utils import load_config_strict, set_seed, setup_logging, split_dataset, get_saved_model_path
from src.data.dataset import RadioMapDataset
from src.models.radio_unet import RadioWNet 

class _EarlyStopping:
    def __init__(self, patience, delta, save_path):
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.save_path)

def _train_one_epoch(model, loader, criterion, optimizer, device, use_mask_loss, stage_output_idx):
    """Encapsulated single-round training function"""
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for batch in pbar:
        if use_mask_loss:
            inputs, targets, masks = batch
            inputs, targets, masks = inputs.to(device), masks.to(device), targets.to(device)
        else:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        
        outputs_all = model(inputs)
        outputs = outputs_all[stage_output_idx] # 根据阶段选择对应的输出头

        # Loss Calculation
        if use_mask_loss:
            loss = criterion(outputs[masks.bool()], targets[masks.bool()])
        else:
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"avg_loss": f"{total_loss / (pbar.n + 1):.4f}"})
        
    return total_loss / len(loader)

@torch.no_grad()
def _validate(model, loader, criterion, device, use_mask_loss, stage_output_idx):
    """Encapsulated verification function"""
    model.eval()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Validating", leave=False)
    for batch in pbar:
        if use_mask_loss:
            inputs, targets, masks = batch
            inputs, targets, masks = inputs.to(device), masks.to(device), targets.to(device)
        else:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
        
        outputs_all = model(inputs)
        outputs = outputs_all[stage_output_idx] # 根据阶段选择对应的输出头

        # Loss Calculation
        if use_mask_loss:
            loss = criterion(outputs[masks.bool()], targets[masks.bool()])
        else:
            loss = criterion(outputs, targets)
        
        total_loss += loss.item()
        pbar.set_postfix({"avg_loss": f"{total_loss / (pbar.n + 1):.4f}"})

    return total_loss / len(loader)

def _get_RadioUNet_in_channels_and_first_out_channels(config):
    in_channels = 2
    if config.samples_number > 0:
        in_channels = in_channels + 1
    if config.cars_input:
        in_channels = in_channels + 1
    
    first_out_channels = 6 if in_channels <= 3 else 10

    return in_channels, first_out_channels

def _recover_check_point(config, model, optimizer, early_stopping, logger, device):
    start_epoch = 0
    checkpoint_pattern = get_saved_model_path(config, suffix="checkpoint_*")
    checkpoint_dir = os.path.dirname(checkpoint_pattern)
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, f"{train_id}_checkpoint_*.pt")))
    
    if checkpoint_files:
        latest_checkpoint = checkpoint_files[-1]
        logger.info(f"Find the latest check point: {latest_checkpoint}, recover training state")
        
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        early_stopping.best_score = checkpoint.get('best_score', None)
        
        logger.info(f"Successfully recover training in epoch {start_epoch}")

def trainRadioUNet(config_path:str, train_id:str, model=None):
    """
    model: 
    """
    # 1. Load configuration, set logs and random seed
    cfg = load_config_strict(config_path) 
    logger = setup_logging(cfg.train.log_dir, train_id)
    set_seed(cfg.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    logger.info(f"Start Training: {train_id} | device: {device}")

    # 2. Prepare dataset
    full_dataset = RadioMapDataset(config=cfg.data)
    train_subset, val_subset, _ = split_dataset(full_dataset, cfg.load.train_ratio, cfg.load.val_ratio, cfg.seed)
    
    train_loader = DataLoader(
        train_subset, 
        batch_size=cfg.load.train_batch_size, 
        shuffle=True, 
        num_workers=cfg.load.num_workers,
        pin_memory=True)
    val_loader = DataLoader(
        val_subset, 
        batch_size=cfg.load.val_batch_size, 
        shuffle=False,
        num_workers=cfg.load.num_workers,
        pin_memory=True)

    # 3. Initialize the model, loss, optimizer and early_stopping
    in_channels, first_out_channels = _get_RadioUNet_in_channels_and_first_out_channels(cfg.data)
    model = RadioWNet(in_channels=in_channels, first_out_channels=first_out_channels).to(device) if model is None else model
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.train.scheduler.step_size, gamma=cfg.train.scheduler.gamma)
    early_stopping = _EarlyStopping(
        patience=cfg.train.early_stop.patience, 
        delta=cfg.train.early_stop.delta, 
        save_path=get_saved_model_path(cfg.data, suffix="best")
        )

    # 4. Main Training Loop
    for epoch in range(cfg.train.epoch):
        if cfg.data.sparse_IRT4_number == 0:
            stage_output_idx = 0
            use_mask_loss = False
            for param in model.first_unet.parameters(): param.requires_grad = True
            for param in model.second_unet.parameters(): param.requires_grad = False
        else:
            stage_output_idx = 1
            use_mask_loss = True
            for param in model.first_unet.parameters(): param.requires_grad = False
            for param in model.second_unet.parameters(): param.requires_grad = True
        
        logger.info(f"\n--- Training Epoch {epoch+1} ---")

        avg_train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device, use_mask_loss, stage_output_idx)
        avg_val_loss = _validate(model, val_loader, criterion, device, use_mask_loss, stage_output_idx)
        
        logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        scheduler.step()

        early_stopping(avg_val_loss, model)

        if (epoch + 1) % cfg.train.save_interval == 0:
            checkpoint_path = get_saved_model_path(cfg.data, suffix=f"checkpoint_{epoch+1}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'best_score': early_stopping.best_score,
            }, checkpoint_path)
            logger.info(f"Checkpoint saved at epoch {epoch+1}")

        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break

    logger.info("Training Finish")