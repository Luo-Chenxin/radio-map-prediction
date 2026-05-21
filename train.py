import torch
from pathlib import Path
from src.dataset import RadioSeerDataset
from src.models.radio_unet import RadioWnet 
from src.utils.config import load_config_strict

def _get_radio_unet_model(config):
    in_channels = 2
    if config.samples_number > 0:
        in_channels = in_channels + 1
    if config.cars_input:
        in_channels = in_channels + 1
    
    first_out_channels = 6 if in_channels <= 3 else 10


def _mkdir(config):
    out_dir = Path(config.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

def trainRadioUNet(config_path:str, train_id:str, model=None):

    config = load_config_strict(config_path)
    _mkdir(config)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # Prepare dataset
    full_dataset = RadioSeerDataset(config=config.data)
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
    model = RadioWnet(in_channels=in_channels, first_out_channels=first_out_channels).to(device) if model is None else model
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

