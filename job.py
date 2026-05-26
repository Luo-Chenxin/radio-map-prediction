import torch
from pathlib import Path
from src.dataset import RadioSeerDataset
from src.datamodule import RadioSeerDataModule 
from src.models.radio_unet import RadioUnet, RadioWnet
from src.utils.config import load_config_strict
from src.utils.utils import get_radiounet_model, get_dataset_desc, append_record
from src.trainers.unmasked_trainer import UnmaskedTrainer
from src.trainers.masked_trainer import MaskedTrainer

EXPERIMENT_RESULTS = 'outputs/experiment_results.csv'

def run_experiment(config_path, model_class, trainer_class, pretrained_path=None, mode='train'):
    """
    Unified experimental actuator

    Paramaters:
      mode: 'train' (training + testing) or 'test' (testing only)
    """
    config_path = Path(config_path)
    experiment_id = config_path.stem
    config = load_config_strict(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Initialize data
    datamodule = RadioSeerDataModule(RadioSeerDataset, config.load, config.data, config.seed)
    
    # 2. Initialize the model
    model = get_radiounet_model(config.data, model_class)
    model.to(device=device)
    
    # 3. Load pre-trained weights (if any)
    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location=device)
        if model_class == RadioWnet:
            model.first_unet.load_state_dict(state_dict)
            for param in model.first_unet.parameters():
                param.requires_grad = False
            model.first_unet.eval()
        else:
            model.load_state_dict(state_dict)

    # 4. Initialize Trainer
    trainer = trainer_class(model, device, experiment_id, config.train)

    # 5. Execution Process
    if mode == 'train':
        train_loader = datamodule.get_train_dataloader()
        val_loader = datamodule.get_val_dataloader()
        trainer.fit(train_loader, val_loader)

    # 6. Testing and Recording
    test_loader = datamodule.get_test_dataloader()
    metrics = trainer.test(test_loader)
    
    dataset_field = get_dataset_desc(config.data)
    append_record(EXPERIMENT_RESULTS, model_class.__name__, dataset_field, metrics)


if __name__ == "__main__":
    CFG_UNET = 'config/radiounet_dpm_nocars_missing0_samples0.yaml'
    CFG_WNET = 'config/radiownet_dpm_nocars_missing0_samples0.yaml'
    CKPT_UNET = 'outputs/radiounet_dpm_nocars_missing0_samples0/best_model.pt'

    # 1. Test the trained RadioUnet
    run_experiment(
        config_path=CFG_UNET, 
        model_class=RadioUnet, 
        trainer_class=UnmaskedTrainer, 
        pretrained_path=CKPT_UNET, 
        mode='test'
    )
    
    # 2. Train and Test RadioWnet
    run_experiment(
        config_path=CFG_WNET, 
        model_class=RadioWnet, 
        trainer_class=MaskedTrainer, 
        pretrained_path=CKPT_UNET,
        mode='train'
    )