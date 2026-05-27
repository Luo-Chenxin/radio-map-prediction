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
    Unified experimental actuator supporting training and testing workflows.

    Parameters:
        config_path (str/Path): Path to the configuration YAML file.
        model_class (class): Model class to instantiate (RadioUnet or RadioWnet).
        trainer_class (class): Trainer class to handle training/testing loops.
        pretrained_path (str, optional): Path to pretrained checkpoint (.pt).
        mode (str): 'train' for training + testing, 'test' for testing only.
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
    
    # 3. Load pre-trained weights based on the experiment mode and model type
    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location=device)
        
        if mode == 'test':
            # Mode 1: Testing only
            # Load the complete trained weights for the corresponding model
            model.load_state_dict(state_dict)
            
        elif mode == 'train':
            # Mode 2: Training + Testing
            if model_class == RadioWnet:
                # Mode 2.1: Load RadioUnet checkpoint into RadioWnet's first_unet and freeze it
                model.first_unet.load_state_dict(state_dict)
                for param in model.first_unet.parameters():
                    param.requires_grad = False
                model.first_unet.eval()
            else:
                # Mode 2.2: RadioUnet training from scratch
                # (pretrained_path is left as None)
                pass

    # 4. Initialize Trainer
    trainer = trainer_class(model, device, experiment_id, config.train)

    # 5. Execution Process (Training phase)
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
    # Define configuration and checkpoint paths
    CFG_UNET = 'config/radiounet_dpm_nocars_missing0_samples0.yaml'
    CFG_WNET = 'config/radiownet_dpm_nocars_missing0_samples0.yaml'
    
    CKPT_UNET = 'outputs/radiounet_dpm_nocars_missing0_samples0/best_model.pt'
    CKPT_WNET = 'outputs/radiownet_dpm_nocars_missing0_samples0/best_model.pt'

    # ---------------------------------------------------------
    # Mode 1: Testing Only
    # ---------------------------------------------------------
    # 1.1 Test the trained RadioWnet
    # run_experiment(
    #     config_path=CFG_WNET, 
    #     model_class=RadioWnet, 
    #     trainer_class=MaskedTrainer, 
    #     pretrained_path=CKPT_WNET, 
    #     mode='test'
    # )

    # 1.2 Test the trained RadioUnet
    # run_experiment(
    #     config_path=CFG_UNET, 
    #     model_class=RadioUnet, 
    #     trainer_class=UnmaskedTrainer, 
    #     pretrained_path=CKPT_UNET, 
    #     mode='test'
    # )
    
    # ---------------------------------------------------------
    # Mode 2: Training + Testing
    # ---------------------------------------------------------
    # 2.1 Train and Test RadioWnet (Load from frozen RadioUnet)
    run_experiment(
        config_path=CFG_WNET, 
        model_class=RadioWnet, 
        trainer_class=MaskedTrainer, 
        pretrained_path=CKPT_UNET, # RadioUnet checkpoint
        mode='train'
    )

    # 2.2 Train and Test RadioUnet from scratch
    # run_experiment(
    #     config_path=CFG_UNET, 
    #     model_class=RadioUnet, 
    #     trainer_class=UnmaskedTrainer, 
    #     pretrained_path=None,      # Train from scratch
    #     mode='train'
    # )