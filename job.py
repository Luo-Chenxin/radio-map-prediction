import torch
from pathlib import Path
from src.dataset import RadioSeerDataset
from src.datamodule import RadioSeerDataModule 
from src.utils.config import load_config_strict
from src.utils.utils import get_radiounet_model, get_dataset_desc, append_record
from src.trainers.unmasked_trainer import UnmaskedTrainer
from src.trainers.early_stopping import BEST_MODEL_PARTERN

EXPERIMENT_RESULTS = 'experiment_results.csv'
RADIOUNET_DPM_NOCARS_MISSING0_SAMPLES0 = 'config/radiounet_dpm_nocars_missing0_samples0.yaml'

def train_radiounet_dpm_nocars_missing0_samples0():
    configPath = Path(RADIOUNET_DPM_NOCARS_MISSING0_SAMPLES0)
    id = configPath.stem
    config = load_config_strict(configPath)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    datamodule = RadioSeerDataModule(RadioSeerDataset, config.load, config.data, config.seed)
    train_loader = datamodule.get_train_dataloader()
    val_loader = datamodule.get_val_dataloader()

    model = get_radiounet_model(config.data)

    trainer = UnmaskedTrainer(model, device, id, config.train)

    trainer.fit(train_loader, val_loader)

def test_radiounet(config_path):
    configPath = Path(config_path)
    id = configPath.stem
    config = load_config_strict(configPath)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    datamodule = RadioSeerDataModule(RadioSeerDataset, config.load, config.data, config.seed)
    test_loader = datamodule.get_test_dataloader()

    model = get_radiounet_model(config.data)
    model_file = Path(config.train.out_dir) / id / BEST_MODEL_PARTERN
    state_dict = torch.load(model_file, map_location=device)
    model.load_state_dict(state_dict)

    trainer = UnmaskedTrainer(model, device, id, config.train)

    metrics = trainer.test(test_loader)
    dataset_field = get_dataset_desc(config.data)
    
    append_record(EXPERIMENT_RESULTS, "RadioUnet", dataset_field, metrics)

if __name__ == "__main__":
    test_radiounet(RADIOUNET_DPM_NOCARS_MISSING0_SAMPLES0)