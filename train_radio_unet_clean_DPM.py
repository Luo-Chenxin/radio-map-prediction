import torch
from src.dataset import RadioSeerDataset
from src.datamodule import RadioSeerDataModule 
from src.utils.config import load_config_strict
from src.trainers.unmasked_trainer import UnmaskedTrainer
from src.utils.train import make_output_dir, get_radio_unet_model

CLEAN_DPM_CONFIG = 'config/clean_DPM.yaml'

def train_radio_unet_clean_DPM():
    config = load_config_strict(CLEAN_DPM_CONFIG)
    make_output_dir(config)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    dataset = RadioSeerDataset(config=config.data)
    datamodule = RadioSeerDataModule(dataset, config.seed, config.load)
    train_loader = datamodule.get_train_dataloader()
    val_loader = datamodule.get_val_dataloader()

    model = get_radio_unet_model(config.data)

    trainer = UnmaskedTrainer(model, device, config.train)

    trainer.fit(train_loader, val_loader)

if __name__ == "__main__":
    train_radio_unet_clean_DPM()