import torch
from pathlib import Path
from src.dataset import RadioSeerDataset
from src.datamodule import RadioSeerDataModule
from src.models.radio_unet import RadioUnet 
from src.utils.config import load_config_strict
from src.trains.unmasked_trainer import UnmaskedTrainer

def _get_radio_unet_model(config):
    in_channels = 1 + 1
    if config.samples_number > 0:
        in_channels = in_channels + 1
    if config.cars_input:
        in_channels = in_channels + 1
    
    first_out_channels = 6 if in_channels <= 3 else 10
    model = RadioUnet(in_channels, first_out_channels)
    return model


def _mkdir(config):
    out_dir = Path(config.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

def train(config_path):
    config = load_config_strict(config_path)
    _mkdir(config)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    dataset = RadioSeerDataset(config=config.data)
    datamodule = RadioSeerDataModule(dataset, config.seed, config.load)
    train_loader = datamodule.get_train_dataloader()
    val_loader = datamodule.get_val_dataloader()

    model = _get_radio_unet_model(config.data)

    trainer = UnmaskedTrainer(model, device, config.train)

    trainer.fit(train_loader, val_loader)

if __name__ == "__main__":
    train('config/clean_DPM.yaml')