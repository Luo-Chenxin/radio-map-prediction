from pathlib import Path
from src.models.radio_unet import RadioUnet

def get_radio_unet_model(config):
    in_channels = 1 + 1
    if config.samples_number > 0:
        in_channels = in_channels + 1
    if config.cars_input:
        in_channels = in_channels + 1
    
    first_out_channels = 6 if in_channels <= 3 else 10
    model = RadioUnet(in_channels, first_out_channels)
    return model


def make_output_dir(config):
    out_dir = Path(config.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)