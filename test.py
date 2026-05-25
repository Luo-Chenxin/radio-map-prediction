import torch
from pathlib import Path
from src.dataset import RadioSeerDataset
from src.datamodule import RadioSeerDataModule 
from src.utils.config import load_config_strict
from src.trainers.unmasked_trainer import UnmaskedTrainer
from src.utils.train import make_output_dir, get_radio_unet_model
from src.trainers.early_stopping import BEST_MODEL_PARTERN
from src.utils.experiment_results import append_record, get_trainset_field

CLEAN_DPM_CONFIG = 'config/clean_DPM.yaml'
EXPERIMENT_RESULTS = 'experiment_results.csv'

def test_radio_unet_clean_DPM():
    config = load_config_strict(CLEAN_DPM_CONFIG)
    make_output_dir(config)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    dataset = RadioSeerDataset(config=config.data, seed=config.seed)
    datamodule = RadioSeerDataModule(dataset, config.seed, config.load)
    test_loader = datamodule.get_test_dataloader()

    model = get_radio_unet_model(config.data)
    model_file = Path(config.train.out_dir) / BEST_MODEL_PARTERN
    state_dict = torch.load(model_file, map_location=device)
    model.load_state_dict(state_dict)

    trainer = UnmaskedTrainer(model, device, config.train)

    metrics = trainer.test(test_loader)
    train_set = get_trainset_field(config.data)
    test_set = 

    append_record(EXPERIMENT_RESULTS, "RadioUnet", train_set, test_set, metrics)

if __name__ == "__main__":
    test_radio_unet_clean_DPM()