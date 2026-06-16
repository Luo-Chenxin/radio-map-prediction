import csv
import json
import random
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from src.dataset import ParisH5Dataset
from src.datamodule import ParisDataModule
from src.models.radio_unet import RadioUnet
from src.trainers.unmasked_trainer import UnmaskedTrainer
from src.utils.config import load_paris_config_strict
from src.utils.utils import append_record

CONFIG_PATH = Path('config/radiounet_paris.yaml')
EXPERIMENT_RESULTS = 'outputs/experiment_results.csv'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_loss_artifacts(history, out_dir):
    csv_path = out_dir / 'loss_curve.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['epoch', 'train_loss', 'val_loss'])
        writer.writeheader()
        writer.writerows(history)


@torch.no_grad()
def save_test_visualization_data(model, device, datamodule, out_dir, seed):
    random_generator = random.Random(seed)
    selected_positions = random_generator.sample(range(len(datamodule.test_dataset)), k=10)
    selected_indices = []
    transmitters = []
    buildings = []
    targets = []
    predictions = []

    model.eval()
    for position in selected_positions:
        dataset_index = datamodule.split_indices['test'][position]
        inputs, target = datamodule.test_dataset[position]
        prediction = model(inputs.unsqueeze(0).to(device)).squeeze(0).cpu()
        selected_indices.append(dataset_index)
        transmitters.append(inputs[0].numpy())
        buildings.append(inputs[1].numpy())
        targets.append(target[0].numpy())
        predictions.append(prediction[0].numpy())

    np.savez_compressed(
        out_dir / 'test_visualization_data.npz',
        indices=np.asarray(selected_indices),
        transmitters=np.stack(transmitters),
        buildings=np.stack(buildings),
        targets=np.stack(targets),
        predictions=np.stack(predictions))

    return selected_indices


def run_experiment():
    config = load_paris_config_strict(CONFIG_PATH)
    set_seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    datamodule = ParisDataModule(ParisH5Dataset, config.data.h5_path, config.load, config.seed)
    model = RadioUnet(in_channels=2, first_out_channels=6).to(device)
    trainer = UnmaskedTrainer(model, device, CONFIG_PATH.stem, config.train)

    history = trainer.fit(
        datamodule.get_train_dataloader(),
        datamodule.get_val_dataloader())

    state_dict = torch.load(trainer.early_stopping.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    metrics = trainer.test(datamodule.get_test_dataloader())

    run_dir = trainer.tb_dir
    save_loss_artifacts(history, run_dir)
    selected_indices = save_test_visualization_data(model, device, datamodule, run_dir, config.seed)

    with (run_dir / 'test_metrics.json').open('w', encoding='utf-8') as file:
        json.dump(metrics, file, indent=2)
    with (run_dir / 'split_indices.json').open('w', encoding='utf-8') as file:
        json.dump(datamodule.split_indices, file, indent=2)
    with (run_dir / 'visualized_test_indices.json').open('w', encoding='utf-8') as file:
        json.dump(selected_indices, file, indent=2)

    append_record(
        file_path=EXPERIMENT_RESULTS,
        model_arch=RadioUnet.__name__,
        dataset_desc='Paris_H5|Global_MinMax|TX_And_Buildings',
        metrics=metrics,
        timestamp=trainer.timestamp)

    subprocess.run(
        [sys.executable, 'scripts/plot_paris_results.py', '--run-dir', str(run_dir)],
        check=True)

    print(json.dumps({
        'device': str(device),
        'epochs_completed': len(history),
        'split_sizes': {key: len(value) for key, value in datamodule.split_indices.items()},
        'metrics': metrics,
        'run_dir': str(run_dir),
    }, indent=2))


if __name__ == '__main__':
    run_experiment()
