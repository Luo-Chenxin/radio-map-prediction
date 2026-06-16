import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_loss(run_dir):
    with (run_dir / 'loss_curve.csv').open('r', encoding='utf-8') as file:
        history = list(csv.DictReader(file))

    epochs = [int(item['epoch']) for item in history]
    train_loss = [float(item['train_loss']) for item in history]
    val_loss = [float(item['val_loss']) for item in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label='Train loss')
    ax.plot(epochs, val_loss, label='Validation loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE loss')
    ax.set_yscale('log')
    ax.set_title('RadioUnet Paris training loss')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / 'loss_curve.png', dpi=200)
    plt.close(fig)


def plot_test_samples(run_dir):
    visualization_dir = run_dir / 'test_visualizations'
    visualization_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(run_dir / 'test_visualization_data.npz')

    for position, dataset_index in enumerate(data['indices'], start=1):
        target = data['targets'][position - 1]
        prediction = data['predictions'][position - 1]
        radiomap_vmax = max(float(target.max()), float(prediction.max()), 1e-8)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        panels = [
            (data['transmitters'][position - 1], 'Transmitter input', 'gray', 1.0),
            (data['buildings'][position - 1], 'Building input', 'gray', 1.0),
            (target, 'Ground-truth normalized radiomap', 'viridis', radiomap_vmax),
            (prediction, 'Predicted normalized radiomap', 'viridis', radiomap_vmax),
        ]
        for ax, (image, title, color_map, vmax) in zip(axes, panels):
            ax.imshow(image, cmap=color_map, vmin=0.0, vmax=vmax)
            ax.set_title(title)
            ax.axis('off')
        fig.suptitle(f'Test sample index {dataset_index}')
        fig.tight_layout()
        fig.savefig(visualization_dir / f'sample_{position:02d}_index_{dataset_index}.png', dpi=180)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    plot_loss(run_dir)
    plot_test_samples(run_dir)
