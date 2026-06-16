import argparse
from pathlib import Path
import h5py
import numpy as np


def get_radiomap_stats(dataset):
    data_min = np.inf
    data_max = -np.inf
    nan_count = 0

    for start in range(0, len(dataset), 16):
        values = dataset[start:start + 16]
        finite_values = values[np.isfinite(values)]
        data_min = min(data_min, float(finite_values.min()))
        data_max = max(data_max, float(finite_values.max()))
        nan_count += int(np.isnan(values).sum())

    return data_min, data_max, nan_count


def prepare_h5(source_path, output_path):
    source_path = Path(source_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(source_path, 'r') as source:
        data_min, data_max, nan_count = get_radiomap_stats(source['radiomap'])
        if data_max <= data_min:
            raise ValueError("Radiomap maximum must be greater than its minimum")

        normalization = source.attrs.get('radiomap_normalization', '')
        if normalization == 'global_minmax' and np.isclose(data_min, 0.0) and np.isclose(data_max, 1.0):
            raise ValueError("Source radiomap is already globally min-max normalized")

        with h5py.File(output_path, 'w') as output:
            for key, value in source.attrs.items():
                output.attrs[key] = value
            output.attrs['radiomap_normalization'] = 'global_minmax'
            output.attrs['radiomap_original_min'] = data_min
            output.attrs['radiomap_original_max'] = data_max
            output.attrs['radiomap_nan_count'] = nan_count
            output.attrs['radiomap_nan_fill_value'] = data_min
            output.attrs['source_file'] = source_path.name

            for name in ['buildings', 'transmitters']:
                source.copy(name, output)

            source_radiomap = source['radiomap']
            output_radiomap = output.create_dataset(
                'radiomap',
                shape=source_radiomap.shape,
                dtype=np.float32,
                chunks=source_radiomap.chunks,
                compression='gzip')

            scale = data_max - data_min
            for start in range(0, len(source_radiomap), 16):
                values = source_radiomap[start:start + 16]
                values = np.nan_to_num(values, nan=data_min)
                output_radiomap[start:start + 16] = (values - data_min) / scale

    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    print(f"Original finite range: [{data_min}, {data_max}]")
    print(f"NaN values filled with {data_min}: {nan_count}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='dataset_paris/block.h5')
    parser.add_argument('--output', default='dataset_paris/block_normalized.h5')
    args = parser.parse_args()
    prepare_h5(args.source, args.output)
