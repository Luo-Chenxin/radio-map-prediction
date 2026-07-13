from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode

class RadioSeerDataset(Dataset):
    def __init__(self, config, seed, is_test=False):
        self.config = config
        self.device = 'cpu'
        self.tensor_dtype = torch.float32
        self.is_test = is_test
        np.random.seed(seed)

    def __len__(self):
        return self.config.maps_number*self.config.transmitters_number

    def __getitem__(self, idx):
        map_idx = idx // self.config.transmitters_number
        tx_idx = idx % self.config.transmitters_number
        
        buildings = self._load_buildings(map_idx)
        transmitters = self._load_transmitters(map_idx, tx_idx)
        
        gain = self._load_gain(map_idx, tx_idx)
        gain = self._apply_threshold(gain)
        
        inputs = torch.cat([buildings, transmitters], dim=0)
        mask = None

        if self.config.sparse_IRT4_number > 0:
            mask, h_coords, w_coords = self._generate_mask()
            if self.config.samples_number > 0:
                samples_gain = self._generate_samples(gain, self.config.sparse_IRT4_number, h_coords, w_coords)
                inputs = torch.cat([inputs, samples_gain], dim=0)
        else:
            if self.config.samples_number > 0:
                samples_gain = self._generate_samples(gain)
                inputs = torch.cat([inputs, samples_gain], dim=0)

        if self.config.cars_exist:
            cars = self._load_cars(map_idx)
            inputs = torch.cat([inputs, cars], dim=0)
        
        return (inputs, gain) if mask is None else (inputs, gain, mask)

    def _load_buildings(self, map_idx):
        path = Path(self.config.root_dir)
        name = f"{str(map_idx)}.png"

        if self.config.city_map == 'complete':
            path = path / self.config.buildings_complete_dir
        else:
            if self.config.city_map == 'rand':
                self.config.missing = np.random.randint(low=1, high=5)
            self.version = np.random.randint(low=1, high=7)

            path = path / f"{self.config.buildings_missing_dir}{self.config.missing}" / str(self.version)
        
        path = path / name
        return read_image(str(path), ImageReadMode.GRAY).to(device=self.device, dtype=self.tensor_dtype) / 255.0
    
    def _load_cars(self, map_idx):
        path = Path(self.config.root_dir) / self.config.cars_dir / f"{str(map_idx)}.png"
        return read_image(str(path), ImageReadMode.GRAY).to(device=self.device, dtype=self.tensor_dtype) / 255.0
    
    def _load_transmitters(self, map_idx, tx_idx):
        path = Path(self.config.root_dir) / self.config.antennas_dir / f"{map_idx}_{tx_idx}.png"
        return read_image(str(path), ImageReadMode.GRAY).to(device=self.device, dtype=self.tensor_dtype) / 255.0


    def _load_gain(self, map_idx, tx_idx):
        path = Path(self.config.root_dir)
        name = f"{map_idx}_{tx_idx}.png"

        pathDPM = Path(self.config.DPM_cars_dir) if self.config.cars_exist else Path(self.config.DPM_dir)
        pathDPM = path / pathDPM / name
        pathIRT2 = Path(self.config.IRT2_cars_dir) if self.config.cars_exist else Path(self.config.IRT2_dir)
        pathIRT2 = path / pathIRT2 / name
        pathIRT4 = Path(self.config.IRT4_cars_dir) if self.config.cars_exist else Path(self.config.IRT4_dir)
        pathIRT4 = path / pathIRT4 / name

        if self.config.sparse_IRT4_number > 0 or self.is_test:
            return read_image(str(pathIRT4), ImageReadMode.GRAY).to(device=self.device, dtype=self.tensor_dtype) / 255.0
        elif self.config.simulation == 'DPM':
            return read_image(str(pathDPM), ImageReadMode.GRAY).to(device=self.device, dtype=self.tensor_dtype) / 255.0
        elif self.config.simulation == 'IRT2':
            return read_image(str(pathIRT2), ImageReadMode.GRAY).to(device=self.device, dtype=self.tensor_dtype) / 255.0
        else:
            dpm = read_image(str(pathDPM), ImageReadMode.GRAY).to(device=self.device, dtype=self.tensor_dtype) / 255.0
            irt2 = read_image(str(pathIRT2), ImageReadMode.GRAY).to(device=self.device, dtype=self.tensor_dtype) / 255.0
            return self.config.IRT2_weight * irt2 + (1-self.config.IRT2_weight) * dpm
    
    def _apply_threshold(self, gain):
        thr = self.config.threshold
        return (torch.clip(gain, min=thr) - thr) / (1 - thr)

    def _generate_mask(self):
        H, W = self.config.img_size[0], self.config.img_size[1]
        mask = torch.zeros(H, W, device=self.device, dtype= torch.float32)
        perm = torch.randperm(H * W, device=self.device)[:self.config.sparse_IRT4_number]
        h_coords = perm // W
        w_coords = perm % W
        mask[h_coords, w_coords] = 1.0

        return mask.unsqueeze(0), h_coords, w_coords
    
    def _generate_samples(
        self, 
        gain,
        total_num = None, 
        h_coords = None, 
        w_coords = None):
        
        H, W = self.config.img_size[0], self.config.img_size[1]
        if total_num is None and h_coords is None and w_coords is None:
            torch_num = H * W
            perm = torch.randperm(torch_num, device=self.device)
            h_coords = perm // W
            w_coords = perm % W

        samples_gain = torch.zeros(H, W, device=self.device, dtype= torch.float32)
        selected_indices = torch.randperm(total_num, device=self.device)[:self.config.samples_number]
        selected_h_coords = h_coords[selected_indices]
        selected_w_coords = w_coords[selected_indices]
        samples_gain[selected_h_coords, selected_w_coords] = gain[0, selected_h_coords, selected_w_coords]

        return samples_gain

class H5Dataset(Dataset):
    def __init__(self, config, seed=None, is_test=False):
        self.h5_path = Path(config.h5_path)
        self.h5_file = None
        self.thr = config.threshold

        with h5py.File(self.h5_path, 'r') as h5_file:
            self.dataset_size = len(h5_file['radiomap'])
            expected_shape = h5_file['radiomap'].shape
            if h5_file['buildings'].shape != expected_shape or h5_file['transmitters'].shape != expected_shape:
                raise ValueError("Paris H5 datasets must have matching shapes")

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        h5_file = self._get_h5_file()
        buildings = torch.from_numpy(h5_file['buildings'][idx].astype(np.float32))
        transmitters = torch.from_numpy(h5_file['transmitters'][idx].astype(np.float32))
        radiomap = torch.from_numpy(h5_file['radiomap'][idx].astype(np.float32))

        inputs = torch.stack([transmitters, buildings], dim=0)
        target = self._get_target(radiomap)
        return inputs, target
    
    def _get_target(self, radiomap):
        thr = self.thr
        target = (torch.clip(radiomap, min=thr) - thr) / (1 - thr)
        return target.unsqueeze(0)

    def _get_h5_file(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
        return self.h5_file

    def __getstate__(self):
        state = self.__dict__.copy()
        state['h5_file'] = None
        return state

    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()
