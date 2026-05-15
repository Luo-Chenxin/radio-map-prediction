from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode

class RadioMapDataset(Dataset):
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.tensor_dtype = torch.float32

    def __len__(self):
        return self.config.maps_number*self.config.transmitters_number

    def __getitem__(self, idx):
        map_idx = idx / self.config.transmitters_number
        tx_idx = idx % self.config.transmitters_number
        
        buildings = self._load_buildings(map_idx)
        transmitters = self._load_transmitters(map_idx, tx_idx)
        
        gain = self._load_gain(map_idx, tx_idx)
        gain = self._apply_threshold(gain)
        
        inputs = [buildings, transmitters]

        if self.config.sparse_IRT4_number > 0:
            mask, h_coords, w_coords = self._generate_mask()
            if self.config.samples_number > 0:
                samples_gain = self._generate_samples(gain, self.config.sparse_IRT4_number, h_coords, w_coords)

        if self.config.cars_input:
            cars = self._load_cars(map_idx)
            inputs.append(cars)

        #--------------------------

        if self.config.samples_number > 0:
            samples = self._generate_random_samples(gain)
        
        if self.mode == 'sparse_random':
            # samples_img = self._generate_random_samples(gain_img)
            inputs.append(samples_img)
        elif self.mode == 'sparse_fixed':
            samples_img, mask_img = self._generate_fixed_samples(gain_img)
            # 注意: IRT4 模式通常返回 mask 作为辅助信息
            # 这里可以将 mask 存入 gain_img 的某个属性，或直接作为第 3 个返回值
            inputs.append(samples_img)
        
        inputs = np.stack(inputs, axis=2)
        if self.config.get('transform'):
            inputs = self.config['transform'](inputs)
            gain_img = self.config['transform'](gain_img)
        
        if self.mode == 'sparse_fixed':
            return inputs, gain_img, mask_img
        else:
            return inputs, gain_img

    def _load_buildings(self, map_idx):
        path = Path(self.config.root_dir)
        name = f"{str(map_idx)}.png"

        if self.config.city_map == 'complete':
            path = path / self.config.buildings_complete_dir
        else:
            if self.config.city_map == 'rand':
                self.config.missing = np.random.randint(low=1, high=5)
            self.config.version = np.random.randint(low=1, high=7)

            path = path / f"{self.config.buildings_missing_dir}{self.config.missing}" / str(self.config.version)
        
        path = path / name
        return read_image(str(path), ImageReadMode.RGB).to(device=self.device, dtype=self.tensor_dtype) / 255.0
    
    def _load_cars(self, map_idx):
        path = Path(self.config.root_dir) / self.config.cars_dir / f"{str(map_idx)}.png"
        return read_image(str(path), ImageReadMode.RGB).to(device=self.device, dtype=self.tensor_dtype) / 255.0
    
    def _load_transmitters(self, map_idx, tx_idx):
        path = Path(self.config.root_dir) / self.config.antennas_dir / f"{map_idx}_{tx_idx}.png"
        return read_image(str(path), ImageReadMode.RGB).to(device=self.device, dtype=self.tensor_dtype) / 255.0


    def _load_gain(self, map_idx, tx_idx):
        path = Path(self.config.root_dir)
        name = f"{map_idx}_{tx_idx}.png"

        pathDPM = path / Path(self.config.DPM_cars_dir) if self.config.cars_simulation else Path(self.config.DPM_dir) / name
        pathIRT2 = path / Path(self.config.IRT2_cars_dir) if self.config.cars_simulation else Path(self.config.IRT2_dir) / name
        pathIRT4 = path / Path(self.config.IRT4_cars_dir) if self.config.cars_simulation else Path(self.config.IRT4_dir) / name

        if self.config.sparse_IRT4_number > 0:
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

        return mask, h_coords, w_coords
    
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
        samples_gain[selected_h_coords, selected_w_coords] = gain[selected_h_coords, selected_w_coords]

        return samples_gain

   #-----------------------------------------------
    def _generate_random_samples(self, gain_img):
        # 对应 RadioUNet_s 中的随机采样逻辑
        num_samples = np.random.randint(low=10, high=300)
        # ... 生成逻辑
        pass

    def _generate_fixed_samples(self, gain_img):
        # 对应 RadioUNet_s_sprseIRT4 中的固定采样逻辑
        # 使用 map_idx 作为 seed 生成固定样本
        # ... 生成逻辑
        pass