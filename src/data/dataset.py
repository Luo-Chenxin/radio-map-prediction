from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode

class RadioMapDataset(Dataset):
    def __init__(self, config):
        self.config = config

    def __len__(self):
        return self.config.maps_number*self.config.transmitters_number

    def __getitem__(self, idx):
        map_idx = idx / self.config.transmitters_number
        tx_idx = idx % self.config.transmitters_number
        
        buildings_img = self._load_buildings(map_idx)
        tx_img = self._load_transmitters(map_idx, tx_idx)
        
        gain_img = self._load_gain(map_idx, tx_idx)
        gain_img = self._apply_threshold(gain_img)
        
        inputs = [buildings_img, tx_img]
        if self.cars_input:
            cars_img = self._load_cars(map_idx)
            inputs.append(cars_img)
        
        if self.mode == 'sparse_random':
            samples_img = self._generate_random_samples(gain_img)
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
        return read_image(str(path), ImageReadMode.RGB).float() / 255.0
    
    def _load_transimtters(self, map_idx, tx_idx):
        path = Path(self.config.root_dir) / self.config.antennas_dir / f"{map_idx}_{tx_idx}.png"
        return read_image(str(path), ImageReadMode.RGB).float() / 255.0


    def _load_gain(self, map_idx, tx_idx):
        # 包含原代码中 simulation == 'rand' (混合 DPM 和 IRT2) 的逻辑处理
        pass

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