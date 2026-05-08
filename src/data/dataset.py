import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class RadioMapDataset(Dataset):
    def __init__(self, config):
        """
        A unified data loader with different behaviors configured via config.
        
        config parameter description is in config files, e.g. config/default.yaml
        """
        self.config = config
        
        self._parse_basic_params()
        
        self._setup_paths()

    def _parse_basic_params(self):
        cfg = self.config

    def _setup_paths(self):
        """根据 simulation 和 carsSimul 设置文件夹路径"""
        # 建筑物路径 (通用)
        city_map = self.config.get('cityMap', 'complete')
        if city_map == 'complete':
            self.dir_buildings = os.path.join(self.data_root, "png", "buildings_complete")
        else:
            self.dir_buildings = os.path.join(self.data_root, "png", "buildings_missing")
        
        # 发射机路径 (通用)
        self.dir_Tx = os.path.join(self.data_root, "png", "antennas")
        
        # 增益 (标签) 路径 (根据 simulation 变化)
        # 注意: 这里需要处理 carsSimul 逻辑，如果 config 中有 carsSimul=True 则加 cars 前缀
        gain_base = "gain"
        cars_prefix = "cars" if self.config.get('carsSimul', False) else ""
        
        if self.simulation in ['DPM', 'IRT2', 'IRT4']:
            self.dir_gain = os.path.join(self.data_root, gain_base, f"{cars_prefix}{self.simulation}")
        elif self.simulation == 'rand':
            # 随机混合模式需要两个路径
            self.dir_gain_DPM = os.path.join(self.data_root, gain_base, f"{cars_prefix}DPM")
            self.dir_gain_IRT2 = os.path.join(self.data_root, gain_base, f"{cars_prefix}IRT2")
        
        # 车辆路径 (如果需要作为输入)
        if self.cars_input:
            self.dir_cars = os.path.join(self.data_root, "png", "cars")

    def _build_indices(self):
        """构建数据索引列表 (对应原代码中 maps_inds 和 phase 划分逻辑)"""
        # 原代码中的 maps_inds=np.zeros(1) 逻辑
        if 'maps_inds' not in self.config or len(self.config['maps_inds']) == 1:
            maps_inds = np.arange(0, 700, dtype=np.int16)
            np.random.seed(42)
            np.random.shuffle(maps_inds)
        else:
            maps_inds = self.config['maps_inds']
        
        # 原代码中的 phase 逻辑 (train/val/test)
        phase = self.config.get('phase', 'train')
        if phase == "train":
            ind1, ind2 = 0, 500
        elif phase == "val":
            ind1, ind2 = 501, 600
        elif phase == "test":
            ind1, ind2 = 601, 699
        else: # custom
            ind1, ind2 = self.config.get('ind1', 0), self.config.get('ind2', 0)
        
        # 构建最终的索引列表
        self.data_list = []
        for map_idx in maps_inds[ind1:ind2+1]:
            for tx_idx in range(self.numTx):
                self.data_list.append((map_idx, tx_idx))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        map_idx, tx_idx = self.data_list[idx]
        
        # 1. 加载通用组件: 建筑物 & 发射机
        buildings_img = self._load_buildings(map_idx)
        tx_img = self._load_tx(map_idx, tx_idx)
        
        # 2. 加载标签 (Radio Map)
        gain_img = self._load_gain(map_idx, tx_idx)
        
        # 3. 处理阈值 (Thresh)
        gain_img = self._apply_threshold(gain_img)
        
        # 4. 构建输入 (Inputs) - 这里是差异最大的地方
        inputs = [buildings_img, tx_img]
        
        # 条件 A: 是否包含车辆输入
        if self.cars_input:
            cars_img = self._load_cars(map_idx)
            inputs.append(cars_img)
        
        # 条件 B: 根据 mode 添加稀疏采样点
        if self.mode == 'sparse_random':
            samples_img = self._generate_random_samples(gain_img)
            inputs.append(samples_img)
        elif self.mode == 'sparse_fixed':
            samples_img, mask_img = self._generate_fixed_samples(gain_img)
            # 注意: IRT4 模式通常返回 mask 作为辅助信息
            # 这里可以将 mask 存入 gain_img 的某个属性，或直接作为第 3 个返回值
            inputs.append(samples_img)
        
        # 5. 后处理与返回
        inputs = np.stack(inputs, axis=2)
        if self.config.get('transform'):
            inputs = self.config['transform'](inputs)
            gain_img = self.config['transform'](gain_img)
        
        # 根据 mode 返回不同数量的参数
        if self.mode == 'sparse_fixed':
            # IRT4 模式通常需要返回 samples mask
            return inputs, gain_img, mask_img
        else:
            return inputs, gain_img

    # --- 以下是具体的私有方法，对应原代码中的细节逻辑 ---

    def _load_buildings(self, map_idx):
        # 包含原代码中 cityMap == 'rand' 的逻辑处理
        pass

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