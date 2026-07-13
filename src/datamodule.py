import torch
from torch.utils.data import DataLoader, Subset
from abc import ABC, abstractmethod

class _BaseDataModule(ABC):
    """
    The common base class for all DataModules. 
    Subclasses only need to implement `_build_datasets` to 
    populate the `train_dataset`, `val_dataset`, and `test_dataset` attributes;
    the logic for constructing DataLoaders (such as `batch_size` and `num_workers`) 
    is handled centrally in the base class, eliminating the need for repetitive code.
    """
    def __init__(self, dataset_class, config_load, config_data, seed):
        self.config_load = config_load
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._build_datasets(dataset_class, config_data, seed)

    @abstractmethod
    def _build_datasets(self, dataset_class, config_data, seed):
        """
        Subclasses must implement: populating 
        self.train_dataset / self.val_dataset / self.test_dataset.
        """
        ...

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config_load.train_batch_size,
            shuffle=True,
            num_workers=self.config_load.num_workers)

    def get_val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config_load.val_batch_size,
            num_workers=self.config_load.num_workers)

    def get_test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config_load.test_batch_size,
            num_workers=self.config_load.num_workers)

class RadioSeerDataModule(_BaseDataModule):
    MAX_TX_IRT4 = 2

    def _build_datasets(self, dataset_class, config_data, seed):
        # Divide map_idx into groups (ensuring that the three map_idx values ​​do not overlap).
        total_maps = config_data.maps_number
        train_map_size = int(self.config_load.train_ratio * total_maps)
        val_map_size = int(self.config_load.val_ratio * total_maps)
        
        # Use seed to ensure the reproducibility of map partitioning.
        g = torch.Generator().manual_seed(seed)
        shuffled_maps = torch.randperm(total_maps, generator=g).tolist()
        
        train_maps = set(shuffled_maps[:train_map_size])
        val_maps = set(shuffled_maps[train_map_size:train_map_size + val_map_size])
        test_maps = set(shuffled_maps[train_map_size + val_map_size:])
        
        #idx = map_idx * transmitters_number + tx_idx
        train_indices, val_indices, test_indices = [], [], []
        
        max_tx = config_data.transmitters_number 
        
        for map_idx in range(total_maps):
            if map_idx in train_maps:
                # Train: tx_idx is in [0,max_tx)
                for tx_idx in range(max_tx):
                    idx = map_idx * max_tx + tx_idx
                    train_indices.append(idx)
                    
            elif map_idx in val_maps:
                # Val: tx_idx is in [0,max_tx)
                for tx_idx in range(max_tx):
                    idx = map_idx * max_tx + tx_idx
                    val_indices.append(idx)
                    
            elif map_idx in test_maps:
                # Test: tx_idx is in [0,2)
                for tx_idx in range(self.MAX_TX_IRT4): 
                    idx = map_idx * max_tx + tx_idx
                    test_indices.append(idx)
        
        self.train_dataset = Subset(dataset_class(config_data, seed), train_indices)
        self.val_dataset = Subset(dataset_class(config_data, seed), val_indices)
        self.test_dataset = Subset(dataset_class(config_data, seed, is_test=True), test_indices)

class H5DataModule(_BaseDataModule):
    def _build_datasets(self, dataset_class, config_data, seed):
        dataset = dataset_class(config_data)
        total_size = len(dataset)
        train_size = int(self.config_load.train_ratio * total_size)
        val_size = int(self.config_load.val_ratio * total_size)

        g = torch.Generator().manual_seed(seed)
        shuffled_indices = torch.randperm(total_size, generator=g).tolist()

        self.train_dataset = Subset(dataset, shuffled_indices[:train_size])
        self.val_dataset = Subset(dataset, shuffled_indices[train_size:train_size + val_size])
        self.test_dataset = Subset(dataset, shuffled_indices[train_size + val_size:])
