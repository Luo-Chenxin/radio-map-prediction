import torch
from torch.utils.data import DataLoader, Subset

MAX_TX_IRT4 = 2

class RadioSeerDataModule:
    def __init__(self, dataset_class, config_load, config_data, seed) -> None:
        # Divide map_idx into groups (ensuring that the three map_idx values ​​do not overlap).
        self.config = config_load
        total_maps = config_data.maps_number
        train_map_size = int(config_load.train_ratio * total_maps)
        val_map_size = int(config_load.val_ratio * total_maps)
        
        # Use seed to ensure the reproducibility of map partitioning.
        g = torch.Generator().manual_seed(seed)
        shuffled_maps = torch.randperm(total_maps, generator=g).tolist()
        
        train_maps = set(shuffled_maps[:train_map_size])
        val_maps = set(shuffled_maps[train_map_size:train_map_size + val_map_size])
        test_maps = set(shuffled_maps[train_map_size + val_map_size:])
        
        #idx = map_idx * transmitters_number + tx_idx
        train_indices = []
        val_indices = []
        test_indices = []
        
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
                for tx_idx in range(MAX_TX_IRT4): 
                    idx = map_idx * max_tx + tx_idx
                    test_indices.append(idx)

        # Shuffle train dataset
        train_indices = [train_indices[i] for i in torch.randperm(len(train_indices), generator=g).tolist()]
        
        self.train_dataset = Subset(dataset_class(config_data, seed), train_indices)
        self.val_dataset = Subset(dataset_class(config_data, seed), val_indices)
        self.test_dataset = Subset(dataset_class(config_data, seed, is_test=True), test_indices)

    def get_train_dataloader(self):
        return DataLoader(
        self.train_dataset, 
        batch_size=self.config.train_batch_size,
        num_workers=self.config.num_workers)
    
    def get_val_dataloader(self):
        return DataLoader(
        self.val_dataset, 
        batch_size=self.config.val_batch_size,
        num_workers=self.config.num_workers)
    
    def get_test_dataloader(self):
        return DataLoader(
        self.test_dataset, 
        batch_size=self.config.test_batch_size,
        num_workers=self.config.num_workers)