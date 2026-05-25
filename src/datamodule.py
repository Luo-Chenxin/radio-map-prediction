import torch
from torch.utils.data import DataLoader, Subset

class RadioSeerDataModule:
    def __init__(self, dataset_class, seed, config) -> None:
        # Calculate total length and slice index. Get the total length using a temporary dataset
        _tmp_dataset = dataset_class(config, seed)
        total_len = len(_tmp_dataset)
        
        train_size = int(config.train_ratio * total_len)
        val_size = int(config.val_ratio * total_len)
        
        # Generating index list under random permutations
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(total_len, generator=generator).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Instantiate independent Datasets for different stages and bind the corresponding indexes through Subsets.
        self.train_dataset = Subset(dataset_class(config, seed), train_indices)
        self.val_dataset = Subset(dataset_class(config, seed), val_indices)
        self.test_dataset = Subset(dataset_class(config, seed, True), test_indices)
        self.config = config

    def get_train_dataloader(self):
        return DataLoader(
        self.train_dataset, 
        batch_size=self.config.train_batch_size, 
        shuffle=True, 
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