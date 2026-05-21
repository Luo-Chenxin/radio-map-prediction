import torch
from torch.utils.data import DataLoader, random_split

class RadioSeerDataModule:
    def __init__(self, dataset, seed, config) -> None:
        generator = torch.Generator().manual_seed(seed)
        train_size = int(config.train_ratio * len(dataset))
        val_size = int(config.val_ratio * len(dataset))
        test_size = len(dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, 
            [train_size, val_size, test_size], 
            generator=generator
        )
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