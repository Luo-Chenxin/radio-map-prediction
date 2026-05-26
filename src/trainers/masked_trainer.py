import torch
from numpy import ndarray
from torch._tensor import Tensor
from src.trainers.base_trainer import BaseTrainer

class MaskedTrainer(BaseTrainer):
    """
    This Trainer is used for both RadioUnet and RadioWnet with mask.
    """
    
    def __init__(self, model, device, id, config):
        super().__init__(model, device, id, config)
    
    def _train_step(self, batch, _) -> Tensor:
        inputs, targets, mask = batch
        inputs, targets, mask = inputs.to(self.device), targets.to(self.device), mask.to(self.device)
        outputs = self.model(inputs)
        bool_mask = mask.to(torch.bool)
        masked_outputs = torch.masked_select(outputs, bool_mask)
        masked_targets = torch.masked_select(targets, bool_mask)
        loss = self.criterion(masked_outputs, masked_targets)
        return loss
    
    def _val_step(self, batch, _) -> float:
        loss = self._train_step(batch, _)
        return loss.item()
    
    def _test_step(self, batch, _) -> tuple[ndarray, ndarray, int]:
        inputs, targets, _ = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        outputs = self.model(inputs)
        
        target = targets.detach().cpu().numpy()
        prediction = outputs.detach().cpu().numpy()
        samples_size = int(inputs.size(0))
        
        return target, prediction, samples_size