from numpy import ndarray
from torch._tensor import Tensor
from src.trainers.base_trainer import BaseTrainer

class UnmaskedTrainer(BaseTrainer):
    """
    This Trainer is used for both RadioUnet and RadioWnet without mask.
    """
    
    def __init__(self, model, device, config):
        super().__init__(model, device, config)
    
    def _train_step(self, batch, _) -> Tensor:
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        return loss
    
    def _val_step(self, batch, _) -> float:
        loss = self._train_step(batch, _)
        return loss.item()
    
    def _test_step(self, batch, _) -> tuple[ndarray, ndarray, int]:
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        outputs = self.model(inputs)
        target = targets.detach().cpu().numpy()
        prediction = outputs.detach().cpu().numpy()
        samples_size = int(inputs.size(0))
        return target, prediction, samples_size