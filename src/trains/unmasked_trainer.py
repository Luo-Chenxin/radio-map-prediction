from src.trains.base_trainer import BaseTrainer

class UnmaskedTrainer(BaseTrainer):
    """
    This Trainer is used for both RadioUnet and RadioWnet without mask.
    """
    
    def __init__(self, model, device, config):
        super().__init__(model, device, config)
    
    def _train_step(self, batch, _):
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        return loss
    
    def _val_step(self, batch, _):
        return self._train_step(batch, _)