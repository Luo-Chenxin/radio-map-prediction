import torch

BEST_MODEL_PARTERN = "best_model.pt"

class EarlyStopping:
    def __init__(self, patience, delta, out_dir):
        self.patience = patience
        self.delta = delta
        self.out_dir = out_dir
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.out_dir / BEST_MODEL_PARTERN)