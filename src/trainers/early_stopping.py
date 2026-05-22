import torch
from pathlib import Path

BEST_MODEL_PARTERN = "best_model.pt"

class EarlyStopping:
    def __init__(self, patience, delta, out_dir):
        self.patience = patience
        self.delta = delta
        self.model_path = Path(out_dir) / BEST_MODEL_PARTERN
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_model(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_model(model)
            self.counter = 0

    def save_model(self, model):
        torch.save(model.state_dict(), self.model_path)