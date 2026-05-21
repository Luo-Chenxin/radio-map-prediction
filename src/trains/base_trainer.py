import torch
import logging
from tqdm import tqdm
from src.trains.early_stopping import EarlyStopping

class BaseTrainer:
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config

        self._set_logger()
        self._set_criterion()
        self._set_optimizer()
        self._set_scheduler()
        self._set_early_stopping()
    
    def _set_logger(self):
        """
        [Hook Function] Subclasses can override this method to set own logger function
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.log_file, mode='w'),
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _set_criterion(self):
        """
        [Hook Function] Subclasses can override this method to set own criterion function
        """
        self.criterion = torch.nn.MSELoss()
    
    def _set_optimizer(self):
        """
        [Hook Function] Subclasses can override this method to set own optimizer function
        """
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def _set_scheduler(self):
        """
        [Hook Function] Subclasses can override this method to set own scheduler function
        """
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=self.config.scheduler.step_size, 
            gamma=self.config.scheduler.gamma)
    
    def _set_early_stopping(self):
        """
        [Hook Function] Subclasses can override this method to set own early stopping function
        """
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stop.patience, 
            delta=self.config.early_stop.delta, 
            out_dir=self.config.out_dir
        )
 
    def _train_step(self, batch, batch_idx):
        """
        [Hook Function] Subclasses must override this method to define the specific logic for a single training iteration
        """
        raise NotImplementedError("Subclasses must implement the _train_step method")
    
    def _val_step(self, batch, batch_idx):
        """
        [Hook Function] Subclasses must override this method to define the specific logic for a single validation iteration
        """
        raise NotImplementedError("Subclasses must implement the _val_step method")

    def _train_one_epoch(self, loader):
        """
        General one epoch training process
        """
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc="Training", leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()

            loss = self._train_step(batch, batch_idx)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"avg_loss": f"{total_loss / (batch_idx + 1):.4f}"})
            
        return total_loss / len(loader)

    @torch.no_grad()
    def _validate_one_epoch(self, loader):
        """
        General one epoch validation process
        """

        self.model.eval()
        total_loss = 0.0
        pbar = tqdm(loader, desc="Validating", leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            loss = self._val_step(batch, batch_idx)

            total_loss += loss.item()
            pbar.set_postfix({"avg_loss": f"{total_loss / (batch_idx + 1):.4f}"})
            
        return total_loss / len(loader)

    # [Main Loop] The master switch that starts training
    def fit(self, train_loader, val_loader, epochs):
        self.logger.info(f"Start Training | Device: {self.device}")

        for epoch in range(epochs):

            self.logger.info(f"Start Training Epoch {epoch+1}")

            avg_train_loss = self._train_one_epoch(train_loader)
            avg_val_loss = self._validate_one_epoch(val_loader)

            self.logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            self.scheduler.step()

            self.early_stopping(avg_val_loss, self.model)
            if self.early_stopping.early_stop:
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        self.logger.info("Training Finish")