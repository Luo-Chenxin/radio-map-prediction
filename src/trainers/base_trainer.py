import torch
import numpy as np
import logging
import time
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from src.trainers.early_stopping import EarlyStopping
from torch.utils.tensorboard import SummaryWriter

TB_STEP_INTERVAL = 10

class BaseTrainer:
    def __init__(self, model, device, id, config):
        """
        Note: Config is not required only during testing
        """
        self.model = model
        self.device = device
        self.id = id
        self.config = config
        self.global_step = 0    # Track total training steps across epochs
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._make_output_dir()
        self._set_logger()
        self._set_tensorboard()
        self._set_criterion()
        self._set_optimizer()
        self._set_scheduler()
        self._set_early_stopping()
    
    
    def _make_output_dir(self):
        self.out_dir = Path(self.config.out_dir)
        self.model_dir = self.out_dir / self.id
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.tb_dir = self.out_dir / self.id / self.timestamp
        self.tb_dir.mkdir(parents=True, exist_ok=True)

    def _set_logger(self):
        """
        [Hook Function] Subclasses can override this method to set own logger function
        """
        log_file = self.out_dir / f"{self.id}.log"
        logger_name = f"{__name__}.{self.id}"
        self.logger = logging.getLogger(logger_name)

        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.propagate = False
    
    def _set_tensorboard(self):
        """
        Initialize TensorBoard SummaryWriter under self.tb_dir
        """

        # SummaryWriter takes a string path
        self.writer = SummaryWriter(log_dir=str(self.tb_dir))
    
    def _set_criterion(self):
        """
        [Hook Function] Subclasses can override this method to set own criterion function
        """
        self.criterion = torch.nn.MSELoss()
    
    def _set_optimizer(self):
        """
        [Hook Function] Subclasses can override this method to set own optimizer function
        """
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(trainable_params, lr=self.config.learning_rate)

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
            out_dir=self.model_dir
        )
 
    def _train_step(self, batch, batch_idx) -> torch.Tensor:
        """
        [Hook Function] Subclasses must override this method to define the specific logic for a single training iteration
        
        Return:
          loss: torch.Tensor
        """
        raise NotImplementedError("Subclasses must implement the _train_step method")
    
    def _val_step(self, batch, batch_idx) -> float:
        """
        [Hook Function] Subclasses must override this method to define the specific logic for a single validation iteration
        
        Return:
          loss: float
        """
        raise NotImplementedError("Subclasses must implement the _val_step method")

    def _test_step(self, batch, batch_idx) -> tuple[np.ndarray, np.ndarray, int]:
        """
        [Hook Function] Subclasses must override this method to define the specific logic for a single testing iteration
        
        Return: 
          target: numpy.ndarray
          prediction: numpy.ndarray
          samples_size: int
        """
        raise NotImplementedError("Subclasses must implement the _test_step method")

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

            if self.global_step % TB_STEP_INTERVAL == 0:
                self.writer.add_scalar("loss/train_step", loss.item(), self.global_step)
            
            self.global_step += 1
            
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

            total_loss += loss
            pbar.set_postfix({"avg_loss": f"{total_loss / (batch_idx + 1):.4f}"})
            
        return total_loss / len(loader)

    @torch.no_grad()
    def _test_one_epoch(self, loader):
        """
        General one epoch testing process
        """

        self.model.eval()
        all_targs_list = []
        all_preds_list = []
        total_samples = 0
        inference_time = 0.0
        pbar = tqdm(loader, desc="Testing", leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            start_batch = time.time()

            target, prediction, samples_size = self._test_step(batch, batch_idx)

            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            end_batch = time.time()

            inference_time += (end_batch - start_batch)
            total_samples += samples_size

            all_targs_list.append(target)
            all_preds_list.append(prediction)

        all_targs = np.concatenate(all_targs_list, axis=0)
        all_preds = np.concatenate(all_preds_list, axis=0)
            
        return all_targs, all_preds, total_samples, inference_time

    def fit(self, train_loader, val_loader):
        """
        [Main Loop] The master switch that starts training
        """
        self.logger.info(f"Start Training... | Device: {self.device}")

        for epoch in range(self.config.epoch):

            self.logger.info(f"Start Training Epoch {epoch+1}")

            avg_train_loss = self._train_one_epoch(train_loader)
            avg_val_loss = self._validate_one_epoch(val_loader)

            self.logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            self.writer.add_scalar("loss/train_epoch", avg_train_loss, epoch)
            self.writer.add_scalar("loss/val_epoch", avg_val_loss, epoch)

            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar("hyperparameter/lr", current_lr, epoch)

            self.scheduler.step()

            self.early_stopping(avg_val_loss, self.model)
            if self.early_stopping.early_stop:
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        self.writer.close()
        self.logger.info("Training Finish")
    
    @torch.no_grad()
    def test(self, test_loader):
        """
        [Main Function] The master switch that starts testing; calculate RMSE, NMSE and inferring time per sample.
        """
        self.logger.info("Start Testing...")

        all_targs, all_preds, total_samples, total_time = self._test_one_epoch(test_loader)

        time_per_sample = total_time / total_samples if total_samples > 0 else 0

        all_targs_flat = all_targs.flatten()
        all_preds_flat = all_preds.flatten()

        mse = np.mean((all_targs_flat - all_preds_flat) ** 2)
        rmse = np.sqrt(mse)

        mean_square_targ = np.mean(all_targs_flat ** 2)
        nmse = mse / mean_square_targ

        metrics = {
            "RMSE": float(rmse),
            "NMSE": float(nmse),
            "Time_Per_Sample_Sec": float(time_per_sample),
        }
        
        self.logger.info("Test Finish.")
        self.logger.info(
            f"[Test Results] -> "
            f"RMSE: {metrics['RMSE']:.4f} | "
            f"NMSE: {metrics['NMSE']:.4f} | "
            f"Total Samples: {total_samples} | "
            f"Time/Sample: {metrics['Time_Per_Sample_Sec'] * 1000:.2f} ms"
        )
        
        return metrics