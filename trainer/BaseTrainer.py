import os
import warnings
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from pathlib import Path
from misc.misc import RunningAverageDict, is_rank_zero, skip_first_batches


class BaseTrainer:
    def __init__(self, config, model, train_loader, test_loader=None, device=None):
        """ Base Trainer class for training a model."""
        
        self.config = config
        self.metric_criterion = None    # TODO: need to override !!!
        if device is None:
            self.device = torch.device(
                'cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = torch.device(device)
        self.n_batch_in_epoch = 0
        self.epoch = 0
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.run_index = 0

        self.model = model
        ############################# initialize save_dir ########################################################
        # Initialize the directory where runs are saved.
        runs_dir = Path('runs')

        # Determine if the current process should handle saving (primary process)
        is_primary = (not self.config['distributed'] or is_rank_zero(self.config))

        if is_primary and self.config['train']:
            # Find the next available run index by incrementing until a non-existent directory is found.
            while (runs_dir / f"run{self.run_index}").exists():
                self.run_index += 1

            if self.config['save_dir'] is None:
                # If no specific save_dir is provided, create a new one within the runs directory.
                self.save_dir = runs_dir / f"run{self.run_index}"
                self.save_dir.mkdir(exist_ok=False, parents=True)
            else:
                # Otherwise, use the provided save_dir from the configuration.
                self.save_dir = Path(self.config['save_dir'])
                if not self.save_dir.exists():
                    self.save_dir.mkdir(exist_ok=False, parents=True)
        
        self.experiment_id = f"{self.config['version_name']}_{self.run_index}"


    def load_ckpt(self, checkpoint_dir="./runs", ckpt_type="best"):
        """
        Load a checkpoint based on configuration settings.
        
        Args:
            checkpoint_dir (str): Directory to search for checkpoint files.
            ckpt_type (str): Type of checkpoint to load (e.g., "best").
        """
        import glob
        import os

        if self.config['checkpoint'] is not None:
            checkpoint = self.config['checkpoint']
        elif self.config['ckpt_pattern'] is not None:
            pattern = self.config['ckpt_pattern']
            # Search for checkpoint files matching the pattern and type.
            matches = glob.glob(os.path.join(checkpoint_dir, f"*{pattern}*{ckpt_type}*"))
            if not matches:
                raise ValueError(f"No matches found for the pattern {pattern}")
            checkpoint = matches[0]
        else:
            # No checkpoint configuration provided; nothing to load.
            return

        # Load weights if resuming or if not in training mode.
        if self.config['resume'] or not self.config['train']:
            self.load_wts(checkpoint)


    def load_wts(self, checkpoint):
        """
        Load model weights from a checkpoint file.
        
        Args:
            checkpoint (str): Path to the checkpoint file.
        """
        # Load checkpoint data using the device mapping.
        checkpoint_data = torch.load(checkpoint, map_location=self.device)

        self.epoch = checkpoint_data["epoch"]
        self.n_batch_in_epoch = checkpoint_data["n_batch_in_epoch"]

        # Load model weights based on multi-GPU configuration.
        if self.config['multigpu']:
            self.model.module.load_state_dict(checkpoint_data["model"])
        else:
            self.model.load_state_dict(checkpoint_data["model"])

        # Only the primary process prints the confirmation.
        if (not self.config['distributed']) or self.config.get('rank', 0) == 0:
            print(f"Loaded weights from {checkpoint}")

    def init_optimizer(self):
        params = self.model.parameters()
        return optim.AdamW(params, lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])

    def init_scheduler(self):
        raise NotImplementedError
    
    def train_on_batch(self, train_step, batch):
        raise NotImplementedError

    def validate_on_batch(self, val_step, batch)->tuple[Dict|None, Dict|None]: # loss, metric
        raise NotImplementedError

    def raise_if_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError(f"loss is NaN, Stopping training")

    @property
    def iters_per_epoch(self):
        return len(self.train_loader)

    @property
    def total_iters(self):
        return self.config['epochs'] * self.iters_per_epoch

    def should_early_stop(self):
        if self.config['early_stop'] != False and self.step > self.config['early_stop']:
            return True

    def train(self):
        """
        Train the model, logging metrics and saving checkpoints along the way.
        """
        print("Training")

        # Setup logging and checkpoint conditions.
        self.should_write = not self.config['distributed'] or is_rank_zero(self.config)
        self.should_log = self.should_write  # This can be expanded later if needed.
        
        if self.should_log:
            wandb.init(
                name=self.experiment_id,
                config=self.config,
                dir=str(self.save_dir),
                settings=wandb.Settings(start_method="fork")
            )

        self.step = 0
        best_loss = np.inf
        validate_interval = int(self.config['validate_every'] * self.iters_per_epoch)

        for epoch in range(self.epoch, self.config['epochs']):
            if self.should_early_stop():
                break

            self.epoch = epoch

            # Log the beginning of a new epoch.
            if self.should_log:
                wandb.log({"Epoch": epoch}, step=self.step)

            # Prepare training data loader with skipped batches if needed.
            train_loader = skip_first_batches(self.train_loader, self.n_batch_in_epoch)

            # Setup progress bar for the primary process.
            if is_rank_zero(self.config):
                pbar = tqdm(
                    enumerate(train_loader),
                    desc=f"Epoch: {epoch + 1}/{self.config['epochs']}. Loop: Train",
                    total=len(train_loader)
                )
            else:
                pbar = enumerate(self.train_loader)

            # Process each batch.
            for i, batch in pbar:
                if self.should_early_stop():
                    print("Early stopping")
                    break

                loss = self.train_on_batch(i, batch)
                self.raise_if_nan(loss)

                # Update progress bar description.
                if is_rank_zero(self.config):
                    pbar.set_description(
                        f"Epoch: {epoch + 1}/{self.config['epochs']}. Loop: Train. Loss: {loss.item()}"
                    )

                # Log training loss every 50 steps.
                if self.should_log and self.step % 50 == 0:
                    wandb.log({'Train/loss': loss.item()}, step=self.step)

                self.step += 1
                ################################ validate ####################################################
                # Perform validation if a test loader is available.
                if self.test_loader and self.step % validate_interval == 0:
                    metrics, test_losses = self.validate()

                    if self.should_log:
                        # Log test losses and metrics.
                        wandb.log({f"Test/{name}": tloss for name, tloss in test_losses.items()}, step=self.step)
                        wandb.log({f"Metrics/{k}": v for k, v in metrics.items()}, step=self.step)

                        # Save a checkpoint if a new best loss is achieved.
                        if metrics[self.metric_criterion] < best_loss and self.should_write:
                            self.save_checkpoint(f"{self.experiment_id}_best.pth")
                            best_loss = metrics[self.metric_criterion]

                    # Synchronize processes in distributed training.
                    if self.config['distributed']:
                        dist.barrier()

            # Save checkpoint at regular epoch intervals.
            if (epoch + 1) % self.config['save_every'] == 0:
                if self.should_write:
                    self.save_checkpoint("latest.pth")
                    print(f"Saved checkpoint to {str(self.save_dir)}/latest.pth")
                if self.config['distributed']:
                    dist.barrier()

            # Reset batch counter for the next epoch.
            self.n_batch_in_epoch = 0

        # Final logging step.
        self.step += 1

        
    @torch.no_grad()
    def validate(self):

        losses_avg = RunningAverageDict()
        metrics_avg = RunningAverageDict()
        for i, batch in tqdm(enumerate(self.test_loader), desc=f"Loop: Validation",
                                total=len(self.test_loader), disable=not is_rank_zero(self.config)):
            
            losses, metrics = self.validate_on_batch(i, batch)

            if losses:
                losses_avg.update(losses)
            if metrics:
                metrics_avg.update(metrics)

        return metrics_avg.get_value(), losses_avg.get_value()

    def save_checkpoint(self, filename):
        if not self.should_write:
            return

        fpath = self.save_dir / filename
        m = self.model.module if self.config['multigpu'] else self.model
        torch.save(
            {
                "model": m.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": self.epoch,
                "n_batch_in_epoch": self.n_batch_in_epoch,
                "config": self.config,
            }, str(fpath))
