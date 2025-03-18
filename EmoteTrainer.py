from base_trainer import BaseTrainer
import torch
import torch.amp as amp
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import random
from torch.nn.parameter import Parameter
from torch.amp import autocast, GradScaler
import os
import logging
import tqdm
from torchvision.transforms import InterpolationMode, Resize

class EmoteTrainer(BaseTrainer):
    def __init__(self, config, model, train_loader, test_loader=None, device=None):
        super().__init__(config, model, train_loader, test_loader, device)

        self.metric_criterion = None

        #set trainability
        for param in model.parameters():
            param.requires_grad = True   # full finetune

        self.optimizer = self.init_optimizer()
        self.lr_scheduler = self.init_scheduler()
        self.scaler = GradScaler()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        if config['resume'] or not config['train']:
            self.load_ckpt()
        self.gradient_accumulation_steps = self.config['gradient_accumulation_steps']



    def train_on_batch(self, train_step, batch):
        # Set model to training mode
        self.model.train()
        
        # If the batch contains a 'strategies' field, separate it out and move to device.
        if 'strategies' in batch:
            strategies = batch.pop('strategies').to(self.device)
        if 'images' in batch:
            images: list = batch.pop('images').to(self.device)

        # Move the remaining batch data to the device.
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with autocast('cuda',enabled=self.config['half_precision']):

            # Forward pass: get model outputs
            outputs = self.model(**batch)
            
            # Compute loss:
            # Here we assume binary classification, so we convert labels to one-hot encoding.
            loss = self.loss_fn(
                outputs.logits,
                torch.nn.functional.one_hot(batch['labels'].long(), num_classes=2).float()
            )
            loss = loss.mean()

            loss = loss / self.gradient_accumulation_steps

        
        self.scaler.scale(loss).backward()
        # Perform optimization step
        if (train_step + 1) % self.gradient_accumulation_steps == 0:
            # self.optimizer.step()
            self.scaler.step(self.optimizer)
            self.lr_scheduler.step()
            # self.scaler.step(self.lr_scheduler)
            self.scaler.update()
            self.optimizer.zero_grad()

        self.n_batch_in_epoch += 1

        return loss



    def init_scheduler(self):
        from torch.optim.lr_scheduler import ConstantLR
        if self.config['lr_scheduler'] == 'const':
            lr_scheduler = ConstantLR(optimizer=self.optimizer, factor=1)
        else:
            raise NameError
        return lr_scheduler