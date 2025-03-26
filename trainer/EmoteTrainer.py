from trainer.BaseTrainer import BaseTrainer
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
        if not config['resume'] and config['train']:
            # Initialize specific layers by name.
            for name, module in model.named_modules():
                if "classifier" in name and isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                    print(f"Initialized {name} with Kaiming normal.")
                elif "decoder" in name and isinstance(module, nn.TransformerDecoder):
                    # Iterate over all parameters in the TransformerDecoder.
                    for param in module.parameters():
                        if param.dim() > 1:
                            nn.init.kaiming_normal_(param)
                    print(f"Initialized {name} with Kaiming normal.")
                elif "swin_proj" in name and isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                    print(f"Initialized {name} with Kaiming normal.")
                elif "self_attention" in name and isinstance(module, nn.TransformerEncoder):
                    for param in module.parameters():
                        if param.dim() > 1:
                            nn.init.kaiming_normal_(param)
                            print(f"Initialized {name} with Kaiming normal.")

        #set trainability
        for param in self.model.fusion_model.swin.parameters():
            param.requires_grad = False
        for param in self.model.fusion_model.bertweet.parameters():
            param.requires_grad = False
        for param in self.model.eng_encoder.parameters():
            param.requires_grad = False


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
        
        with autocast('cuda',enabled=self.config['half_precision']):

            # Forward pass: get model outputs
            outputs = self.model(batch['images'], batch['emoji_tokens'].to(self.device), batch['text_tokens'].to(self.device))
            
            # Compute loss:
            # Here we assume binary classification, so we convert labels to one-hot encoding.
            loss = self.loss_fn(
                outputs,
                torch.nn.functional.one_hot(batch['labels'].to(self.device), num_classes=2).float()
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