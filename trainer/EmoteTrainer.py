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
from torchmetrics.classification import F1Score

class EmoteTrainer(BaseTrainer):
    def __init__(self, config, model, train_loader, test_loader=None, device=None):
        super().__init__(config, model, train_loader, test_loader, device)

        self.metric_criterion = "accuracy"
        self.f1_metric = F1Score(task="binary").to(self.device)

        if config['train']:
            self.__set_trainability()
            if not config['resume']:
                self.__init_model_weights()

        self.optimizer = self._init_optimizer()
        self.lr_scheduler = self._init_scheduler()
        self.scaler = GradScaler()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        if config['resume'] or not config['train']:
            self.load_ckpt()
        self.gradient_accumulation_steps = self.config['gradient_accumulation_steps']



    def _train_on_batch(self, train_step, batch):
        # Set model to training mode
        self.model.train()
        
        with autocast('cuda',enabled=self.config['half_precision']):

            outputs = self.model(batch['images'], batch['emoji_tokens'].to(self.device), batch['text_tokens'].to(self.device))
            
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

    def _validate_on_batch(self, val_step, batch) -> tuple[dict, dict]:
        # Set the model to evaluation mode.
        self.model.eval()
        # Forward pass
        outputs = self.model(
            batch['images'],
            batch['emoji_tokens'].to(self.device),
            batch['text_tokens'].to(self.device)
        )
        # Compute loss with one-hot encoded targets
        loss = self.loss_fn(
            outputs,
            torch.nn.functional.one_hot(batch['labels'].to(self.device), num_classes=2).float()
        )
        loss = loss.mean()
        
        # Compute predictions by taking the argmax over class logits.
        preds = torch.argmax(outputs, dim=1)
        labels = batch['labels'].to(self.device)
        
        # Calculate accuracy: mean of correctly predicted labels.
        accuracy = (preds == labels).float().mean()
        # Compute F1 Score for binary classification
        self.f1_metric.update(preds, labels)
        f1_score = self.f1_metric.compute()

        # Package the loss and metric dictionaries.
        losses = {"loss": loss.item()}
        metrics = {"accuracy": accuracy.item(), "f1_score": f1_score.item()}
        return losses, metrics



    def _init_scheduler(self):
        from torch.optim.lr_scheduler import ConstantLR
        if self.config['lr_scheduler'] == 'const':
            lr_scheduler = ConstantLR(optimizer=self.optimizer, factor=1)
        else:
            raise NameError
        return lr_scheduler
    
    def __init_model_weights(self):
        # Initialize specific layers by name.
        for name, module in self.model.named_modules():
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

    def __set_trainability(self):
        #set trainability
        for param in self.model.fusion_model.swin.parameters():
            param.requires_grad = False
        for param in self.model.fusion_model.bertweet.parameters():
            param.requires_grad = False
        for param in self.model.eng_encoder.parameters():
            param.requires_grad = False