import os
import random
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from emote_config import Emote_Config


def save_model(model, tokenizer, save_path):
    """Helper function to save any model type"""
    if hasattr(model, 'save_pretrained'):
        # For standard Hugging Face models
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
    else:
        # Fallback for other models - just save the state dict
        os.makedirs(save_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
        if tokenizer is not None:
            tokenizer.save_pretrained(save_path)

class EmoteTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device, new_emote_config, tokenizer):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.new_emote_config = new_emote_config
        self.tokenizer = tokenizer

        self.batch_size = self.new_emote_config.bs
        self.model_name = self.new_emote_config.model_name
        self.portion = self.new_emote_config.portion
        self.seed = self.new_emote_config.seed
        self.strategy_map = self.new_emote_config.strategy_map
        self.strategy_map_inverted = {v: k for k, v in self.strategy_map.items()}
        self.max_epochs = 10

        # Configure optimizer for trainable parameters
        trainable_params = []
        
        # Check which parameters exist and are trainable
        if hasattr(model, 'sequence_model'):
            trainable_params.extend(model.sequence_model.parameters())
            print(f"Training sequence_model: {sum(p.numel() for p in model.sequence_model.parameters())} params")
        elif hasattr(model, 'rnn'):
            trainable_params.extend(model.rnn.parameters())
            print(f"Training rnn: {sum(p.numel() for p in model.rnn.parameters())} params")
        
        if hasattr(model, 'projection'):
            trainable_params.extend(model.projection.parameters())
            print(f"Training projection: {sum(p.numel() for p in model.projection.parameters())} params")
        
        if hasattr(model, 'classifier'):
            trainable_params.extend(model.classifier.parameters())
            print(f"Training classifier: {sum(p.numel() for p in model.classifier.parameters())} params")
        
        # Fallback to all trainable parameters if nothing found
        if not trainable_params:
            print("Warning: No specific trainable parameters found, training all that require_grad")
            trainable_params = [p for p in model.parameters() if p.requires_grad]
        
        total_params = sum(p.numel() for p in trainable_params)
        print(f"Total trainable parameters: {total_params}")
        
        self.optimizer = AdamW(trainable_params, lr=2e-4, weight_decay=0.01)
        
        self.best_val_loss = np.inf
        trainable_params.extend(model.rnn.parameters())
        trainable_params.extend(model.classifier.parameters())

        # Print parameter counts
        rnn_params = sum(p.numel() for p in model.rnn.parameters())
        cls_params = sum(p.numel() for p in model.classifier.parameters())
        print(f"Training {rnn_params} RNN params and {cls_params} classifier params")

        self.optimizer = AdamW(trainable_params, lr=2e-4, weight_decay=0.01)
        
        self.best_val_loss = np.inf
        self.best_val_acc = 0
        self.best_epoch = -1

        self.save_dir = self.new_emote_config.sft_dir
        self.fp_txt = '{}/portion_{}_seed_{}.txt'.format(self.save_dir, self.portion, self.seed)
        self.fp_csv = '{}/portion_{}_seed_{}.csv'.format(self.save_dir, self.portion, self.seed)
        self.fp_csv_best = '{}/portion_{}_seed_{}_best.csv'.format(self.save_dir, self.portion, self.seed)

    def train(self, epochs, debug_overfit_batch=False):
        # --- Debug: Overfit Single Batch ---
        if debug_overfit_batch:
            print("\n!!! DEBUG MODE: Attempting to overfit a single batch !!!\n")
            try:
                single_batch = next(iter(self.train_loader))
                print(f"Debug: Loaded single batch. Keys: {single_batch.keys()}")
            except StopIteration:
                print("Error: Could not get a batch from train_loader.")
                return

            # --- Data Inspection (Keep this) ---
            print(f"Debug: Image shape: {single_batch['images'].shape}")
            print(f"Debug: Labels shape: {single_batch['labels'].shape}")
            print(f"Debug: Unique labels in batch: {torch.unique(single_batch['labels'])}")
            print(f"Debug: Strategy shape: {single_batch['strategies'].shape}")
            if torch.isnan(single_batch['images']).any():
                 print("ERROR: NaNs found in input images!")
                 return
            print("Debug: Data inspection passed.")
            # --- End Data Inspection ---


            # Move the single batch to device once
            single_batch_strategies = single_batch.pop('strategies').to(self.device) # Pop strategies
            single_batch_labels = single_batch['labels'].to(self.device)
            single_batch_to_model = {k: v.to(self.device) for k, v in single_batch.items()}


            # Initialize the *actual target model* (Partial Swin + Transformer Head)
            print("Initializing Partial Swin + Transformer Head model for debug...")
            from multimodal_model import EmoteMultimodalModel
            # Ensure config is defined
            debug_model = EmoteMultimodalModel(config, num_labels=2)
            debug_model.to(device)
            debug_model.train() # Set to train mode


            # --- Parameter Check Setup ---
            # Find *any* trainable parameter to track its change
            tracked_param_name = None
            tracked_param_before = None
            for name, param in debug_model.named_parameters():
                 if param.requires_grad:
                     tracked_param_name = name
                     tracked_param_before = param.data.clone()
                     print(f"Debug: Tracking parameter '{tracked_param_name}' for updates.")
                     break
            if tracked_param_name is None:
                 print("ERROR: No trainable parameters found in the model!")
                 return
            # --- End Parameter Check Setup ---

            num_overfit_steps = 50
            overfit_pbar = tqdm(range(num_overfit_steps), desc="Overfitting Single Batch", unit="step")

            for step in overfit_pbar:
                # Re-initialize optimizer for this specific debug model instance
                # Use the same differential LR setup as in __init__
                lr_backbone = 2e-6
                lr_head = 1e-4
                optimizer_grouped_parameters = []
                swin_params = []
                head_params = []
                for name, param in debug_model.named_parameters():
                     if not param.requires_grad: continue
                     if 'image_model.features.6.' in name or 'image_model.features.7.' in name:
                          swin_params.append(param)
                     elif 'pos_embed_after' in name or 'rnn.' in name or \
                          'final_norm.' in name or 'classifier.' in name:
                          head_params.append(param)
                     else: head_params.append(param) # Default to head group
                if swin_params: optimizer_grouped_parameters.append({'params': swin_params, 'lr': lr_backbone})
                if head_params: optimizer_grouped_parameters.append({'params': head_params, 'lr': lr_head})
                debug_optimizer = AdamW(optimizer_grouped_parameters, weight_decay=0.01)
                # --- End Debug Optimizer Setup ---

                debug_optimizer.zero_grad() # Use the debug optimizer

                outputs = debug_model(**single_batch_to_model)

                # --- Logits Inspection (Keep this) ---
                if step % 10 == 0:
                     print(f"\nDebug Step {step+1}: Logits sample: {outputs.logits.flatten()[:10].detach().cpu().numpy()}")
                     if torch.isnan(outputs.logits).any():
                          print("ERROR: NaNs found in logits!")
                          break
                # --- End Logits Inspection ---

                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(outputs.logits, single_batch_labels)

                # --- Loss Inspection (Keep this) ---
                if torch.isnan(loss):
                     print("ERROR: Loss is NaN!")
                     break
                if step % 10 == 0:
                    print(f"Debug Step {step+1}: Loss: {loss.item():.6f}")
                # --- End Loss Inspection ---

                loss.backward()

                # --- CORRECTED Gradient Inspection ---
                # Check ALL trainable parameters now
                if step % 10 == 0: # Log grads periodically
                    print(f"--- Debug Step {step+1}: Gradient Stats ---")
                    total_grad_norm_sq = 0.0
                    max_grad = -float('inf')
                    min_grad = float('inf')
                    nan_grads = False
                    zero_grads_count = 0
                    num_trainable = 0
                    none_grads = False
                    for name, param in debug_model.named_parameters():
                        if param.requires_grad:
                            num_trainable += 1
                            if param.grad is None:
                                print(f"  ERROR: Grad is None for trainable param: {name}")
                                none_grads = True
                            else:
                                grad_norm = param.grad.data.norm(2)
                                total_grad_norm_sq += grad_norm.item() ** 2
                                max_grad = max(max_grad, param.grad.data.max().item())
                                min_grad = min(min_grad, param.grad.data.min().item())
                                if torch.isnan(param.grad).any(): nan_grads = True
                                if grad_norm.item() < 1e-9: zero_grads_count +=1 # Count near-zero grads
                    if not none_grads: # Avoid sqrt of zero if grads are None
                        total_grad_norm = total_grad_norm_sq ** 0.5
                        print(f"  Total Grad Norm (before clip): {total_grad_norm:.4e}")
                        print(f"  Max Grad Val: {max_grad:.4e}, Min Grad Val: {min_grad:.4e}")
                        print(f"  NaN Grads Found: {nan_grads}")
                        print(f"  Near-Zero Grads: {zero_grads_count}/{num_trainable}")
                    else:
                        print("  Skipping grad norm calculation due to None grads.")
                    print("------------------------------------")
                if none_grads: # Stop if gradients are missing
                     print("Stopping debug due to None gradients.")
                     break
                # --- End CORRECTED Gradient Inspection ---

                torch.nn.utils.clip_grad_norm_(debug_model.parameters(), 1.0) # Clip grads of debug model
                debug_optimizer.step() # Step the debug optimizer

                # --- Parameter Update Check (Keep this) ---
                if step % 10 == 0 or step == num_overfit_steps - 1:
                    tracked_param_after = None
                    for name, param in debug_model.named_parameters():
                         if name == tracked_param_name:
                              tracked_param_after = param.data.clone()
                              break
                    if tracked_param_after is not None:
                         param_diff = torch.sum(torch.abs(tracked_param_after - tracked_param_before)).item()
                         print(f"Debug Step {step+1}: Param '{tracked_param_name}' change since start: {param_diff:.6f}")
                         if param_diff == 0 and step > 0:
                              print(f"  Warning: Tracked Parameter did not change after optimizer step!")
                    else:
                         print(f"  Warning: Could not re-find tracked parameter '{tracked_param_name}'")
                # --- End Parameter Update Check ---

                _, preds = torch.max(outputs.logits, 1)
                acc = (preds == single_batch_labels).float().mean().item()
                overfit_pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{acc:.2f}"})


            print("\n!!! DEBUG MODE: Finished overfitting attempt !!!\n")
            # --- Final Parameter Check (Keep this) ---
            tracked_param_final = None
            for name, param in debug_model.named_parameters():
                 if name == tracked_param_name:
                      tracked_param_final = param.data.clone()
                      break
            if tracked_param_final is not None and tracked_param_before is not None:
                 final_diff = torch.sum(torch.abs(tracked_param_final - tracked_param_before)).item()
                 print(f"Debug: Total change in tracked parameter '{tracked_param_name}' after {num_overfit_steps} steps: {final_diff:.6f}")
            # --- End Final Parameter Check ---
            return # Exit after debug run

        # Continue with normal training
        # Original training loop (Keep this unchanged)
        epoch_pbar = tqdm(range(epochs), desc="Training", unit="epoch")
        
        for epoch in epoch_pbar:
            # Update the progress bar description
            epoch_pbar.set_description(f"Epoch {epoch+1}/{epochs}")
            
            # Run training epoch with progress bar
            self._run_epoch(epoch, train=True)
            
            # Run validation with progress bar
            val_acc, val_acc_strategy = self._run_epoch(epoch=epoch, train=False, loader=self.val_loader)
            
            # Run test with progress bar
            test_acc, test_acc_strategy = self._run_epoch(epoch=epoch, train=False, loader=self.test_loader)
            
            # Update progress bar with current metrics
            epoch_pbar.set_postfix({
                'val_acc': f"{val_acc:.2f}%", 
                'test_acc': f"{test_acc:.2f}%"
            })
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                small_dict = {}
                small_dict['Overall'] = test_acc
                small_dict['Direct'] = test_acc_strategy[0]
                small_dict['Metaphorical'] = test_acc_strategy[1]
                small_dict['Semantic list'] = test_acc_strategy[2]
                small_dict['Reduplication'] = test_acc_strategy[3]
                small_dict['Single'] = test_acc_strategy[4]
                small_dict['Negative'] = test_acc_strategy[5]
                # save it into a csv
                with open(self.fp_csv_best, 'w') as f:
                    f.write('strategy,accuracy\n')
                    for key in small_dict.keys():
                        f.write('{},{}\n'.format(key, small_dict[key]))

            # if epoch is 0, then save the model
            if epoch == 0 and self.portion == 1: ## we now only save fully trained models
                model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                save_path = "{}/seed_{}_epoch_{}.pt".format(
                    self.new_emote_config.model_save_dir, self.seed, epoch)
                save_model(model_to_save, self.tokenizer, save_path)
                print(f"Saving model checkpoint to {save_path}")
            if epoch > 0 and val_acc > self.best_val_acc and self.portion == 1:
                model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                save_path = "{}/seed_{}_epoch_{}.pt".format(
                    self.new_emote_config.model_save_dir, self.seed, epoch)
                save_model(model_to_save, self.tokenizer, save_path)
                print(f"Saving model checkpoint to {save_path}")

        # Print the epoch with the best validation loss
        print(f'Best Epoch: {self.best_epoch}')

        # Print the best val accuracy and its epoch number
        print(f'Best Val Accuracy: {self.best_val_acc} at Epoch {self.best_epoch}')
        # Write it into fp_txt
        with open(self.fp_txt, 'a') as f:
            f.write(f'Best Val Accuracy: {self.best_val_acc} at Epoch {self.best_epoch}\n')


    def _run_epoch(self, epoch, train, loader=None, debug_epoch=None):
        if train:
            self.model.train()
            loader = self.train_loader
            str_ = 'Train'
        else:
            self.model.eval()
            str_ = 'Valid' if loader == self.val_loader else 'Test'

        total_loss = 0
        predictions, true_labels, strategy_types = [], [], []
        strategies_correct = [0]*7
        strategies_total = [0]*7

        # Create a progress bar for batches
        batch_pbar = tqdm(loader, desc=f"{str_} Epoch {epoch+1}", leave=False)
        
        for batch_idx, batch in enumerate(batch_pbar):
            strategies = batch.pop('strategies').to(self.device)

            # Prepare batch for the model
            batch_to_model = {}
            # Copy only relevant keys for an image-only model
            image_model_keys = ['images', 'labels', 'strategies']
            for key in image_model_keys:
                if key in batch:
                    batch_to_model[key] = batch[key]

            with torch.set_grad_enabled(train):
                # Pass labels to the model's forward pass
                # Ensure your model's forward accepts 'labels' when needed
                outputs = self.model(**batch_to_model) # Pass only relevant keys
                
                # Use the loss calculated and returned by the model
                # Requires model's forward to calculate loss when labels are passed
                if outputs.loss is None:
                     # Fallback if model didn't calculate loss (e.g., during eval without labels)
                     # This indicates an issue if it happens during training
                     if train: raise ValueError("Model did not return loss during training.")
                     # For eval, calculate loss manually if needed for logging (using CrossEntropy)
                     loss_fct_eval = nn.CrossEntropyLoss()
                     loss = loss_fct_eval(outputs.logits, batch_to_model['labels'])
                else:
                     loss = outputs.loss

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    # Add Gradient Clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # Clip norm at 1.0
                    self.optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs.logits, 1)
            predictions.extend(preds.cpu().numpy().tolist())
            # Use the original labels tensor we moved to device earlier for consistency
            true_labels.extend(batch_to_model['labels'].cpu().numpy().tolist()) 
            strategy_types.extend(strategies.cpu().numpy().tolist())

            current_true_labels = batch_to_model['labels'].cpu().numpy().tolist() # Use the same labels
            current_strategies = strategies.cpu().numpy().tolist()

            for i in range(len(preds)):
                strategies_total[current_strategies[i]] += 1
                if preds[i] == current_true_labels[i]:
                    strategies_correct[current_strategies[i]] += 1
                    
            # Update batch progress bar with running loss
            batch_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(loader)
        print(f'{str_} Loss: {avg_loss}')
        print(classification_report(true_labels, predictions, zero_division=1, digits=4))

        # Calculate overall accuracy
        overall_accuracy = round(100 * (np.array(predictions) == np.array(true_labels)).mean(), 1)
        print('Overall accuracy: {}%'.format(overall_accuracy))

        with open(self.fp_txt, 'a') as f:
            # Write the epoch number
            f.write(f'Epoch {epoch+1} {str_}:\n')
            # Write the classification report
            f.write('Classification Report:\n')
            f.write(classification_report(true_labels, predictions, zero_division=1, digits=4))
            # Write the overall accuracy
            f.write('Overall accuracy: {}%\n'.format(overall_accuracy))

        with open(self.fp_csv, 'a') as f:
            # Write the epoch number, split, and accuracy
            f.write('{},{},{}\n'.format(epoch+1, str_, overall_accuracy))

        # Calculate accuracy for each strategy
        strategy_acc = []
        
        # Finally, after the loop, print the accuracies for each strategy:
        for i in range(len(self.strategy_map)):
            if strategies_total[i] != 0:
                # print accuracy for strategy i, with its correct count and totol number of the strategy
                accuracy = round(strategies_correct[i]/strategies_total[i] * 100, 1)
                strategy_acc.append(accuracy)
                print(f'Strategy {i} {self.strategy_map_inverted[i]} accuracy: {accuracy}, {strategies_correct[i]} / {strategies_total[i]}')
                # If str_ is in ['Valid', 'Test'], then write the accuracy for strategy i into fp_txt
                if str_ in ['Valid', 'Test']:
                    with open(self.fp_txt, 'a') as f:
                        f.write(f'{i}, {accuracy}, {strategies_correct[i]} / {strategies_total[i]}\n')
                        if i == len(self.new_emote_config.strategy_map) - 1:
                            f.write('\n')
            else:
                print(f'No instances of strategy {i} {self.strategy_map_inverted[i]} found')
                strategy_acc.append(0)  # Append 0 for strategies with no instances

        return overall_accuracy, strategy_acc
