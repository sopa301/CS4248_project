import os
import random
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import AutoTokenizer

from emote_multimodal_config import Emote_Multimodal_Config
from multimodal_model import EmoteMultimodalModel
from multimodal_dataset import MultimodalDataset
from finetune import EmoteTrainer
from unsupervised import UnsupervisedEval

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def encode_data(tokenizer, sentences_1, sentences_2, labels, strategies):
    """Encode the sentences and labels into a format that the model can understand."""
    inputs = tokenizer(list(sentences_1), list(sentences_2), truncation=True, padding=True, return_tensors="pt")
    labels = torch.tensor(labels)
    strategies = torch.tensor(strategies)
    
    return inputs, labels, strategies

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--finetune', type=int, default=0, help='0 means unsupervised, 1 means fine-tuning.')
    parser.add_argument('--model_name', type=str, default='bert-base', help='Options include bert-base, roberta-base, roberta-large, or bart-large')
    parser.add_argument('--portion', type=float, default=1, help='A float indicating the portion of training data used, default is 1.')
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--hfpath', type=str, default='YOUR_PATH', help='directory for huggingface cache')
    parser.add_argument('--use_images', action='store_true', help='Use multimodal model with images')
    parser.add_argument('--image_model', type=str, default='swin_tiny_patch4_window7_224', 
                        help='Image model to use (swin_tiny_patch4_window7_224, swin_small_patch4_window7_224, swin_base_patch4_window7_224)')
    args = parser.parse_args()

    # Create config
    config = Emote_Multimodal_Config(
        args.finetune, 
        args.model_name, 
        args.portion, 
        args.seed, 
        args.hfpath,
        use_images=args.use_images,
        image_model_name=args.image_model
    )

    seed = args.seed
    set_seed(seed)

    os.environ["HF_HOME"] = config.hfpath

    print('model_name: ', config.model_name)
    print('model_path: ', config.model_path)
    print('use_images: ', config.use_images)
    if config.use_images:
        print('image_model: ', config.image_model_name)

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    tokenizer.add_tokens(['[EM]'])

    # Load the data
    if args.portion == 1.0:
        train_data = pd.read_csv(config.train_csv_fp)
    else:
        train_data = pd.read_csv(config.train_csv_fp)
        train_data = train_data.sample(frac=config.portion, random_state=1)
        train_data = train_data.reset_index(drop=True)
        print('train_data.shape: ', train_data.shape)
        print(train_data.head(10))

    val_data = pd.read_csv(config.val_csv_fp)
    test_data = pd.read_csv(config.test_csv_fp)

    # merge train_data and val_data and test_data into all_data
    all_data = pd.concat([train_data, val_data, test_data], ignore_index=True)

    # Encode the data
    train_inputs, train_labels, train_strategies = encode_data(tokenizer, train_data['sent1'], train_data['sent2'], train_data['label'], train_data['strategy'])
    val_inputs, val_labels, val_strategies = encode_data(tokenizer, val_data['sent1'], val_data['sent2'], val_data['label'], val_data['strategy'])
    test_inputs, test_labels, test_strategies = encode_data(tokenizer, test_data['sent1'], test_data['sent2'], test_data['label'], test_data['strategy'])
    all_inputs, all_labels, all_strategies = encode_data(tokenizer, all_data['sent1'], all_data['sent2'], all_data['label'], all_data['strategy'])

    # Create the datasets
    if config.use_images:
        train_dataset = MultimodalDataset(train_inputs, train_labels, train_strategies, config.train_img_dir)
        val_dataset = MultimodalDataset(val_inputs, val_labels, val_strategies, config.val_img_dir)
        test_dataset = MultimodalDataset(test_inputs, test_labels, test_strategies, config.test_img_dir)
        all_dataset = MultimodalDataset(all_inputs, all_labels, all_strategies, config.train_img_dir)  # Using train_img_dir as a fallback
    else:
        # Use the original dataset class for text-only
        from emote import CustomizedDataset
        train_dataset = CustomizedDataset(train_inputs, train_labels, train_strategies)
        val_dataset = CustomizedDataset(val_inputs, val_labels, val_strategies)
        test_dataset = CustomizedDataset(test_inputs, test_labels, test_strategies)
        all_dataset = CustomizedDataset(all_inputs, all_labels, all_strategies)

    # Define the batch size
    batch_size = config.bs

    # Create the DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    all_loader = DataLoader(all_dataset, batch_size=batch_size)

    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- START MANUAL DEBUG BLOCK ---
    print("\n--- Starting Manual Debug Check ---")
    try:
        debug_batch = next(iter(train_loader))
        print(f"Loaded one batch. Keys: {debug_batch.keys()}")

        debug_images = debug_batch['images']
        debug_labels = debug_batch['labels']
        debug_strategies = debug_batch.pop('strategies') # Pop strategies manually

        print(f"Image shape: {debug_images.shape}, dtype: {debug_images.dtype}")
        print(f"Labels shape: {debug_labels.shape}, dtype: {debug_labels.dtype}")
        print(f"Labels in batch: {debug_labels.cpu().numpy()}")
        print(f"Label distribution: {np.bincount(debug_labels.cpu().numpy())}")
        print(f"Image Stats: Min={debug_images.min():.4f}, Max={debug_images.max():.4f}, Mean={debug_images.mean():.4f}, Std={debug_images.std():.4f}")
        if torch.isnan(debug_images).any():
            print("ERROR: NaN found in debug_images!")
        else:
            print("Image NaN check passed.")

        # Initialize the *simplest working model* (Frozen Swin + MLP Head)
        # Ensure config is defined before this block
        print("Initializing simplified model for debug...")
        # Assumes 'config' is the Emote_Multimodal_Config object
        from multimodal_model import EmoteMultimodalModel
        debug_model = EmoteMultimodalModel(config, num_labels=2)
        debug_model.to(device)
        debug_model.train() # Set to train mode

        # Manual forward pass
        print("Performing manual forward pass...")
        debug_images = debug_images.to(device)
        debug_labels = debug_labels.to(device)
        # Pass only images, as the simplified model's forward expects that
        # Note: The internal loss calculation in forward won't run without labels passed,
        # but we calculate loss manually below.
        logits = debug_model(images=debug_images).logits # Get logits directly

        print(f"Logits shape: {logits.shape}, dtype: {logits.dtype}")
        print(f"Logits sample: {logits[:5].detach().cpu().numpy()}")
        print(f"Logits Stats: Min={logits.min().item():.4f}, Max={logits.max().item():.4f}, Mean={logits.mean().item():.4f}, Std={logits.std().item():.4f}")
        if torch.isnan(logits).any():
            print("ERROR: NaN found in logits!")

        # Manual loss calculation
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, debug_labels)
        print(f"Manual Loss Calculated: {loss.item():.6f}")
        if torch.isnan(loss):
            print("ERROR: Loss is NaN!")

        # Manual backward pass
        print("Performing manual backward pass...")
        loss.backward()

        # Inspect gradients of TRAINABLE head parameters
        print("--- Head Gradient Inspection ---")
        head_grads_found = False
        head_grads_zero = True
        for name, param in debug_model.named_parameters():
            if 'head.' in name and param.requires_grad: # Focus on MLP head
                print(f"Param: {name}, Requires Grad: {param.requires_grad}")
                if param.grad is None:
                    print("  ERROR: Gradient is None!")
                else:
                    head_grads_found = True
                    grad_norm = param.grad.norm().item()
                    print(f"  Gradient Norm: {grad_norm:.6f}")
                    if torch.isnan(param.grad).any():
                        print("  ERROR: Gradient is NaN!")
                    if grad_norm > 1e-9: # Check if non-zero
                         head_grads_zero = False
        if not head_grads_found: print("ERROR: No head parameters found or none require grad!")
        if head_grads_zero and head_grads_found: print("ERROR: All head gradients are zero!")


        # Optional: Check if any FROZEN params got gradients (shouldn't happen)
        # print("--- Frozen Gradient Check ---")
        # frozen_grads_found = False
        # for name, param in debug_model.named_parameters():
        #     if 'head.' not in name and not param.requires_grad: # Frozen parts
        #          if param.grad is not None:
        #               print(f"ERROR: Gradient found for FROZEN parameter: {name}")
        #               frozen_grads_found = True
        # if not frozen_grads_found: print("OK: No gradients found for frozen parameters.")

    except Exception as e:
        print(f"An error occurred during manual debug: {e}")
        import traceback
        traceback.print_exc()

    print("--- Finished Manual Debug Check ---")
    # --- END MANUAL DEBUG BLOCK ---

    if args.finetune == 0:
        # Unsupervised evaluation is not supported for multimodal model
        if config.use_images:
            print("Unsupervised evaluation is not supported for multimodal model. Please use --finetune 1")
            exit()
            
        # Which means unsupervised evaluation
        model = AutoModelForSequenceClassification.from_pretrained(config.model_path)
        # Resize the token embeddings of the model
        model.resize_token_embeddings(len(tokenizer))
        new_token_id = tokenizer.convert_tokens_to_ids('[EM]')
        print('new_token_id: ', new_token_id)
        embeddings = model.get_input_embeddings()
        with torch.no_grad():
            # Since our special [EM] token is a new token, we need to initialize its embedding
            embeddings.weight[new_token_id] = torch.normal(0, 0.01, (embeddings.embedding_dim,))
        print('Unsupervised evaluation')
        model.to(device)
        unsupervised_eval = UnsupervisedEval(model, test_loader, device, 'test', config)
        unsupervised_eval.unsupervised_evaluation()
        exit()

    # Initialize the model
    model = EmoteMultimodalModel(config, num_labels=2)
    
    model.to(device)

    # Initialize the trainer
    trainer = EmoteTrainer(model, train_loader, val_loader, test_loader, device, new_emote_config=config, tokenizer=tokenizer)

    # Train the model
    trainer.train(epochs=config.max_epoch) 