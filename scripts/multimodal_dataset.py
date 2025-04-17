import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import ast

class MultimodalDataset(Dataset):
    def __init__(self, encodings, labels, strategies, img_dir, df, img_transform=None, debug=False):
        self.encodings = encodings
        self.labels = labels
        self.strategies = strategies
        self.img_dir = img_dir
        self.df = df
        self.image_files = df['separate_filenames']
        self.debug = False 
        # Load tokenizer once if debug is enabled
        if self.debug and any('input_ids' in enc for enc in [encodings]):
            from transformers import AutoTokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            except Exception as e:
                print(f"Warning: Could not load tokenizer: {e}")
                self.tokenizer = None
        else:
            self.tokenizer = None
        
        # Default image transformation if none provided
        if img_transform is None:
            self.img_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.img_transform = img_transform
            
        # Create a placeholder image (black image)
        self.placeholder_image = torch.zeros(3, 224, 224)
        
        # Apply normalization to the placeholder image to match other images
        norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.placeholder_image = (self.placeholder_image - norm_mean) / norm_std
        

    def __getitem__(self, idx):
        # Map the requested index to a valid index
        # original_idx = self.valid_indices[idx]
        original_idx = idx
        item = {key: val[original_idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = self.labels[original_idx].clone().detach()
        item['strategies'] = self.strategies[original_idx].clone().detach()
        # Load grid_image
        grid_img_path = os.path.join(self.img_dir, f"{original_idx}.png")
        
        grid_img_path = os.path.join("dataset/google_dataset/", self.df.iloc[original_idx]['filename'])
        # Debug info
        if self.debug:
            # Extract text info if possible (from input_ids)
            if 'input_ids' in item and self.tokenizer is not None:
                try:
                    # Convert to CPU and integer if needed
                    input_ids = item['input_ids'].cpu().int().tolist()
                    text = self.tokenizer.decode(input_ids)
                    print(f"Index: {original_idx}, Grid image: {grid_img_path}, Label: {item['labels'].item()}, Strategy: {item['strategies'].item()}")
                    print(f"Text: {text}")
                    print("-" * 50)
                except Exception as e:
                    print(f"Index: {original_idx}, Grid image: {grid_img_path}, Label: {item['labels'].item()}, Strategy: {item['strategies'].item()}")
                    print(f"Could not decode text: {str(e)}")
                    print("-" * 50)
            else:
                print(f"Index: {original_idx}, Grid image: {grid_img_path}, Label: {item['labels'].item()}, Strategy: {item['strategies'].item()}")
                print("-" * 50)
        
        # The grid_image should always exist since we filtered the indices
        try:
            grid_image = Image.open(grid_img_path).convert('RGB')
            grid_image = self.img_transform(grid_image)
        except Exception as e:
            print(f"Warning: Could not process image {grid_img_path} despite it existing. Error: {e}")
            grid_image = self.placeholder_image
        
        item['grid_image'] = grid_image
        item['images'] = self._get_list_images(original_idx)
        return item
    
    def _get_list_images(self, index):
        images = []
        for image_filename in ast.literal_eval(self.image_files[index]):
            image_path = os.path.join("dataset/", image_filename)
            try:
                img = Image.open(image_path).convert("RGB")
                img = self.img_transform(img)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue
            images.append(img)
        return images
    
    def __len__(self):
        return len(self.labels)  # Return the original length for consistency