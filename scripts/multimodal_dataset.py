import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import glob

class MultimodalDataset(Dataset):
    def __init__(self, encodings, labels, strategies, img_dir, max_seq_len=9, img_transform=None):
        self.encodings = encodings
        self.labels = labels
        self.strategies = strategies
        self.img_dir = img_dir
        self.max_seq_len = max_seq_len
        
        # Default image transformation
        if img_transform is None:
            self.img_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.img_transform = img_transform
        
        # Create a placeholder/padding image (black image with normalization)
        placeholder = torch.zeros(3, 224, 224)
        norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.placeholder_image = (placeholder - norm_mean) / norm_std
        
        # Filter indices where composite image exists (we'll decompose it)
        self.valid_indices = []
        for idx in range(len(self.labels)):
            img_path = os.path.join(self.img_dir, f"{idx}.png")
            if os.path.exists(img_path):
                self.valid_indices.append(idx)
        
        if len(self.valid_indices) < len(self.labels):
            print(f"Filtered out {len(self.labels) - len(self.valid_indices)} datapoints without images.")
            print(f"Dataset size reduced from {len(self.labels)} to {len(self.valid_indices)}.")
        
        # Cache extracted sequences to avoid repeated processing
        self.sequence_cache = {}

    def extract_emoji_sequence(self, composite_image_path):
        """
        Extract a sequence of individual emoji images from a composite grid image
        This is a simplified version - in production you'd use more robust image processing
        """
        # Load the composite image
        composite = Image.open(composite_image_path).convert('RGB')
        width, height = composite.size
        
        # For a 3x3 grid of emojis
        emoji_size = width // 3
        emoji_sequence = []
        
        # Extract each emoji from the grid
        for row in range(3):
            for col in range(3):
                left = col * emoji_size
                upper = row * emoji_size
                right = left + emoji_size
                lower = upper + emoji_size
                
                emoji = composite.crop((left, upper, right, lower))
                
                # Skip if emoji is empty (black)
                pixel_sum = np.array(emoji).sum()
                if pixel_sum > 100:  # Non-black threshold
                    emoji_sequence.append(self.img_transform(emoji))
                
                # Limit sequence length
                if len(emoji_sequence) >= self.max_seq_len:
                    break
                    
        # If sequence is empty (no emojis found), add at least one placeholder
        if not emoji_sequence:
            emoji_sequence.append(self.placeholder_image)
            
        return emoji_sequence

    def __getitem__(self, idx):
        # Map to valid index
        original_idx = self.valid_indices[idx]
        
        # Get text inputs and labels
        item = {key: val[original_idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = self.labels[original_idx].clone().detach()
        item['strategies'] = self.strategies[original_idx].clone().detach()
        
        # Get image path
        img_path = os.path.join(self.img_dir, f"{original_idx}.png")
        
        # Use cached sequence if available
        if original_idx in self.sequence_cache:
            emoji_sequence = self.sequence_cache[original_idx]
        else:
            try:
                emoji_sequence = self.extract_emoji_sequence(img_path)
                self.sequence_cache[original_idx] = emoji_sequence
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                # Fallback: create a sequence with just placeholders
                emoji_sequence = torch.stack([self.placeholder_image] * self.max_seq_len)
            
        # Pad sequence to fixed length
        seq_len = len(emoji_sequence)
        while len(emoji_sequence) < self.max_seq_len:
            emoji_sequence.append(self.placeholder_image)
            
        # Stack into tensor [seq_len, C, H, W]
        emoji_sequence = torch.stack(emoji_sequence)
        
        item['images'] = emoji_sequence
        item['seq_len'] = torch.tensor(seq_len)
        
        # Add image ID for caching
        item['img_ids'] = torch.tensor(original_idx)
        
        return item

    def __len__(self):
        return len(self.valid_indices) 