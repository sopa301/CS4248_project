from torch.utils.data import Dataset
import pandas as pd
import torch
from PIL import Image
import os
from enum import Enum
from torchvision import transforms
import regex
import ast
from transformers import AutoImageProcessor

class DatasetMode(Enum):
    EVAL = "evaluate"
    TRAIN = "train"

class EmoteDataset(Dataset):
    def __init__(self, csv_file: str, dataset_dir: str, portion=1.0, random_state=1, mode=DatasetMode.TRAIN, **kwargs):
        """
        Args:
            csv_file (str): Path to the CSV file.
            tokenizer: Hugging Face tokenizer for processing the text.
            max_length (int): Maximum sequence length for tokenization.
            portion (float): Fraction of data to use from the CSV (default 1.0 for all data).
            random_state (int): Random seed for reproducibility when sampling.
        """
        # Read the CSV file
        csv_file = os.path.join(dataset_dir, csv_file)
        data = pd.read_csv(csv_file)
        
        # Optionally sample a portion of the data
        if portion < 1.0:
            data = data.sample(frac=portion, random_state=random_state).reset_index(drop=True)
        
        self.sent1 = list(data['sent1'])
        self.sent2 = list(data['sent2'])
        self.labels = list(data['label'])
        self.image_files = data['separate_filenames']
        self.unicodes = list(data['unicode'])
        self.strategies = list(data['strategy'])
        self.emojis = list(data['emoji'])

        self.base_dir = dataset_dir

        # Define image transformations (apply it no matter it is training or not)
        self.transform = transforms.Compose([
            #TODO: resize the image if needed
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ImageNet models
        ])

        self.mode = mode

    def __getitem__(self, idx):
        # Retrieve the tokenized inputs and the corresponding labels/strategies
        batch = {}
        batch['EN'] = self.sent2[idx]
        batch['labels'] = torch.tensor(self.labels[idx]).clone()
        batch['strategies'] = self.strategies[idx]
        batch['images'] = self._get_images(idx)
        batch['unicodes'] = self.unicodes[idx]
        batch['emojis'] = regex.findall(r'\X', self.emojis[idx]) # Split emoji into individual characters in a list
        if DatasetMode.TRAIN == self.mode:
            batch['images'] = self._training_preprocess(batch['images'])
        return batch
    
    def _get_images(self, index):
        """
        Load images for the given index based on the filenames stored in the list.
        Returns a list of Tensor objects.
        """
        images = []
        for image_filename in ast.literal_eval(self.image_files[index]):
            image_path = os.path.join(self.base_dir, image_filename)
            try:
                img = Image.open(image_path).convert("RGB")
                processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-base-patch4-window12-192-22k")
                processed = processor(img, return_tensors="pt")
                img = processed["pixel_values"]
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                return None
            images.append(img)
        return images

    def _training_preprocess(self, images):
        """
        Apply any training-specific preprocessing (e.g., data augmentation) to the images.
        Currently, this method returns the images unchanged.
        """
        # Implement any augmentation or processing here if desired.
        return images

    def __len__(self):
        return len(self.labels)
