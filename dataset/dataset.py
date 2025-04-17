from torch.utils.data import Dataset
import pandas as pd
import torch
from PIL import Image
import os
from enum import Enum
from torchvision import transforms
import regex
import ast
from transformers import AutoImageProcessor, AutoFeatureExtractor
from transformers import AutoTokenizer, AutoModel
from run_trainer import get_text_tokenizer
class DatasetMode(Enum):
    EVAL = "evaluate"
    TRAIN = "train"

class EmoteDataset(Dataset):
    def __init__(self, csv_file: str, dataset_dir: str, portion=1.0, random_state=1, mode=DatasetMode.TRAIN, **kwargs):
        """
        Args:
            csv_file (str): Path to the CSV file.
            dataset_dir (str): Directory where the dataset files are stored.
            portion (float): Fraction of data to use from the CSV (default 1.0 for all data).
            random_state (int): Random seed for reproducibility when sampling.
            mode (DatasetMode): Mode of the dataset (train or evaluate).
        """
        # Read the CSV file
        csv_file = os.path.join(dataset_dir, csv_file)
        data = pd.read_csv(csv_file)
        
        self.processor = AutoImageProcessor.from_pretrained(
            "microsoft/swin-base-patch4-window12-384-in22k", use_fast=False
        )
        # Optionally sample a portion of the data
        if portion < 1.0:
            data = data.sample(frac=portion, random_state=random_state).reset_index(drop=True)
        
        self.en = list(data['sent2'])
        self.text1 = list(data['sent1'])
        self.labels = list(data['label'])
        self.image_files = data['separate_filenames']
        self.grid_image = data['filename']
        self.unicodes = list(data['unicode'])
        self.strategies = list(data['strategy'])
        
        self.emojis = data['sent2']
        text_tokenizer = get_text_tokenizer()
        self.encodings = text_tokenizer(self.text1, self.en, padding=True, truncation=True, return_tensors="pt")
        
        self.base_dir = dataset_dir
        
        self.mode = mode

    def __getitem__(self, idx):
        # Retrieve the tokenized inputs and the corresponding labels/strategies
        batch = {}
        # batch['sent2'] = self.en[idx]
        # batch['sent1'] = self.text1[idx]
        batch['labels'] = torch.tensor(self.labels[idx]).clone()
        # batch['strategies'] = self.strategies[idx]
        # batch['images'] = self._get_list_images(idx)
        batch['images'] = self._get_images(idx)
        batch.update({key: val[idx].clone().detach() for key, val in self.encodings.items()})
        # batch['unicodes'] = self.unicodes[idx]
        # batch['emojis'] = self.emojis[idx]    # put all emojis in a string, will be tokenized in collate_fn

        if DatasetMode.TRAIN == self.mode:
            batch['images'] = self._training_preprocess(batch['images'])
        return batch
    
    def _get_list_images(self, index):
        images = []
        for image_filename in ast.literal_eval(self.image_files[index]):
            image_path = os.path.join(self.base_dir, image_filename)
            try:
                img = Image.open(image_path).convert("RGB")
                processed = self.processor(img, return_tensors="pt")
                img = processed["pixel_values"].squeeze(0)  # [3, H, W]
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue
            images.append(img)
        return images
    
    def _get_images(self, index):
        image_filename = self.grid_image[index]
        image_path = os.path.join(self.base_dir + "google_dataset/", image_filename)
        try:
            img = Image.open(image_path).convert("RGB")
            processed = self.processor(img, return_tensors="pt")
            img = processed["pixel_values"].squeeze(0)  # [3, H, W]
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            img = None
        return img
    
    def _training_preprocess(self, images):
        """
        Apply any training-specific preprocessing (e.g., data augmentation) to the images.
        Currently, this method returns the images unchanged.
        """
        # Implement any augmentation or processing here if desired.
        return images

    def __len__(self):
        return len(self.labels)
