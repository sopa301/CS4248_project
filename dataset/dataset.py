from torch.utils.data import Dataset
import pandas as pd
import torch
from PIL import Image
import os
from enum import Enum
from torchvision import transforms

class DatasetMode(Enum):
    EVAL = "evaluate"
    TRAIN = "train"

class EmoteDataset(Dataset):
    def __init__(self, filename_ls_path: str, csv_file: str, tokenizer, dataset_dir: str, portion=1.0, random_state=1, mode=DatasetMode.TRAIN, **kwargs):
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
        self.strategies = list(data['strategy'])
        
        self.encodings = tokenizer(
            self.sent1,
            self.sent2,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        self.base_dir = dataset_dir
        # store the emoji image path
        self.filename_ls_path = os.path.join(self.base_dir, filename_ls_path)
        # Load filenames
        with open(self.filename_ls_path, "r") as f:
            self.filenames = [
                s.split() for s in f.readlines()
            ]  # [['.../emoji1.png', '.../emoji2.png'], [], ...] (each line is an input with a list of one or more images)
        
        # Define image transformations (apply it no matter it is training or not)
        self.transform = transforms.Compose([
            #TODO: resize the image if needed
            # transforms.Resize(()),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ImageNet models
        ])

        self.mode = mode

    def __getitem__(self, idx):
        # Retrieve the tokenized inputs and the corresponding labels/strategies
        batch = {key: val[idx].clone() for key, val in self.encodings.items()}
        batch['labels'] = torch.tensor(self.labels[idx]).clone()
        batch['strategies'] = torch.tensor(self.strategies[idx]).clone()
        # TODO: uncomment after implementing the image filname split
        # batch['images'] = self._get_data_item(idx)
        # if DatasetMode.TRAIN == self.mode:
        #     batch['images'] = self._training_preprocess(batch['images'])
        return batch
    
    def _get_data_item(self, index):
        """
        Load images for the given index based on the filenames stored in the list.
        Returns a list of PIL.Image objects.
        """
        image_filenames = self.filenames[index]
        images = []
        for image_filename in image_filenames:
            image_path = os.path.join(self.base_dir, image_filename)
            try:
                img = Image.open(image_path).convert("RGB")
                img = self.transform(img)
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
