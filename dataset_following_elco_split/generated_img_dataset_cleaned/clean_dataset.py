import os
import pandas as pd

train_dataset = pd.read_csv("generated_img_dataset/train.csv")
valid_dataset = pd.read_csv("generated_img_dataset/val.csv")
test_dataset = pd.read_csv("generated_img_dataset/test.csv")


def file_exists(filepath):
    return os.path.exists(filepath)

## train
mask = train_dataset['image'].apply(lambda x: file_exists(x))
cleaned_dataset = train_dataset[mask].copy()
cleaned_dataset.to_csv("generated_img_dataset/train.csv", index=False)
print(f"Cleaned dataset: {len(train_dataset) - len(cleaned_dataset)} rows removed, {len(cleaned_dataset)} remaining.")

## val
mask = valid_dataset['image'].apply(lambda x: file_exists(x))
cleaned_dataset = valid_dataset[mask].copy()
cleaned_dataset.to_csv("generated_img_dataset/val.csv", index=False)
print(f"Cleaned dataset: {len(valid_dataset) - len(cleaned_dataset)} rows removed, {len(cleaned_dataset)} remaining.")

## test
mask = test_dataset['image'].apply(lambda x: file_exists(x))
cleaned_dataset = test_dataset[mask].copy()
cleaned_dataset.to_csv("generated_img_dataset/test.csv", index=False)
print(f"Cleaned dataset: {len(test_dataset) - len(cleaned_dataset)} rows removed, {len(cleaned_dataset)} remaining.")
