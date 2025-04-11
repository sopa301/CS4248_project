# This is a script that synchronises the positive samples in the csvs in ../dataset-positional-sensitivity with
# the data in ELCo_with_pos_sens.csv. It edits the csv files in ../dataset-positional-sensitivity.

import csv
import os
import pandas as pd
import shutil
import numpy as np
import random
from tqdm import tqdm

import ast

def emoji_to_unicode(emoji_str):
    return ' '.join([f"U+{ord(char):X}" for char in emoji_str])

def process_emoji_list_to_str(emoji_list):
    desc_processed = ' [EM] '.join(desc.strip(':') for desc in emoji_list)
    return f"{desc_processed}."

def unprocess_emoji_list_from_str(emoji_str):
    s = emoji_str.split(' [EM] ')
    return [f":{desc}:" for desc in s]



def get_csv_files(directory):
    """
    Get a list of all CSV files in the given directory.
    """
    csv_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            csv_files.append(os.path.join(directory, filename))
    return csv_files

files = get_csv_files('./csvs')
# Read the ELCo_with_pos_sens.csv file
elco_df = pd.read_csv('./elco_csv/ELCo_with_pos_sens.csv')

for file in files:
    df = pd.read_csv(file)
    # Get the name of the file without the path
    filename = os.path.basename(file)
    # Get the name of the file without the .csv extension
    filename = filename[:-4]
    print("processing file: ", filename)
    # iterate over the rows of the df, if "label" == 1, then take sent2, remove the "This is " prefix, and get the
    # positional_sensitivity from the elco_df in the same row as the EN value that is equal to sent2
    # throw an error if the EN value is not found in the elco_df
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        # print("KEKWKWKW",row)
        if row['label'] == 1:
            # Process sent1
            sent1 = row['sent1'][8:].rstrip('.')  # remove prefix and trailing dot
            key1 = str(unprocess_emoji_list_from_str(sent1))
            # print("KEY1:", key1)

            # Process sent2
            sent2 = row['sent2'][8:].rstrip('.')  # same treatment
            # print("EN phrase:", sent2)

            # Ensure scalars
            if isinstance(sent2, (list, np.ndarray)):
                sent2 = sent2[0]
            if isinstance(key1, (list, np.ndarray)):
                key1 = key1[0]

            # Strip to avoid any invisible mismatch
            sent2 = sent2.strip()
            key1 = key1.strip()

            # Lookup
            match = elco_df[
                (elco_df['EN'].str.strip() == sent2) &
                (elco_df['Description'].str.strip() == key1)
            ]

            if not match.empty:
                positional_sensitivity = match['positional sensitivity'].values[0]
                # print("POS SENS:", positional_sensitivity)
            else:
                raise ValueError(f"⚠️ No match found for EN='{sent2}' and Description='{key1}'")

            # Update the df with the positional sensitivity
            df.at[index, 'positional_sensitivity'] = positional_sensitivity
            # exit(0)
    # save the df to the same file
    df.to_csv(file, index=False)