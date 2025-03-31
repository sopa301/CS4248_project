# OpenFlamingo

* Passed test set with zero shot, few shot
* Performance is not good, as good as random classification (or assigning all rows the same label)
Kaggle Notebook here: https://www.kaggle.com/code/timothyleow12/openflamingo-test

it's also in this repo

## How to use the dataset

## Dataset following the ElCo split
* It's in the directory `dataset/`
* This directory contains the scripts `generate_images.ipynb` and `create_csv.ipynb` which should be run in that order if anything needs to be recreated
* and also the original elco dataset split if you need to refer to it
  
### Your folder of concern is `dataset/generated_img_dataset/`, which contains:

* The augmented dataset in `generated_img_dataset/`, which contains the relevant `.csv` files and folders containing the images corresponding to each EM
    * The image file path relevant to the row is in the last column of each `.csv` file.
