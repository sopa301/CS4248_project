# How to use the dataset
## Latest update 12 Apr 0016hrs
* Use dataset-latest (follow the structure of this)
* Use this notebook: https://www.kaggle.com/code/timothyleow12/nlp-project-chengyue-model
* Updated scripts such that it's doing the performance analysis across sensitivity for positive labels only

## Each Dataset follows the ElCo split
The image file path relevant to each row is in the last column of the `.csv` files.

### Your folders of concern are as follows:

#### **Fixed-order (3x3)**: 
Located in `dataset-fixed-order/`.  
Contains `.csv` files and image folders where emojis are arranged in a fixed 3x3 grid.  

![Fixed-order (3x3)](dataset-fixed-order/generated_img_dataset/test_google/0.png)

#### **Random-order-dense (3x3)**:
Located in `dataset-random-order-dense/`.  
Contains `.csv` files and image folders where emojis are arranged in a random dense 3x3 grid (emojis are close together).

![Random-order-dense (3x3)](dataset-random-order-dense/generated_img_dataset/test_google/0.png)

#### **Random-order-sparse (3x3)**: 

Located in `dataset-random-order-sparse/`.  
Contains `.csv` files and image folders where emojis are arranged in a random sparse 3x3 grid (emojis are spaced apart randomly).

![Random-order-sparse (3x3)](dataset-random-order-sparse/generated_img_dataset/test_google/0.png)

#### **Dynamic size and positioning (maximize-used-space, 3x3)**:

Located in `dataset-maximise-used-space/`.  
Contains `.csv` files and image folders where emojis are randomly sized and positioned to maximize the used space in a 3x3 grid.

![Random size and positioning (maximize-used-space, 3x3)](dataset-maximise-used-space/generated_img_dataset/test_google/0.png)
