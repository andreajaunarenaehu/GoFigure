# Running counting_code_git.py 
First, you need to download the images from V-FLUTE. You can find them here: 

1) TRAIN split: https://drive.google.com/file/d/1EgCFjiyZeqCTr5aC7tPRwKwdXWCMsosw/view?usp=sharing

2) VALIDATION split: https://drive.google.com/file/d/1Jac9Jm3QX2-vfNqaM7EfyzrfDLFy_6uk/view?usp=sharing

3) TEST split: https://drive.google.com/file/d/1lpORwO9-DngOR5iR8m-5w8PhVhGzycg3/view?usp=sharing

Once you have the images downloaded, change these lines in the code to add your path: 

`` BASE_DIRS = {
    "train": "/home/andrea/Desktop/TRAIN_VFLUTE",
    "validation": "/home/andrea/Desktop/VALIDATION_VFLUTE",
    "test": "/home/andrea/Desktop/TEST_VFLUTE",
} ``

Finally, you can run your code normally. 
