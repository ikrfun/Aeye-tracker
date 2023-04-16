from dataclasses import dataclass
import random
import os
import random
import shutil
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


def make_dataset_dir(image_root:str, out_dir:str='dataset',split:tuple = (9,1),over_write:bool = False)->None:
    if os.path.exists(out_dir) and over_write == False:
        print(f'{out_dir} already exist')
        print('if you want overwrite, plz use -o option')
    else:
        print('makeing new dataset folder')
        os.makedirs(out_dir,exist_ok=True)
        # get the list of label directories in the image_root directory
        label_dirs = [os.path.join(image_root, d) for d in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, d))]
        # create the train/test directories and label subdirectories
        # plz make validation_dataset in 
        for d in ["train","test"]:
            dir_path = os.path.join(out_dir, d)
            os.makedirs(dir_path,exist_ok = True)
            for label_dir in label_dirs:
                label_path = os.path.join(dir_path, os.path.basename(label_dir))
                os.makedirs(label_path,exist_ok = True)

        # copy the images to the appropriate directories based on the split ratio
        for label_dir in label_dirs:
            images = os.listdir(label_dir)
            num_images = len(images)
            num_train = int(num_images * split[0] / sum(split))
            num_test = num_images - num_train

            random.shuffle(images)

            # copy the images to the train directory
            for img_name in images[:num_train]:
                src_path = os.path.join(label_dir, img_name)
                dst_path = os.path.join(out_dir, "train", os.path.basename(label_dir), img_name)
                shutil.copy(src_path, dst_path)

            # copy the images to the test directory
            for img_name in images[num_train:]:
                src_path = os.path.join(label_dir, img_name)
                dst_path = os.path.join(out_dir, "test", os.path.basename(label_dir), img_name)
                shutil.copy(src_path, dst_path)

    print('done')



import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify a folder containing video files that will be the source of data used for AI training, and output each label to output_dir")
    parser.add_argument("-i", "--input", type=str, required=True, help="path to directory containing images stored by label.")
    parser.add_argument("-n", "--name", type=str, required=True, help="output dir name")
    parser.add_argument("-o", "--overwrite", type=bool, default=False, const=True, nargs='?',help='Whether to overwrite the directory if it already exists.')
    args = parser.parse_args()

    input = args.input
    name = args.name
    overwrite = args.overwrite

    make_dataset_dir(image_root=input,out_dir=name,over_write = overwrite)