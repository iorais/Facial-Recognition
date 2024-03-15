import os
import argparse
import statistics

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image
from collections import defaultdict

# path to Git Repo from Google CoLab
path = 'drive/Shareddrives/CSEN240_Group11/Facial-Recognition'

root_path = path if os.path.isdir(path) else ''
train_path = os.path.join(root_path, 'trainingset0206')
val_path = os.path.join(root_path,'training_validation_set_0226')

train_dst = os.path.join(root_path, 'sorted_data')
val_dst = os.path.join(root_path, 'sorted_val_data')


parser = argparse.ArgumentParser(description='Sort Raw Image Files')

src_dst = parser.add_mutually_exclusive_group(required=True)
src_dst.add_argument('--training_set', action='store_true',
                    help='boolean, sort the training set')
src_dst.add_argument('--validation_set', action='store_true', 
                    help='boolean, sort the validation set')
src_dst.add_argument('--src_dst', nargs=2, type=str,
                     help='source and destination of data to be sorted separated by a space')

opt = parser.parse_args()

src = ''
dst = ''

if opt.src_dst:
    src, dst = opt.src_dst
elif opt.training_set:
    src = train_path
    dst = train_dst 
elif opt.validation_set:
    src = val_path
    dst = val_dst

def is_image_file(filename: str) -> bool:
    extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    return any(filename.endswith(extension) for extension in extensions)

# mapping from filename to label
filename_to_label = {}

# mapping from filename to PIL Image object
filename_to_img = {}

# mapping from label to list of image files
label_to_filenames = defaultdict(list)

with open(src + '/file_mapping.txt') as file_mapping:
    for line in file_mapping:
        filename, label = line.split()
        if is_image_file(filename):
            filename_to_label[filename] = label
            filename_to_img[filename] = Image.open(f'{src}/{filename}')
            label_to_filenames[label].append(filename)

for label in label_to_filenames.keys():
    label_to_filenames[label].sort()

# list of labels
labels = list(label_to_filenames.keys())

def save_image(filename: str):
    '''
    sorts the data into directories
    '''
    # create parent directory if needed
    parent = os.path.join(root_path, dst)
    os.makedirs(parent, exist_ok=True)

    # subdirectories
    folders = ['Raw']

    for i, folder in enumerate(folders):
        # create directory if needed
        dir = os.path.join(parent, folder)
        os.makedirs(dir, exist_ok=True)
        
    print(f'saving {filename} in {folders[-1]} folder')
    print('...')

    img = filename_to_img[filename]

    label = filename_to_label[filename]
    date = filename.split('_')[0]
    new_filename = '_'.join([label, date + '.jpeg'])

    img.save(f'{parent}/{folders[-1]}/{new_filename}')

# main function
def sort_data():
    exclude_labels = ['wufangyuan']
    exclude = defaultdict(lambda : False)
    for label in exclude_labels:
        exclude[label] = True

    for label in label_to_filenames.keys():
        if exclude[label]:
            print(f'excluding {label}')
            continue
        else:
            print(f'sorting {label}')
        
        for filename in label_to_filenames[label]:
            save_image(filename)

sort_data()