import os
import time
import configparser
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from collections import defaultdict

'''
Run this file in an interactive window
'''
 
def get_paths():
    config = configparser.ConfigParser()

    # path to 'config.ini' file
    path = 'drive/Shareddrives/CSEN240_Group11/'

    if not os.path.isdir(path):
    # for local machine
        path = 'configure.ini'
    else:
    # for Google CoLab
        path += 'configure.ini'

    config.read(path)

    root_path = config['PATHS']['root']
    train_path = root_path + config['PATHS']['train']
    grade_path = root_path + "/grade"

    return train_path, grade_path

def is_image_file(filename: str) -> bool:
    extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    return any(filename.endswith(extension) for extension in extensions)

def get_maps(mapping_filename='/file_mapping.txt', train_path=get_paths()[0]):
    # mapping from filename to label
    filename_to_label = {}

    # mapping from filename to PIL Image object
    filename_to_img = {}

    # mapping from label to list of image files
    label_to_filenames = defaultdict(list)

    with open(train_path + mapping_filename) as file_mapping:
        for line in file_mapping:
            filename, label = line.split()
            if is_image_file(filename):
                filename_to_label[filename] = label
                filename_to_img[filename] = Image.open(f'{train_path}/{filename}')
                label_to_filenames[label].append(filename)

    for label in label_to_filenames.keys():
        label_to_filenames[label].sort()

    return filename_to_label, filename_to_img, label_to_filenames

def sort_data(filename):
    # create parent directory if needed
    if not os.path.isdir('sorted_data'):
        os.makedirs('sorted_data')

    # subdirectories
    folders = ['Good', 'LR', 'Tilted']

    for i, folder in enumerate(folders):
        # print options
        print(f'[{i}]', folder)

        # create directory if needed
        if not os.path.isdir(f'sorted_data/{folder}'):
            os.makedirs(f'sorted_data/{folder}')
    
    # get user input
    while True:
        idx = int(input('Which folder should this image go into?'))
        
        if idx not in range(len(folders)):
            print('index was out of range')
        else:
            break

    print(f'saving {filename} in {folders[idx]} folder')
    print('...')

    img = filename_to_img[filename]
    img.save(f'sorted_data/{folders[idx]}/{filename}')

def show_all(image_files: list[str]):
    '''
    shows a subplot for each image file    
    '''

    if len(image_files) <= 6:
        rows = 2
    else:
        rows = 3
    
    fig, axs = plt.subplots(rows, 3)
    for idx, filename in enumerate(image_files):
        i = idx // 3
        j = idx % rows
        fig.suptitle(label)
        img = mpimg.imread(f'{train_path}/{filename}')
        axs[i][j].set_title(filename)
        axs[i][j].imshow(img)
        axs[i][j].tick_params(left = False, right = False , labelleft = False , 
                    labelbottom = False, bottom = False)
        
        img = filename_to_img[filename]
        img_vec = np.asarray(img).flatten()
        axs[i][j].set(ylabel=f'{img_vec.shape[0]//3000}K px')
    plt.show()


train_path = get_paths()[0]

# get maps
filename_to_label, filename_to_img, label_to_filenames = get_maps()

# list of labels
labels = list(label_to_filenames.keys())

# shows image to be sorted
for label in labels:
    image_files = label_to_filenames[label]
    show_all(image_files)

# label = labels[0]
# image_files = label_to_filenames[label]

# filename: str
# for filename in image_files:
#     show_all(image_files)
#     img = mpimg.imread(f'{train_path}/{filename}')
#     imgplot = plt.imshow(img)
#     date = filename.split('_')[0]
#     plt.title(filename)
#     plt.xlabel(label)
#     plt.tick_params(left = False, right = False , labelleft = False , 
#                 labelbottom = False, bottom = False)
#     plt.show()

#     sort_data(filename)