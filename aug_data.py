import os
import configparser

from PIL import Image

import sort_data
import aug_image_tilt
import aug_image_color

def get_paths():
    config = configparser.ConfigParser()

    # path to 'config.ini' file in Google
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

    return root_path, train_path

root_path, train_path = get_paths()

# sorting data
sort_data.sort_data()

# create directory for augmented data
aug_path = os.path.join(root_path, 'augmented_data')
os.makedirs(aug_path, exist_ok=True)

sorted_path = os.path.join(root_path, 'sorted_data')

# iterate through subdirectories of sorted_data
for subdir in os.listdir(sorted_path):
    if subdir == 'LR':
        continue

    # create matching subdirectory of sorted_data 
    aug_subdir = os.path.join(aug_path, subdir)
    os.makedirs(aug_subdir, exist_ok=True)

    sorted_subdir = os.path.join(sorted_path, subdir)

    # saves default data into augmented_data subdirectory
    for filename in os.listdir(sorted_subdir):
        file_path = os.path.join(sorted_subdir, filename)

        img = Image.open(file_path)
        img.save(f'{aug_subdir}/{filename}')

    # saves tilted images into augmented_data subdirectory
    aug_image_tilt.process_folder(sorted_subdir, aug_subdir)

    # saves colored images into augmented_data subdirectory
    aug_image_color.process_folder(sorted_subdir, aug_subdir)