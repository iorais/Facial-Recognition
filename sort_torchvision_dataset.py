import os
import configparser

from autocrop import Cropper
from PIL import Image

import sort_data

# sorts dataset for torchvision.dataset.ImageFolder
# sorts data into torchvision_dataset
# each subdirectory of torchvision_dataset represents a class

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

src = os.path.join(root_path, 'augmented_data/Raw')
dst = os.path.join(root_path, 'torchvision_dataset')
os.makedirs(dst, exist_ok=True)

cropper = Cropper()

for filename in os.listdir(src):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        # creates subdirectory by labels
        label = filename.split('_')[0]
        subdir = os.path.join(dst, label)
        os.makedirs(subdir, exist_ok=True)

        # creates rejection directory
        rej = os.path.join(root_path, 'rejected')
        os.makedirs(rej, exist_ok=True)

        # crops image
        cropped_array = cropper.crop(f'{src}/{filename}')

        if type(cropped_array) != type(None):
            # saves successfully cropped image in subdir
            img = Image.fromarray(cropped_array)
            img.save(f'{subdir}/{filename}')
        else:
            # saves saves rejected image in rej
            img = Image.open(f'{src}/{filename}')
            img.save(f'{rej}/{filename}')