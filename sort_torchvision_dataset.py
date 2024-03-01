import os
import argparse

from autocrop import Cropper
from PIL import Image

# sorts dataset for torchvision.dataset.ImageFolder
# sorts data into torchvision_dataset
# each subdirectory of torchvision_dataset represents a class

# path to Git Repo from Google CoLab
path = 'drive/Shareddrives/CSEN240_Group11/Facial-Recognition'

root_path = path if os.path.isdir(path) else ''

train_path = os.path.join(root_path, 'augmented_data/Raw')
train_dst = os.path.join(root_path, 'training')

val_path = os.path.join(root_path, 'sorted_val_data/Raw')
val_dst = os.path.join(root_path, 'validation')

parser = argparse.ArgumentParser(description='Sort Raw Image Files')

src_dst = parser.add_mutually_exclusive_group(required=True)
src_dst.add_argument('--training_set', action='store_true',
                    help='boolean, sort the training set')
src_dst.add_argument('--validation_set', action='store_true', 
                    help='boolean, sort the validation set')
src_dst.add_argument('--src_dst', nargs=2, type=str,
                     help='source and destination of data to be sorted')

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

os.makedirs(dst, exist_ok=True)

cropper = Cropper(244, 244)

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
            rejsubdir = os.path.join(rej, label)
            os.makedirs(rejsubdir, exist_ok=True)

            # saves saves rejected image in rej/label/
            img = Image.open(f'{src}/{filename}')
            img.save(f'{rejsubdir}/{filename}')