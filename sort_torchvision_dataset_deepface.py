import os
import argparse

import numpy as np
from PIL import Image

from tqdm import tqdm

from deepface import DeepFace
from autocrop import Cropper


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

rej_path = os.path.join(root_path, 'rejected')

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
rej = ''

if opt.src_dst:
    src, dst = opt.src_dst
elif opt.training_set:
    src = train_path
    dst = train_dst
elif opt.validation_set:
    src = val_path
    dst = val_dst

rej = os.path.join(rej_path, dst)

os.makedirs(dst, exist_ok=True)
os.makedirs(rej, exist_ok=True)

cropper = Cropper(244, 244)

def df_cropper(img_path):
    backends = [
    'opencv', 
    'ssd', 
    'dlib', 
    'mtcnn', 
    'retinaface', 
    'mediapipe',
    'yolov8',
    'yunet',
    'fastmtcnn',
    ]

    face_objs = DeepFace.extract_faces(
        img_path=img_path, 
        target_size = (224, 224), 
        detector_backend = backends[4],
        enforce_detection=False
    )

    image_array = np.array(face_objs[0]['face'] * 255).astype(np.uint8)

    return image_array

for filename in tqdm(os.listdir(src)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        # creates subdirectory by labels
        label = filename.split('_')[0]
        subdir = os.path.join(dst, label)
        os.makedirs(subdir, exist_ok=True)

        # crops image
        # cropped_array = cropper.crop(f'{src}/{filename}')
        print(f'attempting to save {src}/{filename}')
        cropped_array = df_cropper(f'{src}/{filename}')

        if type(cropped_array) != type(None):
            # saves successfully cropped image in subdir
            img = Image.fromarray(cropped_array)
            img.save(f'{subdir}/{filename}')
            print(f'saved to {subdir}/{filename}')
        else:
            rejsubdir = os.path.join(rej, label)
            os.makedirs(rejsubdir, exist_ok=True)

            # saves saves rejected image in rej/label/
            img = Image.open(f'{src}/{filename}')
            img.save(f'{rejsubdir}/{filename}')
            print(f'rejected to {rejsubdir}/{filename}')