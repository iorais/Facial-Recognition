import os

from PIL import Image

import aug_image_tilt
import aug_image_color

# path to Git Repo from Google CoLab file
path = 'drive/Shareddrives/CSEN240_Group11/Facial-Recognition'

root_path = path if os.path.isdir(path) else ''
train_path = os.path.join(root_path, 'trainingset0206')
val_path = os.path.join(root_path,'training_validation_set_0226')

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

    # saves tilted images into augmented_data subdirectories
    aug_subdir_tilt = os.path.join(aug_subdir, 'Tilt')
    aug_image_tilt.process_folder(sorted_subdir, aug_subdir_tilt)
    aug_image_tilt.process_folder(sorted_subdir, aug_subdir)

    # saves colored images into augmented_data subdirectory
    aug_subdir_colored = os.path.join(aug_subdir, 'Colored')
    aug_image_color.process_folder(sorted_subdir, aug_subdir_colored)
    aug_image_color.process_folder(aug_subdir_tilt, aug_subdir)