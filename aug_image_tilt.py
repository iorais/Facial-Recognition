import os
import cv2

import numpy as np
import matplotlib.image as mpimg


def tilt_right(image):
    rows, cols = image.shape[:2]
    tilt_matrix = np.float32([[1, 0.2, 0], [0, 1, 0]])
    # Adjust translation to fit the entire tilted image within the original dimensions
    translation_matrix = np.float32([[1, 0, -0.2 * cols], [0, 1, 0]])
    combined_matrix = np.dot(tilt_matrix, np.vstack((translation_matrix, [0, 0, 1])))
    # Warp the image
    tilted_image = cv2.warpAffine(image, combined_matrix[:2, :], (cols, rows))
    return tilted_image

def tilt_left(image):
    rows, cols = image.shape[:2]
    tilt_matrix = np.float32([[1, -0.2, 0], [0, 1, 0]])
    # Adjust translation to fit the entire tilted image within the original dimensions
    translation_matrix = np.float32([[1, 0, 0.2 * cols], [0, 1, 0]])
    combined_matrix = np.dot(tilt_matrix, np.vstack((translation_matrix, [0, 0, 1])))
    # Warp the image
    tilted_image = cv2.warpAffine(image, combined_matrix[:2, :], (cols, rows))
    return tilted_image

def tilt_front(image):
    rows, cols = image.shape[:2]
    tilt_matrix = np.float32([[1, 0, 0], [0, 1, 0]])
    tilted_image = cv2.warpAffine(image, tilt_matrix, (cols, rows))
    return tilted_image

def tilt_back(image):
    rows, cols = image.shape[:2]
    tilt_matrix = np.float32([[-1, 0, cols], [0, 1, 0]])
    tilted_image = cv2.warpAffine(image, tilt_matrix, (cols, rows))
    return tilted_image

def process_folder(folder_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    filename: str
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            file_path = os.path.join(folder_path, filename)
            image = mpimg.imread(file_path)

            # tilt images
            tilted_right = tilt_right(image)
            tilted_left = tilt_left(image)
            tilted_front = tilt_front(image)
            tilted_back = tilt_back(image)

            # save tilted iamges
            name, ext = filename.split('.')
            mpimg.imsave(fname=os.path.join(output_folder, f'{name}_tilted_right.{ext}'), arr=tilted_right)
            mpimg.imsave(fname=os.path.join(output_folder, f'{name}_tilted_left.{ext}'), arr=tilted_left)
            mpimg.imsave(fname=os.path.join(output_folder, f'{name}_tilted_front.{ext}'), arr=tilted_front)
            mpimg.imsave(fname=os.path.join(output_folder, f'{name}_tilted_back.{ext}'), arr=tilted_back)

folder_path = 'trainingset0206'
output_folder = 'tilted'

# process_folder(folder_path, output_folder)