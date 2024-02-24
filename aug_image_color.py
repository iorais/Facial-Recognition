import os
import cv2
from PIL import Image, ImageEnhance

def adjust_image_properties_opencv(image_path, 
                                   output_folder, 
                                   saturation_factor, 
                                   brightness_factor, 
                                   contrast_factor, 
                                   hue_shift_value):
    try:
        image = cv2.imread(image_path)
        base_name = os.path.basename(image_path)
        file_name, extension_type = base_name.split(".")

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)

        h = (h + hue_shift_value) % 180 
        hsv_image = cv2.merge([h, s, v])

        adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        adjusted_image_pil = Image.fromarray(cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2RGB))

        enhancer = ImageEnhance.Color(adjusted_image_pil)
        adjusted_image_pil = enhancer.enhance(saturation_factor)

        enhancer = ImageEnhance.Brightness(adjusted_image_pil)
        adjusted_image_pil = enhancer.enhance(brightness_factor)

        enhancer = ImageEnhance.Contrast(adjusted_image_pil)
        adjusted_image_pil = enhancer.enhance(contrast_factor)

        color_adjusted_filename = f"{file_name}_h{round(hue_shift_value,2)}_s{round(saturation_factor,2)}_b{round(brightness_factor,2)}_c{round(contrast_factor,2)}.{extension_type}"
        color_adjusted_path = os.path.join(output_folder, color_adjusted_filename)
        adjusted_image_pil.save(color_adjusted_path)
        print(f"Color adjusted and saved: {color_adjusted_path}")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def process_folder(folder_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    filename: str
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            file_path = os.path.join(folder_path, filename)

            hue_range = [10 * i for i in range(1)]  # hue shift locked to 10
            saturation_range = [0.9 + 0.3*i for i in range(2)]  # range from 0.9 to 1.5
            brightness_range = [0.6 + 0.5*i for i in range(2)]  # range from 0.6 to 1.6
            contrast_range = [0.8 + 0.5*i for i in range(2)]  # range from 0.8 to 1.6

            for hue_shift_value in hue_range:
                for saturation_factor in saturation_range:
                    for brightness_factor in brightness_range:
                        for contrast_factor in contrast_range:
                            adjust_image_properties_opencv(file_path, 
                                                           output_folder, 
                                                           saturation_factor, 
                                                           brightness_factor, 
                                                           contrast_factor, 
                                                           hue_shift_value)

            # grayscale images
            adjust_image_properties_opencv(file_path, 
                                           output_folder, 
                                           0, 
                                           brightness_range[1], 
                                           contrast_range[0], 
                                           hue_range[0])