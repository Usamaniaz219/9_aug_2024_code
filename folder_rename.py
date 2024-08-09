import os
import cv2
import numpy as np

folder_path = "/home/usama/9_aug_2024/ca_dublin/"
output_dir = "/home/usama/9_aug_2024/folder_for_testing/"
folder_name = os.path.basename(os.path.dirname(folder_path))
print("folder name",folder_name)

all_masks = os.listdir(folder_path)
masks_renamed = [mask.replace(".jpg","").replace(".png","") for mask in all_masks]
for renamed_mask in masks_renamed:
    mask_path = f"{folder_path}/{renamed_mask}.jpg"
    mask_image = cv2.imread(mask_path)
    output_subdir = os.path.join(output_dir, os.path.basename(os.path.dirname(folder_path)))
    os.makedirs(output_subdir, exist_ok=True)
    output_file_path = os.path.join(output_subdir, f"{renamed_mask}_output_mask.jpg")
    cv2.imwrite(output_file_path,mask_image)