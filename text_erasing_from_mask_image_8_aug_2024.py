import numpy as np
import cv2
import os
import time


def text_eraser_from_mask_images(source_image,mask_image):
    
    image_Gray = cv2.cvtColor(source_image,cv2.COLOR_BGR2GRAY)
    # cv2.medianBlur(image_Gray,5)
    height, width = image_Gray.shape[:2]
    # Create a blank (black) image
    blank_image = np.zeros((height, width), dtype=np.uint8)
    # cv2.imwrite("blank_image.jpg",blank_image)
    _, thresh = cv2.threshold(image_Gray, 20, 255, cv2.THRESH_BINARY & cv2.THRESH_OTSU)
    thresh = cv2.bitwise_not(thresh)
    # cv2.imwrite("ca_dana_point_thresh.jpg",thresh)
    

    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    print("Original contours length",len(contours))
    retained_contours = []
    for i,cont in enumerate(contours):
   
        image_with_text = blank_image.copy()
        cnt = np.array([cont], np.int32)
        cnt = cnt.reshape((-1, 1, 2))

        cv2.fillPoly(image_with_text, [cnt], (255,255,255))

        filled_area = cv2.countNonZero(image_with_text)
        # total image area
        total_image_area = image_Gray.shape[0]*image_Gray.shape[1]
        if filled_area <= 0.01*total_image_area:

            result_image = cv2.bitwise_and(image_with_text, mask_image) # Perform logical AND operation with the source mask image
            
            if np.any(result_image):  # Check if the result image is blank
                retained_contours.append(cont)
    return retained_contours

intersected_contours = []

def retain_intersected_contours(retained_contours,source_mask_image):
    height,width = source_mask_image.shape[:2]
    # print("Source Mask Image shape",source_mask_image)
    blank_image = np.zeros((height,width),dtype=np.uint8)
        
    
    for i,cnt211 in enumerate(retained_contours):
        # print(cnt211)
        blank_image_with_text = blank_image.copy()
        cnt11 = np.array([cnt211], np.int32)
        cnt11 = cnt11.reshape((-1, 1, 2))

        cv2.fillPoly(blank_image_with_text, [cnt11], (255))
    
        intersections = cv2.bitwise_and(blank_image_with_text,source_mask_image)
        intersection_area = np.sum(intersections)
        bbox_mask_intersection_area = np.sum(blank_image_with_text)
        # print("bbox_mask_intersection_area",bbox_mask_intersection_area)
        if bbox_mask_intersection_area==0:
            return 0
        intersection_percentage = intersection_area/bbox_mask_intersection_area
        # print("intersection percentage",intersection_percentage)
        if intersection_percentage >=0.0009:
            intersected_contours.append(cnt211)
            # print("appended")

    return intersected_contours


# print("length of intersected Contours",len(intersected_contours))

def draw_intersected_bounding_box_on_mask_image(mask_image,intersected_contours):

    height,width = mask_image.shape[:2]
    blank_mask_image = np.zeros((height,width),dtype = np.uint8)
    kernel = np.ones((3, 3), np.uint8)
        
    for i,cnt1 in enumerate(intersected_contours):
        cnt11 = np.array([cnt1], np.int32)
        cnt11 = cnt11.reshape((-1, 1, 2))

        cv2.fillPoly(blank_mask_image, [cnt11], (255))
        # cv2.dilate(blank_mask_image,kernel,iterations=1)
        cv2.fillPoly(mask_image, [cnt11], (255))
       
    kernel = np.ones((3, 3), np.uint8)

    cv2.imwrite("text_area_masks_ca_colma_1.jpg",blank_mask_image)
    mask_image_dilated = cv2.dilate(blank_mask_image,kernel,iterations=2)
    merged_result = cv2.bitwise_or(mask_image_dilated,mask_image)
   
    return merged_result

def process_image(source_image_path, source_mask_path, output_dir):
    ori_image_name = os.path.splitext(os.path.basename(source_image_path))[0]
    print(f"Processing ori image name image: {ori_image_name}")

    source_image = cv2.imread(source_image_path)
    
    # cv2.imwrite("mask-temp.jpg",source_image)
    if source_image is None:
        print(f"Error reading mask image: {source_image_path}")
        return None
    
 
    mask_image = cv2.imread(source_mask_path)

    if mask_image is None:
        print(f"Error reading bounding box image: {source_mask_path}")
        return None
    mask_image = cv2.cvtColor(mask_image,cv2.COLOR_BGR2GRAY)
    retained_contours = text_eraser_from_mask_images(source_image,mask_image)
    intersected_contours = retain_intersected_contours(retained_contours,mask_image)
    print("intersected_retained_contours",len(intersected_contours))
    print("Length of retained Contours",len(retained_contours))
    merged_result = draw_intersected_bounding_box_on_mask_image(mask_image,intersected_contours)
   
    intersected_contours.clear()
    return merged_result



def process_images(input_dir, output_dir, bounding_box_dir):
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()
    file_count = 0
    
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                # original_file_list.append(filename)
                file_count += 1
                image_path = os.path.join(input_dir, filename)
                ori_image_name = os.path.splitext(os.path.basename(image_path))[0]
                print("mask image name :",ori_image_name)
            
                for root,dirs, files in os.walk(bounding_box_dir):              
                    # all_masks = os.listdir(bounding_box_dir)
                    for dir in dirs:
                        if dir==ori_image_name:
                            dir1 = os.path.join(root,dir)
                            # mask_dirs.append(dir1)
                            
                            all_masks = os.listdir(dir1)
                            masks_renamed = [mask.replace(".jpg","").replace(".png","") for mask in all_masks]

                            for renamed_mask in masks_renamed:
                                mask_path = f"{bounding_box_dir}/{dir}/{renamed_mask}.jpg"
                                output_ = process_image(image_path, mask_path, output_dir)
                                if output_ is None:
                                    continue

                                # output_subdir = os.path.join(output_dir, os.path.basename(os.path.dirname(image_path)))
                                output_subdir = os.path.join(output_dir, ori_image_name)
                                # output_subdir = f"{output_subdir}_{renamed_mask}_intersection_of_0.1"
                                os.makedirs(output_subdir, exist_ok=True)
                                
                                output_file_path = os.path.join(output_subdir, f"{renamed_mask}_output_mask.jpg")
                                cv2.imwrite(output_file_path,output_)
                                
                                print(f"Processed {filename} in {time.time() - start_time:.2f} seconds")
                        continue
                    break    



if __name__ == "__main__":

    input__image_directory = '/home/usama/Converted_1_jpg_from_tiff_july3_2024_updated/'
    mask_image_dir = '/home/usama/9_aug_2024/'
    output_directory = '/media/usama/6EDEC3CBDEC389B3/10_aug_2024_results_11/'
    
    process_images(input__image_directory, output_directory, mask_image_dir)


