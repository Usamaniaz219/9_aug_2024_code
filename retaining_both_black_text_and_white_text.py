import numpy as np
import cv2

image = cv2.imread("/home/usama/Converted_jpg_from_tiff_july3_2024/ca_santee.jpg")
image_Gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
inverted_image_Gray = cv2.bitwise_not(image_Gray)
# cv2.medianBlur(image_Gray,5)
height, width = image_Gray.shape[:2]

# blank_image = np.zeros((height, width), dtype=np.uint8)
# cv2.imwrite("blank_image.jpg",blank_image)
_, thresh1 = cv2.threshold(image_Gray, 30, 255, cv2.THRESH_BINARY)
thresh1 = cv2.bitwise_not(thresh1)
_,thresh = cv2.threshold(inverted_image_Gray,30,255,cv2.THRESH_BINARY)
thresh = cv2.bitwise_not(thresh)

merged_thresh = cv2.bitwise_or(thresh1,thresh)

cv2.imwrite("ca_santee_inverted_thresh.jpg",thresh)
cv2.imwrite("ca_santee_thresh.jpg",thresh1)
cv2.imwrite("ca_santee_merged.jpg",merged_thresh)
