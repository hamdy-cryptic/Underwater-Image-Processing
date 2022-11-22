import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg



left_img  = cv2.imread('p1_13_L_2034.jpg', cv2.IMREAD_GRAYSCALE)# reading left image as gray image
right_img = cv2.imread('p1_13_R_2034.jpg', cv2.IMREAD_GRAYSCALE)# reading right image as gray image



stereo = cv2.StereoBM_create(numDisparities=16, blockSize=19)# creat stereoBM object 
disparity = stereo.compute(left_img, right_img)# compute the disparity for the left and right gray images


filt_disparity = cv2.medianBlur(disparity,(5))# filtring the output map using median filter
for i in range(35):
    filt_disparity = cv2.medianBlur(filt_disparity,(5)) # repeat the filter for more enhancement



cv2.imwrite('Filtered_Disparity_Map.jpeg', filt_disparity)# save the map

