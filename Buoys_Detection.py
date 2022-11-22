import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg




def read_video(video):
    
    images = [] #to save video frams

    vidcap = cv2.VideoCapture(video)# creat video capture object

    while (vidcap.isOpened): # while video is open and being captured 
        ret, frame = vidcap.read() # read video frame
    
        if ret:     
            np_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # convert each fram to BGR 
            images.append(np_frame) # save fram in images
        
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break
        
    vidcap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    
    all_frames = np.array(images)# convert images to numpy array
    
    return all_frames # return all video frams as nump array



def white_balance(img):
    
    # convert image to LAB Color space  
    #"Lightness(Intensity), color component ranging from Green to Magenta, color component ranging from Blue to Yellow"
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    avg_a = np.average(result[:, :, 1])# average for color component ranging from Green to Magenta
    avg_b = np.average(result[:, :, 2])# average for color component ranging from Blue to Yellow"
    
    # modify the color channels based on its intensity
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)# return the image to the BGR color space
    return result
    



def Detect_Red(img):


    # convert to HSV color space "Hue(Dominant Wavelength), Saturation(Purity/shades of the color), Value(Intensity)"
    hsvFrame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Set range for red color and define mask
    red_lower = np.array([55, 110, 165], np.uint8)
    red_upper = np.array([255, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)
    
    circle_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(12,12))# Structur Element for the morphologycal operation
    
    # For red color
    res_red = cv2.bitwise_and(img, img, mask = red_mask)# bitwise the image and color mask 
    res_red = cv2.medianBlur(res_red,5)# applying median filter to enhance the image 
    res_red = cv2.morphologyEx(res_red, cv2.MORPH_OPEN, circle_kernel)# applying open opration to enhance the image 
    res_red = cv2.dilate(res_red,circle_kernel,iterations = 1)# applying dilation to enhance the image 
    
        
    thresh = cv2.cvtColor(res_red, cv2.COLOR_RGB2GRAY)# convert to gray scale 
    
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)# find the contours in the image

    try:
        cnts = sorted(contours, key = cv2.contourArea, reverse = True)
        (x,y),radius = cv2.minEnclosingCircle(cnts[0])# extracting the center and redius for circle around the buoy
        center = (int(x),int(y))
        radius = int(radius)
        cv2.circle(img, center, radius, (255,0,0), 2)# draw circle around the red buoy 
        
    except:
        pass




def Detect_Green(img):
    
    # convert to HSV color space "Hue(Dominant Wavelength), Saturation(Purity/shades of the color), Value(Intensity)"
    hsvFrame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Set range for green color and define mask
    green_lower = np.array([40, 105, 150], np.uint8)
    green_upper = np.array([60, 255, 255], np.uint8)
    green_mask  = cv2.inRange(hsvFrame, green_lower, green_upper)
    
    circle_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(12,12))# Structur Element for the morphologycal operation
    
    # For green color
    res_green = cv2.bitwise_and(img, img, mask = green_mask)# bitwise the image and color mask
    res_green = cv2.medianBlur(res_green,5)# applying median filter to enhance the image 
    res_green = cv2.morphologyEx(res_green, cv2.MORPH_OPEN, circle_kernel)# applying open opration to enhance the image
    res_green = cv2.dilate(res_green,circle_kernel,iterations = 1)# applying dilation to enhance the image 
    
    
    thresh = cv2.cvtColor(res_green, cv2.COLOR_RGB2GRAY)# convert to gray scale 

    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)# find the contours in the image

    try:
        cnts = sorted(contours, key = cv2.contourArea, reverse = True)
        (x,y),radius = cv2.minEnclosingCircle(cnts[0])# extracting the center and redius for circle around the buoy
        center = (int(x),int(y))
        radius = int(radius)
        cv2.circle(img, center, radius, (0,255,0), 2)# draw circle around the green buoy 
    except:
        pass




def Detect_Yellow(img):
    
    hsvFrame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Set range for yellow color and define mask
    yellow_lower = np.array([55, 27, 150], np.uint8)
    yellow_upper = np.array([90, 255, 254], np.uint8)
    yellow_mask  = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)
    
    circle_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(12,12))
    
    # For yellow color
    res_yellow = cv2.bitwise_and(img, img, mask = yellow_mask)# bitwise the image and color mask
    res_yellow = cv2.medianBlur(res_yellow,5) # applying median filter to enhance the image 
    res_yellow = cv2.morphologyEx(res_yellow, cv2.MORPH_OPEN, circle_kernel)# applying open opration to enhance the image
    res_yellow = cv2.dilate(res_yellow,circle_kernel,iterations = 1)# applying dilation to enhance the image 
    
    
    thresh = cv2.cvtColor(res_yellow, cv2.COLOR_RGB2GRAY)# convert to gray scale 

    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)# find the contours in the image

    try:
        cnts = sorted(contours, key = cv2.contourArea, reverse = True)
        (x,y),radius = cv2.minEnclosingCircle(cnts[0])# extracting the center and redius for circle around the buoy
        center = (int(x),int(y))
        radius = int(radius)
        cv2.circle(img, center, radius, (255,255,0), 2)# draw circle around the yellow buoy 
    except:
        pass




def Creat_Video(images, video_name):
    
    num, height, width, layers = images.shape# extracting images shape
    size = (width,height)# video fram size
     
    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 25, size)# write video with mp4 format and 25 frame per second
 
    for i in range(0, images.shape[0]):
        
        np_frame = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)# convert to RGB color space
        out.write(np_frame)
    
    out.release()




videos_filepath = ["Bouys_1.mp4", "Buoys_2.mp4", "Red_Buoy.mp4"]# file path for mission videos    


for video in videos_filepath :
    images = read_video(video)# read the video and return it as a numpy array
    
    for i in range (0, images.shape[0]):
        images[i] = white_balance(images[i])# apply white balance on all vidoe's frams
        
        Detect_Red(images[i])# detect the red buoy 
        Detect_Green(images[i])# detect the green buoy
        Detect_Yellow(images[i])# detect the yellow buoy
        
    video_name = video.replace('.mp4','_Results.mp4')# name the resulting video to be saved
    Creat_Video(images, video_name)# creat video

