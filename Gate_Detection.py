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
    



def Detect_Orange(img):
    
    # convert to HSV color space "Hue(Dominant Wavelength), Saturation(Purity/shades of the color), Value(Intensity)"
    hsvFrame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Set range for orange color and define mask
    orange_lower = np.array([37, 0, 100], np.uint8)
    orange_upper = np.array([255, 255, 255], np.uint8)
    orange_mask = cv2.inRange(hsvFrame, orange_lower, orange_upper)
        
    square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,10))# Structur Element for the morphologycal operation

    # For orange color
    res_orange = cv2.bitwise_and(img, img, mask = orange_mask)# bitwise the image and color mask 
    res_orange = cv2.medianBlur(res_orange,5) # applying median filter to enhance the image 
    res_orange = cv2.morphologyEx(res_orange, cv2.MORPH_OPEN, square_kernel)# applying open opration to enhance the image 
    res_orange = cv2.erode(res_orange,square_kernel,iterations = 3)# applying erosion to enhance the image 
    res_orange = cv2.dilate(res_orange,square_kernel,iterations = 6)# applying dilation to enhance the image 
    
    
    thresh = cv2.cvtColor(res_orange, cv2.COLOR_RGB2GRAY)# convert to gray scale 

    contours, hierarchy = cv2.findContours(thresh, 1, 2)# find the contours in the image

    try:
        for j in range (len(contours)):
            cnt = contours[j]
            x,y,w,h = cv2.boundingRect(cnt)# extracting all bounding rectangles in the image
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            if h > 65:# thresh for rectangle height
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,69,0), 2)# draw rectangles around the orange gat legs   
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




videos_filepath = ["Gate_1.mp4", "Gate_2.mp4"]# file path for mission videos 


for video in videos_filepath :
    images = read_video(video)# read the video and return it as a numpy array
    
    for i in range (0, images.shape[0]):
        images[i] = white_balance(images[i])# apply white balance on all vidoe's frams
        
        Detect_Orange(images[i])# detect the orange gate legs
        
    video_name = video.replace('.mp4','_Results.mp4')# name the resulting video to be saved
    Creat_Video(images, video_name)# creat video


