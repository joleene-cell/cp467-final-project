import numpy as np
import cv2
import time
from contour import detect_contour
import pandas as pd
from utils import load_frame, gt_corners, load_split

#### finding the actual distance between the predicted document corners and actual corners

def calculate_corner_difference(predicted, ground_truth):


###### if there was no document detected return "Nan"
    if predicted is None:
        return np.nan
####### calculating the distance between each corner 
    distances = np.linalg.norm(
        predicted - ground_truth, axis = 1
) 
########## return the average distance 
    return np.mean(distances)



######## IoU implementation


### calculating the overlap from the predicted document and actual document
def IoU_calculation(predicted, ground_truth, image_shape):
    if predicted is None :
        return 0.0
    height, width = image_shape[:2]

###### creating 2 empty black masks 
    mask_on = np.zeros((height, width), dtype=np.uint8)
    mask_off = np.zeros((height, width), dtype=np.uint8)

######## filling document areas with white
    cv2.fillPoly(mask_on, [predicted.astype(np.int32)], 1)
    cv2.fillPoly(mask_off, [ground_truth.astype(np.int32)], 1)


### calculating the overlapping regions 
    meeting = np.logical_and(mask_off, mask_on).sum()

#### calculating combined area
    union = np.logical_or(mask_on, mask_off).sum()

#### making sure there is not division by zero case 
    if union == 0:
        return 0.0
    
###### returning the IoU
    return meeting/union


####### implementing processing time

def record_runtime(detection_method, image):

######## recording the start time 

    start = time.perf_counter()

####### compiling the detection method

    result = detection_method(image)

######## recording time it took to finish 

    completion_time = time.perf_counter()

########## calculating elapsed time ins econds

    elapsed = completion_time - start

    return result, elapsed

######### Testing evaluation on SmartDoc Image

validation = load_split("validation")

######### creating arrays to store the results 

elapsed_time = []
corner_sum = []
iou_scores = []

####### count the successful and non successful detections
detected_it = 0
nothing_here = 0

######## going through every image in the dataset 
for index, row in validation.iterrows():
######## checking actual document corners for the image
    image, scale = load_frame(row)
    actual_corners = gt_corners(row, scale)

########## measuring how long contour detection runs for 
    result, elapsed = record_runtime(detect_contour, image)
    predicted_corners, edges= result

############ calculating how well contour detection is performing 
    corner_difference = calculate_corner_difference(predicted_corners, actual_corners)
    iou = IoU_calculation(predicted_corners, actual_corners, image.shape)

########### storing the results   
    corner_sum.append(corner_difference)
    iou_scores.append(iou)
    elapsed_time.append(elapsed)
    
######## counter for number of successful and non successful detections

    if predicted_corners is None:
        nothing_here += 1
    else:
        detected_it += 1
    print("Processed image: ", index + 1)
    
######### calculation of results afer successful run

average_iou = np.mean(iou_scores)
average_elapsedTime = np.mean(elapsed_time)
average_cornerDiff = np.nanmean(corner_sum)
total_images = detected_it + nothing_here
success_rate = (detected_it / total_images) * 100


####### Show finding results 
print("\n=== SmartDoc Validation Results ====")
print("Total Images: ",total_images)
print("Successful Detections: ", detected_it)
print("Non-successful Detections: ", nothing_here)
print("Success Rate of Detections: ", success_rate, "%")
print("Difference of Corners: ", average_cornerDiff, "PIXELS")
print("Average IoU: ", average_iou)
print("Average Runtime: ", average_elapsedTime, "SECONDS")
