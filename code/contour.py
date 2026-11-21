from pathlib import Path 
import cv2 
import numpy as np 

from utils import (load_frame, load_split, gt_corners, order_corners, draw_corners)

def detect_contour(img): 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grayscale
    blur = cv2.GaussianBlur(gray, (5,5), 0)  #blur to reduce noise
    edges = cv2.Canny(blur, 50, 150)        #detect edges
    k = np.ones((5,5), np.uint8)            # close small gaps in edges

    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations= 3)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   #find contours
    contours = sorted(contours, key = cv2.contourArea, reverse=True)        #look at larger contours first 

    img_area= (img.shape[0]*img.shape[1])

    for c in contours[:20]:
        a = cv2.contourArea(c)
        if a < 0.05*img_area:
            continue 
        p = cv2.arcLength(c, True)

        approx = cv2.approxPolyDP(c, 0.02*p, True)
        if len(approx) ==4: #document should have 4 corners
            if not cv2.isContourConvex(approx):
                continue
            corners = approx.reshape(4, 2).astype(np.float32)
            corners = order_corners(corners)

            return corners, edges 
        
    return None, edges #if no document found 
 

#testing on validation images
if __name__ == "__main__":
    out = Path("../output_images/contour")
    out.mkdir(parents=True, exist_ok=True)
    validation = load_split("validation")
    sample = validation.sample(min(50, len(validation)), random_state=0)
    
    success = 0 
    fail = 0 
    for i, row in sample.iterrows():
        img, scale = load_frame(row)
        gt = gt_corners(row, scale)

        predicted, edges = detect_contour(img)
        if predicted is None:
            fail+=1
            print(f"{i}: no contour detected")
            output = draw_corners(img, gt, color =(0,0,255))
        else: 
            success+=1 
            print(f"{i}: contour detected")
            output= draw_corners(img, gt, color=(0,0,255))

            output = draw_corners(output, predicted, color=(0,255,0))
        cv2.imwrite(str(out/f"{i}_result.jpg"), output)
        cv2.imwrite(str(out/f"{i}_edges.jpg"), edges)
    
    
    print()
    print("Contour test finished")
    print("Sucessful: ", success)
    print("Failed: ", fail)


