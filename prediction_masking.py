import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

def fixMasks(image, table_mask, column_mask):
    
    table_mask = table_mask.reshape(1024,1024).astype(np.uint8)
    column_mask = column_mask.reshape(1024,1024).astype(np.uint8)
    
    #get contours of the mask to get number of tables
    contours, table_heirarchy = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    table_contours = []
    #ref: https://www.pyimagesearch.com/2015/02/09/removing-contours-image-using-python-opencv/
    #remove bad contours

    for c in contours:
        if cv2.contourArea(c) > 2000:
            table_contours.append(c)
    
    if len(table_contours) == 0:
        return None

    #ref : https://docs.opencv.org/4.5.2/da/d0c/tutorial_bounding_rects_circles.html
    #get bounding box for the contour
    
    table_boundRect = [None]*len(table_contours)
    for i, c in enumerate(table_contours):
        polygon = cv2.approxPolyDP(c, 3, True)
        table_boundRect[i] = cv2.boundingRect(polygon)
    
    #table bounding Box
    table_boundRect.sort()
    
    return table_boundRect

plt.imshow(tf.keras.preprocessing.image.array_to_img(image))
plt.axis('off')
plt.show()