import numpy as np
import cv2 as cv
import os.path

filename = "" 
img = cv.imread(filename)


# Adaptive Histogram Equalization
clahe = cv.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
# WhiteBalance
"""
White Balance based on Gray Scale 
"""
img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
"""
Mean Shift Color Value
Dark Color Prior 
"""

dark_elm = np.percentile(img_gray,2,method="inverted_cdf")


