import os
import glob
import numpy as np
import pandas as pd
import imageio
import matplotlib.pyplot as plt
from skimage.measure import label as connected_components
from skimage.measure import regionprops
from skimage.color import rgb2gray
from skimage import exposure, color
from tqdm import tqdm
import cv2

from micro_sam import util
from micro_sam.instance_segmentation import (
    InstanceSegmentationWithDecoder,
    get_predictor_and_decoder,
    mask_data_to_segmentation
)

def optimize_circle_detection(image):
    """
    Optimized circle detection using OpenCV's HoughCircles
    
    Args:
        image: Input microscopy image
        
    Returns:
        Mask of the plate region and plate info (center_x, center_y, radius)
    """
    # Convert to BGR if grayscale (OpenCV expects BGR for colored images)
    if len(image.shape) == 2:
        img_for_cv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        img_for_cv = image.copy()
        
        # If image has 4 channels (RGBA), convert to BGR
        if img_for_cv.shape[2] == 4:
            img_for_cv = cv2.cvtColor(img_for_cv, cv2.COLOR_RGBA2BGR)
    
    # Convert to grayscale for circle detection
    gray = cv2.cvtColor(img_for_cv, cv2.COLOR_BGR2GRAY)
    
    # Create a copy for output
    img_with_mask = img_for_cv.copy()
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Try multiple parameter sets for HoughCircles to ensure detection
    circle_params = [
        # dp, minDist, param1, param2, minRadius, maxRadius
        (1.2, min(gray.shape) // 2, 50, 30, min(gray.shape) // 4, min(gray.shape) // 2),
        (1.5, min(gray.shape) // 2, 100, 40, min(gray.shape) // 4, min(gray.shape) // 2),
        (1.2, min(gray.shape) // 2, 70, 20, min(gray.shape) // 4, min(gray.shape) // 2),
    ]
    
    detected_circles = None
    
    for dp, minDist, param1, param2, minRadius, maxRadius in circle_params:
        try:
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius
            )
            
            if circles is not None and len(circles[0]) > 0:
                # Take the circle with the highest prominence (first one returned by HoughCircles)
                detected_circles = circles[0][0]
                break
        except Exception as e:
            print(f"Circle detection attempt failed with parameters {dp}, {minDist}, {param1}, {param2}: {e}")
    
    # If no circles detected, use fallback to estimate the circle
    if detected_circles is None:
        print("Warning: No circle detected, using estimated circle in center of image.")
        h, w = gray.shape
        center_x, center_y = w // 2, h // 2
        radius = min(w, h) // 2 - 20  # Slightly smaller than half the minimum dimension
        circle_info = (center_x, center_y, radius)
    else:
        x, y, r = np.round(detected_circles).astype(int)
        circle_info = (x, y, r)
    
    # Create a mask for the detected circle
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(mask, (circle_info[0], circle_info[1]), circle_info[2], 255, thickness=-1)
    
    return mask > 0, circle_info