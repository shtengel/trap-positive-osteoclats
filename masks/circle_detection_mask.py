import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from joblib import Memory

memory = Memory(location=os.path.join(os.getcwd(), "joblib_cache"), verbose=0)

# @memory.cache
def optimize_circle_detection(image, scale=0.15):
    """
    Optimized circle detection using OpenCV's HoughCircles with image scaling.
    
    Args:
        image: Input image (RGB or grayscale)
        scale: Scaling factor to downsample image for faster detection
    
    Returns:
        mask (bool ndarray): Binary mask of detected plate
        circle_info (tuple): (center_x, center_y, radius)
    """
    # Convert to BGR for OpenCV
    if len(image.shape) == 2:
        img_bgr = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        img_bgr = image.copy()
    
    # Grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Resize
    gray_small = cv2.resize(gray, (0, 0), fx=scale, fy=scale)
    
    # Blur
    blurred = cv2.GaussianBlur(gray_small, (5, 5), 0)
    
    # Parameters for scaled image
    dp = 1.2
    minDist = int(gray_small.shape[0] // 2)
    param1 = 50
    param2 = 30
    minRadius = int(gray_small.shape[0] // 4)
    maxRadius = int(gray_small.shape[0] // 2)
    
    detected_circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
        param1=param1, param2=param2,
        minRadius=minRadius, maxRadius=maxRadius
    )
    
    if detected_circles is not None and len(detected_circles[0]) > 0:
        x, y, r = detected_circles[0][0] / scale  # rescale to original size
        circle_info = tuple(np.round([x, y, r]).astype(int))
    else:
        print("Warning: No circle detected. Using fallback.")
        h, w = gray.shape
        circle_info = (w // 2, h // 2, min(w, h) // 2 - 20)
    
    # Create binary mask
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(mask, (circle_info[0], circle_info[1]), circle_info[2], 255, thickness=-1)
    
    return mask > 0, circle_info


def visualize_circle_detection_from_path(image_path, scale=0.25):
    """
    Load image, detect plate circle, and visualize result with total timing.
    
    Args:
        image_path: Path to image file
        scale: Downscaling factor for detection (e.g., 1.0 = no scaling)
    """
    start_time = time.perf_counter()
    
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not load image from path: {image_path}")
    
    # Convert to RGB for display
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect circle
    mask, (x, y, r) = optimize_circle_detection(image, scale=scale)
    
    elapsed = time.perf_counter() - start_time
    print(f"[INFO] Total time (load + detect + display): {elapsed:.2f} seconds with scale={scale}")

    # Plot result
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image_rgb)
    circle = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
    ax.add_patch(circle)
    ax.set_title(f"Detected Plate Circle (scale={scale})")
    ax.axis('off')
    plt.show()

    


# visualize_circle_detection_from_path("scan_Plate_TM_p00_0_B05f00d0.TIF", scale=0.25)  # Fast
# visualize_circle_detection_from_path('/Volumes/Extreme SSD/BMDM-INVITRO2/31.5 NO CHANGE M/scan.2025-06-04-08-30-47/scan_Plate_TM_p00_0_B09f00d0.TIF', scale=0.15)   # Full resolution