import re
import numpy as np
from skimage import io, exposure
import matplotlib.pyplot as plt
from skimage.morphology import white_tophat, disk
from skimage.color import rgb2hed
from skimage.exposure import rescale_intensity
import numpy as np
import cv2
from skimage.morphology import white_tophat, disk


def apply_clahe(image):
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced


def subtract_background(image, radius=30):
    # Structuring element size should match the size of the background variations
    selem = disk(radius)
    background_removed = white_tophat(image, selem)
    return background_removed



def denoise_with_blur(image, kernel_size=3):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def normalize_image(image):
    image = image.astype(np.float32)
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def enhance_purple(image):
        
    # Convert to float for processing
    image_float = image.astype(np.float32)

    # Extract channels
    R = image_float[:, :, 0]
    G = image_float[:, :, 1]
    B = image_float[:, :, 2]

    # Step 1: Enhance purple
    purple_score = 0.5 * R + 0.5 * B - G
    purple_score = np.clip(purple_score, 0, 255)

    # Step 2: Normalize purple score to [0, 1]
    purple_score_norm = (purple_score - purple_score.min()) / (purple_score.max() - purple_score.min())

    # Step 3: Multiply original image by purple emphasis
    # This will darken non-purple areas and brighten purple ones
    enhanced_image = image_float * purple_score_norm[:, :, np.newaxis]
    enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)

    # Optional: Improve contrast
    enhanced_image = exposure.equalize_adapthist(enhanced_image)  # returns float in [0,1]
    enhanced_image = (enhanced_image * 255).astype(np.uint8)
    return enhanced_image

def preprocess_image(image_path):
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Step 1: Subtract background
    img_bg_subtracted = white_tophat(img, disk(30))
    
    # Step 2: CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_bg_subtracted)
    
    # Step 3: Normalize
    img_norm = (img_clahe - np.min(img_clahe)) / (np.max(img_clahe) - np.min(img_clahe))
    
    # Step 4: Optional denoising
    img_blur = cv2.GaussianBlur((img_norm * 255).astype(np.uint8), (3, 3), 0)
    
    return img_blur

def preprocess_trap_image(image_path):
    # Load RGB image
    img_rgb = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

    # Convert RGB to HED (Hematoxylin-Eosin-DAB)
    img_hed = rgb2hed(img_rgb)

    # DAB/TRAP signal is in the third channel; invert it (darker = stronger stain)
    trap_channel = -img_hed[:, :, 2]  # Now darker areas = more TRAP signal

    # Rescale intensity to 0–255 range for better visualization and processing
    trap_rescaled = rescale_intensity(trap_channel, out_range=(0, 255)).astype(np.uint8)

    # Optional: CLAHE to enhance contrast (gentle)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    trap_clahe = clahe.apply(trap_rescaled)

    # Normalize to [0, 1] float image for model input
    trap_normalized = trap_clahe.astype(np.float32) / 255.0

    return trap_normalized


# Step 4: Visualize the result
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
image = io.imread("scan_Plate_TM_p00_0_B05f00d0.TIF")  # or directly use your image variable
enhanced_image = preprocess_trap_image("scan_Plate_TM_p00_0_B05f00d0.TIF")
ax[0].imshow(image)
ax[0].set_title("Original Image")
ax[1].imshow(enhanced_image)
ax[1].set_title("Purple-Enhanced Image")
plt.tight_layout()
plt.show()
