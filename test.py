import numpy as np
from skimage import io, exposure
import matplotlib.pyplot as plt

# Load image
# 

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


# Step 4: Visualize the result
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
image = io.imread("scan_Plate_TM_p00_0_D03f00d0.TIF")  # or directly use your image variable
enhanced_image = enhance_purple(image)
ax[0].imshow(image)
ax[0].set_title("Original Image")
ax[1].imshow(enhanced_image)
ax[1].set_title("Purple-Enhanced Image")
plt.tight_layout()
plt.show()
