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


def enhance_purple(image):
    image_float = image.astype(np.float32)

    R = image_float[:, :, 0]
    G = image_float[:, :, 1]
    B = image_float[:, :, 2]

    # Step 1: Compute stronger purple score
    # purple_score = 0.75 * R + 0.75 * B - 1.5 * G
    purple_score = 0.5 * R + 0.5 * B - G
    purple_score = np.clip(purple_score, 0, 255)

    # Step 2: Normalize and exaggerate purple score
    purple_score_norm = (purple_score - purple_score.min()) / (purple_score.max() - purple_score.min() + 1e-6)
    purple_score_boosted = purple_score_norm ** 2.0  # adjust this exponent as needed

    # Step 3: Create a purple-tinted boost image
    purple_tint = np.stack([
        np.full_like(R, 255),  # red channel full
        np.full_like(G, 50),   # green low
        np.full_like(B, 255)   # blue full
    ], axis=2).astype(np.float32)

    # Step 4: Blend original image and purple tint using boosted purple score
    alpha = purple_score_boosted[:, :, np.newaxis]
    enhanced_image = (1 - alpha) * image_float + alpha * purple_tint

    enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)

    # Optional: Global contrast adjustment
    enhanced_image = exposure.equalize_adapthist(enhanced_image, clip_limit=0.01)
    enhanced_image = (enhanced_image * 255).astype(np.uint8)

    return enhanced_image