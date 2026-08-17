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

def extract_shape_features(segmentation, original_image, plate_mask=None):
    """
    Extract shape features (area, perimeter, and mean intensity) from segmentation mask

    Args:
        segmentation: Instance segmentation mask
        original_image: Original input image for intensity measurements
        min_area: Minimum area threshold for cell filtering
        plate_mask: Optional binary mask of the plate region

    Returns:
        DataFrame with area, perimeter, and mean intensity for each cell
        Filtered segmentation mask
    """
    # Apply plate mask if provided
    if plate_mask is not None:
        filtered_segmentation = segmentation.copy()
        filtered_segmentation[~plate_mask] = 0
    else:
        filtered_segmentation = segmentation.copy()

    # Get region properties for each labeled region
    props = regionprops(filtered_segmentation, intensity_image=original_image)

    # Extract area, perimeter, and mean intensity for each cell
    features = []
    for prop in props:
        # Skip background (label 0)
        if prop.label > 0:
            # Handle different types of intensity values
            if hasattr(prop.mean_intensity, '__iter__'):
                mean_intensity = np.sum(prop.mean_intensity)
            else:
                mean_intensity = prop.mean_intensity

            features.append({
                'cell_id': prop.label,
                'area': prop.area,
                'perimeter': prop.perimeter,
                'mean_intensity': mean_intensity
            })

    return pd.DataFrame(features), filtered_segmentation