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

from joblib import Memory
memory = Memory(location=os.path.join(os.getcwd(), "joblib_cache"), verbose=0)

# @memory.cache
def run_automatic_instance_segmentation(image_input, model_type="vit_b_lm"):
    """Automatic Instance Segmentation by training an additional instance decoder in SAM.

    Args:
        image: The input image.
        model_type: The choice of the `µsam` model.

    Returns:
        The instance segmentation.
    """
    # Step 1: Initialize the model attributes using the pretrained µsam model weights.
    predictor, decoder = get_predictor_and_decoder(
        model_type=model_type,
        checkpoint_path=None,
    )

    # Step 2: Computation of the image embeddings from the vision transformer-based image encoder.
    image_embeddings = util.precompute_image_embeddings(
        predictor=predictor,
        input_=image_input,
        ndim=2,
    )

    # Step 3: Combining the decoder with the Segment Anything backbone for automatic instance segmentation.
    ais = InstanceSegmentationWithDecoder(predictor, decoder)

    # Step 4: Initializing the precomputed image embeddings to perform faster automatic instance segmentation.
    ais.initialize(
        image=image_input,
        image_embeddings=image_embeddings,
    )

    # Step 5: Getting automatic instance segmentations for the given image and applying the relevant post-processing steps.
    prediction = ais.generate()
    prediction = mask_data_to_segmentation(prediction, with_background=True)

    return prediction, image_input