import os
import glob
import numpy as np
import pandas as pd
import imageio
import matplotlib.pyplot as plt
from skimage.measure import label as connected_components
from skimage.measure import regionprops
from skimage.color import rgb2gray
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


def run_automatic_instance_segmentation(image, model_type="vit_b_lm"):
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
        input_=image,
        ndim=2,
    )

    # Step 3: Combining the decoder with the Segment Anything backbone for automatic instance segmentation.
    ais = InstanceSegmentationWithDecoder(predictor, decoder)

    # Step 4: Initializing the precomputed image embeddings to perform faster automatic instance segmentation.
    ais.initialize(
        image=image,
        image_embeddings=image_embeddings,
    )

    # Step 5: Getting automatic instance segmentations for the given image and applying the relevant post-processing steps.
    prediction = ais.generate()
    prediction = mask_data_to_segmentation(prediction, with_background=True)

    return prediction


def extract_shape_features(segmentation, original_image, min_area=200, plate_mask=None):
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
            # Filter by area
            if prop.area >= min_area:
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
            else:
                # Remove small cells from segmentation
                filtered_segmentation[filtered_segmentation == prop.label] = 0
    
    return pd.DataFrame(features), filtered_segmentation


def visualize_segmentation(original_image, segmentation):
    """
    Create visualization of the segmentation without drawing the circle
    
    Args:
        original_image: Original input image
        segmentation: Instance segmentation mask
        
    Returns:
        RGB visualization of the segmentation
    """
    # Generate random colors for visualization
    max_label = np.max(segmentation) if np.max(segmentation) > 0 else 1
    random_colors = np.random.randint(0, 255, size=(max_label + 1, 3))
    # Make background black
    random_colors[0] = [0, 0, 0]
    
    # Create RGB segmentation visualization
    segmentation_vis = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
    for label in range(1, max_label + 1):
        mask = segmentation == label
        if np.any(mask):  # Only process if mask contains any True values
            segmentation_vis[mask] = random_colors[label]
    
    return segmentation_vis


def calculate_plate_coverage(features_df, plate_radius):
    """
    Calculate the percentage of the plate area covered by cells
    
    Args:
        features_df: DataFrame with cell features
        plate_radius: Radius of the plate in pixels
        
    Returns:
        Percentage of plate area covered by cells
    """
    if features_df.empty:
        return 0.0
    
    # Calculate total cell area
    total_cell_area = features_df['area'].sum()
    
    # Calculate plate area
    plate_area = np.pi * (plate_radius ** 2)
    
    # Calculate coverage percentage
    coverage_percentage = (total_cell_area / plate_area) * 100
    
    return coverage_percentage


def process_image(image_path, output_dir, model_type="vit_b_lm", intensity_percentile=None, min_area=200):
    """
    Process a single image with MicroSAM and save outputs
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save outputs
        model_type: MicroSAM model type
        intensity_percentile: Optional percentile threshold for filtering cells by intensity
        min_area: Minimum area threshold for cell filtering
    
    Returns:
        Dictionary with image statistics for final CSV
    """
    # Get image filename without extension
    filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Read the image
    try:
        image = imageio.imread(image_path)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None
    
    # Detect plate region using optimized circle detection
    plate_mask, plate_info = optimize_circle_detection(image)
    
    # Run instance segmentation
    try:
        segmentation = run_automatic_instance_segmentation(image, model_type=model_type)
    except Exception as e:
        print(f"Error in segmentation for {filename}: {e}")
        return None
    
    # Extract shape features and apply area filtering
    features_df, area_filtered_segmentation = extract_shape_features(
        segmentation, image, min_area=min_area, plate_mask=plate_mask
    )
    
    # Filter cells based on intensity percentile if specified
    filtered_segmentation = area_filtered_segmentation.copy()
    if intensity_percentile is not None and not features_df.empty:
        # Calculate intensity threshold based on percentile
        intensity_threshold = 400 # 550 # np.percentile(features_df['mean_intensity'], intensity_percentile)
        
        # Filter out cells with intensity above the threshold
        high_intensity_cells = features_df[features_df['mean_intensity'] > intensity_threshold]['cell_id'].values
        for cell_id in high_intensity_cells:
            filtered_segmentation[filtered_segmentation == cell_id] = 0
        
        # Update features DataFrame to include only remaining cells
        features_df = features_df[features_df['mean_intensity'] <= intensity_threshold].reset_index(drop=True)
    
    # Create visualizations (without drawing the circle)
    segmentation_vis = visualize_segmentation(image, segmentation)
    area_filtered_vis = visualize_segmentation(image, area_filtered_segmentation)
    final_filtered_vis = visualize_segmentation(image, filtered_segmentation)
    
    # Create output paths
    os.makedirs(output_dir, exist_ok=True)
    original_output_path = os.path.join(output_dir, f"{filename}_original.png")
    segmentation_output_path = os.path.join(output_dir, f"{filename}_segmentation.png")
    area_filtered_output_path = os.path.join(output_dir, f"{filename}_area_filtered.png")
    final_filtered_output_path = os.path.join(output_dir, f"{filename}_final_filtered.png")
    features_output_path = os.path.join(output_dir, f"{filename}_features.csv")
    
    # Save outputs
    plt.figure(figsize=(20, 5))
    
    # Original image
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")
    
    # Original segmentation visualization
    plt.subplot(1, 4, 4)
    plt.imshow(segmentation_vis)
    plt.title(f"All Cells ({np.max(segmentation)})")
    plt.axis("off")
    
    # Area-filtered segmentation visualization
    plt.subplot(1, 4, 3)
    plt.imshow(area_filtered_vis)
    plt.title(f"Area Filtered (>{min_area} px²)")
    plt.axis("off")
    
    # Final filtered segmentation visualization
    plt.subplot(1, 4, 2)
    plt.imshow(final_filtered_vis)
    plt.title(f"Final Filtered ({len(features_df)} cells)")
    plt.axis("off")
    
    # Save combined visualization
    plt.tight_layout()
    plt.savefig(final_filtered_output_path)
    plt.close()
    
    # Save original image and segmentations separately
    #imageio.imwrite(original_output_path, image)
    #imageio.imwrite(segmentation_output_path, segmentation_vis)
    #imageio.imwrite(area_filtered_output_path, area_filtered_vis)
    
    # Save features CSV
    features_df.to_csv(features_output_path, index=False)
    
    # Calculate plate coverage
    plate_coverage = calculate_plate_coverage(features_df, plate_info[2])
    
    # Calculate image statistics for final CSV
    image_stats = {
        'image_name': filename,
        'num_cells': len(features_df),
        'mean_area': features_df['area'].mean() if not features_df.empty else 0,
        'mean_perimeter': features_df['perimeter'].mean() if not features_df.empty else 0,
        'plate_coverage_percent': plate_coverage
    }
    
    return image_stats


def process_directory(input_dir, output_dir, model_type="vit_b_lm", intensity_percentile=None, min_area=200):
    """
    Process all images in a directory with MicroSAM with progress bar
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save outputs
        model_type: MicroSAM model type
        intensity_percentile: Optional percentile threshold for filtering cells by intensity
        min_area: Minimum area threshold for cell filtering
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files in input directory
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.TIF", "*.tiff", "*.tif"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    print(f"Found {len(image_files)} images in {input_dir}")
    print(f"Intensity filtering: {'Enabled at {intensity_percentile}th percentile' if intensity_percentile is not None else 'Disabled'}")
    print(f"Area filtering: Enabled (min area = {min_area} pixels²)")
    
    # Process each image and collect statistics with progress bar
    all_image_stats = []
    
    for image_path in tqdm(image_files, desc="Processing images", unit="image"):
        print(f"\nProcessing {os.path.basename(image_path)}")
        image_stats = process_image(image_path, output_dir, model_type, intensity_percentile, min_area)
        if image_stats:
            all_image_stats.append(image_stats)
            print(f"  Cells detected: {image_stats['num_cells']}")
            print(f"  Plate coverage: {image_stats['plate_coverage_percent']:.2f}%")
    
    # Create final statistics CSV
    if all_image_stats:
        stats_df = pd.DataFrame(all_image_stats)
        final_csv_path = os.path.join(output_dir, "FINAL_STATS.csv")
        stats_df.to_csv(final_csv_path, index=False)
        
        print(f"\nSaved final statistics to {final_csv_path}")
        
        # Print summary
        print("\nSummary:")
        print(f"Total images processed: {len(all_image_stats)}")
        print(f"Average cells per image: {stats_df['num_cells'].mean():.2f}")
        print(f"Average plate coverage: {stats_df['plate_coverage_percent'].mean():.2f}%")
        
        return stats_df
    else:
        print("No images were successfully processed.")
        return None


if __name__ == "__main__":
    import argparse
    from time import time
    
    parser = argparse.ArgumentParser(description="Process directory of images with MicroSAM")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing images")
    parser.add_argument("--output", type=str, required=True, help="Output directory for results")
    parser.add_argument("--model", type=str, default="vit_b_lm", 
                       help="MicroSAM model type (vit_b_lm, vit_h_lm, etc.)")
    parser.add_argument("--intensity-percentile", type=float, default=80,
                       help="Filter cells by keeping only those below the specified percentile of mean intensity (e.g., 80 keeps the lowest 80%)")
    parser.add_argument("--min-area", type=int, default=500,
                       help="Minimum cell area in pixels² for filtering small objects")
    
    args = parser.parse_args()
    
    start_time = time()
    # process_directory(args.input, args.output, args.model, args.intensity_percentile, args.min_area)
    process_directory(args.input, args.output, args.model, None, args.min_area)
    end_time = time()
    
    # conda activate micro-sam
    print(f"\nTotal execution time: {(end_time - start_time) / 60:.2f} minutes")