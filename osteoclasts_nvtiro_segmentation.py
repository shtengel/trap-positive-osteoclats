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

from segmentation.msam_segmentation import run_automatic_instance_segmentation
from preprocess.features import extract_shape_features
from masks.circle_detection_mask import optimize_circle_detection
from filters.filter import filter_dataframe

FIXED_COLOR = [123,43,250]


def save_float_image_as_png(image_float, filename):
    """
    Save a float image (0.0 to 1.0) as an 8-bit PNG.
    """
    image_uint8 = (np.clip(image_float, 0, 1) * 255).astype(np.uint8)
    imageio.imwrite(filename, image_uint8)


def visualize_segmentation(original_image, segmentation, random_colors=None):
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
    if random_colors is None:
        random_colors = np.random.randint(0, 255, size=(max_label + 1, 3))
        # Make background black
        random_colors[0] = [0, 0, 0]
    
    # Create RGB segmentation visualization
    segmentation_vis = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
    for label in range(1, max_label + 1):
        mask = segmentation == label
        if np.any(mask):  # Only process if mask contains any True values
            segmentation_vis[mask] = random_colors[label] #FIXED_COLOR
    
    return segmentation_vis, random_colors


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

def add_numbers_to_image(visualize_segmentation_image, filtered_segmentation, features_df): 
    """
    Adds a numbering to each segmented cell

    Args:
        visualize_segmentation_image (_type_): _description_
        filtered_segmentation (_type_): _description_
        features_df (_type_): _description_
    """
    # Make a copy to annotate
    vis_with_numbers = visualize_segmentation_image # visualize_segmentation_image.copy()
    
    # Get region properties for the filtered segmentation
    props = regionprops(filtered_segmentation)

    # Map label to centroid for quick lookup
    label_to_centroid = {prop.label: prop.centroid for prop in props}

    # Iterate through the features DataFrame in CSV order
    for idx, row in features_df.iterrows():
        cell_label = row['cell_id']
        if cell_label in label_to_centroid:
            y, x = map(int, label_to_centroid[cell_label])
            # Write the CSV row number (starting from 1) at the centroid
            cv2.putText(
                vis_with_numbers,
                str(int(cell_label)),         # CSV row number (1-based)
                (x, y),               # (x, y) coordinates
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,                  # Font scale
                (255, 255, 255),      # White text
                3,                    # Thickness
                cv2.LINE_AA
            )

def process_image(image_path, output_dir, model_type="vit_b_lm", intensity_threshold=0, min_area=200, numbered=False, image_stream=None):
    """
    Process a single image with MicroSAM and save outputs
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save outputs
        model_type: MicroSAM model type
        intensity_threshold: Optional threshold for filtering cells by intensity
        min_area: Minimum area threshold for cell filtering
        numbered: Write cell number on each cell
    
    Returns:
        Dictionary with image statistics for final CSV
    """
    # Get image filename without extension
    filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Read the image
    try:
        image = imageio.imread(image_path)
    except Exception as e:
        try:
            image = imageio.imread(image_stream)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            return (
                None,
                {
                    'image_name': filename,
                    'num_cells': 0,
                    'mean_area': 0,
                    'mean_perimeter': 0,
                    'plate_coverage_percent': 0.0,
                },
                [],
                pd.DataFrame(),
            )

    # Run instance segmentation
    try:
        segmentation, segmentation_image = run_automatic_instance_segmentation(image, model_type=model_type)
    except Exception as e:
        print(f"Error in segmentation for {filename}: {e}")
        return (
            None,
            {
                'image_name': filename,
                'num_cells': 0,
                'mean_area': 0,
                'mean_perimeter': 0,
                'plate_coverage_percent': 0.0,
            },
            [],
            pd.DataFrame(),
        )
    
    # Detect plate region using optimized circle detection
    plate_mask, plate_info = optimize_circle_detection(image)
    
    # Extract shape features and apply area filtering
    features_df, plate_segmentation = extract_shape_features(
        segmentation, segmentation_image, plate_mask=plate_mask
    )
    
    all_cells_count = len(features_df)
    
    # Filter cells based on intensity threshold if specified
    # filtered_segmentation = plate_segmentation.copy()
    # dead_cells_segmentation = plate_segmentation.copy()
    # dead_cells_features_df = features_df.copy()
    
    # min area filter
    features_df, area_filtered_segmentation = filter_dataframe(features_df, features_df['area'] > min_area, plate_segmentation) 
    
    # perimeter filter 
    features_df, area_filtered_segmentation = filter_dataframe(features_df, features_df['perimeter'] < 700, area_filtered_segmentation) 
    
    area_filtered_cells_count = len(features_df)
    
    # mean intensity filter
    features_df, intensity_filtered_segmentation = filter_dataframe(features_df, features_df['mean_intensity'] < intensity_threshold, area_filtered_segmentation)
    
    # Create visualizations (without drawing the circle)
    segmentation_vis, random_colors = visualize_segmentation(image, segmentation)
    area_filtered_vis, _ = visualize_segmentation(image, area_filtered_segmentation, random_colors)
    final_filtered_vis, _ = visualize_segmentation(image, intensity_filtered_segmentation, random_colors)
    
    if numbered:
        add_numbers_to_image(final_filtered_vis, segmentation, features_df)
    
    if output_dir:
        # Create output paths
        os.makedirs(output_dir, exist_ok=True)
        original_output_path = os.path.join(output_dir, f"{filename}_original.png")
        final_filtered_output_path = os.path.join(output_dir, f"{filename}_final_filtered.png")
        features_output_path = os.path.join(output_dir, f"{filename}_features.csv")
        
        # Save outputs
        plt.figure(figsize=(20, 5))
        
        # Original image
        plt.subplot(1, 4, 1)
        plt.imshow(segmentation_image)
        plt.title("Original Image")
        plt.axis("off")
        
        # Original segmentation visualization
        plt.subplot(1, 4, 3)
        plt.imshow(segmentation_vis)
        plt.title(f"All Cells ({all_cells_count})")
        plt.axis("off")
        
        # Area-filtered segmentation visualization
        plt.subplot(1, 4, 4)
        plt.imshow(area_filtered_vis)
        plt.title(f"Area Filtered ({area_filtered_cells_count}) (>{min_area} px²)")
        plt.axis("off")
        
        # Dead Cells segmentation visualization
        # plt.subplot(1, 5, 2)
        # plt.imshow(deadcells_filtered_vis)
        # plt.title(f"Dead Cells ({len(dead_cells_features_df)})")
        # plt.axis("off")
        
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
        imageio.imwrite(original_output_path, segmentation_image)
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
        # 'num_dead_cells': len(dead_cells_features_df),
        'mean_area': features_df['area'].mean() if not features_df.empty else 0,
        'mean_perimeter': features_df['perimeter'].mean() if not features_df.empty else 0,
        'plate_coverage_percent': plate_coverage
    }
    
    return [final_filtered_vis], image_stats, [f"Final Filtered ({len(features_df)} cells)"], features_df


def process_directory(input_dir, output_dir, model_type="vit_b_lm", intensity_threshold=0, min_area=200, numbered=False):
    """
    Process all images in a directory with MicroSAM with progress bar
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save outputs
        model_type: MicroSAM model type
        intensity_threshold: Optional intensity ceiling; cells with mean intensity >= this value are dropped
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
    print(f"Intensity filtering: {'Enabled at threshold {intensity_threshold}' if intensity_threshold is not None else 'Disabled'}")
    print(f"Area filtering: Enabled (min area = {min_area} pixels²)")
    
    # Process each image and collect statistics with progress bar
    all_image_stats = []
    
    for image_path in tqdm(image_files, desc="Processing images", unit="image"):
        print(f"\nProcessing {os.path.basename(image_path)}")
        image_stats = process_image(image_path, output_dir, model_type, intensity_threshold, min_area, numbered)
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
    parser.add_argument("--intensity-filter", type=float, default=600,
                       help="Filter cells by keeping only those below the specified mean intensity (e.g., 600 filter cells with higher intensity(more white))")
    parser.add_argument("--min-area", type=int, default=500,
                       help="Minimum cell area in pixels² for filtering small objects")
    parser.add_argument("--numbered", action="store_true", help="Show cells with numbers")
    
    args = parser.parse_args()
    
    start_time = time()
    process_directory(args.input, args.output, args.model, args.intensity_filter, args.min_area, args.numbered)
    end_time = time()
    
    # conda activate micro-sam
    print(f"\nTotal execution time: {(end_time - start_time) / 60:.2f} minutes")