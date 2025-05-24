import pandas as pd

def filter_dataframe(df, filter_condition, segmentation):
    """
    Filters a DataFrame using a given boolean Series.
    
    Parameters:
    - df: The input DataFrame.
    - filter_condition: A boolean Series indicating which rows to keep.
    
    Returns:
    - A tuple containing:
        1. Filtered DataFrame
        2. List of 'cell_id' from rows that were dropped
    """
    # Copy Segmentation
    filtered_segmentation = segmentation.copy()
    
    # Invert the filter to find dropped rows
    dropped_ids = df.loc[~filter_condition, 'cell_id'].tolist()
    
    # Apply the filter
    filtered_df = df[filter_condition].reset_index(drop=True)
    
    for cell_id in dropped_ids:
        filtered_segmentation[filtered_segmentation == cell_id] = 0
    
    return filtered_df, filtered_segmentation