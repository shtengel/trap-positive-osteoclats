import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import io
import math
import pandas as pd
from osteoclasts_nvtiro_segmentation import process_image


def display_image_batch(images, titles=None, columns=3):
    num_images = len(images)
    rows = math.ceil(num_images / columns)

    fig, axes = plt.subplots(rows, columns, figsize=(columns * 4, rows * 4))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < num_images:
            img = images[i]
            if img.ndim == 2:  # grayscale
                ax.imshow(img, cmap='gray')
            else:              # RGB/RGBA
                ax.imshow(img)
            ax.axis("off")
            if titles:
                ax.set_title(titles[i], fontsize=12)
        else:
            ax.remove()  # Remove unused subplots

    st.pyplot(fig)
    
# Set allowed file types
allowed_extensions = ["png", "tiff", "tif", "jpeg", "jpg"]

# Streamlit layout
st.set_page_config(layout="wide")
st.title("Image Processing App")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=allowed_extensions)

results_df = pd.DataFrame(columns=["Cell ID", "Area", "Confidence"])

if uploaded_file:
    # Create two columns
    col1, col2 = st.columns([1, 2])

    # Left pane: parameters
    with col1:
        st.subheader("Parameters")
        min_area = st.number_input("Min Area", min_value=0, value=500)
        intensity_threshold = st.number_input("Intensity Filter", min_value=0.0, value=600.0)
        numbered = st.checkbox("Numbered", value=True)
        model_type = st.selectbox("Model Type", ["vit_b_lm"])

        if st.button("Process Image"):
            # Save uploaded file to a temporary path-like object
            image_bytes = uploaded_file.read()
            image_stream = io.BytesIO(image_bytes)

            processed_img_array, result_dict, titles, features_df = process_image(
                image_path=uploaded_file.name,
                image_stream=image_stream,
                output_dir=None,
                model_type=model_type,
                intensity_threshold=intensity_threshold,
                min_area=min_area,
                numbered=numbered
            )
            
            results_df = features_df

            # Display the output image and dictionary below the columns
            st.markdown("---")
            st.subheader("Processed Batch")
            display_image_batch(images=processed_img_array, titles=titles)

    # Right pane: show the uploaded image
    with col2:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)


if st.button("Show Results Table"):
    st.session_state["show_modal"] = True

if st.session_state.get("show_modal", False):
    with st.expander("Show Results Table"):
        st.dataframe(results_df)
