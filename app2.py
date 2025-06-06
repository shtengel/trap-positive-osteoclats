import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import io
import math
import pandas as pd
import tempfile
import os
import zipfile
import shutil
from tqdm import tqdm
from osteoclasts_nvtiro_segmentation import process_image
from utilities.utils import sort_images_by_group_and_column

# --- Utility: display images ---
def display_image_batch(images, titles=None, columns=3):
    num_images = len(images)
    rows = math.ceil(num_images / columns)

    fig, axes = plt.subplots(rows, columns, figsize=(columns * 4, rows * 4))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < num_images:
            img = images[i]
            if img.ndim == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
            ax.axis("off")
            if titles:
                ax.set_title(titles[i], fontsize=12)
        else:
            ax.remove()
    st.pyplot(fig)

# --- Utility: process batch of uploaded files ---
def process_uploaded_files(uploaded_files, model_type="vit_b_lm", intensity_threshold=0, min_area=200, numbered=False):
    with tempfile.TemporaryDirectory() as tmp_input_dir:
        output_dir = os.path.join(tmp_input_dir, "results")
        os.makedirs(output_dir, exist_ok=True)

        input_paths = []
        for file in uploaded_files:
            save_path = os.path.join(tmp_input_dir, file.name)
            with open(save_path, "wb") as f:
                f.write(file.read())
            input_paths.append(save_path)

        all_image_stats = []
        total = len(input_paths)

        # Streamlit progress bar and status
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, image_path in enumerate(input_paths):
            status_text.text(f"Processing {os.path.basename(image_path)} ({idx + 1}/{total})")

            visArr, image_stats, titles, features = process_image(
                image_path=image_path,
                output_dir=output_dir,
                model_type=model_type,
                intensity_threshold=intensity_threshold,
                min_area=min_area,
                numbered=numbered
            )
            if image_stats:
                all_image_stats.append(image_stats)

            progress_bar.progress((idx + 1) / total)

        status_text.text("Processing complete.")
        progress_bar.empty()

        if all_image_stats:
            sorted_image_stats = sort_images_by_group_and_column(all_image_stats)
            stats_df = pd.DataFrame(sorted_image_stats)
            final_csv_path = os.path.join(output_dir, "FINAL_STATS.csv")
            stats_df.to_csv(final_csv_path, index=False)

            zip_path = os.path.join(tmp_input_dir, "results.zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, output_dir)
                        zipf.write(file_path, arcname)

            final_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
            shutil.copy(zip_path, final_zip.name)

            return final_zip.name, stats_df

        return None, None

# --- Streamlit layout ---
st.set_page_config(layout="wide")
st.title("Image Processing App")

allowed_extensions = ["png", "tiff", "tif", "jpeg", "jpg"]
results_df = pd.DataFrame(columns=["Cell ID", "Area", "Confidence"])

# --- Sidebar Parameters ---
st.sidebar.header("Processing Parameters")
min_area = st.sidebar.number_input("Min Area", min_value=0, value=500)
st.sidebar.caption("Filter out small cells by pixel size")
intensity_threshold = st.sidebar.number_input("Intensity Filter", min_value=0.0, value=600.0)
st.sidebar.caption("Filter out dead cells by how 'White' they are, higher values -> more dead cells")
numbered = st.sidebar.checkbox("Numbered Labels", value=True)
model_type = st.sidebar.selectbox("Model Type", ["vit_b_lm", "vit_t_lm", "vit_l_lm"])
model_desc = {
    "vit_t_lm": "ViT-T: Tiny model for fastest processing, lower accuracy",
    "vit_b_lm": "ViT-B: Base model with good balance between speed and accuracy",
    "vit_l_lm": "ViT-L: Large model with highest accuracy, slower inference",
}
st.sidebar.caption(model_desc[model_type])

# --- Tabs ---
tab1, tab2 = st.tabs(["🖼 Single Image", "📂 Batch Processing"])

# --- Tab 1: Single Image ---
with tab1:
    st.subheader("Single Image Processing")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        uploaded_file = st.file_uploader("Choose a single image file", type=allowed_extensions, key="single")
        process_clicked = st.button("Process Image")

    with col_right:
        if uploaded_file:
            uploaded_image_preview = Image.open(uploaded_file)
            st.image(uploaded_image_preview, width=150, caption="Preview")

    if uploaded_file and process_clicked:
        with st.spinner("Processing image..."):
            uploaded_file.seek(0)  # reset pointer
            image_bytes = uploaded_file.read()
            process_stream = io.BytesIO(image_bytes)

            processed_img_array, result_dict, titles, features_df = process_image(
                image_path=uploaded_file.name,
                image_stream=process_stream,
                output_dir=None,
                model_type=model_type,
                intensity_threshold=intensity_threshold,
                min_area=min_area,
                numbered=numbered
            )

        st.subheader("Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_image_preview, caption="Uploaded Image", use_column_width=True)
        with col2:
            st.subheader("Processed Image")
            display_image_batch(images=processed_img_array, titles=titles)

        with st.expander("📊 Show Results Table"):
            st.dataframe(features_df.reset_index(drop=True), hide_index=True)
        
# --- Tab 2: Batch Processing ---
with tab2:
    st.subheader("Batch Processing (Multiple Images)")
    uploaded_files = st.file_uploader("Upload multiple image files from a folder", type=allowed_extensions, accept_multiple_files=True, key="multi")

    if uploaded_files and st.button("Process Uploaded Batch", key="process_batch"):
        with st.spinner("Processing batch..."):
            zip_path, batch_stats_df = process_uploaded_files(
                uploaded_files,
                model_type=model_type,
                intensity_threshold=intensity_threshold,
                min_area=min_area,
                numbered=numbered
            )

        if zip_path:
            st.success("Batch processing complete.")
            with open(zip_path, "rb") as f:
                st.download_button("📦 Download Results (ZIP)", f, file_name="results.zip", mime="application/zip")

            with st.expander("📊 Show Summary Table"):
                st.dataframe(batch_stats_df)
        else:
            st.warning("No images were successfully processed.")
