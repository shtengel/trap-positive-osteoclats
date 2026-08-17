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
                # Add FINAL_STATS.csv first so it appears at the top of the archive
                zipf.write(final_csv_path, "FINAL_STATS.csv")
                for root, _, files in os.walk(output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, output_dir)
                        if arcname != "FINAL_STATS.csv":
                            zipf.write(file_path, arcname)

            final_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
            shutil.copy(zip_path, final_zip.name)

            return final_zip.name, stats_df

        return None, None

# --- Streamlit layout ---
st.set_page_config(layout="wide")
st.title("TRAP stained Osteoclasts Analyzer")

allowed_extensions = ["png", "tiff", "tif", "jpeg", "jpg"]
results_df = pd.DataFrame(columns=["Cell ID", "Area", "Confidence"])

# --- Sidebar Parameters ---
st.sidebar.header("Processing Parameters")
min_area = st.sidebar.number_input("Min Area", min_value=0, value=500)
st.sidebar.caption("Filter out cells by pixel size — any cell with an area **lower** than this value will be dropped.")
intensity_threshold = st.sidebar.number_input("Intensity Filter", min_value=0.0, value=600.0)
st.sidebar.caption("Cells with intensity **higher** than this value are considered dead (too bright/white) and dropped. Decrease the value to filter out more cells, increase it to keep more.")
numbered = st.sidebar.checkbox("Numbered Labels", value=True)
st.sidebar.caption("Overlay numeric IDs on each detected cell to match them to the results table.")
model_type = st.sidebar.selectbox("Model Type", ["vit_b_lm", "vit_t_lm", "vit_l_lm"])
model_desc = {
    "vit_t_lm": "ViT-T: Tiny model for fastest processing, lower accuracy",
    "vit_b_lm": "ViT-B: Base model with good balance between speed and accuracy",
    "vit_l_lm": "ViT-L: Large model with highest accuracy, slower inference",
}
st.sidebar.caption(model_desc[model_type])

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["🖼 Single Image", "📂 Batch Processing", "📖 Documentation"])

# --- Tab 3: Documentation ---
with tab3:
    st.subheader("Documentation & Guides")

    ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

    def show_image(filename, caption=None, width=None):
        """Render an image from the assets/ folder if it exists, else show a notice."""
        path = os.path.join(ASSETS_DIR, filename)
        if os.path.isfile(path):
            st.image(path, caption=caption, width=width)
        else:
            st.info(f"📷 Add screenshot at `assets/{filename}` to display here.")

    section = st.radio(
        "Jump to section:",
        ["Overview", "Recommended Workflow", "Parameters Guide", "Single Image Tutorial", "Batch Processing Tutorial"],
        horizontal=True,
    )

    if section == "Overview":
        st.markdown("""
        ### What this tool does
        This app automatically detects and analyzes **TRAP-stained osteoclasts** in microscopy images.

        TRAP (Tartrate-Resistant Acid Phosphatase) staining highlights multinucleated
        osteoclasts — the bone-resorbing cells. This tool:

        - Segments individual osteoclasts from your images using a vision foundation model.
        - Measures each cell's **area** and **confidence score**.
        - Lets you filter out small artifacts or over-stained (dead) cells.
        - Exports per-cell statistics and annotated overlay images.

        **Intended users:** researchers in bone biology, dental research, and related fields
        who need consistent, quantitative cell measurements across many images.
        """)
        show_image("overview_example.png", caption="Example: detected osteoclasts overlaid on a TRAP-stained image.")

    elif section == "Recommended Workflow":
        st.markdown("""
        ### Recommended Workflow: calibrate before batching

        Before running a full batch, **calibrate the parameters on a few
        representative images** using the 🖼 Single Image tab:

        1. Start with the most permissive values:
           **`Min Area = 0`** and **`Intensity Filter = max`** — so nothing is dropped.
        2. Process a handful of representative images.
        3. Manually inspect the annotated overlays. Note the smallest *true*
           osteoclasts and the brightest *dead / over-stained* cells.
        4. Set **`Min Area`** just **below** the smallest true cell you want to keep.
        5. Set **`Intensity Filter`** just **above** the brightest dead cell you want to drop.
        6. Re-run the same single images to confirm the filters look correct.
        7. Only then switch to 📂 Batch Processing with the chosen values.
        """)
        show_image("workflow_calibration.png", caption="Calibrate filters on single images before running a batch.")

    elif section == "Parameters Guide":
        st.markdown("""
        ### Parameters Guide
        All parameters live in the left sidebar.

        #### Min Area
        Filter cells by pixel size. Any cell with an area **lower** than this value is dropped.
        Use a higher value to remove small fragments; use a lower value to keep small cells.

        #### Intensity Filter
        Cells with mean intensity **higher** than this value are considered over-stained (dead)
        and dropped. Decrease to filter out more, increase to keep more.

        #### Numbered Labels
        Overlay numeric IDs on each detected cell so you can match them to the results table.

        #### Model Type
        Choose between three SAM-based backbones:
        - **ViT-T (Tiny)** — fastest, lowest accuracy.
        - **ViT-B (Base)** — balanced speed and accuracy. *Default.*
        - **ViT-L (Large)** — highest accuracy, slower inference.

        For more details on choosing a model, see the
        [micro-SAM model guide](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#choosing-a-model).
        """)
        show_image("parameters_sidebar.png", caption="The parameter sidebar.", width=400)

    elif section == "Single Image Tutorial":
        st.markdown("""
        ### Single Image Tutorial

        1. Open the **🖼 Single Image** tab.
        2. Click *Choose a single image file* and select a `.png`, `.tif`, or `.jpg`.
        3. Adjust the parameters in the sidebar if needed.
        4. Click **Process Image**.
        5. Review the side-by-side comparison: your original image and the annotated overlay.
        6. Expand **📊 Show Results Table** to see per-cell measurements.
        """)
        show_image("tutorial_single_upload.png", caption="Step 2: Upload a single image.")
        show_image("tutorial_single_output.png", caption="Step 5: Original vs. processed output.")

    elif section == "Batch Processing Tutorial":
        st.markdown("""
        ### Batch Processing Tutorial

        1. Open the **📂 Batch Processing** tab.
        2. Click *Upload multiple image files* and select many files at once (Ctrl/⌘-click).
        3. Click **Process Uploaded Batch** — a progress bar will update as each image runs.
        4. When complete, click **📦 Download Results (ZIP)**.
        5. The ZIP contains annotated overlays, mask images, and a `FINAL_STATS.csv`
           summary table. The CSV is the first file in the archive for easy access.
        6. Expand **📊 Show Summary Table** to preview the CSV inside the app.

        #### What is in `FINAL_STATS.csv`?
        - `image_name`: name of the input image.
        - `num_cells`: number of cells that passed the filters.
        - `mean_area`: average cell area in pixels.
        - `mean_perimeter`: average cell perimeter in pixels.
        - `plate_coverage_percent`: estimated area covered by detected cells.

        Rows are ordered by plate group/column for `_B01f`-style filenames, and naturally
        (1, 2, 3, ..., 10) for plain numbered filenames.

        > **Note:** Segmentation results may vary slightly between different computers
        due to hardware differences, floating-point behavior, and dependency versions.
        Always verify outputs on your own system.
        """)
        show_image("tutorial_batch_upload.png", caption="Step 2: Select multiple files.")
        show_image("tutorial_batch_results.png", caption="Step 4: Download the results ZIP.")


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
            display_df = features_df.drop(columns=["plate_coverage_percent"], errors="ignore")
            st.dataframe(display_df.reset_index(drop=True), hide_index=True)
        
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
