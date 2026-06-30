# app/pages/1_Image_Inspector.py

import io
import streamlit as st
from pathlib import Path
from PIL import Image

from streamlit_image_comparison import image_comparison
from components.dual_view import render_dual_view
from components.sidebar import render_sidebar
from components.header import render_header
from utils.api_client import ApiClient
from utils.image_ops import image_to_base64, image_to_bytes, figure_to_bytes, get_difference_heatmap, get_pixel_intensity_histogram

# Page Config
st.set_page_config(
    page_title="Image Inspector | Denoiser Studio",
    page_icon="🔍",
    layout="wide"
)

client = ApiClient()

# Render the Sidebar
render_sidebar(show_controls=True)

if "denoised_image_bytes" not in st.session_state:
    st.session_state.denoised_image_bytes = None
if "original_image_bytes" not in st.session_state:
    st.session_state.original_image_bytes = None
if "b64_original" not in st.session_state:
    st.session_state.b64_original = None
if "b64_denoised" not in st.session_state:
    st.session_state.b64_denoised = None
if "processed_file_name" not in st.session_state:
    st.session_state.processed_file_name = None

# MAIN TITLE
render_header(
    title="🔍 Image Inspector", 
    subtitle="Upload a high-resolution scan to remove noise using the GPU-accelerated engine."
)

# FILE UPLOADER
upl_col1, upl_col2 = st.columns(2)
with upl_col1:
    uploaded_file = st.file_uploader(
        "Drag & Drop Scan (PNG, JPG, TIF)", 
        type=["png", "jpg", "jpeg", "tif"],
        key="scan_uploader",
        width="stretch"
    )

    # ACTION BUTTON
    server_config = client.get_config()
    is_gpu = server_config.get("enable_cuda", False)
    model_name = server_config.get("model_name", "Unknown Model")
    device_label = "GPU ⚡" if is_gpu else "CPU 🔳"

    is_denoiser_disabled = (uploaded_file is None) and (st.session_state.get("scan_uploader") is None)
    denoiser_button_help = "Upload a file to enable processing" if is_denoiser_disabled else "Click to process on " + device_label

    if st.button("✨ Denoise Scan", type="primary", width="stretch", disabled=is_denoiser_disabled, help=denoiser_button_help):
        if not client.check_health():
            st.error("❌ Engine is Offline. Check sidebar.")

        elif uploaded_file is not None:

            live_file_bytes = uploaded_file.getvalue()

            status_msg = f"Processing on {device_label} ({model_name})..."
            with st.spinner(status_msg):

                result_bytes = client.predict(live_file_bytes, uploaded_file.name)
                
                if result_bytes:
                    # SUCCESS: Update ALL State here
                    st.session_state.denoised_image_bytes = result_bytes
                    st.session_state.original_image_bytes = live_file_bytes
                    st.session_state.processed_file_name = uploaded_file.name
                    
                    # Clear caches to force regeneration for the new pair
                    st.session_state.b64_original = None
                    st.session_state.b64_denoised = None
                    
                    st.rerun()
                else:
                    st.error("Processing failed.")

with upl_col2:
    if uploaded_file is not None:
        # Open the image to create a preview
        preview_image = Image.open(uploaded_file)
        
        # thumbnail modifies the image in-place, keeping aspect ratio intact
        preview_image.thumbnail((500, 500)) 
        
        # Display it centered in the column
        st.image(
            preview_image, 
            caption=f"Preview: {uploaded_file.name}", 
            use_container_width=False
        )
    else:
        # Show a helpful placeholder when empty
        st.info("🖼️ Image preview will appear here.")


# RESULT DISPLAY
if st.session_state.denoised_image_bytes and st.session_state.original_image_bytes:

    # Reconstruct images from memory
    res_original = Image.open(io.BytesIO(st.session_state.original_image_bytes)).convert("RGB")
    res_denoised = Image.open(io.BytesIO(st.session_state.denoised_image_bytes)).convert("RGB")
    
    # OPTIMIZATION: Generate Base64 strings ONCE
    if st.session_state.b64_original is None:
        with st.spinner("Preparing interactive viewer..."):
            st.session_state.b64_original = image_to_base64(res_original)
            st.session_state.b64_denoised = image_to_base64(res_denoised)

    str_b64_original = st.session_state.b64_original or ""
    str_b64_denoised = st.session_state.b64_denoised or ""

    # COMPARISON SLIDER
    st.divider()
    st.subheader("⚡ Quick Comparison")
    
    # Warn if the displayed result is different from the currently uploaded file
    if uploaded_file and (uploaded_file.getvalue() != st.session_state.original_image_bytes):
        st.caption("⚠️ **Note:** Displaying results for the *previously* processed file. Click 'Denoise' to update.")

    image_comparison(
        img1=res_original, # type: ignore
        img2=res_denoised, # type: ignore
        label1="Original",
        label2="Denoised",
        starting_position=50,
        show_labels=True,
        make_responsive=True,
        in_memory=True
    )
    
    # INTERACTIVE VIEWPORT (DUAL)
    st.divider()
    st.subheader("🔬 Deep Inspection (Interactive)")
    st.markdown("Use your mouse wheel to **Zoom** and drag to **Pan**. Both views are locked 1:1.")
    
    render_dual_view(
        img1_b64=str_b64_original, 
        img2_b64=str_b64_denoised, 
        height=600 
    )

    # ANALYSIS
    st.divider()
    st.subheader("📊 Analytical Tools")
    st.markdown("Quantify and visualize the exact structural differences between the original and denoised scans.")
    
    with st.spinner("Calculating pixel distributions and difference heatmaps..."):
        # # Generate and display the Heatmap
        # heatmap_img = get_difference_heatmap(res_original, res_denoised)
        # st.image(
        #     heatmap_img, 
        #     caption="Difference Heatmap (Inferno Colormap). Bright spots indicate high noise removal.",
        #     width=500
        # )

        # # Generate and display the Histogram Plot
        # fig = get_histogram_plot(res_original, res_denoised)
        # st.pyplot(fig, width=800)

        stat_col1, stat_col2 = st.columns(2, vertical_alignment='bottom')
        
        with stat_col1:
            # Generate and display the Heatmap
            heatmap_img = get_difference_heatmap(res_original, res_denoised)
            heatmap_bytes = image_to_bytes(heatmap_img)
            
            if st.session_state.processed_file_name:
                orig_name = Path(st.session_state.processed_file_name).stem
                download_filename = f"{orig_name}_heatmap.png"
            else:
                download_filename = "heatmap_result.png"

            st.download_button(
                label="Heatmap",
                icon="⬇️",
                data=heatmap_bytes,
                file_name=download_filename,
                mime="image/png",
                width="content"
            )

            st.image(
                heatmap_img, 
                caption="Difference Heatmap (Inferno Colormap). Bright spots indicate high noise removal.",
                width="stretch"
            )
            
            
        with stat_col2:
            # Generate and display the Histogram Plot
            fig = get_pixel_intensity_histogram(res_original, res_denoised)
            hist_bytes = figure_to_bytes(figure=fig)

            if st.session_state.processed_file_name:
                orig_name = Path(st.session_state.processed_file_name).stem
                download_filename = f"{orig_name}_intensity_distribution_histogram.png"
            else:
                download_filename = "intensity_distribution_histogram_result.png"

            st.download_button(
                label="Histogram",
                icon="⬇️",
                data=hist_bytes,
                file_name=download_filename,
                mime="image/png",
                width="content"
            )
            st.pyplot(fig, width="stretch")


    # DOWNLOAD
    st.divider()
    d_col1, d_col2 = st.columns([1, 4])
    with d_col1:
        if st.session_state.processed_file_name:
            orig_name = Path(st.session_state.processed_file_name).stem
            download_filename = f"{orig_name}_denoised.png"
        else:
            download_filename = "denoised_result.png"

        st.download_button(
            label="Download Result",
            icon="⬇️",
            data=st.session_state.denoised_image_bytes,
            file_name=download_filename,
            mime="image/png",
            width="content"
        )
    with d_col2:
        if st.button("Clear Results", icon="🔄", width="content"):
            st.session_state.denoised_image_bytes = None
            st.session_state.original_image_bytes = None
            st.session_state.processed_file_name = None
            st.session_state.b64_original = None
            st.session_state.b64_denoised = None

            # Optional: Clear the file uploader widget itself!
            # del st.session_state["scan_uploader"]

            st.rerun()
    
    

elif uploaded_file is None:
    st.info("waiting for upload...")