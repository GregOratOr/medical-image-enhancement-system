# app/pages/2_Batch_Processor.py

import io
import os
import gc
import zipfile
import streamlit as st
from pathlib import Path

from components.sidebar import render_sidebar
from components.header import render_header
from utils.api_client import ApiClient

st.set_page_config(
    page_title="Batch Processor | Denoiser Studio",
    page_icon="📦", 
    layout="wide"
)

client = ApiClient()

render_sidebar(show_controls=True)

render_header(
    title="📦 Batch Processor", 
    subtitle="Process entire volumes or directories of medical scans automatically."
)

# Optional: Set this environment variable in your cloud deployment to hide the Local Tab
is_cloud_env = os.environ.get("CLOUD_DEPLOYMENT", "false").lower() == "true"

# Initialize tab2 to None
tab2 = None

if not is_cloud_env:
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        def get_local_folder_path():
            """Opens a native OS folder browser dialog and returns the path."""
            # Set up a hidden Tkinter root window
            root = tk.Tk()
            root.withdraw()
            
            # Force the window to appear on top of the web browser
            root.wm_attributes('-topmost', 1)
            
            # Open the dialog
            folder_path = filedialog.askdirectory(parent=root, title="Select a Folder of Medical Scans")
            
            # Destroy the root window after selection
            root.destroy()
            
            return folder_path
        
    except ImportError:
        pass
    
    tab1, tab2 = st.tabs(["☁️ File Uploader **[Lite]**", "📁 Local Folder **[Heavy-Duty]**"])
else:
    tab1, = st.tabs(["☁️ File Uploader **[Lite]**"]) 


# ==========================================
# TAB 1: WEB UPLOADER (IN-MEMORY ZIP) - Always Exists
# ==========================================
with tab1:
    st.info("💡 **Best for Cloud/Web:** Upload files from anywhere. Results are zipped for download. (Small Batches ONLY!!!)")
    
    uploaded_files = st.file_uploader(
        "Drag & Drop Scans (PNG, JPG, TIF)", 
        type=["png", "jpg", "jpeg", "tif"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.write(f"**{len(uploaded_files)} files staged for processing.**")
        
        if st.button("🚀 Start Web Batch", type="primary", width="stretch"):
            progress_bar = st.progress(0, text="0%")
            
            # Create an in-memory ZIP file
            zip_buffer = io.BytesIO()
            
            with st.status("Initializing batch...", expanded=True) as status:
                log_box = st.container(height=250)
                
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for i, file in enumerate(uploaded_files):
                        # Update the spinner's text dynamically!
                        status.update(label=f"⏳ Processing {file.name}... [{i+1}/{len(uploaded_files)}]")
                        
                        log_box.write(f"⏳ Reading: `{file.name}`")
                        
                        img_bytes = file.getvalue()
                        log_box.write(f"⚙️ Denoising: `{file.name}`")
                        res_bytes = client.predict(img_bytes, file.name)
                        
                        if res_bytes:
                            # Write the denoised bytes directly into the zip archive
                            save_name = f"{Path(file.name).stem}_denoised.png"
                            zip_file.writestr(save_name, res_bytes)
                            log_box.write(f"✅ Successfully added to zip: `{save_name}`")
                            
                        progress_val = (i + 1) / len(uploaded_files)
                        progress_bar.progress(progress_val, text=f"{progress_val * 100:.0f}%")
                        
                        # Aggressive memory cleanup for the cloud tab!
                        del img_bytes, res_bytes, progress_val
                        gc.collect()
                
                # Update the spinner to a green checkmark when finished
                status.update(
                    label="Batch complete! Archive is ready for download.", 
                    state="complete", 
                    expanded=False
                )
            
            # Render the download button
            st.download_button(
                label="Download Denoised Batch (ZIP)",
                icon="⬇️",
                data=zip_buffer.getvalue(),
                file_name="denoised_batch.zip",
                mime="application/zip",
                type="secondary",
                width="content"
            )

# ==========================================
# TAB 2: LOCAL FOLDER (LAZY LOAD) - Conditional
# ==========================================
if "local_input_dir" not in st.session_state:
    st.session_state.local_input_dir = ""

if tab2 is not None:
    with tab2:
        st.info("💡 **Best for Local/Standalone Application:** Reads and writes directly to your hard drive.")
        
        # UI Layout: Button on the left, Path on the right
        btn_col, path_col = st.columns([1, 4], vertical_alignment="bottom")
        
        with btn_col:
            if st.button("📁 Browse Folder", width="content"):
                # Open the native OS dialog and save to session state
                selected_path = get_local_folder_path() # type: ignore
                if selected_path:
                    st.session_state.local_input_dir = selected_path
                    st.rerun() # Refresh the UI to show the new path
                    
        with path_col:
            # The text input reads directly from session state
            input_folder = st.text_input(
                "Input Folder Path:", 
                value=st.session_state.local_input_dir,
                placeholder="e.g., C:/Scans/Patient_01"
            )
            # Update state if the user manually types in the box instead
            st.session_state.local_input_dir = input_folder
        
        if input_folder:
            in_path = Path(input_folder)
            
            if in_path.exists() and in_path.is_dir():
                valid_exts = {'.png', '.jpg', '.jpeg', '.tif'}
                files = [f for f in in_path.iterdir() if f.suffix.lower() in valid_exts]
                
                if not files:
                    st.warning("No compatible images found in this directory.")
                else:
                    st.success(f"Found {len(files)} compatible images.")
                    
                    # Auto-generate the output path
                    default_out = str(in_path.parent / (in_path.name + "_Denoised"))
                    output_folder = st.text_input("Output Folder Path:", value=default_out)
                    
                    st.divider()
                    
                    if st.button("🚀 Start Local Batch", type="primary", width="stretch"):
                        out_path = Path(output_folder)
                        out_path.mkdir(parents=True, exist_ok=True)
                        
                        progress_bar = st.progress(0,text="0%")
                        with st.status("Initializing batch...", expanded=True) as status:
                            log_box = st.container(height=250)
                            # THE LAZY LOOP
                            for i, file_path in enumerate(files):
                                # Update the spinner's text dynamically!
                                status.update(label=f"⏳ Processing {file_path.name}... [{i+1}/{len(files)}]")

                                log_box.write(f"⏳ Reading: `{file_path.name}`")

                                with open(file_path, "rb") as f:
                                    img_bytes = f.read()
                                
                                log_box.write(f"⚙️ Denoising: `{file_path.name}`")
                                res_bytes = client.predict(img_bytes, file_path.name)
                                
                                if res_bytes:
                                    save_path = out_path / f"{file_path.stem}_denoised.png"
                                    with open(save_path, "wb") as f:
                                        f.write(res_bytes)
                                    log_box.write(f"✅ Successfully saved: `{save_path.name}`")
                                
                                
                                progress_val = (i + 1) / len(files)
                                progress_bar.progress(progress_val, text=f"{progress_val * 100:.0f}%")
                                del img_bytes, res_bytes, progress_val
                                gc.collect()
                            
                            # Update the spinner to a green checkmark when finished
                            status.update(
                                label=f"Batch complete! All files saved to: {out_path}", 
                                state="complete", 
                                expanded=False
                            )
            else:
                st.error("Directory not found. Please check the path.")