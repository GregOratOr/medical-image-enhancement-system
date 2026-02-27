# app/pages/0_Dashboard.py

import streamlit as st
from routes import inspector_page, batch_page
from components.sidebar import render_sidebar
from components.header import render_header

# Page Config (Must be the first Streamlit command)
st.set_page_config(
    page_title="Medical Denoiser Studio",
    page_icon="🏥",
    layout="centered"
)

# Render the Sidebar
render_sidebar(show_controls=False)

render_header(
    title="🏥 Medical Denoiser Studio", 
    subtitle=   """
                    ### Welcome to the Medical Image Enhancement System. 

                    This platform utilizes hardware-accelerated ONNX inference to remove noise from high-resolution medical scans. 
                """,
    is_dashboard=True
)

st.markdown(
    """ ---
    ### Available Tools:
    """
)

# Spacer
st.write("")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.subheader("🔍 Image Inspector")
        st.markdown("""
        * Upload single high-resolution scans.
        * Interactive Image viewer.
        * Instant Before/After comparison.
        """)
        st.write("") # Spacer
        
        if st.button("Launch Inspector", type="primary", width="stretch"):
            st.switch_page(inspector_page)

with col2:
    with st.container(border=True):
        st.subheader("📦 Batch Processor")
        st.markdown("""
        * Process entire folders of slices or selection of files uploaded.
        * Auto-zipping and bulk processing.
        """)
        st.write("") # Spacer
        
        if st.button("Launch Batch Processor", type="primary", width="stretch"):
            st.switch_page(batch_page)

st.markdown("---")