# app/pages/0_Dashboard.py

import streamlit as st
from routes import inspector_page
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
        
        st.button("Coming Soon", disabled=True, width="stretch")

with col2:
    with st.container(border=True):
        st.subheader("📦 Batch Processor")
        st.markdown("""
        * Process entire folders of slices.
        * Auto-zipping and bulk processing.
        * _(Currently in development)_
        """)
        st.write("") # Spacer
        
        # Disabled button for features not yet built
        st.button("Coming Soon", disabled=True, width="stretch")

st.markdown("---")