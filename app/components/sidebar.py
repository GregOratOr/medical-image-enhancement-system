# app/components/sidebar.py

import time
import streamlit as st

from routes import APP_PAGES
from utils.api_client import ApiClient

def render_sidebar(show_controls: bool=True):
    """ Renders the persistent sidebar with status checks and global settings.

    Args:
        show_controls: If True, shows model swapping controls. If False, only shows status.
    """
    client = ApiClient()
    
    with st.sidebar:
        st.header("🩻 MedDenoise AI")
        
        for page in APP_PAGES:
            st.page_link(page)

        st.markdown("---")
        
        # SERVER STATUS CHECK
        # Perform a quick health check every time the app reruns.
        server_config = client.get_config()
        is_online = bool(server_config)
        is_gpu = server_config.get("has_gpu", False)
        # is_online = client.check_health()

        col1, col2 = st.columns([1, 2], vertical_alignment="center", gap=None)
        with col1:
            st.markdown("**Server:**")
        
        with col2:
            if is_online:    
                st.success("Online", icon="🟢")
            else:
                st.error("Offline", icon="🔴")
        
        if is_online:
            # Show specific active model info
            active_model = server_config.get("model_name", "Unknown")
            active_wrapped = server_config.get("wrapped_model", False)
            active_device = "GPU ⚡" if server_config.get("enable_cuda") else "CPU 🔳"
            
            # Clean up display string
            wrapped_str = "[Wrapped]" if active_wrapped else "[Base]"
            st.caption(f"**Running:** `{active_model}`")
            st.caption(f"**Type:** {wrapped_str} | **Device:** {active_device}")
        else:
            st.caption("⚠️ Please start the backend: `uv run uvicorn api.api_server:app`")

        st.markdown("---")
        
        if show_controls:
            # MODEL CONFIGURATION (UI Only for now)
            st.subheader("⚙️ Configuration")
            
            model_map = {
                "UNet - [Base]": ("medical_denoiser_dyno", False),
                "UNet - [Wrapped]": ("medical_denoiser_dyno_wrap", True),
                "(Legacy) UNet - [Base]": ("medical_denoiser_legacy", False),
                "(Legacy) UNet - [Wrapped]": ("medical_denoiser_legacy_wrap", True),
            }

            model_choice = st.selectbox(
                "Active Model",
                options=list(model_map.keys()),
                index=0,
                disabled=not is_online # Disable if server is down
            )
            
            # COMPUTE MODE
            if is_gpu:
                device_options = ["GPU (CUDA)", "CPU"]
                device_help = "Switching to CPU will be significantly slower."
            else:
                device_options = ["CPU"]
                device_help = "No GPU detected. Locked to CPU mode."
                
            selected_device = st.radio(
                "Compute Backend",
                device_options,
                index=0,
                help=device_help
            )

            if st.button("Apply Settings", icon="🔄", width="content", disabled=not is_online):
                # Unpack the tuple from your map
                target_model_name, target_is_wrapped = model_map[model_choice]
                
                target_cuda = True if selected_device == "GPU (CUDA)" else False
                
                with st.spinner("Reloading Engine..."):
                    # Pass all 3 parameters
                    success = client.update_config(
                        model_name=target_model_name, 
                        enable_cuda=target_cuda,
                        wrapped_model=target_is_wrapped
                    )
                    
                if success:
                    st.success("Engine Updated!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Failed to update engine.")
            
            if st.button("Free VRAM (Unload)", icon="🫗", width="content", disabled=not is_online):
                with st.spinner("Clearing GPU memory..."):
                    if client.unload_model():
                        st.success("VRAM Cleared!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Failed to clear VRAM.")

            st.markdown("---")
        
        # APP INFO
        st.caption(f"Backend: `{client.base_url}`")
        st.caption("v0.5.0 - Connected")
