# app/components/header.py

import streamlit as st
from routes import dashboard_page

def render_header(title: str, subtitle: str | None = None, is_dashboard: bool = False):
    """ Renders a uniform title header across all pages.
        Provides a safe, state-preserving link back to the dashboard.
    """
    if not is_dashboard:
        st.page_link(dashboard_page, label="**MedDenoise AI Studio**", icon="🏥")
    
    st.title(title)
    
    st.markdown("---")

    if subtitle:
        st.markdown(subtitle)