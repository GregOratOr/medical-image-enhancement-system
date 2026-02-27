# app/routes.py

import streamlit as st

# Define all pages here
dashboard_page = st.Page(
    "pages/0_Dashboard.py", 
    title="Dashboard", 
    icon="🏠", 
    default=True
)

inspector_page = st.Page(
    "pages/1_Image_Inspector.py", 
    title="Image Inspector", 
    icon="🔍"
)

batch_page = st.Page(
    "pages/2_Batch_Processor.py", 
    title="Batch Processor", 
    icon="📦"
)

APP_PAGES = [
    dashboard_page, 
    inspector_page,
    batch_page
]
