# app/routes.py

import streamlit as st

# Define all pages here
dashboard_page = st.Page(
    "pages/0_Dashboard.py", 
    title="Dashboard", 
    icon="🏠", 
    default=True
)


APP_PAGES = [
    dashboard_page, 
]
