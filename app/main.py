# app/main.py

import streamlit as st
from routes import APP_PAGES

# Build the navigation menu
pg = st.navigation(APP_PAGES, position='hidden')

# Run the selected page
pg.run()