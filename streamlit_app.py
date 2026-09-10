"""Streamlit entry point that registers the dashboard pages and starts navigation."""
import streamlit as st

# This must run before other Streamlit commands because it configures the browser tab.
st.set_page_config(
    page_title='Person ReID Control Center',
    page_icon='👥',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ``st.navigation`` creates the sidebar router; each page is a normal Python file.
navigation = st.navigation(
    [
        st.Page('app/dashboard/home.py', title='Home', icon='🏠', default=True),
        st.Page('app/dashboard/analytics.py', title='Analytics', icon='📊', url_path='analytics'),
        st.Page('app/dashboard/admin.py', title='Admin', icon='⚙️', url_path='admin'),
    ],
    position='sidebar',
)
navigation.run()
