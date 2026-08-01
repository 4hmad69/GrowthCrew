"""Streamlit entry point for the GrowthCrew frontend."""

import streamlit as st

from frontend.config import get_frontend_settings
from frontend.views.home import render_home


def main() -> None:
    """Configure the page and render the foundation view."""

    st.set_page_config(
        page_title="GrowthCrew",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    render_home(get_frontend_settings())


if __name__ == "__main__":
    main()
