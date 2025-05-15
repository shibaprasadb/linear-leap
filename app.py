import streamlit as st
import os

# Import UI components
from utils.ui_components import load_css, set_favicon

# Setup page configuration FIRST, before any other imports or Streamlit commands
st.set_page_config(
    page_title="LinearLeap",
    page_icon="📈",  # Use a text emoji as fallback
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set custom favicon (works better than page_icon)
set_favicon()

# Load custom CSS
load_css()

# Import everything else AFTER setting the page config
from navigation.navbar import display_navigation_banner
from views.landing import show_landing_page
from views.data_input import show_data_input
from views.eda import show_eda
from views.model_training import show_model_training
from views.results import show_results
from views.recommendations import show_recommendation

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'landing'
if 'view' not in st.session_state:
    st.session_state.view = 'data_input'

# Initialize data state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'input_columns' not in st.session_state:
    st.session_state.input_columns = []
if 'regression_type' not in st.session_state:
    st.session_state.regression_type = 'linear'  # Default to linear regression

# Main app logic
def main():
    if st.session_state.page == 'landing':
        show_landing_page()
    else:
        # Display navigation banner at the top
        display_navigation_banner()
        
        # Show the appropriate view based on selection
        view_mapping = {
            'data_input': show_data_input,
            'eda': show_eda,
            'model_training': show_model_training,
            'results': show_results,
            'recommendation': show_recommendation
        }
        
        # Execute the selected view function
        current_view = st.session_state.view
        if current_view in view_mapping:
            view_mapping[current_view]()
        else:
            st.error(f"View '{current_view}' not found")

if __name__ == "__main__":
    main()