import streamlit as st
import os

# Import the CSS loader
from utils.ui_components import load_css

# Setup page configuration FIRST, before any other imports or Streamlit commands
# Use the LinearLeap logo as the page icon
st.set_page_config(
    page_title="LinearLeap",
    page_icon="assets/LinearLeap_logo.png", 
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
load_css()

# Import everything else AFTER setting the page config
from navigation.navbar import display_navigation_banner
from views.landing import show_landing_page
from views.data_input import show_data_input
from views.eda import show_eda
from views.model_training import show_model_training
# Remove these imports and replace with model_insights
# from views.results import show_results
# from views.recommendations import show_recommendation
from views.model_insights import show_model_insights  # New combined view

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
            'model_insights': show_model_insights  # Replace results and recommendations
        }
        
        # Execute the selected view function
        current_view = st.session_state.view
        if current_view in view_mapping:
            view_mapping[current_view]()
        else:
            st.error(f"View '{current_view}' not found")

if __name__ == "__main__":
    main()