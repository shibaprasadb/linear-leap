import streamlit as st
from utils.ui_components import show_footer
import os
import base64

def get_image_as_base64(image_path):
    """
    Convert an image to base64 encoding.
    """
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        print(f"Error encoding image: {e}")
        return ""

def show_landing_page():
    """
    Display the landing page of the LinearLeap application.
    """
    # Add logo at the top right using direct HTML with base64 encoding
    try:
        logo_path = "assets/LinearLeap_logo.png"
        if os.path.exists(logo_path):
            logo_base64 = get_image_as_base64(logo_path)
            st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
                    <img src="data:image/png;base64,{logo_base64}" width="150px">
                </div>
                """, 
                unsafe_allow_html=True
            )
    except Exception as e:
        print(f"Error displaying logo: {e}")
    
    # Main content in the center column
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.title("Welcome to LinearLeap 📈")
        st.markdown("""
        ### Your Interactive Linear Regression Analysis Tool
        
        LinearLeap allows you to:
        - Upload and analyze your datasets with ease
        - Perform linear and multilinear regression analysis
        - Visualize relationships between variables
        - Get detailed statistical insights and predictions
        - Receive tailored recommendations based on your data (GenAI generated)
        
        Get started by clicking the button below!
        """)
        
        if st.button("Start Analysis", key="start_button", use_container_width=True):
            st.session_state.page = 'app'
            st.session_state.view = 'data_input'
            st.rerun()
            
        st.markdown("---")
        
        st.markdown("""
        ### About LinearLeap
        
        LinearLeap is a GenAI powered application designed to simplify regression analysis for data scientists, 
        analysts, and students. Simply upload your data, select your variables and regression type, and the app 
        will guide you through the entire analysis process with clear visualizations and insights.
        
        LinearLeap handles both:
        - **Simple Linear Regression**: Analyze the relationship between one predictor and one outcome variable
        - **Multiple Linear Regression**: Explore how multiple predictors influence an outcome variable
        """)
    
    # Create a separate container outside the columns for the footer
    footer_container = st.container()
    with footer_container:
        show_footer()