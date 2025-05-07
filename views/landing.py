import streamlit as st

def show_landing_page():
    """
    Display the landing page of the LinearLeap application.
    """
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
        - Receive tailored recommendations based on your data
        
        Get started by clicking the button below!
        """)
        
        if st.button("Start Analysis", key="start_button", use_container_width=True):
            st.session_state.page = 'app'
            st.session_state.view = 'data_input'
            st.rerun()  # Changed from st.experimental_rerun()
            
        st.markdown("---")
        
        st.markdown("""
        ### About LinearLeap
        
        LinearLeap is a streamlined application designed to simplify regression analysis for data scientists, 
        analysts, and students. Simply upload your data, select your variables and regression type, and the app 
        will guide you through the entire analysis process with clear visualizations and insights.
        
        LinearLeap handles both:
        - **Simple Linear Regression**: Analyze the relationship between one predictor and one outcome variable
        - **Multiple Linear Regression**: Explore how multiple predictors influence an outcome variable
        """)