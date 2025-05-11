import streamlit as st
import pandas as pd
import numpy as np
from utils.ui_components import show_footer

def show_data_input():
    """
    Display the data input page where users can upload CSV files and select variables.
    """
    st.markdown("## Data Input")
    st.markdown("Upload your CSV file and select the variables for analysis.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.data = data
            
            # Display sample of the data
            st.markdown("### Data Preview")
            st.dataframe(data.head())
            
            # Basic data info
            st.markdown("### Data Information")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"Rows: {data.shape[0]}")
            with col2:
                st.info(f"Columns: {data.shape[1]}")
            
            # Get numeric columns for regression
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_columns) < 2:
                st.error("Your dataset needs at least 2 numeric columns for regression analysis.")
                return
                
            # Regression type selection - FIRST
            regression_type = st.radio(
                "Select regression type",
                options=["linear", "multilinear"],
                index=0 if st.session_state.regression_type == "linear" else 1,
                horizontal=True,
                key="regression_type_select"
            )
            
            # Variable selection - Based on regression type
            st.markdown("### Variable Selection")
            
            # Target variable selection
            target_column = st.selectbox(
                "Select target variable (y)",
                options=numeric_columns,
                index=0 if st.session_state.target_column is None else numeric_columns.index(st.session_state.target_column),
                key="target_var_select"
            )
            
            # Input variables selection (exclude target)
            available_inputs = [col for col in numeric_columns if col != target_column]
            
            if regression_type == 'linear':
                input_column = st.selectbox(
                    "Select input variable (x)",
                    options=available_inputs,
                    index=0 if not st.session_state.input_columns else available_inputs.index(st.session_state.input_columns[0]) if st.session_state.input_columns and st.session_state.input_columns[0] in available_inputs else 0,
                    key="input_var_select"
                )
                input_columns = [input_column]
            else:
                input_columns = st.multiselect(
                    "Select input variables (x)",
                    options=available_inputs,
                    default=st.session_state.input_columns if st.session_state.input_columns else [],
                    key="input_vars_select"
                )
                
                # Validate at least one input selected for multilinear
                if not input_columns:
                    st.warning("Please select at least one input variable for multiple regression.")
            
            # Save button
            if st.button("Save Selections", key="save_selections", use_container_width=True):
                st.session_state.target_column = target_column
                st.session_state.input_columns = input_columns
                st.session_state.regression_type = regression_type
                st.success("Selections saved successfully!")
                
                # Auto-navigate to next step
                st.session_state.view = "eda"
                st.rerun()  # FIXED: Changed from st.experimental_rerun()
                
        except Exception as e:
            st.error(f"Error loading data: {e}")
    else:
        st.info("Please upload a CSV file to begin.")
        
        # Sample data option
        st.markdown("### No data? Try our sample datasets")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Load Simple Linear Sample", key="linear_sample", use_container_width=True):
                load_linear_sample_data()
                st.success("Linear regression sample data loaded!")
                st.rerun()  # FIXED: Changed from st.experimental_rerun()
                
        with col2:
            if st.button("Load Multiple Linear Sample", key="multilinear_sample", use_container_width=True):
                load_multilinear_sample_data()
                st.success("Multiple linear regression sample data loaded!")
                st.rerun()  # FIXED: Changed from st.experimental_rerun()
    # Add footer
    show_footer()

def load_linear_sample_data():
    """
    Load a sample dataset for simple linear regression.
    """
    # Create a sample dataset
    np.random.seed(42)
    sample_size = 100
    X = np.random.rand(sample_size) * 10
    y = 2 * X + 1 + 0.5 * np.random.randn(sample_size)
    
    sample_data = pd.DataFrame({
        'X': X,
        'Y': y
    })
    
    st.session_state.data = sample_data
    st.session_state.target_column = 'Y'
    st.session_state.input_columns = ['X']
    st.session_state.regression_type = 'linear'

def load_multilinear_sample_data():
    """
    Load a sample dataset for multiple linear regression.
    """
    # Create a sample dataset
    np.random.seed(42)
    sample_size = 100
    X1 = np.random.rand(sample_size) * 10
    X2 = np.random.rand(sample_size) * 5
    X3 = np.random.rand(sample_size) * 3
    y = 2 * X1 + 3 * X2 - 1.5 * X3 + 2 + 0.5 * np.random.randn(sample_size)
    
    sample_data = pd.DataFrame({
        'Feature1': X1,
        'Feature2': X2,
        'Feature3': X3,
        'Target': y
    })
    
    st.session_state.data = sample_data
    st.session_state.target_column = 'Target'
    st.session_state.input_columns = ['Feature1', 'Feature2', 'Feature3']
    st.session_state.regression_type = 'multilinear'


