import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def show_model_training():
    """
    Display the model training page with functionality based on regression type.
    """
    st.markdown("## Model Training")
    
    # Check if data is available
    if st.session_state.data is None:
        st.warning("No data available. Please upload data in the Data Input section.")
        if st.button("Go to Data Input"):
            st.session_state.view = "data_input"
            st.rerun()
        return
    
    data = st.session_state.data
    target = st.session_state.target_column
    inputs = st.session_state.input_columns
    regression_type = st.session_state.regression_type
    
    # Check if variables are selected
    if not target or not inputs:
        st.warning("Target or input variables not selected. Please complete variable selection in Data Input section.")
        if st.button("Go to Data Input"):
            st.session_state.view = "data_input"
            st.rerun()
        return
    
    # Display different training options based on regression type
    if regression_type == 'linear':
        train_linear_model(data, inputs[0], target)
    else:
        train_multilinear_model(data, inputs, target)
    
    # Button to proceed to results page
    st.markdown("---")
    if 'model' in st.session_state:
        if st.button("View Results", key="proceed_to_results", use_container_width=True):
            st.session_state.view = "results"
            st.rerun()

def train_linear_model(data, input_var, target_var):
    """
    Train and display simple linear regression model.
    Placeholder function.
    """
    st.markdown("### Simple Linear Regression Training")
    
    # Training options
    st.markdown("#### Training Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider(
            "Test set size (%)",
            min_value=10,
            max_value=50,
            value=20,
            step=5,
            help="Percentage of data to use for testing"
        )
        
    with col2:
        random_state = st.number_input(
            "Random state",
            min_value=0,
            max_value=999,
            value=42,
            step=1,
            help="Random seed for reproducibility"
        )
    
    # Train button
    if st.button("Train Linear Regression Model", key="train_linear_model_button", use_container_width=True):
        with st.spinner("Training model..."):
            # Placeholder for model training logic - in a real implementation, you would:
            # 1. Prepare data (split features and target)
            # 2. Split into training and test sets
            # 3. Train the model
            # 4. Calculate metrics
            # 5. Store model and data in session state
            
            # For this placeholder, we'll do a minimal implementation:
            X = data[input_var].values.reshape(-1, 1)
            y = data[target_var].values
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, random_state=random_state
            )
            
            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_metrics = {
                'MSE': mean_squared_error(y_train, y_train_pred),
                'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'MAE': mean_absolute_error(y_train, y_train_pred),
                'R²': r2_score(y_train, y_train_pred)
            }
            
            test_metrics = {
                'MSE': mean_squared_error(y_test, y_test_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'MAE': mean_absolute_error(y_test, y_test_pred),
                'R²': r2_score(y_test, y_test_pred)
            }
            
            # Store model and data in session state
            st.session_state.model = model
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.y_train_pred = y_train_pred
            st.session_state.y_test_pred = y_test_pred
            st.session_state.train_metrics = train_metrics
            st.session_state.test_metrics = test_metrics
            
            # Display success message
            st.success("Model trained successfully!")
            
            # Display basic results
            st.markdown("#### Model Results")
            st.markdown(f"**Coefficient:** {model.coef_[0]:.4f}")
            st.markdown(f"**Intercept:** {model.intercept_:.4f}")
            st.markdown(f"**R² (test set):** {test_metrics['R²']:.4f}")

def train_multilinear_model(data, input_vars, target_var):
    """
    Train and display multiple linear regression model.
    Placeholder function.
    """
    st.markdown("### Multiple Linear Regression Training")
    
    # Training options
    st.markdown("#### Training Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider(
            "Test set size (%)",
            min_value=10,
            max_value=50,
            value=20,
            step=5,
            help="Percentage of data to use for testing"
        )
        
    with col2:
        random_state = st.number_input(
            "Random state",
            min_value=0,
            max_value=999,
            value=42,
            step=1,
            help="Random seed for reproducibility"
        )
    
    # Feature scaling option
    scale_features = st.checkbox(
        "Apply feature scaling (recommended)",
        value=True,
        help="Standardize features by removing the mean and scaling to unit variance"
    )
    
    # Train button
    if st.button("Train Multiple Regression Model", key="train_multilinear_model_button", use_container_width=True):
        with st.spinner("Training model..."):
            # Placeholder for model training logic
            # For this placeholder, we'll do a minimal implementation:
            X = data[input_vars]
            y = data[target_var].values
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, random_state=random_state
            )
            
            # Apply scaling if selected
            if scale_features:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Save for later use
                st.session_state.scaler = scaler
                st.session_state.X_train_original = X_train.copy()
                st.session_state.X_test_original = X_test.copy()
                
                # Use scaled data for training
                X_train_model = X_train_scaled
                X_test_model = X_test_scaled
            else:
                st.session_state.scaler = None
                X_train_model = X_train
                X_test_model = X_test
            
            # Train model
            model = LinearRegression()
            model.fit(X_train_model, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train_model)
            y_test_pred = model.predict(X_test_model)
            
            # Calculate metrics
            train_metrics = {
                'MSE': mean_squared_error(y_train, y_train_pred),
                'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'MAE': mean_absolute_error(y_train, y_train_pred),
                'R²': r2_score(y_train, y_train_pred)
            }
            
            test_metrics = {
                'MSE': mean_squared_error(y_test, y_test_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'MAE': mean_absolute_error(y_test, y_test_pred),
                'R²': r2_score(y_test, y_test_pred)
            }
            
            # Store model and data in session state
            st.session_state.model = model
            st.session_state.X_train = X_train_model
            st.session_state.X_test = X_test_model
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.y_train_pred = y_train_pred
            st.session_state.y_test_pred = y_test_pred
            st.session_state.train_metrics = train_metrics
            st.session_state.test_metrics = test_metrics
            st.session_state.input_vars = input_vars
            
            # Display success message
            st.success("Model trained successfully!")
            
            # Display basic results
            st.markdown("#### Model Results")
            
            # Show coefficients
            coef_df = pd.DataFrame({
                'Variable': input_vars,
                'Coefficient': model.coef_
            })
            st.dataframe(coef_df)
            
            st.markdown(f"**Intercept:** {model.intercept_:.4f}")
            st.markdown(f"**R² (test set):** {test_metrics['R²']:.4f}")