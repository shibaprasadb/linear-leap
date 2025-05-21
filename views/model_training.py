import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from utils.ui_components import show_footer


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
        if st.button("View Insights", key="proceed_to_results", use_container_width=True):
            st.session_state.view = "model_insights"
            st.rerun()

def train_linear_model(data, input_var, target_var):
    """
    Train simple linear regression model using k-fold cross-validation.
    """
    st.markdown("### Simple Linear Regression Training")
    
    # Training options
    st.markdown("#### Training Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_folds = st.slider(
            "Number of folds for cross-validation",
            min_value=3,
            max_value=10,
            value=5,
            step=1,
            help="Number of folds to use for cross-validation. Higher values give more stable metrics but take longer to compute."
        )
        
    with col2:
        shuffle = st.checkbox(
            "Shuffle data before splitting",
            value=True,
            help="Randomize data order before creating folds (recommended)"
        )
        
        if shuffle:
            random_seed = st.number_input(
                "Random seed",
                min_value=0,
                max_value=999,
                value=42,
                step=1,
                help="Random seed for reproducibility"
            )
        else:
            random_seed = None
    
    # Train button
    if st.button("Train Linear Regression Model", key="train_linear_model_button", use_container_width=True):
        with st.spinner("Training model with cross-validation..."):
            # Prepare data
            X = data[input_var].values.reshape(-1, 1)
            y = data[target_var].values
            
            # Initialize k-fold cross-validation
            kf = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_seed)
            
            # Initialize containers for metrics
            r2_scores = []
            mse_scores = []
            rmse_scores = []
            mae_scores = []
            
            # Initialize main model (trained on all data) for later use
            final_model = LinearRegression()
            final_model.fit(X, y)
            
            # Store the final model
            st.session_state.model = final_model
            
            # Containers for fold-specific data
            fold_predictions = []
            fold_actual = []
            fold_coefficients = []
            fold_intercepts = []
            
            # Perform cross-validation
            for i, (train_idx, test_idx) in enumerate(kf.split(X)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Train model for this fold
                fold_model = LinearRegression()
                fold_model.fit(X_train, y_train)
                
                # Store coefficient and intercept
                fold_coefficients.append(fold_model.coef_[0])
                fold_intercepts.append(fold_model.intercept_)
                
                # Make predictions
                y_pred = fold_model.predict(X_test)
                
                # Calculate and store metrics
                r2_scores.append(r2_score(y_test, y_pred))
                mse_scores.append(mean_squared_error(y_test, y_pred))
                rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
                mae_scores.append(mean_absolute_error(y_test, y_pred))
                
                # Store predictions and actual values for this fold
                for j in range(len(y_test)):
                    fold_predictions.append(y_pred[j])
                    fold_actual.append(y_test[j])
            
            # Get cross-validated predictions for all data points
            all_cv_predictions = cross_val_predict(LinearRegression(), X, y, cv=kf)
            
            # Calculate average metrics
            metrics = {
                'R²': {
                    'mean': np.mean(r2_scores),
                    'std': np.std(r2_scores),
                    'values': r2_scores
                },
                'MSE': {
                    'mean': np.mean(mse_scores),
                    'std': np.std(mse_scores),
                    'values': mse_scores
                },
                'RMSE': {
                    'mean': np.mean(rmse_scores),
                    'std': np.std(rmse_scores),
                    'values': rmse_scores
                },
                'MAE': {
                    'mean': np.mean(mae_scores),
                    'std': np.std(mae_scores),
                    'values': mae_scores
                }
            }
            
            # Calculate coefficient stability
            coef_mean = np.mean(fold_coefficients)
            coef_std = np.std(fold_coefficients)
            intercept_mean = np.mean(fold_intercepts)
            intercept_std = np.std(fold_intercepts)
            
            # Store all data in session state
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.cv_metrics = metrics
            st.session_state.cv_predictions = all_cv_predictions
            st.session_state.fold_coefficients = fold_coefficients
            st.session_state.fold_intercepts = fold_intercepts
            st.session_state.coef_mean = coef_mean
            st.session_state.coef_std = coef_std
            st.session_state.intercept_mean = intercept_mean
            st.session_state.intercept_std = intercept_std
            st.session_state.input_var = input_var
            st.session_state.target_var = target_var
            
            # Display success message
            st.success(f"Model trained successfully with {n_folds}-fold cross-validation!")
            
            # Display basic results
            st.markdown("#### Cross-Validation Results")
            
            # Create metrics display
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric(
                    "R² Score", 
                    f"{metrics['R²']['mean']:.4f}",
                    delta=f"±{metrics['R²']['std']:.4f}"
                )
            
            with metric_cols[1]:
                st.metric(
                    "MSE", 
                    f"{metrics['MSE']['mean']:.4f}",
                    delta=f"±{metrics['MSE']['std']:.4f}",
                    delta_color="inverse"
                )
            
            with metric_cols[2]:
                st.metric(
                    "RMSE", 
                    f"{metrics['RMSE']['mean']:.4f}",
                    delta=f"±{metrics['RMSE']['std']:.4f}",
                    delta_color="inverse"
                )
            
            with metric_cols[3]:
                st.metric(
                    "MAE", 
                    f"{metrics['MAE']['mean']:.4f}",
                    delta=f"±{metrics['MAE']['std']:.4f}",
                    delta_color="inverse"
                )
            
            # Display model equation with stability info
            st.markdown("#### Model Equation")
            equation = f"y = {intercept_mean:.4f} ± {intercept_std:.4f} + ({coef_mean:.4f} ± {coef_std:.4f})x"
            st.markdown(f"**{equation}**")
            
            st.info("""
            **Note:** The ± values show the standard deviation across different folds, indicating 
            model stability. Lower values indicate more stable coefficient estimates.
            """)
            
            # Visualization tabs - MODIFIED to remove Metrics Distribution tab
            viz_tab1, viz_tab2 = st.tabs(["Regression Visualization", "Fold Comparison"])
            
            with viz_tab1:
                # Plot regression line with data and prediction intervals
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot the data points
                ax.scatter(X, y, alpha=0.5, label='Data points')
                
                # Plot the regression line with confidence band
                x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                y_pred_mean = final_model.predict(x_range)
                
                # Calculate a simple confidence band based on coefficient variation
                y_pred_upper = (intercept_mean + 2*intercept_std) + (coef_mean + 2*coef_std) * x_range.flatten()
                y_pred_lower = (intercept_mean - 2*intercept_std) + (coef_mean - 2*coef_std) * x_range.flatten()
                
                ax.plot(x_range, y_pred_mean, color='blue', label='Regression line')
                ax.fill_between(x_range.flatten(), y_pred_lower, y_pred_upper, color='blue', alpha=0.1, label='Model variation')
                
                ax.set_xlabel(input_var)
                ax.set_ylabel(target_var)
                ax.set_title(f"Linear Regression: {input_var} vs {target_var}")
                ax.legend()
                
                st.pyplot(fig)
                
                # Show the prediction vs actual plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(y, all_cv_predictions, alpha=0.5)
                
                # Add identity line
                min_val = min(y.min(), all_cv_predictions.min())
                max_val = max(y.max(), all_cv_predictions.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Identity line')
                
                ax.set_xlabel("Actual Values")
                ax.set_ylabel("Predicted Values")
                ax.set_title("Actual vs Predicted Values (Cross-Validated)")
                ax.legend()
                
                st.pyplot(fig)
            
            with viz_tab2:
                # Plot all fold regression lines
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot data points
                ax.scatter(X, y, alpha=0.3, color='gray', label='Data points')
                
                # Plot regression line for each fold
                x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                
                for i in range(n_folds):
                    y_fold = fold_intercepts[i] + fold_coefficients[i] * x_range.flatten()
                    ax.plot(x_range, y_fold, alpha=0.5, label=f'Fold {i+1}')
                
                # Plot the average line
                y_avg = intercept_mean + coef_mean * x_range.flatten()
                ax.plot(x_range, y_avg, color='red', linewidth=2, label='Average')
                
                ax.set_xlabel(input_var)
                ax.set_ylabel(target_var)
                ax.set_title(f"Regression Lines Across {n_folds} Folds")
                ax.legend()
                
                st.pyplot(fig)
                
                # Plot coefficient and intercept variation
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                sns.stripplot(y=fold_coefficients, jitter=True, ax=ax1)
                ax1.axhline(y=coef_mean, color='r', linestyle='--', label=f'Mean: {coef_mean:.4f}')
                ax1.set_title(f"Coefficient Values Across Folds")
                ax1.set_ylabel(f"Coefficient of {input_var}")
                ax1.set_xticks([])
                
                sns.stripplot(y=fold_intercepts, jitter=True, ax=ax2)
                ax2.axhline(y=intercept_mean, color='r', linestyle='--', label=f'Mean: {intercept_mean:.4f}')
                ax2.set_title(f"Intercept Values Across Folds")
                ax2.set_ylabel("Intercept")
                ax2.set_xticks([])
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Educational component explaining cross-validation
                st.markdown("#### Understanding Cross-Validation")
                st.markdown("""
                Cross-validation helps assess how the model will generalize to independent data by:
                
                1. Dividing the data into multiple folds
                2. Training the model multiple times, each time holding out a different fold as the test set
                3. Averaging the performance metrics across all iterations
                
                **Benefits:**
                - More reliable performance estimates
                - Insight into model stability
                - Better use of available data
                - Reduced risk of overfitting
                
                The variation you see in the regression lines above shows how the model parameters can
                change based on which data points are used for training. A stable model will have
                lines that are close together.
                """)

def train_multilinear_model(data, input_vars, target_var):
    """
    Train and display multiple linear regression model with cross-validation.
    """
    st.markdown("### Multiple Linear Regression Training")
    
    # Training options
    st.markdown("#### Training Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_folds = st.slider(
            "Number of folds for cross-validation",
            min_value=3,
            max_value=10,
            value=5,
            step=1,
            help="Number of folds to use for cross-validation. Higher values give more stable metrics but take longer to compute."
        )
        
    with col2:
        shuffle = st.checkbox(
            "Shuffle data before splitting",
            value=True,
            help="Randomize data order before creating folds (recommended)"
        )
        
        if shuffle:
            random_seed = st.number_input(
                "Random seed",
                min_value=0,
                max_value=999,
                value=42,
                step=1,
                help="Random seed for reproducibility"
            )
        else:
            random_seed = None
    
    # Feature scaling option
    scale_features = st.checkbox(
        "Apply feature scaling (recommended)",
        value=True,
        help="Standardize features by removing the mean and scaling to unit variance"
    )
    
    # Train button
    if st.button("Train Multiple Regression Model", key="train_multilinear_model_button", use_container_width=True):
        with st.spinner("Training model with cross-validation..."):
            # Prepare data
            X = data[input_vars]
            y = data[target_var].values
            
            # Apply scaling if selected
            if scale_features:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X_model = X_scaled
                
                # Save for later use
                st.session_state.scaler = scaler
                st.session_state.X_original = X.copy()
            else:
                st.session_state.scaler = None
                X_model = X.values
            
            # Initialize k-fold cross-validation
            kf = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_seed)
            
            # Initialize containers for metrics
            r2_scores = []
            mse_scores = []
            rmse_scores = []
            mae_scores = []
            
            # Initialize main model (trained on all data) for later use
            final_model = LinearRegression()
            final_model.fit(X_model, y)
            
            # Store the final model
            st.session_state.model = final_model
            
            # Containers for fold-specific data
            fold_coefficients = []
            fold_intercepts = []
            
            # Perform cross-validation
            for i, (train_idx, test_idx) in enumerate(kf.split(X_model)):
                X_train, X_test = X_model[train_idx], X_model[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Train model for this fold
                fold_model = LinearRegression()
                fold_model.fit(X_train, y_train)
                
                # Store coefficient and intercept
                fold_coefficients.append(fold_model.coef_)
                fold_intercepts.append(fold_model.intercept_)
                
                # Make predictions
                y_pred = fold_model.predict(X_test)
                
                # Calculate and store metrics
                r2_scores.append(r2_score(y_test, y_pred))
                mse_scores.append(mean_squared_error(y_test, y_pred))
                rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
                mae_scores.append(mean_absolute_error(y_test, y_pred))
            
            # Get cross-validated predictions for all data points
            all_cv_predictions = cross_val_predict(LinearRegression(), X_model, y, cv=kf)
            
            # Calculate average metrics
            metrics = {
                'R²': {
                    'mean': np.mean(r2_scores),
                    'std': np.std(r2_scores),
                    'values': r2_scores
                },
                'MSE': {
                    'mean': np.mean(mse_scores),
                    'std': np.std(mse_scores),
                    'values': mse_scores
                },
                'RMSE': {
                    'mean': np.mean(rmse_scores),
                    'std': np.std(rmse_scores),
                    'values': rmse_scores
                },
                'MAE': {
                    'mean': np.mean(mae_scores),
                    'std': np.std(mae_scores),
                    'values': mae_scores
                }
            }
            
            # Calculate coefficient stability
            coef_means = np.mean(fold_coefficients, axis=0)
            coef_stds = np.std(fold_coefficients, axis=0)
            intercept_mean = np.mean(fold_intercepts)
            intercept_std = np.std(fold_intercepts)
            
            # Store all data in session state
            st.session_state.X = X_model
            st.session_state.X_df = X  # Original dataframe
            st.session_state.y = y
            st.session_state.cv_metrics = metrics
            st.session_state.cv_predictions = all_cv_predictions
            st.session_state.fold_coefficients = fold_coefficients
            st.session_state.fold_intercepts = fold_intercepts
            st.session_state.coef_means = coef_means
            st.session_state.coef_stds = coef_stds
            st.session_state.intercept_mean = intercept_mean
            st.session_state.intercept_std = intercept_std
            st.session_state.input_vars = input_vars
            st.session_state.target_var = target_var
            
            # Display success message
            st.success(f"Model trained successfully with {n_folds}-fold cross-validation!")
            
            # Display basic results
            st.markdown("#### Cross-Validation Results")
            
            # Create metrics display
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric(
                    "R² Score", 
                    f"{metrics['R²']['mean']:.4f}",
                    delta=f"±{metrics['R²']['std']:.4f}"
                )
            
            with metric_cols[1]:
                st.metric(
                    "MSE", 
                    f"{metrics['MSE']['mean']:.4f}",
                    delta=f"±{metrics['MSE']['std']:.4f}",
                    delta_color="inverse"
                )
            
            with metric_cols[2]:
                st.metric(
                    "RMSE", 
                    f"{metrics['RMSE']['mean']:.4f}",
                    delta=f"±{metrics['RMSE']['std']:.4f}",
                    delta_color="inverse"
                )
            
            with metric_cols[3]:
                st.metric(
                    "MAE", 
                    f"{metrics['MAE']['mean']:.4f}",
                    delta=f"±{metrics['MAE']['std']:.4f}",
                    delta_color="inverse"
                )
            
            # Display coefficient stability
            st.markdown("#### Coefficient Stability")
            
            # Create a dataframe of coefficients with their standard deviations
            coef_df = pd.DataFrame({
                'Variable': input_vars,
                'Coefficient': coef_means,
                'Std Dev': coef_stds,
                'Coefficient Range': [f"{coef_means[i]:.4f} ± {coef_stds[i]:.4f}" for i in range(len(input_vars))]
            })
            
            st.dataframe(coef_df[['Variable', 'Coefficient Range']])
            
            # Add intercept information
            st.markdown(f"**Intercept:** {intercept_mean:.4f} ± {intercept_std:.4f}")
            
            st.info("""
            **Note:** The ± values show the standard deviation across different folds, indicating 
            coefficient stability. Lower values indicate more stable estimates.
            """)
            
            # Visualization tabs - MODIFIED to update tab names and content
            viz_tab1, viz_tab2 = st.tabs(["Prediction Analysis", "Coefficient Analysis"])
            
            with viz_tab1:
                # Show the prediction vs actual plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(y, all_cv_predictions, alpha=0.5)
                
                # Add identity line
                min_val = min(y.min(), all_cv_predictions.min())
                max_val = max(y.max(), all_cv_predictions.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Identity line')
                
                ax.set_xlabel("Actual Values")
                ax.set_ylabel("Predicted Values")
                ax.set_title("Actual vs Predicted Values (Cross-Validated)")
                ax.legend()
                
                st.pyplot(fig)
                
                # Residual plot (actual - predicted)
                residuals = y - all_cv_predictions
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(all_cv_predictions, residuals, alpha=0.5)
                ax.axhline(y=0, color='r', linestyle='--')
                ax.set_xlabel("Predicted Values")
                ax.set_ylabel("Residuals")
                ax.set_title("Residual Plot")
                
                st.pyplot(fig)
                
                # Explain residual plot
                st.markdown("""
                **About the Residual Plot:**
                
                The residual plot shows the difference between predicted and actual values. 
                In an ideal model:
                
                - Points should be randomly scattered around the horizontal line at y=0
                - No clear patterns should be visible
                - Residuals should have similar spread across all predicted values
                
                Patterns in the residual plot can indicate model issues such as non-linearity 
                or heteroscedasticity (non-constant variance).
                """)
            
            with viz_tab2:
                # Coefficient variation visualization
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Prepare data for box plot
                box_data = []
                labels = []
                
                for i, var in enumerate(input_vars):
                    var_coefs = [fold_coef[i] for fold_coef in fold_coefficients]
                    box_data.append(var_coefs)
                    labels.append(var)
                
                # Create box plot
                ax.boxplot(box_data, labels=labels, vert=True)
                ax.set_title("Coefficient Stability Across Folds")
                ax.set_ylabel("Coefficient Value")
                ax.set_xlabel("Input Variable")
                
                # Rotate x labels if many variables
                if len(input_vars) > 4:
                    plt.xticks(rotation=45, ha='right')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Feature importance plot (based on absolute coefficient values)
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Sort by absolute mean coefficient value
                importance_df = pd.DataFrame({
                    'Variable': input_vars,
                    'Absolute Coefficient': np.abs(coef_means)
                }).sort_values('Absolute Coefficient', ascending=False)
                
                # Plot
                sns.barplot(x='Absolute Coefficient', y='Variable', data=importance_df, ax=ax)
                ax.set_title("Feature Importance (Based on Absolute Coefficient Value)")
                
                st.pyplot(fig)
                
                # Educational component explaining multilinear regression
                st.markdown("#### Understanding Multiple Linear Regression and Cross-Validation")
                st.markdown("""
                In multiple linear regression:
                
                - Each coefficient represents the change in the target variable when the corresponding input 
                  variable increases by one unit, *holding all other variables constant*
                
                - The variation in coefficients across folds helps identify which features have a consistent
                  impact on the target variable
                
                - Features with high coefficient values but also high standard deviations may be less reliable
                  predictors
                
                **Cross-validation benefits:**
                - Provides more reliable performance estimates
                - Helps identify which features are consistently important
                - Shows the stability of the model's predictive power
                - Reduces the risk of overfitting to a specific train-test split
                """)

# Add footer
show_footer()