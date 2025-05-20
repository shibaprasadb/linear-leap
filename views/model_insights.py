import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.ui_components import show_footer

def show_model_insights():
    """
    Display combined model results, diagnostics, and recommendations.
    """
    st.markdown("## Model Insights & Recommendations")
    
    # Check if data and model are available
    if st.session_state.data is None:
        st.warning("No data available. Please upload data in the Data Input section.")
        if st.button("Go to Data Input"):
            st.session_state.view = "data_input"
            st.rerun()
        return
    
    if 'model' not in st.session_state:
        st.warning("No model has been trained yet. Please train a model first.")
        if st.button("Go to Model Training"):
            st.session_state.view = "model_training"
            st.rerun()
        return
    
    # Get regression type
    regression_type = st.session_state.regression_type
    
    # Display tabs for different insights
    tab1, tab2, tab3, tab4 = st.tabs([
        "Model Summary", 
        "Model Diagnostics", 
        "Predictions",
        "Business Recommendations"
    ])
    
    with tab1:
        if regression_type == 'linear':
            show_linear_summary()
        else:
            show_multilinear_summary()
    
    with tab2:
        if regression_type == 'linear':
            show_linear_diagnostics()
        else:
            show_multilinear_diagnostics()
    
    with tab3:
        if regression_type == 'linear':
            show_linear_predictions()
        else:
            show_multilinear_predictions()
    
    with tab4:
        if regression_type == 'linear':
            show_linear_recommendations()
        else:
            show_multilinear_recommendations()
    
    # Final call to action
    st.markdown("---")
    st.markdown("### Ready to Try Again?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Start New Analysis", key="start_new", use_container_width=True):
            # Reset session state
            for key in ['data', 'target_column', 'input_columns', 'regression_type', 'model']:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Redirect to data input
            st.session_state.view = "data_input"
            st.rerun()
    
    with col2:
        st.button("Export Results (Coming Soon)", key="export_results", use_container_width=True, disabled=True)
    
    # Add footer
    show_footer()


def show_linear_summary():
    """
    Display summary for simple linear regression model with focus on equation and coefficient stability.
    """
    st.markdown("### Simple Linear Regression Summary")
    
    # Check if we have the necessary data
    if 'model' not in st.session_state:
        st.warning("Please train a model first to see the summary.")
        return
    
    # Get data from session state
    model = st.session_state.model
    input_var = st.session_state.input_var
    target_var = st.session_state.target_var
    
    # Model equation
    st.markdown("#### Model Equation")
    
    # Use cross-validated values instead of direct model coefficients
    if 'coef_mean' in st.session_state and 'intercept_mean' in st.session_state:
        coef = st.session_state.coef_mean
        intercept = st.session_state.intercept_mean
        
        # Format the equation
        equation = f"{target_var} = {intercept:.4f} + {coef:.4f} × {input_var}"
        
        # Display equation in LaTeX format for better presentation
        st.latex(equation)
        
        # Get CV results if available
        if 'coef_std' in st.session_state and 'intercept_std' in st.session_state:
            coef_std = st.session_state.coef_std
            intercept_std = st.session_state.intercept_std
            
            st.markdown(f"""
            **Coefficient stability:** 
            - Intercept: {intercept:.4f} ± {intercept_std:.4f}
            - {input_var}: {coef:.4f} ± {coef_std:.4f}
            """)
    else:
        st.info("Model equation not available.")
    
    # Model performance
    st.markdown("#### Key Performance Metrics")
    
    if 'cv_metrics' in st.session_state:
        metrics = st.session_state.cv_metrics
        
        # Create columns for metrics
        col1, col2 = st.columns(2)
        
        with col1:
            # R² with interpretation
            r2 = metrics['R²']['mean']
            r2_std = metrics['R²']['std']
            
            st.metric(
                "R² Score (Coefficient of Determination)", 
                f"{r2:.4f}",
                delta=f"±{r2_std:.4f}"
            )
            
            if r2 >= 0.7:
                st.success(f"Strong predictive power: Model explains {r2*100:.1f}% of variation in {target_var}")
            elif r2 >= 0.3:
                st.info(f"Moderate predictive power: Model explains {r2*100:.1f}% of variation in {target_var}")
            else:
                st.warning(f"Limited predictive power: Model explains only {r2*100:.1f}% of variation in {target_var}")
        
        with col2:
            # Error metrics with interpretation
            rmse = metrics['RMSE']['mean']
            
            st.metric(
                "RMSE (Root Mean Squared Error)",
                f"{rmse:.4f}",
                delta=f"±{metrics['RMSE']['std']:.4f}",
                delta_color="inverse"  # Lower is better
            )
            
            # Add context for the error metric
            if 'y' in st.session_state:
                y = st.session_state.y
                y_range = np.max(y) - np.min(y)
                error_percentage = (rmse / y_range) * 100
                
                st.markdown(f"""
                RMSE represents the average prediction error in the units of {target_var}.
                This error is approximately **{error_percentage:.1f}%** of the total range of {target_var}.
                """)
    else:
        st.info("Model performance metrics not available.")


def show_multilinear_summary():
    """
    Display summary for multiple linear regression model with focus on equation and coefficient stability.
    """
    st.markdown("### Multiple Linear Regression Summary")
    
    # Check if we have the necessary data
    if 'model' not in st.session_state:
        st.warning("Please train a model first to see the summary.")
        return
    
    # Get data from session state
    model = st.session_state.model
    input_vars = st.session_state.input_vars if 'input_vars' in st.session_state else []
    target_var = st.session_state.target_var if 'target_var' in st.session_state else "target"
    
    # Model equation
    st.markdown("#### Model Equation")
    
    if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
        coefs = model.coef_
        intercept = model.intercept_
        
        # Format the equation
        equation = f"{target_var} = {intercept:.4f}"
        for i, var in enumerate(input_vars):
            sign = " + " if coefs[i] >= 0 else " - "
            equation += f"{sign}{abs(coefs[i]):.4f} × {var}"
        
        # Display equation in LaTeX format for better presentation
        st.latex(equation)
        
        # Get CV results if available
        if 'coef_means' in st.session_state and 'coef_stds' in st.session_state:
            coef_means = st.session_state.coef_means
            coef_stds = st.session_state.coef_stds
            
            # Create a nicely formatted table of coefficients
            coef_df = pd.DataFrame({
                'Variable': input_vars,
                'Coefficient': coef_means,
                'Std Dev': coef_stds,
                'Coefficient Range': [f"{coef_means[i]:.4f} ± {coef_stds[i]:.4f}" for i in range(len(input_vars))]
            })
            
            st.markdown("**Coefficient Values:**")
            st.dataframe(coef_df[['Variable', 'Coefficient Range']])
            
            # Add intercept info
            if 'intercept_mean' in st.session_state and 'intercept_std' in st.session_state:
                intercept_mean = st.session_state.intercept_mean
                intercept_std = st.session_state.intercept_std
                st.markdown(f"**Intercept:** {intercept_mean:.4f} ± {intercept_std:.4f}")
    else:
        st.info("Model equation not available.")
    
    # Model performance
    st.markdown("#### Key Performance Metrics")
    
    if 'cv_metrics' in st.session_state:
        metrics = st.session_state.cv_metrics
        
        # Create columns for metrics
        col1, col2 = st.columns(2)
        
        with col1:
            # R² with interpretation
            r2 = metrics['R²']['mean']
            r2_std = metrics['R²']['std']
            
            st.metric(
                "R² Score", 
                f"{r2:.4f}",
                delta=f"±{r2_std:.4f}"
            )
            
            if r2 >= 0.7:
                st.success(f"Strong predictive power: Model explains {r2*100:.1f}% of variation")
            elif r2 >= 0.3:
                st.info(f"Moderate predictive power: Model explains {r2*100:.1f}% of variation")
            else:
                st.warning(f"Limited predictive power: Model explains only {r2*100:.1f}% of variation")
        
        with col2:
            # Error metrics with interpretation
            rmse = metrics['RMSE']['mean']
            
            st.metric(
                "RMSE (Root Mean Squared Error)",
                f"{rmse:.4f}",
                delta=f"±{metrics['RMSE']['std']:.4f}",
                delta_color="inverse"  # Lower is better
            )
            
            # Add context for the error metric
            if 'y' in st.session_state:
                y = st.session_state.y
                y_range = np.max(y) - np.min(y)
                error_percentage = (rmse / y_range) * 100
                
                st.markdown(f"""
                RMSE represents the average prediction error in the units of {target_var}.
                This error is approximately **{error_percentage:.1f}%** of the total range of {target_var}.
                """)
    else:
        st.info("Model performance metrics not available.")


def show_linear_diagnostics():
    """
    Display diagnostics for simple linear regression model with diagnostic plots.
    """
    st.markdown("### Model Diagnostics")
    
    # Check if necessary data is available
    if 'model' not in st.session_state or 'X' not in st.session_state or 'y' not in st.session_state:
        st.warning("Complete model data not available. Please retrain the model.")
        return
    
    # Get data from session state
    model = st.session_state.model
    X = st.session_state.X
    y = st.session_state.y
    input_var = st.session_state.input_var
    target_var = st.session_state.target_var
    
    # Calculate predictions and residuals
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Standardized residuals for better analysis
    std_residuals = residuals / np.std(residuals)
    
    # Residual vs Fitted plot
    st.markdown("#### 1. Residuals vs Fitted Values")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred, residuals, alpha=0.6)
    ax.axhline(y=0, color='r', linestyle='--')
    
    # Add a smooth line to help identify patterns
    try:
        from scipy.stats import gaussian_kde
        # Sort the data for smooth line
        sorted_idx = np.argsort(y_pred)
        sorted_x = y_pred[sorted_idx]
        sorted_y = residuals[sorted_idx]
        
        # Calculate smoothed line using rolling average
        window_size = max(int(len(sorted_x) * 0.1), 5)  # 10% of data points or at least 5
        smoothed_y = np.convolve(sorted_y, np.ones(window_size)/window_size, mode='valid')
        valid_x = sorted_x[window_size-1:len(sorted_x)-window_size+1]
        valid_smoothed_y = smoothed_y[:len(valid_x)]
        
        if len(valid_x) > 0:
            ax.plot(valid_x, valid_smoothed_y, color='blue', linestyle='-', linewidth=2)
    except Exception as e:
        st.warning(f"Could not add smoothed trend line: {e}")
    
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted Values")
    
    # Add annotations
    ax.text(0.02, 0.95, "Desired pattern: Random scatter around y=0", 
            transform=ax.transAxes, fontsize=11, 
            bbox=dict(facecolor='white', alpha=0.8))
    
    st.pyplot(fig)
    
    # Analysis of this plot
    st.markdown("""
    **What to look for:**
    - **Random scatter around zero line**: Indicates linearity assumption is met
    - **Patterns or trends**: May indicate non-linear relationship
    - **Funnel shape**: May indicate heteroscedasticity (non-constant variance)
    """)
    
    # QQ Plot (Normal probability plot of residuals)
    st.markdown("#### 2. Q-Q Plot (Normality of Residuals)")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    from scipy import stats
    
    # Create QQ plot
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot of Residuals")
    
    st.pyplot(fig)
    
    # Analysis of this plot
    st.markdown("""
    **What to look for:**
    - **Points following the diagonal line**: Indicates residuals are normally distributed
    - **Deviations from the line**: Indicate departures from normality
    - **S-curve pattern**: May indicate skewness in the residuals
    """)
    
    # Add overall diagnostics summary
    st.markdown("#### Diagnostic Summary")
    
    # Basic normality test
    from scipy import stats
    
    try:
        shapiro_test = stats.shapiro(residuals)
        shapiro_p = shapiro_test[1]
        
        st.markdown(f"""
        **Normality Test (Shapiro-Wilk):**
        - p-value: {shapiro_p:.4f}
        - Interpretation: Residuals are {'normally distributed (p > 0.05)' if shapiro_p > 0.05 else 'not normally distributed (p ≤ 0.05)'}
        """)
    except Exception as e:
        st.warning(f"Could not perform normality test: {e}")


def show_multilinear_diagnostics():
    """
    Display diagnostics for multiple linear regression model with diagnostic plots.
    """
    st.markdown("### Model Diagnostics")
    
    # Check if necessary data is available
    if 'model' not in st.session_state or 'X' not in st.session_state or 'y' not in st.session_state:
        st.warning("Complete model data not available. Please retrain the model.")
        return
    
    # Get data from session state
    model = st.session_state.model
    X = st.session_state.X
    y = st.session_state.y
    input_vars = st.session_state.input_vars if 'input_vars' in st.session_state else []
    target_var = st.session_state.target_var if 'target_var' in st.session_state else "target"
    
    # Calculate predictions and residuals
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Standardized residuals for better analysis
    std_residuals = residuals / np.std(residuals)
    
    # Residual vs Fitted plot
    st.markdown("#### 1. Residuals vs Fitted Values")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred, residuals, alpha=0.6)
    ax.axhline(y=0, color='r', linestyle='--')
    
    # Add a smooth line to help identify patterns
    try:
        # Sort the data for smooth line
        sorted_idx = np.argsort(y_pred)
        sorted_x = y_pred[sorted_idx]
        sorted_y = residuals[sorted_idx]
        
        # Calculate smoothed line using rolling average
        window_size = max(int(len(sorted_x) * 0.1), 5)  # 10% of data points or at least 5
        smoothed_y = np.convolve(sorted_y, np.ones(window_size)/window_size, mode='valid')
        valid_x = sorted_x[window_size-1:len(sorted_x)-window_size+1]
        valid_smoothed_y = smoothed_y[:len(valid_x)]
        
        if len(valid_x) > 0:
            ax.plot(valid_x, valid_smoothed_y, color='blue', linestyle='-', linewidth=2)
    except Exception as e:
        st.warning(f"Could not add smoothed trend line: {e}")
    
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted Values")
    
    # Add annotations
    ax.text(0.02, 0.95, "Desired pattern: Random scatter around y=0", 
            transform=ax.transAxes, fontsize=11, 
            bbox=dict(facecolor='white', alpha=0.8))
    
    st.pyplot(fig)
    
    # Analysis of this plot
    st.markdown("""
    **What to look for:**
    - **Random scatter around zero line**: Indicates linearity assumption is met
    - **Patterns or trends**: May indicate non-linear relationship
    - **Funnel shape**: May indicate heteroscedasticity (non-constant variance)
    """)
    
    # QQ Plot (Normal probability plot of residuals)
    st.markdown("#### 2. Q-Q Plot (Normality of Residuals)")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    from scipy import stats
    
    # Create QQ plot
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot of Residuals")
    
    st.pyplot(fig)
    
    # Analysis of this plot
    st.markdown("""
    **What to look for:**
    - **Points following the diagonal line**: Indicates residuals are normally distributed
    - **Deviations from the line**: Indicate departures from normality
    - **S-curve pattern**: May indicate skewness in the residuals
    """)
    
    # Add overall diagnostics summary
    st.markdown("#### Diagnostic Summary")
    
    # Basic normality test
    from scipy import stats
    
    try:
        shapiro_test = stats.shapiro(residuals)
        shapiro_p = shapiro_test[1]
        
        st.markdown(f"""
        **Normality Test (Shapiro-Wilk):**
        - p-value: {shapiro_p:.4f}
        - Interpretation: Residuals are {'normally distributed (p > 0.05)' if shapiro_p > 0.05 else 'not normally distributed (p ≤ 0.05)'}
        """)
    except Exception as e:
        st.warning(f"Could not perform normality test: {e}")


def show_linear_predictions():
    """
    Display prediction interface for simple linear regression model (placeholder).
    """
    st.markdown("### Make Predictions")
    
    # Create prediction interface
    input_method = st.radio("Select input method", ["Manual Input", "Use Existing Data"], horizontal=True)
    
    if input_method == "Manual Input":
        st.number_input("Enter value for input variable", value=0.0)
        st.button("Predict", key="manual_predict", use_container_width=True)
        st.info("This section will allow users to input values and get predictions.")
    else:  # Use Existing Data
        st.slider("Select data point index", 0, 10, 0)
        st.info("This section will allow users to select existing data points and compare actual vs predicted values.")


def show_multilinear_predictions():
    """
    Display prediction interface for multiple linear regression model (placeholder).
    """
    st.markdown("### Make Predictions")
    
    # Create prediction interface
    input_method = st.radio("Select input method", ["Manual Input", "Use Existing Data"], horizontal=True)
    
    if input_method == "Manual Input":
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Variable 1", value=0.0)
            st.number_input("Variable 3", value=0.0)
        with col2:
            st.number_input("Variable 2", value=0.0)
            st.number_input("Variable 4", value=0.0)
            
        st.button("Predict", key="manual_predict_multi", use_container_width=True)
        st.info("This section will allow users to input multiple values and get predictions.")
    else:  # Use Existing Data
        st.slider("Select data point index", 0, 10, 0)
        st.info("This section will allow users to select existing data points and compare actual vs predicted values.")


def show_linear_recommendations():
    """
    Display recommendations for simple linear regression model (placeholder).
    """
    st.markdown("### Simple Linear Regression Insights")
    
    # Relationship analysis
    st.markdown("#### Relationship Analysis")
    st.info("This section will provide insights about the relationship between variables.")
    
    # Strength of relationship
    st.markdown("#### Strength of Relationship")
    st.info("This section will analyze the strength of the relationship based on model metrics.")
    
    # Practical interpretation
    st.markdown("#### Practical Interpretation")
    st.info("This section will give practical interpretation of the model in business terms.")
    
    # Model assumptions check
    st.markdown("#### Model Assumptions Check")
    st.info("This section will summarize whether the model meets the assumptions of linear regression.")
    
    # Recommendation summary
    st.markdown("#### Business Recommendation")
    st.info("This section will provide actionable business recommendations based on the model results.")


def show_multilinear_recommendations():
    """
    Display recommendations for multiple linear regression model (placeholder).
    """
    st.markdown("### Multiple Linear Regression Insights")
    
    # Key predictors
    st.markdown("#### Key Predictors")
    st.info("This section will identify and explain the most important predictors in the model.")
    
    # Overall model performance
    st.markdown("#### Overall Model Performance")
    st.info("This section will provide a summary assessment of the model's performance.")
    
    # Potential issues
    st.markdown("#### Potential Issues")
    st.info("This section will highlight potential issues with the model and how to address them.")
    
    # Recommendation summary
    st.markdown("#### Business Recommendations")
    st.info("This section will provide actionable business recommendations based on the model results.")