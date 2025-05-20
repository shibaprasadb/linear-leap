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
    Display recommendations for simple linear regression model with AI-powered analysis.
    """
    st.markdown("### Business Recommendations")
    
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
    
    # Calculate predictions and residuals for analysis
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Get API key from Streamlit secrets or user input
    api_key = None
    model_name = "gemini-2.0-flash"  # Default model
    
    if 'user_api_key' in st.session_state and st.session_state.user_api_key:
        api_key = st.session_state.user_api_key
        model_name = st.session_state.user_model_name
    else:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
            model_name = "gemini-2.0-flash"
        except Exception as e:
            st.warning("API key not found. Some recommendations may not be available.")
    
    # SECTION 1: Residuals Analysis
    st.markdown("#### 1. Model Fit Analysis")
    
    if api_key:
        with st.spinner("Analyzing residuals..."):
            # Create residuals vs fitted plot for AI analysis only (won't be displayed)
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
                pass
            
            ax.set_xlabel("Fitted Values")
            ax.set_ylabel("Residuals")
            ax.set_title("Residuals vs Fitted Values")
            
            # Convert plot to PIL image for API
            from utils.plot_analysis import plot_to_pil, generate_text_with_image
            
            pil_image = plot_to_pil(fig)
            plt.close(fig)  # Close the figure since we won't display it
            
            # Define prompt for analysis
            prompt = """
            Summarise this residuals vs fitted plot for the linear regression. 
            
            First, highlight what is going well for the model - where the residuals behave as expected.
            
            Then, identify any cases where the model might be going wrong. Is there any up or down trend at a particular point? That should be taken care of.
            
            Focus on:
            1. Overall pattern of residuals - are they randomly distributed around zero?
            2. Any systematic patterns (curves, trends) that suggest model misspecification
            3. Any sections where residuals are consistently above or below zero
            4. Areas where the prediction might be biased and good
            5. If there are outliers, discuss their potential impact
            
            Provide 3-4 sentences summarizing the key observations and implications. Just give me the summary don't add anything like "Here's a summary of the residuals vs fitted values plot"
            """
            
            # Get analysis from API
            residual_analysis = generate_text_with_image(prompt, pil_image, api_key, model_name=model_name)
            
            # Display analysis
            if residual_analysis and "Error" not in residual_analysis:
                st.markdown(residual_analysis)
            else:
                st.warning(f"Could not generate analysis: {residual_analysis}")
                # Fall back to basic analysis
                st.info("""
                The residual analysis examines if the model fits the data well. Key points to consider:
                - Random scatter around zero line indicates a good linear fit
                - Patterns or curves suggest non-linear relationships
                - Consider transformation if systematic patterns appear
                """)
    else:
        # If no API key, just show basic analysis
        st.info("""
        The residual analysis examines if the model fits the data well. Key points to consider:
        - Random scatter around zero line indicates a good linear fit
        - Patterns or curves suggest non-linear relationships
        - Consider transformation if systematic patterns appear
        """)
    
    # SECTION 2: Q-Q Plot Analysis
    st.markdown("#### 2. Normality Analysis")
    
    if api_key:
        with st.spinner("Analyzing Q-Q plot..."):
            # Create Q-Q plot for API analysis (won't be displayed)
            fig, ax = plt.subplots(figsize=(10, 6))
            from scipy import stats
            
            # Create QQ plot
            stats.probplot(residuals, dist="norm", plot=ax)
            ax.set_title("Q-Q Plot of Residuals")
            
            # Convert plot to PIL image for API
            pil_image = plot_to_pil(fig)
            plt.close(fig)  # Close the figure since we won't display it
            
            # Define prompt for analysis
            prompt = """
            Analyze this Q-Q (quantile-quantile) plot for a linear regression model.
            
            Focus on what it means for the model's reliability, specifically identifying cases where the model might not give good results.
            
            Consider:
            1. How well the points follow the diagonal line
            2. Any deviations at the tails
            3. What these deviations imply about the distribution of residuals
            4. How normality or non-normality of residuals affects prediction reliability
            5. What types of data points might be predicted less accurately based on this plot
            
            Provide a summary in 4-5 sentences (not bullet points). Don't start with "The Q-Q plot shows..." or similar introductory text. Just give the direct analysis.
            """
            
            # Get analysis from API
            qq_analysis = generate_text_with_image(prompt, pil_image, api_key, model_name=model_name)
            
            # Display analysis
            if qq_analysis and "Error" not in qq_analysis:
                st.markdown(qq_analysis)
            else:
                st.warning(f"Could not generate analysis: {qq_analysis}")
                # Fall back to basic analysis
                st.info("""
                The Q-Q plot assesses if residuals follow a normal distribution. Deviations from the diagonal line, especially at the tails, suggest non-normality which can affect the reliability of prediction intervals and hypothesis tests. The model may produce less accurate predictions for extreme values.
                """)
    else:
        # If no API key, just show basic analysis
        st.info("""
        The Q-Q plot assesses if residuals follow a normal distribution. Deviations from the diagonal line, especially at the tails, suggest non-normality which can affect the reliability of prediction intervals and hypothesis tests. The model may produce less accurate predictions for extreme values.
        """)
    
    # SECTION 3: R-squared and Statistical Significance
    st.markdown("#### 3. Model Performance & Significance")
    
    # Get R-squared and other metrics from the trained model results
    if 'cv_metrics' in st.session_state:
        metrics = st.session_state.cv_metrics
        r2 = metrics['R²']['mean']
        r2_std = metrics['R²']['std']
        
        # Get coefficient and p-value (for p-value, check if it was stored during training)
        coef = None
        p_value = None
        significance = "unknown"
        
        if 'coef_mean' in st.session_state:
            coef = st.session_state.coef_mean
            
        if 'p_value' in st.session_state:
            p_value = st.session_state.p_value
            significance = "statistically significant" if p_value < 0.05 else "not statistically significant at the conventional level"
        
        # Create analytics summary based on R-squared value
        if r2 >= 0.7:
            strength = "strong"
            explanation = f"The model explains {r2*100:.1f}% of the variation in {target_var}, indicating a robust predictive relationship."
        elif r2 >= 0.3:
            strength = "moderate"
            explanation = f"The model explains {r2*100:.1f}% of the variation in {target_var}, suggesting a meaningful but not dominant relationship."
        else:
            strength = "weak"
            explanation = f"The model explains only {r2*100:.1f}% of the variation in {target_var}, indicating that other factors have substantial influence."
        
        # Display R-squared interpretation
        st.markdown(f"""
        **R-squared Analysis:**
        
        This model shows a **{strength} relationship** between {input_var} and {target_var}. {explanation}
        """)
        
        # Display p-value information if available
        if p_value is not None:
            st.markdown(f"""
            **Statistical Significance:**
            
            The relationship between {input_var} and {target_var} is {significance} (p-value: {p_value:.4f}). 
            """)
            
            # Nuanced interpretation of p-value
            if p_value < 0.05:
                st.markdown("There is strong evidence that a genuine relationship exists between the variables.")
            elif p_value < 0.1:
                st.markdown("There is moderate evidence of a relationship, though it doesn't meet the conventional significance threshold.")
            else:
                st.markdown("""
                While the p-value exceeds the conventional significance threshold, this doesn't necessarily mean the relationship is meaningless. 
                Consider the practical significance, sample size, and business context when interpreting results.
                """)
        else:
            # If p-value not available, give generic guidance
            st.markdown("""
            **Statistical Significance:**
            
            The statistical significance of this relationship wasn't explicitly calculated during model training. 
            However, the cross-validation results provide evidence about the model's reliability.
            """)
    else:
        st.info("R-squared and significance metrics not available. Please retrain the model.")
    
    # SECTION 4: Coefficient Interpretation
    st.markdown("#### 4. Business Interpretation & Recommendations")
    
    # Check if we have coefficient information
    if 'coef_mean' in st.session_state and 'intercept_mean' in st.session_state:
        coef = st.session_state.coef_mean
        intercept = st.session_state.intercept_mean
        
        # Direction of relationship
        direction = "positive" if coef > 0 else "negative"
        
        # Business interpretation
        st.markdown(f"""
        **Coefficient Interpretation:**
        
        The model equation is: **{target_var} = {intercept:.4f} + {coef:.4f} × {input_var}**
        
        This means:
        
        1. For every one-unit increase in {input_var}, {target_var} {"increases" if coef > 0 else "decreases"} by {abs(coef):.4f} units, on average.
        
        2. When {input_var} is zero, the predicted {target_var} would be {intercept:.4f}.
        
        3. The relationship is {direction}, indicating that higher values of {input_var} are associated with {"higher" if coef > 0 else "lower"} values of {target_var}.
        """)
        
        # Business recommendations based on relationship strength and direction
        if 'cv_metrics' in st.session_state:
            r2 = st.session_state.cv_metrics['R²']['mean']
            
            st.markdown("**Business Recommendations:**")
            
            if r2 >= 0.7:
                if coef > 0:
                    st.markdown(f"""
                    Given the strong positive relationship, investing resources to increase {input_var} is likely to yield significant improvements in {target_var}. Consider developing strategies specifically targeting this variable, as the return on investment should be substantial and predictable.
                    """)
                else:
                    st.markdown(f"""
                    Given the strong negative relationship, focus on reducing {input_var} to improve {target_var}. This relationship is reliable enough to serve as a primary driver for strategic decisions, and interventions targeting {input_var} should produce predictable results.
                    """)
            elif r2 >= 0.3:
                if coef > 0:
                    st.markdown(f"""
                    With this moderate positive relationship, {input_var} should be considered an important factor, but not the only driver of {target_var}. Develop a balanced approach that includes targeting {input_var} alongside other potential factors. Monitor changes to validate the effectiveness of interventions.
                    """)
                else:
                    st.markdown(f"""
                    This moderate negative relationship suggests that reducing {input_var} can help improve {target_var}, but should be part of a broader strategy. Consider this variable as one of several levers to influence outcomes, and continue to explore additional factors not captured in this model.
                    """)
            else:
                st.markdown(f"""
                While there is a {"positive" if coef > 0 else "negative"} relationship, the relatively weak predictive power suggests that {input_var} is just one of many factors influencing {target_var}. This model should be used as a general guideline rather than a precise predictive tool. Consider expanding your analysis to include additional variables that might have stronger relationships with {target_var}.
                """)
    else:
        st.info("Coefficient information not available. Please retrain the model.")


def show_multilinear_recommendations():
    """
    Display recommendations for multiple linear regression model (placeholder).
    """
    st.markdown("### Business Recommendations")
    
    # Key predictors
    st.markdown("#### 1. Key Predictors & Influence")
    st.info("This section will identify and explain the most important predictors in the model.")
    
    # Overall model performance
    st.markdown("#### 2. Model Performance Assessment")
    st.info("This section will provide a summary assessment of the model's performance.")
    
    # Potential issues
    st.markdown("#### 3. Potential Model Issues")
    st.info("This section will highlight potential issues with the model and how to address them.")
    
    # Recommendation summary
    st.markdown("#### 4. Action Recommendations")
    st.info("This section will provide actionable business recommendations based on the model results.")