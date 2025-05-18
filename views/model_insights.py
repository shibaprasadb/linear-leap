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
    Display summary for simple linear regression model with focus on interpretation.
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
    
    if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
        coef = model.coef_[0]
        intercept = model.intercept_
        
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
    
    # Interpretation
    st.markdown("#### Interpretation")
    
    if hasattr(model, 'coef_'):
        coef = model.coef_[0]
        
        # Determine the relationship direction
        if coef > 0:
            relationship = "positive"
            change_direction = "increases"
        else:
            relationship = "negative"
            change_direction = "decreases"
        
        # Create interpretation text
        st.markdown(f"""
        This model shows a **{relationship} relationship** between {input_var} and {target_var}:
        
        - For each one-unit increase in {input_var}, {target_var} {change_direction} by approximately **{abs(coef):.4f} units**
        - This means that {input_var} has a {"direct" if coef > 0 else "inverse"} effect on {target_var}
        """)
        
        # Add strength interpretation
        if 'cv_metrics' in st.session_state:
            r2 = st.session_state.cv_metrics['R²']['mean']
            
            if r2 >= 0.7:
                strength = "strong"
            elif r2 >= 0.3:
                strength = "moderate"
            else:
                strength = "weak"
                
            st.markdown(f"""
            The model shows a **{strength} predictive power** with R² of {r2:.4f}, meaning that 
            approximately {r2*100:.1f}% of the variation in {target_var} can be explained by {input_var}.
            """)
    else:
        st.info("Model interpretation not available.")
    
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
            mae = metrics['MAE']['mean']
            
            # Only show one of RMSE or MAE to avoid clutter
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
    
    # Business insight section
    st.markdown("#### Business Perspective")
    
    if hasattr(model, 'coef_') and 'cv_metrics' in st.session_state:
        coef = model.coef_[0]
        r2 = st.session_state.cv_metrics['R²']['mean']
        
        # Provide business-oriented interpretation
        st.markdown(f"""
        From a business perspective, this model shows that:
        
        1. {input_var} is {"an important factor" if r2 >= 0.3 else "a factor"} in determining {target_var}
        
        2. The relationship is {"reliable and can be used for planning" if r2 >= 0.7 else 
            "moderately reliable and should be used with other factors" if r2 >= 0.3 else 
            "not strong enough to be used alone for decision-making"}
        
        3. {"Focusing on strategies that modify " + input_var + " could have meaningful impacts on " + target_var if abs(coef) > 0.1 and r2 >= 0.3 else 
            "While there is a relationship, other factors likely have stronger influences on " + target_var}
        """)
    else:
        st.info("Business insights not available.")


def show_multilinear_summary():
    """
    Display summary for multiple linear regression model with focus on interpretation.
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
    
    # Interpretation
    st.markdown("#### Key Variables and Their Impact")
    
    if hasattr(model, 'coef_'):
        coefs = model.coef_
        
        # Create dataframe for sorting
        var_impact = pd.DataFrame({
            'Variable': input_vars,
            'Coefficient': coefs,
            'Absolute Impact': np.abs(coefs)
        }).sort_values('Absolute Impact', ascending=False)
        
        # Display top variables
        for i in range(min(3, len(var_impact))):
            var = var_impact.iloc[i]['Variable']
            coef = var_impact.iloc[i]['Coefficient']
            
            if coef > 0:
                relationship = "positive"
                change_direction = "increases"
            else:
                relationship = "negative"
                change_direction = "decreases"
            
            st.markdown(f"""
            **{i+1}. {var}** (Coefficient: {coef:.4f})
            - Has a **{relationship}** relationship with {target_var}
            - For each one-unit increase in {var}, {target_var} {change_direction} by approximately **{abs(coef):.4f} units**
            - This assumes all other variables remain constant
            """)
        
        # Feature importance visualization
        st.markdown("#### Variable Importance")
        
        # Prepare data for visualization
        var_impact = var_impact.head(min(10, len(var_impact)))  # Limit to top 10
        
        # Create color map based on coefficient sign
        colors = ['#4285F4' if x >= 0 else '#EA4335' for x in var_impact['Coefficient']]
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(var_impact['Variable'], var_impact['Absolute Impact'], color=colors)
        ax.set_xlabel('|Coefficient Value|')
        ax.set_title('Variable Importance in the Model')
        
        # Add a legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#4285F4', label='Positive Effect'),
            Patch(facecolor='#EA4335', label='Negative Effect')
        ]
        ax.legend(handles=legend_elements)
        
        st.pyplot(fig)
    else:
        st.info("Model interpretation not available.")
    
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
    
    # Business insight section
    st.markdown("#### Business Perspective")
    
    if hasattr(model, 'coef_') and 'cv_metrics' in st.session_state:
        r2 = st.session_state.cv_metrics['R²']['mean']
        
        # Prepare data for top positive and negative influencers
        coefs = model.coef_
        pos_idx = np.argsort(-coefs)[:3]  # Top 3 positive coefficients
        neg_idx = np.argsort(coefs)[:3]    # Top 3 negative coefficients
        
        # Filter out zero or insignificant coefficients
        pos_drivers = [input_vars[i] for i in pos_idx if coefs[i] > 0.01]
        neg_drivers = [input_vars[i] for i in neg_idx if coefs[i] < -0.01]
        
        # Provide business-oriented interpretation
        st.markdown(f"""
        From a business perspective, this model shows that:
        
        1. {input_vars[np.argmax(np.abs(coefs))]} has the strongest influence on {target_var}
        
        2. The model {'explains a substantial portion' if r2 >= 0.7 else 
           'explains a moderate portion' if r2 >= 0.3 else 
           'explains only a small portion'} of what drives {target_var}
        """)
        
        if pos_drivers:
            pos_drivers_list = ", ".join(pos_drivers)
            st.markdown(f"3. **Positive Drivers:** {pos_drivers_list}")
            
        if neg_drivers:
            neg_drivers_list = ", ".join(neg_drivers)
            st.markdown(f"4. **Negative Drivers:** {neg_drivers_list}")
        
        # Add model reliability statement
        st.markdown(f"""
        5. This model is {'highly reliable and suitable for decision-making' if r2 >= 0.7 else 
          'moderately reliable and should be used alongside other inputs' if r2 >= 0.3 else 
          'of limited reliability and should be used primarily for directional guidance'}
        """)
    else:
        st.info("Business insights not available.")


def show_linear_diagnostics():
    """
    Display diagnostics for simple linear regression model (placeholder).
    """
    st.markdown("### Model Diagnostics")
    
    # Residual analysis
    st.markdown("#### Residual Analysis")
    st.info("This section will display residual plots to check for patterns.")
    
    # Residual distribution
    st.markdown("#### Residual Distribution")
    st.info("This section will show the distribution of residuals to check for normality.")
    
    # Normality test
    st.markdown("#### Normality Test")
    st.info("This section will display Q-Q plots and statistical tests for normality.")
    
    # Homoscedasticity check
    st.markdown("#### Homoscedasticity Check")
    st.info("This section will provide visualizations to check for constant variance of residuals.")


def show_multilinear_diagnostics():
    """
    Display diagnostics for multiple linear regression model (placeholder).
    """
    st.markdown("### Model Diagnostics")
    
    # Residual analysis
    st.markdown("#### Residual Analysis")
    st.info("This section will display residual plots to check for patterns.")
    
    # Residual distribution
    st.markdown("#### Residual Distribution")
    st.info("This section will show the distribution of residuals to check for normality.")
    
    # Normality test
    st.markdown("#### Normality Test")
    st.info("This section will display Q-Q plots and statistical tests for normality.")
    
    # Multicollinearity check
    st.markdown("#### Multicollinearity Check")
    st.info("This section will provide correlation analysis and VIF statistics to check for multicollinearity.")


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