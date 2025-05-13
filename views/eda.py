import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def show_eda():
    """
    Display the Exploratory Data Analysis (EDA) page with different analyses
    based on the selected regression type.
    """
    st.markdown("## Exploratory Data Analysis")
    
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
    
    # Display different EDA based on regression type
    if regression_type == 'linear':
        show_linear_eda(data, inputs[0], target)
    else:
        show_multilinear_eda(data, inputs, target)
    
    # Button to proceed to model training
    st.markdown("---")
    if st.button("Proceed to Model Training", key="proceed_to_training", use_container_width=True):
        st.session_state.view = "model_training"
        st.rerun()


def detect_outliers(x, y, threshold=3.0):
    """
    Detect outliers in x and y variables using z-scores.
    
    Parameters:
    -----------
    x : array-like
        Independent variable
    y : array-like
        Dependent variable
    threshold : float, default=3.0
        Z-score threshold for outlier detection
        
    Returns:
    --------
    dict : Dictionary containing outlier indices and statistics
    """
    # Ensure numpy arrays
    x = np.array(x)
    y = np.array(y)
    
    # Handle empty arrays
    if len(x) == 0 or len(y) == 0:
        return {'status': 'error', 'message': 'Empty input arrays'}
    
    # Calculate z-scores for both x and y
    x_z = np.abs(stats.zscore(x, nan_policy='omit'))
    y_z = np.abs(stats.zscore(y, nan_policy='omit'))
    
    # Find outliers in x and y separately
    x_outliers = np.where(x_z > threshold)[0]
    y_outliers = np.where(y_z > threshold)[0]
    
    # Combined outliers (points that are outliers in either x or y)
    combined_outliers = np.unique(np.concatenate((x_outliers, y_outliers))) if len(x_outliers) > 0 or len(y_outliers) > 0 else np.array([])
    
    result = {
        'x_outliers': x_outliers,
        'y_outliers': y_outliers,
        'combined_outliers': combined_outliers,
        'outlier_count': len(combined_outliers),
        'outlier_percentage': len(combined_outliers) / len(x) * 100,
        'has_outliers': len(combined_outliers) > 0
    }
    
    # Only add values if outliers exist
    if result['has_outliers']:
        result['x_outlier_values'] = x[combined_outliers] if len(combined_outliers) > 0 else np.array([])
        result['y_outlier_values'] = y[combined_outliers] if len(combined_outliers) > 0 else np.array([])
    else:
        result['message'] = 'No outliers detected using z-score threshold of {}'.format(threshold)
    
    return result



# def show_linear_eda(data, input_var, target_var):
#     """
#     Display EDA for simple linear regression.
#     """
#     st.markdown("### Simple Linear Regression EDA")
    
#     # Display tabs for different EDA aspects - removed Correlation Analysis tab
#     tab1, tab2 = st.tabs(["Summary Statistics", "Data Visualization"])
    
#     with tab1:
#         st.markdown("#### Summary Statistics")
        
#         # Basic statistics for each variable
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.markdown(f"**Input Variable: {input_var}**")
#             st.dataframe(pd.DataFrame(data[input_var].describe()).T)
            
#             # Box plot for input variable
#             fig, ax = plt.subplots(figsize=(8, 4))
#             sns.boxplot(x=data[input_var], ax=ax)
#             ax.set_title(f"Distribution of {input_var}")
#             st.pyplot(fig)
        
#         with col2:
#             st.markdown(f"**Target Variable: {target_var}**")
#             st.dataframe(pd.DataFrame(data[target_var].describe()).T)
            
#             # Box plot for target variable
#             fig, ax = plt.subplots(figsize=(8, 4))
#             sns.boxplot(x=data[target_var], ax=ax)
#             ax.set_title(f"Distribution of {target_var}")
#             st.pyplot(fig)
    
#     with tab2:
#         st.markdown("#### Data Visualization")
        
#         # Calculate correlation for display with scatter plot
#         correlation = data[[input_var, target_var]].corr().iloc[0, 1]
        
#         # Outlier detection
#         outlier_threshold = st.slider(
#             "Outlier Detection Z-Score Threshold (Default is 3)",
#             min_value=1.0,
#             max_value=5.0,
#             value=3.0,
#             step=0.1,
#             help="Z-score threshold for outlier detection. Lower values detect more outliers."
#         )
        
#         # Detect outliers using the function
#         outlier_results = detect_outliers(
#             data[input_var].values,
#             data[target_var].values,
#             threshold=outlier_threshold
#         )
        
#         # Show correlation and outlier information
#         col1, col2 = st.columns([3, 1])
        
#         with col2:
#             # Display correlation with descriptive label
#             st.metric(
#                 "Correlation", 
#                 f"{correlation:.4f}", 
#                 delta=f"{'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.3 else 'Weak'}"
#             )
            
#             # Add correlation interpretation
#             if correlation > 0.7:
#                 st.success("Strong positive correlation")
#             elif correlation > 0.3:
#                 st.info("Moderate positive correlation")
#             elif correlation > 0:
#                 st.warning("Weak positive correlation")
#             elif correlation > -0.3:
#                 st.warning("Weak negative correlation")
#             elif correlation > -0.7:
#                 st.info("Moderate negative correlation")
#             else:
#                 st.success("Strong negative correlation")
            
#             # Add outlier information
#             st.metric(
#                 "Outliers Detected", 
#                 f"{outlier_results['outlier_count']}",
#                 delta=f"{outlier_results['outlier_percentage']:.1f}% of data"
#             )
        
#         with col1:
#             # Create scatter plot with outliers highlighted
#             fig, ax = plt.subplots(figsize=(10, 6))
            
#             # Plot non-outlier points
#             if outlier_results['has_outliers']:
#                 # Create mask for non-outlier points
#                 mask = np.ones(len(data), dtype=bool)
#                 mask[outlier_results['combined_outliers']] = False
                
#                 # Plot regular points
#                 ax.scatter(
#                     data[input_var].values[mask], 
#                     data[target_var].values[mask], 
#                     alpha=0.6,
#                     label='Regular data points'
#                 )
                
#                 # Plot outliers in red
#                 ax.scatter(
#                     outlier_results['x_outlier_values'], 
#                     outlier_results['y_outlier_values'], 
#                     color='red', 
#                     alpha=0.8,
#                     s=100,  # Larger size
#                     label='Outliers'
#                 )
#                 ax.legend()
#             else:
#                 # If no outliers, plot all points normally
#                 ax.scatter(data[input_var], data[target_var], alpha=0.6)
            
#             # Include correlation in the title
#             ax.set_title(f"Relationship between {input_var} and {target_var} (Correlation: {correlation:.4f})")
#             ax.set_xlabel(input_var)
#             ax.set_ylabel(target_var)
#             st.pyplot(fig)
            
#             # Display message about outliers
#             if outlier_results['has_outliers']:
#                 st.warning(f"**{outlier_results['outlier_count']} outliers** detected (shown in red) using a z-score threshold of {outlier_threshold}. These outliers may influence the regression model.")
#             else:
#                 st.success(f"No outliers detected using a z-score threshold of {outlier_threshold}.")
        
#         # Histograms
#         st.markdown("#### Distributions")
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
#         # Input variable histogram
#         sns.histplot(data[input_var], kde=True, ax=ax1)
#         ax1.set_title(f"Distribution of {input_var}")
        
#         # Target variable histogram
#         sns.histplot(data[target_var], kde=True, ax=ax2)
#         ax2.set_title(f"Distribution of {target_var}")
        
#         plt.tight_layout()
#         st.pyplot(fig)

def show_linear_eda(data, input_var, target_var):
    """
    Display EDA for simple linear regression.
    """
    st.markdown("### Simple Linear Regression EDA")
    
    # Display tabs for different EDA aspects - removed Correlation Analysis tab
    tab1, tab2 = st.tabs(["Summary Statistics", "Data Visualization"])
    
    with tab1:
        st.markdown("#### Summary Statistics")
        
        # Basic statistics for each variable
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Input Variable: {input_var}**")
            st.dataframe(pd.DataFrame(data[input_var].describe()).T)
            
            # Box plot for input variable
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(x=data[input_var], ax=ax)
            ax.set_title(f"Distribution of {input_var}")
            st.pyplot(fig)
        
        with col2:
            st.markdown(f"**Target Variable: {target_var}**")
            st.dataframe(pd.DataFrame(data[target_var].describe()).T)
            
            # Box plot for target variable
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(x=data[target_var], ax=ax)
            ax.set_title(f"Distribution of {target_var}")
            st.pyplot(fig)
    
    with tab2:
        st.markdown("#### Data Visualization")
        
        # Calculate correlation for display with scatter plot
        correlation = data[[input_var, target_var]].corr().iloc[0, 1]
        
        # Outlier detection
        outlier_threshold = st.slider(
            "Outlier Detection Z-Score Threshold (Default is 3 - Recommended)",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.1,
            help="Z-score threshold for outlier detection. Lower values detect more outliers."
        )
        
        # Detect outliers using the function
        outlier_results = detect_outliers(
            data[input_var].values,
            data[target_var].values,
            threshold=outlier_threshold
        )
        
        # Show correlation and outlier information
        col1, col2 = st.columns([3, 1])
        
        with col2:
            # Display correlation with descriptive label
            st.metric(
                "Correlation", 
                f"{correlation:.4f}", 
                delta=f"{'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.3 else 'Weak'}"
            )
            
            # Add correlation interpretation
            if correlation > 0.7:
                st.success("Strong positive correlation")
            elif correlation > 0.3:
                st.info("Moderate positive correlation")
            elif correlation > 0:
                st.warning("Weak positive correlation")
            elif correlation > -0.3:
                st.warning("Weak negative correlation")
            elif correlation > -0.7:
                st.info("Moderate negative correlation")
            else:
                st.success("Strong negative correlation")
            
            # Add outlier information
            st.metric(
                "Outliers Detected", 
                f"{outlier_results['outlier_count']}",
                delta=f"{outlier_results['outlier_percentage']:.1f}% of data"
            )
        
        with col1:
            # Create scatter plot with outliers highlighted
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot non-outlier points
            if outlier_results['has_outliers']:
                # Create mask for non-outlier points
                mask = np.ones(len(data), dtype=bool)
                mask[outlier_results['combined_outliers']] = False
                
                # Plot regular points
                ax.scatter(
                    data[input_var].values[mask], 
                    data[target_var].values[mask], 
                    alpha=0.6,
                    label='Regular data points'
                )
                
                # Plot outliers in red
                ax.scatter(
                    outlier_results['x_outlier_values'], 
                    outlier_results['y_outlier_values'], 
                    color='red', 
                    alpha=0.8,
                    s=100,  # Larger size
                    label='Outliers'
                )
                ax.legend()
            else:
                # If no outliers, plot all points normally
                ax.scatter(data[input_var], data[target_var], alpha=0.6)
            
            # Include correlation in the title
            ax.set_title(f"Relationship between {input_var} and {target_var} (Correlation: {correlation:.4f})")
            ax.set_xlabel(input_var)
            ax.set_ylabel(target_var)
            st.pyplot(fig)
            
            # Display message about outliers
            if outlier_results['has_outliers']:
                st.warning(f"**{outlier_results['outlier_count']} outliers** detected (shown in red) using a z-score threshold of {outlier_threshold}. These outliers may influence the regression model.")
            else:
                st.success(f"No outliers detected using a z-score threshold of {outlier_threshold}.")
            
            # Pro tip about outliers
            st.info("""
            **Pro Tip:** It is important to understand the business context of why outliers might be present. 
            Removing them without proper context may lead to erroneous conclusions or loss of important information.
            Always investigate outliers to determine if they represent actual errors, rare events, or valuable insights.
            """)
        
        # Histograms
        st.markdown("#### Distributions")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Input variable histogram
        sns.histplot(data[input_var], kde=True, ax=ax1)
        ax1.set_title(f"Distribution of {input_var}")
        
        # Target variable histogram
        sns.histplot(data[target_var], kde=True, ax=ax2)
        ax2.set_title(f"Distribution of {target_var}")
        
        plt.tight_layout()
        st.pyplot(fig)

def show_multilinear_eda(data, input_vars, target_var):
    """
    Display EDA for multiple linear regression.
    """
    # [Keep multilinear_eda implementation as is]
    st.markdown("### Multiple Linear Regression EDA")
    
    # Display tabs for different EDA aspects
    tab1, tab2, tab3, tab4 = st.tabs(["Summary Statistics", "Data Visualization", "Correlation Analysis", "Multicollinearity"])
    
    with tab1:
        st.markdown("#### Summary Statistics")
        
        # All variables statistics
        st.dataframe(data[input_vars + [target_var]].describe())
        
        # Box plots
        st.markdown("#### Box Plots")
        
        # Display only first few variables if there are many
        display_vars = input_vars[:min(3, len(input_vars))] + [target_var]
        
        fig, axes = plt.subplots(1, len(display_vars), figsize=(15, 5))
        
        # Handle case with only one subplot
        if len(display_vars) == 1:
            axes = [axes]
            
        for i, var in enumerate(display_vars):
            sns.boxplot(y=data[var], ax=axes[i])
            axes[i].set_title(f"{var}")
            
        plt.tight_layout()
        st.pyplot(fig)
        
        if len(input_vars) > 3:
            st.info(f"Showing only first 3 of {len(input_vars)} input variables. See Data Visualization tab for more details.")
    
    with tab2:
        st.markdown("#### Data Visualization")
        
        # Select variable to visualize
        selected_var = st.selectbox(
            "Select input variable to visualize against target:",
            options=input_vars
        )
        
        # Scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = sns.scatterplot(x=selected_var, y=target_var, data=data, ax=ax)
        ax.set_title(f"Relationship between {selected_var} and {target_var}")
        ax.set_xlabel(selected_var)
        ax.set_ylabel(target_var)
        st.pyplot(fig)
        
        # Histograms for selected variables
        st.markdown("#### Distributions")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Selected input variable histogram
        sns.histplot(data[selected_var], kde=True, ax=ax1)
        ax1.set_title(f"Distribution of {selected_var}")
        
        # Target variable histogram
        sns.histplot(data[target_var], kde=True, ax=ax2)
        ax2.set_title(f"Distribution of {target_var}")
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Pairplot (optional for datasets with many variables)
        if len(input_vars) <= 5 and st.checkbox("Show pairplot (may be slow for large datasets)", value=False):
            st.markdown("#### Pairwise Relationships")
            
            fig = sns.pairplot(data[input_vars + [target_var]], height=2.5)
            fig.fig.suptitle("Pairwise Relationships", y=1.02)
            st.pyplot(fig.fig)
    
    with tab3:
        st.markdown("#### Correlation Analysis")
        
        # Correlation matrix
        corr_matrix = data[input_vars + [target_var]].corr()
        
        # Display correlation with target
        st.markdown("**Correlation with Target Variable**")
        target_corrs = corr_matrix[target_var].drop(target_var).sort_values(ascending=False)
        
        for var, corr in target_corrs.items():
            strength = 'Strong' if abs(corr) > 0.7 else 'Moderate' if abs(corr) > 0.3 else 'Weak'
            st.metric(
                f"Correlation: {var}",
                f"{corr:.4f}",
                delta=f"{strength} correlation"
            )
        
        # Correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
    
    with tab4:
        st.markdown("#### Multicollinearity Analysis")
        
        # Input variables correlation matrix
        input_corr = data[input_vars].corr()
        
        st.markdown("Checking for high correlations between input variables (multicollinearity)")
        
        # Find high correlations (excluding self-correlations)
        high_corrs = []
        for i in range(len(input_vars)):
            for j in range(i+1, len(input_vars)):
                corr = input_corr.iloc[i, j]
                if abs(corr) > 0.7:  # Threshold for high correlation
                    high_corrs.append((input_vars[i], input_vars[j], corr))
        
        if high_corrs:
            st.warning("Potential multicollinearity detected between input variables!")
            
            for var1, var2, corr in high_corrs:
                st.write(f"- High correlation between **{var1}** and **{var2}**: {corr:.4f}")
                
            st.markdown("""
            **What is multicollinearity?**  
            Multicollinearity occurs when two or more input variables are highly correlated. This can cause issues in the regression model, making it difficult to determine the individual effect of each variable.
            
            **Potential solutions:**
            - Remove one of the correlated variables
            - Combine correlated variables into a single feature
            - Use regularization techniques in the model
            """)
        else:
            st.success("No significant multicollinearity detected among input variables.")
        
        # Correlation heatmap for inputs only
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(input_corr, dtype=bool))  # Mask for upper triangle
        sns.heatmap(input_corr, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
        ax.set_title("Input Variables Correlation Heatmap")
        st.pyplot(fig)