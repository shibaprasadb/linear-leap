"""
Utility functions for simple linear regression analysis.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(data, input_var, target_var, test_size=0.2, random_state=42):
    """
    Preprocess data for linear regression.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    input_var : str
        Name of input variable
    target_var : str
        Name of target variable
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        Processed data (X_train, X_test, y_train, y_test)
    """
    # Extract X and y
    X = data[input_var].values.reshape(-1, 1)
    y = data[target_var].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

def train_linear_model(X_train, y_train):
    """
    Train a simple linear regression model.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training feature data
    y_train : numpy.ndarray
        Training target data
        
    Returns:
    --------
    sklearn.linear_model.LinearRegression
        Trained linear regression model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate the linear regression model.
    
    Parameters:
    -----------
    model : sklearn.linear_model.LinearRegression
        Trained model
    X_train : numpy.ndarray
        Training feature data
    X_test : numpy.ndarray
        Testing feature data
    y_train : numpy.ndarray
        Training target data
    y_test : numpy.ndarray
        Testing target data
        
    Returns:
    --------
    dict
        Dictionary containing model metrics
    """
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
    
    return train_metrics, test_metrics, y_train_pred, y_test_pred

def plot_regression_line(X_train, X_test, y_train, y_test, model, input_var, target_var):
    """
    Plot regression line with data points.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training feature data
    X_test : numpy.ndarray
        Testing feature data
    y_train : numpy.ndarray
        Training target data
    y_test : numpy.ndarray
        Testing target data
    model : sklearn.linear_model.LinearRegression
        Trained model
    input_var : str
        Name of input variable
    target_var : str
        Name of target variable
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object with the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot training data points
    ax.scatter(X_train, y_train, alpha=0.5, label='Training Data')
    
    # Plot test data points
    ax.scatter(X_test, y_test, alpha=0.5, color='red', label='Test Data')
    
    # Plot regression line
    x_range = np.linspace(
        min(X_train.min(), X_test.min()),
        max(X_train.max(), X_test.max()),
        100
    ).reshape(-1, 1)
    
    y_pred = model.predict(x_range)
    ax.plot(x_range, y_pred, color='green', label='Regression Line')
    
    # Add equation
    coef = model.coef_[0]
    intercept = model.intercept_
    equation = f"y = {intercept:.4f} {'+' if coef >= 0 else '-'} {abs(coef):.4f}x"
    ax.text(0.05, 0.95, equation, transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0.5))
    
    ax.set_title(f"Linear Regression: {input_var} vs {target_var}")
    ax.set_xlabel(input_var)
    ax.set_ylabel(target_var)
    ax.legend()
    
    return fig

def plot_residuals(y_true, y_pred, title="Residual Plot"):
    """
    Plot residuals against predicted values.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True target values
    y_pred : numpy.ndarray
        Predicted target values
    title : str, default="Residual Plot"
        Plot title
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object with the plot
    """
    residuals = y_true - y_pred
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_title(title)
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')
    
    return fig

def plot_residual_histogram(residuals, title="Residual Distribution"):
    """
    Plot histogram of residuals.
    
    Parameters:
    -----------
    residuals : numpy.ndarray
        Residual values
    title : str, default="Residual Distribution"
        Plot title
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object with the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(residuals, kde=True, ax=ax)
    ax.axvline(x=0, color='r', linestyle='--')
    ax.set_title(title)
    ax.set_xlabel('Residual Value')
    
    return fig

def plot_qq(residuals, title="Q-Q Plot"):
    """
    Create a Q-Q plot for residuals.
    
    Parameters:
    -----------
    residuals : numpy.ndarray
        Residual values
    title : str, default="Q-Q Plot"
        Plot title
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object with the plot
    """
    from scipy import stats
    
    fig, ax = plt.subplots(figsize=(10, 6))
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title(title)
    
    return fig

def make_prediction(model, input_value):
    """
    Make a prediction using the trained model.
    
    Parameters:
    -----------
    model : sklearn.linear_model.LinearRegression
        Trained model
    input_value : float
        Input value for prediction
        
    Returns:
    --------
    float
        Predicted value
    """
    # Reshape input for prediction
    X_input = np.array([input_value]).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(X_input)[0]
    
    return prediction