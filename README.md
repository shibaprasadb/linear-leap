# LinearLeap

A cutting-edge data science application that demystifies regression analysis through an intuitive Streamlit interface, interactive visualizations, and AI-powered interpretations.

## Overview

LinearLeap is an interactive web application built with Python that makes regression analysis accessible and intuitive. Upload your dataset, select your variables, and get comprehensive analysis results including exploratory data analysis, model training, visualization, and AI-powered recommendations - all in a user-friendly interface.

## Features

- **Simple Linear Regression**: Analyze the relationship between one predictor and one outcome variable
- **Multiple Linear Regression**: Explore how multiple predictors influence an outcome variable
- **Interactive Data Exploration**: Visualize your data with interactive charts and statistics
- **AI-Powered Insights**: Get AI-generated analysis of distributions using Google's Gemini API
- **Detailed Model Diagnostics**: Get comprehensive metrics and residual analysis
- **Predictions**: Use your trained model to make new predictions
- **Recommendations**: Receive tailored recommendations based on your analysis results

## Installation

```bash
# Clone the repository
git clone https://github.com/shibaprasadb/linear-leap.git
cd linear-leap

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure the API key
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Then edit .streamlit/secrets.toml to add your Gemini API key

# Run the application
streamlit run app.py
```

## Configuration

LinearLeap uses Google's Gemini API to provide AI-powered analysis of data distributions. You'll need to:

1. Create or use an existing Google API key with Gemini API access
2. Create a `.streamlit/secrets.toml` file (use the provided example file as a template)
3. Add your Gemini API key to the `secrets.toml` file

Example `.streamlit/secrets.toml` file:
```toml
# Gemini API key for AI-powered analysis
GEMINI_API_KEY = "your_api_key_here"
```

## Usage

1. **Upload Data**: Start by uploading a CSV file with your dataset
2. **Select Regression Type**: Choose between simple linear or multiple linear regression
3. **Variable Selection**: Select your target and input variables
4. **Explore Your Data**: View statistics and visualizations in the EDA section, including AI-powered distribution analysis
5. **Train Model**: Set parameters and train your regression model
6. **View Results**: Analyze model performance with detailed metrics and visualizations
7. **Make Predictions**: Use your model to predict new outcomes
8. **Get Recommendations**: Receive insights and recommendations based on your analysis

## Project Structure

```
LinearLeap/
├── app.py                     # Main application file
├── config.py                  # Configuration settings
├── .streamlit/                # Streamlit configuration
│   ├── secrets.toml           # Contains API keys (gitignored)
│   └── secrets.toml.example   # Example config without actual keys
├── navigation/
│   └── navbar.py              # Navigation components
├── views/
│   ├── landing.py             # Landing page
│   ├── data_input.py          # Data upload and selection
│   ├── eda.py                 # Exploratory data analysis
│   ├── model_training.py      # Model training interface
│   ├── results.py             # Results and diagnostics
│   └── recommendation.py      # Recommendations and insights
├── utils/
│   ├── linear_regression.py   # Linear regression utilities
│   ├── multilinear_regression.py  # Multilinear regression utilities
│   ├── plot_analysis.py       # Utilities for AI-powered plot analysis
│   └── ui_components.py       # File with the footer and other UI components
└── assets/
    ├── LinearLeap_logo.png    # Application logo
    └── styles.css             # Custom styling
```

## AI-Powered Analysis

LinearLeap leverages Google's Gemini API to provide intelligent analysis of data distributions. This feature automatically:

1. Analyzes the shape of distributions (normal, bimodal, etc.)
2. Identifies business implications of the observed distributions
3. Provides concise bullet-point summaries of findings

To use this feature, ensure your API key is correctly configured in the `.streamlit/secrets.toml` file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

If you have any questions or feedback, please reach out to shibaprasad.b.mail@gmail.com