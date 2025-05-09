# LinearLeap

A streamlined Streamlit application for performing linear and multilinear regression analysis with intuitive visualizations and comprehensive insights.

## Overview

LinearLeap is an interactive web application built with Python that makes regression analysis accessible and intuitive. Upload your dataset, select your variables, and get comprehensive analysis results including exploratory data analysis, model training, visualization, and recommendations - all in a user-friendly interface.

## Features

- **Simple Linear Regression**: Analyze the relationship between one predictor and one outcome variable
- **Multiple Linear Regression**: Explore how multiple predictors influence an outcome variable
- **Interactive Data Exploration**: Visualize your data with interactive charts and statistics
- **Detailed Model Diagnostics**: Get comprehensive metrics and residual analysis
- **Predictions**: Use your trained model to make new predictions
- **Recommendations**: Receive tailored recommendations based on your analysis results

## Installation

```bash
# Clone the repository
git clone https://github.com/username/linear-leap.git
cd linear-leap

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Usage

1. **Upload Data**: Start by uploading a CSV file with your dataset
2. **Select Regression Type**: Choose between simple linear or multiple linear regression
3. **Variable Selection**: Select your target and input variables
4. **Explore Your Data**: View statistics and visualizations in the EDA section
5. **Train Model**: Set parameters and train your regression model
6. **View Results**: Analyze model performance with detailed metrics and visualizations
7. **Make Predictions**: Use your model to predict new outcomes
8. **Get Recommendations**: Receive insights and recommendations based on your analysis

## Project Structure

```
LinearLeap/
├── app.py                     # Main application file
├── config.py                  # Configuration settings
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
│   └── multilinear_regression.py  # Multilinear regression utilities
└── assets/
    └── styles.css             # Custom styling
```


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

If you have any questions or feedback, please reach out to shibaprasad.b.mail@gmail.com
