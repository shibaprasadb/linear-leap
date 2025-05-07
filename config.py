"""
Configuration file for the LinearLeap application.
"""

# Streamlit page configuration
PAGE_CONFIG = {
    "page_title": "LinearLeap",
    "page_icon": "📈",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# Application theme colors
COLORS = {
    "primary": "#4285F4",  # Blue
    "secondary": "#34A853",  # Green
    "accent": "#FBBC05",  # Yellow
    "error": "#EA4335",  # Red
    "background": "#F9F9F9",
    "text": "#202124",
}

# Navigation options
NAV_OPTIONS = [
    {"name": "Data Input", "icon": "📁", "view": "data_input"},
    {"name": "Exploratory Analysis", "icon": "🔍", "view": "eda"},
    {"name": "Model Training", "icon": "⚙️", "view": "model_training"},
    {"name": "Results", "icon": "📈", "view": "results"},
    {"name": "Recommendation", "icon": "🎯", "view": "recommendation"},
]