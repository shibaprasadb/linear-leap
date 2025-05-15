import streamlit as st
import base64

def load_css():
    """
    Load custom CSS styles from the styles.css file.
    """
    try:
        with open("assets/styles.css", "r") as f:
            css = f.read()
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except Exception as e:
        # If file doesn't exist, fail silently
        pass

def show_footer():
    """
    Display a consistent footer at the bottom of the page with deep blue styling.
    """
    # Deep blue footer with enhanced styling
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #1E3A8A; padding: 15px; margin-top: 30px; 
                    font-weight: 500; font-family: sans-serif; letter-spacing: 0.5px;'>
            For feedback or suggestions: shibaprasad[dot]b[dot]mail[at]gmail.com
        </div>
        """,
        unsafe_allow_html=True
    )


# Function to set favicon with explicit HTML
def set_favicon():
    try:
        # Read the logo file
        with open("assets/LinearLeap_logo.png", "rb") as f:
            logo_data = f.read()
        
        # Encode the image to base64
        b64_logo = base64.b64encode(logo_data).decode()
        
        # Create the favicon HTML
        favicon_html = f"""
        <link rel="shortcut icon" href="data:image/png;base64,{b64_logo}">
        """
        
        # Inject the favicon HTML
        st.markdown(favicon_html, unsafe_allow_html=True)
    except Exception as e:
        print(f"Error setting favicon: {e}")
        # Fall back to text icon if image fails
        pass

# Add this function to ui_components.py, then call it in app.py before st.set_page_config