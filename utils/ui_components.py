import streamlit as st
import base64
import os

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

def set_page_favicon():
    """
    Set the page favicon using a more direct approach with JavaScript injection.
    This works even when the normal page_icon approach fails.
    """
    try:
        # Path to your logo
        logo_path = "assets/LinearLeap_logo.png"
        
        # Check if file exists
        if not os.path.exists(logo_path):
            print(f"Favicon file not found: {logo_path}")
            return
            
        # Read the image file and convert to base64
        with open(logo_path, "rb") as f:
            img_data = f.read()
            b64_encoded = base64.b64encode(img_data).decode()
        
        # Create JavaScript to set the favicon dynamically
        favicon_js = f"""
        <script>
            // Remove any existing favicons
            var links = document.getElementsByTagName('link');
            for (var i = 0; i < links.length; i++) {{
                if (links[i].rel.includes('icon')) {{
                    links[i].parentNode.removeChild(links[i]);
                }}
            }}
            
            // Create new favicon link
            var link = document.createElement('link');
            link.rel = 'icon';
            link.type = 'image/png';
            link.href = 'data:image/png;base64,{b64_encoded}';
            document.getElementsByTagName('head')[0].appendChild(link);
        </script>
        """
        
        # Inject the JavaScript
        st.markdown(favicon_js, unsafe_allow_html=True)
    except Exception as e:
        # If error occurs, log it but continue execution
        print(f"Error setting favicon: {e}")

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
            For feedback or suggestions: shibaprasad.b.mail@gmail.com
        </div>
        """,
        unsafe_allow_html=True
    )