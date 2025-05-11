import streamlit as st

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