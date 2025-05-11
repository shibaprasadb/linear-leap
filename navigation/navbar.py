import streamlit as st
from config import NAV_OPTIONS, COLORS

def display_navigation_banner():
    """
    Display the navigation banner at the top of the application.
    """
    st.markdown(
        f"""
        <style>
        .nav-container {{
            display: flex;
            justify-content: space-around;
            background-color: {COLORS["background"]};
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .nav-item {{
            padding: 8px 15px;
            text-align: center;
            cursor: pointer;
            border-radius: 5px;
        }}
        .nav-item:hover {{
            background-color: rgba(66, 133, 244, 0.1);
        }}
        .nav-item.active {{
            background-color: {COLORS["primary"]};
            color: white;
        }}
        </style>
        """, 
        unsafe_allow_html=True
    )
    
    # Create navigation container
    cols = st.columns(len(NAV_OPTIONS))
    
    for i, option in enumerate(NAV_OPTIONS):
        with cols[i]:
            # Highlight current view
            is_active = st.session_state.view == option["view"]
            button_style = "primary" if is_active else "secondary"
            
            if st.button(
                f"{option['icon']} {option['name']}", 
                key=f"nav_{option['view']}",
                use_container_width=True,
                type=button_style,
            ):
                st.session_state.view = option["view"]
                st.rerun()  # FIXED: Changed from st.experimental_rerun()
    
    # Display current section title
    current_view = st.session_state.view
    current_section = next((opt["name"] for opt in NAV_OPTIONS if opt["view"] == current_view), "Unknown")
    
    # Display regression type badge
    if 'regression_type' in st.session_state and st.session_state.regression_type:
        reg_type = "Simple Linear Regression" if st.session_state.regression_type == "linear" else "Multiple Linear Regression"
        reg_color = COLORS["primary"] if st.session_state.regression_type == "linear" else COLORS["secondary"]
        
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                <h2>{current_section}</h2>
                <div style="background-color: {reg_color}; color: white; padding: 5px 10px; border-radius: 5px;">
                    {reg_type}
                </div>
            </div>
            <hr>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(f"## {current_section}")
        st.markdown("---")