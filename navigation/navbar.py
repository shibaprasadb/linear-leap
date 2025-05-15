# import streamlit as st
# from config import NAV_OPTIONS, COLORS
# import os
# import logging

# def display_navigation_banner():
#     """
#     Display the navigation banner at the top of the application.
#     """
#     # Display logo with more direct HTML injection
#     try:
#         logo_path = "assets/LinearLeap_logo.png"
#         if os.path.exists(logo_path):
#             # Using markdown with HTML for more reliable rendering
#             st.markdown(
#                 f"""
#                 <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
#                     <img src="data:image/png;base64,{get_image_as_base64(logo_path)}" width="80px">
#                 </div>
#                 """, 
#                 unsafe_allow_html=True
#             )
#     except Exception as e:
#         logging.error(f"Error displaying logo: {e}")
    
#     st.markdown(
#         f"""
#         <style>
#         .nav-container {{
#             display: flex;
#             justify-content: space-around;
#             background-color: {COLORS["background"]};
#             padding: 10px;
#             border-radius: 5px;
#             margin-bottom: 20px;
#         }}
#         .nav-item {{
#             padding: 8px 15px;
#             text-align: center;
#             cursor: pointer;
#             border-radius: 5px;
#         }}
#         .nav-item:hover {{
#             background-color: rgba(66, 133, 244, 0.1);
#         }}
#         .nav-item.active {{
#             background-color: {COLORS["primary"]};
#             color: white;
#         }}
#         </style>
#         """, 
#         unsafe_allow_html=True
#     )
    
#     # Create navigation container
#     cols = st.columns(len(NAV_OPTIONS))
    
#     for i, option in enumerate(NAV_OPTIONS):
#         with cols[i]:
#             # Highlight current view
#             is_active = st.session_state.view == option["view"]
#             button_style = "primary" if is_active else "secondary"
            
#             if st.button(
#                 f"{option['icon']} {option['name']}", 
#                 key=f"nav_{option['view']}",
#                 use_container_width=True,
#                 type=button_style,
#             ):
#                 st.session_state.view = option["view"]
#                 st.rerun()
    
#     # Display current section title
#     current_view = st.session_state.view
#     current_section = next((opt["name"] for opt in NAV_OPTIONS if opt["view"] == current_view), "Unknown")
    
#     # Display regression type badge
#     if 'regression_type' in st.session_state and st.session_state.regression_type:
#         reg_type = "Simple Linear Regression" if st.session_state.regression_type == "linear" else "Multiple Linear Regression"
#         reg_color = COLORS["primary"] if st.session_state.regression_type == "linear" else COLORS["secondary"]
        
#         st.markdown(
#             f"""
#             <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
#                 <h2>{current_section}</h2>
#                 <div style="background-color: {reg_color}; color: white; padding: 5px 10px; border-radius: 5px;">
#                     {reg_type}
#                 </div>
#             </div>
#             <hr>
#             """,
#             unsafe_allow_html=True
#         )
#     else:
#         st.markdown(f"## {current_section}")
#         st.markdown("---")

# def get_image_as_base64(image_path):
#     """
#     Convert an image to base64 encoding.
#     """
#     import base64
#     with open(image_path, "rb") as img_file:
#         return base64.b64encode(img_file.read()).decode()



import streamlit as st
from config import NAV_OPTIONS, COLORS
import os
import base64

def get_image_as_base64(image_path):
    """
    Convert an image to base64 encoding.
    """
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        print(f"Error encoding image: {e}")
        return ""

def display_navigation_banner():
    """
    Display the navigation banner at the top of the application.
    """
    # Add logo in a row with home button
    logo_col1, logo_col2, logo_col3 = st.columns([1, 8, 1])
    
    # Home button in the first column
    with logo_col1:
        if st.button("🏠 Home", key="home_button", use_container_width=True):
            st.session_state.page = 'landing'
            st.rerun()
    
    # Logo in the last column
    with logo_col3:
        try:
            logo_path = "assets/LinearLeap_logo.png"
            if os.path.exists(logo_path):
                logo_base64 = get_image_as_base64(logo_path)
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
                        <img src="data:image/png;base64,{logo_base64}" width="80px">
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        except Exception as e:
            print(f"Error displaying logo: {e}")
    
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
                st.rerun()
    
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