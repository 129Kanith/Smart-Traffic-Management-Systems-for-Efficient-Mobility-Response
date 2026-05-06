import streamlit as st
import os
from modules.traffic_processor import process_video

# Page configuration
st.set_page_config(
    page_title="Smart Traffic Management System",
    layout="wide"
)

# Custom CSS for Dark Mode & Smart City Theme
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0a0a0f;
        color: #ffffff;
    }
    h1, h2, h3 {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] {
        background-color: #111116;
        border-right: 1px solid #1f1f2e;
    }
    .stButton>button {
        background-color: #00f2fe;
        color: #000;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #4facfe;
        box-shadow: 0 0 20px rgba(79, 172, 254, 0.6);
        transform: translateY(-2px);
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
    }
    .css-1v0mbdj > img {
        border-radius: 12px;
        box-shadow: 0 0 20px rgba(0,0,0,0.5);
        border: 1px solid #1f1f2e;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main title
st.title("🌆 Smart City Traffic Control Center")

st.markdown(
    """
    ### AI-Based Intelligent Smart City Traffic System
    Computer Vision | Adaptive Signal Control | Predictive Analytics
    """
)

# Sidebar
st.sidebar.title("⚙️ Control Panel")

# Video selection
VIDEO_SOURCES = {
    "Traffic Cam 1": r"K:\TMSYoLo\videos\Video 1.mp4",
    "Traffic Cam 2": r"K:\TMSYoLo\videos\Video 2.mp4",
    "Traffic Cam 3": r"K:\TMSYoLo\videos\Video 3.mp4",
}

selected_video_name = st.sidebar.selectbox(
    "Select Traffic Video",
    options=list(VIDEO_SOURCES.keys())
)

video_option = VIDEO_SOURCES[selected_video_name]

enable_heatmap = st.sidebar.checkbox("Enable Heatmap", value=False)

if os.path.exists("outputs/traffic_report.csv"):
    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 Analytics Reports")
    with open("outputs/traffic_report.csv", "rb") as file:
        st.sidebar.download_button(
            label="📥 Download CSV Report",
            data=file,
            file_name="smart_city_traffic_report.csv",
            mime="text/csv"
        )

# Start button
start = st.sidebar.button(
    "▶ Start Traffic Analysis"
)

# Dashboard layout
left_col, right_col = st.columns([2.8, 1.2])

# Video section
with left_col:

    st.subheader("📹 Live Traffic Monitoring")

    frame_placeholder = st.empty()
    
    st.markdown("---")
    st.subheader("📈 Live Traffic Trends")
    graphs_placeholder = st.empty()

# Analytics section
with right_col:

    st.subheader("📊 Live Metrics")

    analytics_placeholder = st.empty()

# Start processing
if start:

    process_video(
        video_option,
        frame_placeholder,
        analytics_placeholder,
        graphs_placeholder,
        enable_heatmap
    )