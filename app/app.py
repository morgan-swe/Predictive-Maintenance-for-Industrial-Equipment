import streamlit as st

# Page config
# This controls: Browser tab title, Browser icon, Page width
# Wide gives us more room for charts
st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

# Application title
st.title("Predictive Maintenance System")
st.write(
    "Remaining Useful Life prediction for industrial equipment "
    "using machine learning."
)

st.divider()

# Sidebar
st.sidebar.title("System Information")
st.sidebar.write(
    "Model: Random Forest Regressor"
)
st.sidebar.write(
    "Dataset: NASA CMAPSS FD001"
)

# Model Performance section
st.header("Model Performance")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("MAE", "10.23 cycles")

with col2:
    st.metric("RMSE", "15.23 cycles")

with col3:
    st.metric("R²", "0.867")

# Engine selection section
st.header("Engine Analysis")
engines = list(range(1, 101))

selected_engine = st.selectbox(
    "Select an engine",
    engines
)

# Displaying the section
st.write(f"Selected Engine: {selected_engine}")