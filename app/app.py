import streamlit as st

from predict import (
    MODELS,
    load_features,
    load_metrics,
    held_out_engines,
    load_model,
    predict_for_engine,
)

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
selected_model_name = st.sidebar.selectbox(
    "Model",
    list(MODELS.keys())
)
st.sidebar.write(
    "Dataset: NASA CMAPSS FD001"
)

# Loading the model and feature table is slow, so cache both.
# Streamlit reruns this whole script on every interaction.
@st.cache_resource
def get_model(name):
    return load_model(name)

@st.cache_data
def get_features():
    return load_features()

model = get_model(selected_model_name)
data = get_features()

# Model Performance section
st.header("Model Performance")
col1, col2, col3 = st.columns(3)

# Metrics for whichever model is selected in the sidebar
metrics = load_metrics(selected_model_name)

with col1:
    st.metric("MAE", f"{metrics['MAE']:.2f} cycles")

with col2:
    st.metric("RMSE", f"{metrics['RMSE']:.2f} cycles")

with col3:
    st.metric("R²", f"{metrics['R2']:.3f}")

# Engine selection section
st.header("Engine Analysis")
st.write(
    "These 20 engines were held out from training. Each one ran until it "
    "failed, so move the slider to replay its life and compare the "
    "prediction with the cycles it actually had left."
)
engines = held_out_engines(data)

selected_engine = st.selectbox(
    "Select an engine",
    engines
)

engine_rows = data[data["unit_number"] == selected_engine]
first_cycle = int(engine_rows["time_in_cycles"].min())
last_cycle = int(engine_rows["time_in_cycles"].max())
cycle = st.slider(
    "Current cycle",
    first_cycle,
    last_cycle,
    value=(first_cycle + last_cycle) // 2,
)

predicted_rul = predict_for_engine(model, data, selected_engine, cycle)
actual_rul = int(engine_rows.loc[engine_rows["time_in_cycles"] == cycle, "RUL"].iloc[0])

col1, col2 = st.columns(2)
with col1:
    st.metric("Predicted RUL", f"{predicted_rul:.1f} cycles")
with col2:
    st.metric("Actual RUL", f"{actual_rul} cycles")
st.caption(
    "The model was trained with RUL capped at 125 cycles, so early in an "
    "engine's life it predicts about 125 even when more life remains."
)