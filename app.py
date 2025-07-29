import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# --- Custom Page Styling ---
page_style = """
<style>
/* Page background and font */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    # background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    # background: linear-gradient(135deg, #ff512f, #dd2476, #1a2a6c);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

/* Hide the default Streamlit footer and header */
footer, header {visibility: hidden;}

/* Main title */
h1 {
    color: #ffffff;
    font-size: 3em;
    text-align: center;
    margin-top: 20px;
}

/* Subtitles and section titles */
h2, h3, h4, h5 {
    color: #ffdd57;
    text-align: center;
    font-weight: 600;
    margin-bottom: 0.5em;
}

/* Center and style prediction subtitle */
h5 {
    font-size: 1.2em;
    font-style: italic;
    color: #e0e0e0;
}

/* Input section container */
div[data-testid="column"] {
    padding: 20px;
    background-color: rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    margin-bottom: 15px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
}

/* Labels and form text */
label, .stSelectbox label, .stSlider label {
    font-weight: bold;
    font-size: 1.05em;
    color: #ffffff !important;
    margin-bottom: 5px;
}

/* Sliders */
.stSlider > div {
    color: white !important;
}

/* Buttons */
.stButton > button {
    background-color: #ff914d;
    color: white;
    font-size: 1em;
    font-weight: bold;
    border-radius: 10px;
    padding: 0.6em 1.6em;
    margin-top: 20px;
    transition: 0.3s ease;
    box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    border: none;
}

.stButton > button:hover {
    background-color: #39ae47;
    transform: scale(1.05);
}

/* Success box styling */
.stSuccess {
    background-color: #22bb33 !important;
    color: white !important;
    font-weight: bold;
    border-radius: 8px;
    padding: 15px;
    text-align: center;
    margin-top: 20px;
    font-size: 1.1em;
}

/* Select boxes and input fields */
select, input {
    background-color: #1f4068;
    color: white;
    border-radius: 6px;
}

/* Gauge/plot color */
.plot-container .svg-container {
    color: white;
}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)


# --- Load Random Forest model ---
model = joblib.load("Smart_Home_RF_model.pkl")  # Make sure this path is correct

# --- Page Title ---
st.title("🏠 Smart Home Energy Consumption Forecasting")
st.markdown("""
<div style="text-align: center;">
    <h5><em>Predict hourly or daily energy usage using IoT & weather insights.</em></h5>
</div>
""", unsafe_allow_html=True)

# --- Inputs Section ---
st.header("📋 Input Details")

col1, col2 = st.columns(2)

with col1:
    appliance = st.selectbox("🔌 Appliance Type", [
        "Fridge", "Oven", "Dishwasher", "Heater", 
        "Microwave", "Air Conditioning", 
        "Computer", "TV", "Washing Machine", "Lights"
    ])
    temperature = st.number_input("🌡 Outdoor Temperature (°C)", min_value=-10.0, max_value=50.0, step=0.1)
    household_size = st.slider("👨‍👩‍👧‍👦 Household Size", 1, 10, 4)

with col2:
    season = st.selectbox("🌤️ Season", ["Spring", "Summer", "Fall", "Winter"])
    hour = st.slider("⏰ Hour of the Day", 0, 23, 12)
    month = st.slider("🗓 Month", 1, 12, 6)

# --- Prepare input data ---
input_data = pd.DataFrame({
    "Appliance Type": [appliance],
    "Season": [season],
    "Outdoor Temperature (°C)": [temperature],
    "Household Size": [household_size],
    "Hour": [hour],
    "Month": [month]
})

# --- Prediction Section ---
if st.button("🔍 Predict Energy Consumption"):
    prediction = model.predict(input_data)[0]
    st.success(f"🔋 Predicted Energy Consumption: {prediction:.2f} kWh")

    # --- Gauge Chart ---
    st.subheader("🔧 Energy Prediction Gauge")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        title={'text': "Predicted Energy (kWh)", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 10]},
            'bar': {'color': "#ff914d"},
            'bgcolor': "white",
            'borderwidth': 2,
            'steps': [
                {'range': [0, 3], 'color': "#adebad"},
                {'range': [3, 7], 'color': "#ffff99"},
                {'range': [7, 10], 'color': "#ff9999"},
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    # --- 24-Hour Simulation ---
    st.subheader("⏱ Hourly Energy Usage Pattern")
    hours = list(range(24))
    hourly_data = pd.DataFrame({
        "Appliance Type": [appliance] * 24,
        "Season": [season] * 24,
        "Outdoor Temperature (°C)": [temperature] * 24,
        "Household Size": [household_size] * 24,
        "Hour": hours,
        "Month": [month] * 24
    })
    hourly_predictions = model.predict(hourly_data)

    fig2, ax = plt.subplots()
    ax.plot(hours, hourly_predictions, marker='o', color='#ffdd57', linewidth=2)
    ax.set_facecolor('#2a5298')
    ax.set_title("Predicted Energy Usage Across 24 Hours", fontsize=14, color='white')
    ax.set_xlabel("Hour", color='white')
    ax.set_ylabel("Energy (kWh)", color='white')
    ax.tick_params(axis='both', colors='white')
    fig2.patch.set_facecolor('#2a5298')
    st.pyplot(fig2)
