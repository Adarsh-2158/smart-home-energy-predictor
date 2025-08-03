import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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

# Sidebar Navigation
st.sidebar.title("📂 Navigation")
selected_page = st.sidebar.radio("Go to", ["Home", "About","Team Members","Predict"])




# Main content based on selection
if selected_page == "Home":
    st.title("🏡 Smart Home Energy Forecasting")

    st.markdown("""
    Welcome to the **Smart Home Energy Consumption Forecasting System**.

    This application utilizes historical IoT and weather data to predict hourly and daily energy consumption in a smart home setup.

    **Why this matters:**
    - 🧠 Machine learning-powered predictions
    - 🌍 Support for sustainable energy usage
    - 🏠 Smart home grid optimization

    Use the sidebar to explore different sections of this app.
    """)


elif selected_page == "About":
    st.title("📘 About This Project")

    st.markdown("""
    <style>
    h1 {
        color: #ffffff;
        text-align: center;
    }
    .section-title {
        font-size: 22px;
        font-weight: bold;
        color: #facc15;
        text-align: left;
        margin-top: 25px;
    }
    .section-text, ul {
        font-size: 16px;
        color: #ffffff;
        text-align: left;
    }
    </style>

    <p class="section-title">🔍 Objective:</p>
    <p class="section-text">
    To forecast smart home energy usage based on weather, time, and occupancy features.
    </p>

    <p class="section-title">📦 Dataset:</p>
    <ul>
        <li>Simulated smart home energy dataset</li>
        <li>Includes features: temperature, household size, appliance count, etc.</li>
    </ul>

    <p class="section-title">⚙️ Tech Stack:</p>
    <ul>
        <li>Python, Pandas, Scikit-learn, Streamlit</li>
        <li>Random Forest Regressor</li>
        <li>Matplotlib / Seaborn</li>
    </ul>

    <p class="section-title">💡 Real-world Impact:</p>
    <ul>
        <li>Energy cost savings</li>
        <li>Optimized grid load</li>
        <li>Improved eco-conscious living</li>
    </ul>
    """, unsafe_allow_html=True)

elif selected_page == "Team Members":
    st.title("👥 Meet the Team")

    team_members = [
        {
            "name": "Adarsh Srivastava",
            "role": "Model Building & Model Training",
            "github": "https://github.com/Adarsh-2158",
            "linkedin": "https://www.linkedin.com/in/adarsh-kumar-srivastava/",
            "email": "adarshkumarsrivastava56@gmail.com",
        },
        {
            "name": "Ayush Patel",
            "role": "Data Preprocessing & Frontend Developer",
            "github": "https://github.com/ayush42patel",
            "linkedin": "https://www.linkedin.com/in/ayush-42-patel/",
            "email": "ayush42patel@gmail.com"
        },
        {
            "name": "Aditya Nigam",
            "role": "Visualization & Presentation",
            "github": "https://github.com/adityanigam28",
            "linkedin": "https://www.linkedin.com/in/aditya-nigam-838a8527a/",
            "email": "adityanigam508@gmail.com"
        },
        {
            "name": "Parveen Chaudhary",
            "role": "Documentation",
            "github": "https://github.com/Parveen-8289",
            "linkedin": "https://www.linkedin.com/in/praveen-chaudhary-1a6a42314/",
            "email": "praveenchaudhary067@gmail.com"
        }
    ]

    st.markdown("""
        <style>
            .card {
                background-color: #f0f2f6;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 15px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .name {
                font-size: 20px;
                font-weight: bold;
                color: #2c3e50;
            }
            .role {
                color: #7f8c8d;
                margin-bottom: 10px;
            }
            a {
                text-decoration: none;
            }
        </style>
    """, unsafe_allow_html=True)

    for member in team_members:
        st.markdown(f"""
        <div class="card">
            <div class="name">{member['name']}</div>
            <div class="role">{member['role']}</div>
            🔗 <a href="{member['github']}" target="_blank">GitHub</a> | 
            💼 <a href="{member['linkedin']}" target="_blank">LinkedIn</a> | 
            📧 <a href="mailto:{member['email']}">{member['email']}</a>
        </div>
        """, unsafe_allow_html=True)

elif selected_page == "Predict":
    model = joblib.load("Smart_Home_RF_best.pkl")  # Make sure this path is correct

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
