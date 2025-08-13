# 🔋 Smart Home Energy Consumption Forecasting

A machine learning project that predicts energy consumption in smart homes using Random Forest Regression. The model is trained on a large dataset with features such as outdoor temperature, household size, and appliance usage.

---

## 📂 Project Structure

```bash
📁 Energy-consumption/
│
├── app.py                                # Streamlit app interface
├── Smart_Home.ipynb                      # Jupyter notebook (data prep + model training)
├── smart_home_energy_consumption_large.csv  # Dataset
├── rf_best_model.pkl                     # Final trained Random Forest model
├── Smart_Home_LinRegr_model.pkl         # (ignored) Linear Regression model
├── Smart_Home_RF_model.pkl              # (ignored) Earlier RF model version
├── .gitignore                            # To exclude specific files from version control
```


---

## ⚙️ Features

- Preprocessing pipeline with categorical and numerical handling
- Random Forest Regressor with hyperparameter tuning (`RandomizedSearchCV`)
- Evaluation using R² Score, RMSE, and MAE
- Streamlit web interface for making predictions

---

## 🧪 Model Performance (Tuned Random Forest)

- **R² Score:** 0.7567
- **RMSE:** 0.3421
- **MAE:** 0.4766

---

## 🔗 Online Link
https://smartenergyconsumptionforecast-asapanpc.streamlit.app/
---

## 🚀 How to Run

1. Clone the repository:
   
   -git clone https://github.com/Adarsh-2158/smart-home-energy-predictor.git
   -cd smart-home-energy-predictor
   
3. Install dependencies:
   
   -pip install -r requirements.txt
   
3.Launch the Streamlit app:

-streamlit run app.py
