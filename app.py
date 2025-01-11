import streamlit as st
import joblib
import numpy as np

# Use st.cache_resource to cache the model loading process
@st.cache_resource
def load_model():
    return joblib.load("model/manu_def.pkl")

model = load_model()
labels = ['No', 'Yes']

st.title("Manufacturing Defects - Prediction App")
st.markdown("This app predicts whether a manufacturing defect will occur based on various production parameters.")

# Create a function to collect and organize input data
def get_input():
    st.header("Production Parameters")

    ProductionVolume = st.slider("Number of units produced per day:", 100, 1000, 100, step=1)
    ProductionCost = st.slider("Cost incurred for production per day ($):", 5000.0, 20000.0, 5000.0, step=0.01)
    SupplierQuality = st.slider("Supplier quality rating (%):", 80.0, 100.0, 80.0, step=0.01)
    DeliveryDelay = st.slider("Average delivery delay (days):", 0, 5, 0, step=1)

    st.header("Quality and Efficiency Parameters")

    DefectRate = st.slider("Defects per thousand units:", 0.5, 5.0, 0.5, step=0.01)
    QualityScore = st.slider("Overall quality assessment (%):", 60.0, 100.0, 60.0, step=0.01)
    MaintenanceHours = st.slider("Maintenance hours per week:", 0, 24, 0, step=1)
    DowntimePercentage = st.slider("Production downtime (%):", 0.0, 5.0, 0.0, step=0.01)

    st.header("Inventory and Workforce Parameters")

    InventoryTurnover = st.slider("Inventory turnover ratio:", 2.0, 10.0, 2.0, step=0.01)
    StockoutRate = st.slider("Stockout rate (%):", 0.0, 10.0, 0.0, step=0.01)
    WorkerProductivity = st.slider("Workforce productivity (%):", 80.0, 100.0, 80.0, step=0.01)
    SafetyIncidents = st.slider("Safety incidents per month:", 0, 10, 0, step=1)

    st.header("Energy and Additive Manufacturing Parameters")

    EnergyConsumption = st.slider("Energy consumption (kWh):", 1000.0, 5000.0, 1000.0, step=0.01)
    EnergyEfficiency = st.slider("Energy efficiency factor:", 0.1, 0.5, 0.1, step=0.01)
    AdditiveProcessTime = st.slider("Additive manufacturing time (hours):", 1.0, 10.0, 1.0, step=0.01)
    AdditiveMaterialCost = st.slider("Additive material cost per unit ($):", 100.0, 500.0, 100.0, step=0.01)

    return [ProductionVolume, ProductionCost, SupplierQuality, DeliveryDelay, DefectRate, QualityScore, MaintenanceHours,
            DowntimePercentage, InventoryTurnover, StockoutRate, WorkerProductivity, SafetyIncidents, EnergyConsumption,
            EnergyEfficiency, AdditiveProcessTime, AdditiveMaterialCost]

# Collect input data
input_data = np.array(get_input()).reshape(1, -1)

proba = model.predict_proba(input_data)
st.write("Probabilities:", proba)

# Prediction and Probability display
if st.button("Predict"):
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)
    st.write(f"Prediction: **{labels[prediction[0]]}**")
    st.write(f"Probability of defect: **{proba[0][1]:.2f}**")

