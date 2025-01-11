from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load the model (same as using @st.cache_resource)
def load_model():
    return joblib.load("model/manu_def.pkl")

model = load_model()
labels = ['No', 'Yes']

# Pydantic model for input validation
class ProductionParameters(BaseModel):
    ProductionVolume: int
    ProductionCost: float
    SupplierQuality: float
    DeliveryDelay: int
    DefectRate: float
    QualityScore: float
    MaintenanceHours: int
    DowntimePercentage: float
    InventoryTurnover: float
    StockoutRate: float
    WorkerProductivity: float
    SafetyIncidents: int
    EnergyConsumption: float
    EnergyEfficiency: float
    AdditiveProcessTime: float
    AdditiveMaterialCost: float

@app.get("/ping")
def ping():
    return {"message": "Working!"}

# Root endpoint
@app.get("/")
def root():
    return {"message": "Welcome to the Manufacturing Defects Prediction API"}

# Prediction endpoint
@app.post("/predict")
def predict(parameters: ProductionParameters):
    # Prepare the input data in the required format
    input_data = np.array([[
        parameters.ProductionVolume, parameters.ProductionCost, parameters.SupplierQuality, parameters.DeliveryDelay,
        parameters.DefectRate, parameters.QualityScore, parameters.MaintenanceHours, parameters.DowntimePercentage,
        parameters.InventoryTurnover, parameters.StockoutRate, parameters.WorkerProductivity, parameters.SafetyIncidents,
        parameters.EnergyConsumption, parameters.EnergyEfficiency, parameters.AdditiveProcessTime, parameters.AdditiveMaterialCost
    ]])

    # Make predictions
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)

    return {
        "Prediction": labels[prediction[0]],
        "Probability of Defect": proba[0][1]
    }

# To run the server: uvicorn <filename>:app --reload
