import joblib
import json
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load provinces
with open("provinces.json", "r", encoding="utf-8") as f:
    PROVINCES = json.load(f)

# Load choices
with open("choices.json", "r", encoding="utf-8") as f:
    CHOICES = json.load(f)

# Các mô hình có sẵn
MODEL_PATHS = {
    "Linear Regression": "models/linear_regression.joblib",
    "Ridge Regression": "models/ridge_regression.joblib",
    "Random Forest": "models/random_forest.joblib",
    "Gradient Boosting": "models/gradient_boosting.joblib",
    "XGBoost": "models/xgboost.joblib"
}

def load_model(model_name):
    return joblib.load(MODEL_PATHS[model_name])

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "provinces": PROVINCES,
        "house_directions": CHOICES["house_directions"],
        "balcony_directions": CHOICES["balcony_directions"],
        "legal_statuses": CHOICES["legal_statuses"],
        "furniture_states": CHOICES["furniture_states"],
        "models": list(MODEL_PATHS.keys()),
        "form_data": {},  # form trống lúc đầu
        "prediction": None
    })

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    province: str = Form(...),
    area: float = Form(...),
    frontage: float = Form(...),
    access_road: float = Form(...),
    floors: int = Form(...),
    bedrooms: int = Form(...),
    bathrooms: int = Form(...),
    house_direction: str = Form(...),
    balcony_direction: str = Form(...),
    legal_status: str = Form(...),
    furniture_state: str = Form(...),
    model_choice: str = Form(...)
):
    model = load_model(model_choice)
    input_data = {
        "Province": [province],
        "Area": [area],
        "Frontage": [frontage],
        "Access Road": [access_road],
        "Floors": [floors],
        "Bedrooms": [bedrooms],
        "Bathrooms": [bathrooms],
        "House direction": [house_direction],
        "Balcony direction": [balcony_direction],
        "Legal status": [legal_status],
        "Furniture state": [furniture_state],
    }

    df_input = pd.DataFrame(input_data)
    prediction = model.predict(df_input)[0]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "provinces": PROVINCES,
        "house_directions": CHOICES["house_directions"],
        "balcony_directions": CHOICES["balcony_directions"],
        "legal_statuses": CHOICES["legal_statuses"],
        "furniture_states": CHOICES["furniture_states"],
        "models": list(MODEL_PATHS.keys()),
        "prediction": f"{prediction:.2f} tỷ VNĐ",
        "form_data": {
            "province": province,
            "area": area,
            "frontage": frontage,
            "access_road": access_road,
            "floors": floors,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "house_direction": house_direction,
            "balcony_direction": balcony_direction,
            "legal_status": legal_status,
            "furniture_state": furniture_state,
            "model_choice": model_choice
        }
    })
