import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import pandas as pd
from src.predictor import AfficionadoDemandPredictor

app = FastAPI(
    title="Afficionado Coffee Roasters ML Demand API",
    description="REST API service for real-time demand forecasting, peak rush alerts, and staffing optimization.",
    version="1.0.0"
)

predictor = None

@app.on_event("startup")
def startup_event():
    global predictor
    predictor = AfficionadoDemandPredictor()

class HourlyPredictionRequest(BaseModel):
    store_location: str = Field(..., example="Hell's Kitchen", description="Store location: Hell's Kitchen, Lower Manhattan, Astoria")
    datetime_str: str = Field(..., example="2025-07-01 09:00", description="Datetime in YYYY-MM-DD HH:MM format")

class HourlyPredictionResponse(BaseModel):
    store_location: str
    datetime: str
    hour: int
    predicted_quantity: float
    predicted_revenue: float
    peak_rush_probability: float
    is_peak_demand: bool
    demand_tier: str
    recommended_baristas: int
    recommended_cashiers: int
    total_recommended_staff: int
    operational_guidance: str

class ScheduleRequest(BaseModel):
    store_location: str = Field(..., example="Lower Manhattan")
    target_date: str = Field(..., example="2025-07-01")

class CategoryBreakdownRequest(BaseModel):
    store_location: str = Field(..., example="Astoria")
    datetime_str: str = Field(..., example="2025-07-01 08:00")

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "Afficionado Demand ML API", "version": "1.0.0"}

@app.get("/stores")
def get_stores():
    return {"stores": ["Hell's Kitchen", "Lower Manhattan", "Astoria"]}

@app.post("/predict/hour", response_model=HourlyPredictionResponse)
def predict_hourly_demand(req: HourlyPredictionRequest):
    try:
        res = predictor.predict_hour(req.store_location, req.datetime_str)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/schedule")
def predict_day_schedule(req: ScheduleRequest):
    try:
        schedule_df = predictor.predict_day_schedule(req.store_location, req.target_date)
        return {"schedule": schedule_df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/category")
def predict_category_breakdown(req: CategoryBreakdownRequest):
    try:
        df_breakdown = predictor.predict_category_breakdown(req.store_location, req.datetime_str)
        return {"breakdown": df_breakdown.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
