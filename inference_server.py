from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from typing import Dict, List
from pydantic import BaseModel
import asyncio
from model_inference import ModelInference
from loguru import logger
import os

app = FastAPI(title="DeFi Momentum Inference Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_inference = ModelInference()

class PredictionRequest(BaseModel):
    features: List[List[float]]
    token_address: str = ""
    network: str = ""

class PredictionResponse(BaseModel):
    momentum_score: float
    confidence: float
    predicted_return: float
    predicted_volatility: float
    timestamp: str

@app.on_event("startup")
async def startup_event():
    await model_inference.load_model()
    logger.info("Inference server started")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        features = np.array(request.features)
        
        if features.shape[-1] != 12:
            raise HTTPException(status_code=400, detail="Features must have 12 dimensions")
        
        prediction = model_inference.predict(features)
        
        return PredictionResponse(**prediction)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict")
async def batch_predict(requests: List[PredictionRequest]):
    try:
        results = []
        
        for request in requests:
            features = np.array(request.features)
            prediction = model_inference.predict(features)
            results.append(prediction)
        
        return {"predictions": results}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model_info")
async def get_model_info():
    try:
        return model_inference.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model_inference.interpreter is not None}

if __name__ == "__main__":
    uvicorn.run(
        "inference_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        access_log=False
    )
