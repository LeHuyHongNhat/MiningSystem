from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
from data_preprocessing import DataPreprocessor
from model_trainer import ModelTrainer
import json
import os
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import shap
import joblib

app = FastAPI(title="Data Mining System API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
data_processor = None
model_trainer = ModelTrainer()

class DataUploadRequest(BaseModel):
    file_path: str

class TrainingRequest(BaseModel):
    target_column: str
    test_size: float = 0.2
    algorithm_type: str  # 'classification', 'regression', or 'clustering'
    selected_models: List[str]

class ModelOptimizationRequest(BaseModel):
    model_name: str
    algorithm_type: str

class ModelManagementRequest(BaseModel):
    model_name: str
    file_path: str

@app.post("/api/upload")
async def upload_data(request: DataUploadRequest):
    """Handle data upload and initial processing"""
    global data_processor
    
    try:
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=400, detail="Invalid file path")
            
        data_processor = DataPreprocessor(request.file_path)
        
        # Load and process data
        if not data_processor.load_data():
            raise HTTPException(status_code=400, detail="Failed to load data")
            
        if not data_processor.clean_data():
            raise HTTPException(status_code=400, detail="Failed to clean data")
            
        # Get data summary and feature analysis
        summary = data_processor.get_data_summary()
        feature_importance = data_processor.analyze_features()
        
        return {
            "message": "Data uploaded and processed successfully",
            "summary": summary,
            "feature_importance": feature_importance
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/train")
async def train_models(request: TrainingRequest):
    """Handle model training request"""
    global data_processor
    
    try:
        if data_processor is None:
            raise HTTPException(status_code=400, detail="No data loaded")
            
        # Prepare features
        if not data_processor.prepare_features(request.target_column):
            raise HTTPException(status_code=400, detail="Failed to prepare features")
            
        # Split data
        X_train, X_test, y_train, y_test = data_processor.get_train_test_split(
            test_size=request.test_size
        )
        
        # Train models based on algorithm type
        metrics = model_trainer.train_models(
            X_train, y_train, X_test, y_test,
            algorithm_type=request.algorithm_type,
            selected_models=request.selected_models
        )
        
        # Perform cross-validation
        cv_scores = model_trainer.perform_cross_validation(
            X_train, y_train,
            algorithm_type=request.algorithm_type
        )
        
        # Generate model explanations
        explanations = model_trainer.generate_model_explanations(
            X_test, y_test,
            algorithm_type=request.algorithm_type
        )
        
        return {
            "message": "Models trained successfully",
            "metrics": metrics,
            "cross_validation_scores": cv_scores,
            "model_explanations": explanations
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/optimize")
async def optimize_model(request: ModelOptimizationRequest):
    """Handle model optimization request"""
    try:
        if data_processor is None:
            raise HTTPException(status_code=400, detail="No data loaded")
            
        X_train, X_test, y_train, y_test = data_processor.get_train_test_split()
        
        # Optimize hyperparameters
        best_params = model_trainer.optimize_hyperparameters(
            X_train, y_train,
            request.model_name,
            request.algorithm_type
        )
        
        return {
            "message": "Model optimized successfully",
            "best_params": best_params
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/save-model")
async def save_model(request: ModelManagementRequest):
    """Handle model saving request"""
    try:
        model_trainer.save_model(request.model_name, request.file_path)
        return {
            "message": "Model saved successfully",
            "path": request.file_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/load-model")
async def load_model(request: ModelManagementRequest):
    """Handle model loading request"""
    try:
        model_trainer.load_model(request.model_name, request.file_path)
        return {
            "message": "Model loaded successfully",
            "model_name": request.model_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/available-models")
async def get_available_models():
    """Get list of available models for each algorithm type"""
    return {
        "classification": model_trainer.get_classification_models(),
        "regression": model_trainer.get_regression_models(),
        "clustering": model_trainer.get_clustering_models()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000) 