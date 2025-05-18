from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from backend.data_preprocessing import DataPreprocessor
from backend.model_trainer import ModelTrainer
import json
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Data Mining System API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create data directory if it doesn't exist
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Global variables to store data and components
current_data = None
preprocessed_data = None
preprocessor = None
model_trainer = ModelTrainer()

class PreprocessingConfig(BaseModel):
    target_column: Optional[str] = None
    handle_missing: bool = True
    missing_method: str = 'mean'
    missing_features: Optional[List[str]] = None
    handle_outliers: bool = True
    outlier_method: str = 'iqr'
    outlier_threshold: float = 1.5
    scale_features: bool = True
    scaling_method: str = 'standard'
    scaling_features: Optional[List[str]] = None
    encode_categorical: bool = True
    encoding_method: str = 'label'
    encoding_features: Optional[List[str]] = None
    feature_selection: bool = False
    feature_selection_method: str = 'variance'
    n_features: Optional[int] = None
    algorithm_type: str = 'regression'
    
    class Config:
        schema_extra = {
            "example": {
                "handle_missing": True,
                "missing_method": "mean",
                "missing_features": ["Price", "Quantity"],
                "handle_outliers": True,
                "outlier_method": "iqr",
                "outlier_threshold": 1.5,
                "scale_features": True,
                "scaling_method": "standard",
                "scaling_features": ["Price", "Quantity"],
                "encode_categorical": True,
                "encoding_method": "label",
                "encoding_features": ["Country", "Description"],
                "feature_selection": True,
                "feature_selection_method": "variance",
                "n_features": 10,
                "algorithm_type": "regression"
            }
        }

class TrainingConfig(BaseModel):
    algorithm_type: str
    selected_models: List[str]
    test_size: float = 0.2
    random_state: int = 42
    optimize_hyperparameters: bool = False

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a data file (CSV or Excel)"""
    global current_data, preprocessor
    try:
        # Create data directory if it doesn't exist
        DATA_DIR.mkdir(exist_ok=True)
        
        file_path = DATA_DIR / file.filename
        logger.info(f"Received file: {file.filename}")
        
        # Save the uploaded file
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info(f"File saved to: {file_path}")
        
        # Read the file to validate it
        try:
            if file.filename.endswith('.csv'):
                current_data = pd.read_csv(file_path, encoding='utf-8')
            elif file.filename.endswith(('.xlsx', '.xls')):
                current_data = pd.read_excel(file_path)
            else:
                raise HTTPException(status_code=400, detail="Unsupported file format")
                
            logger.info(f"File read successfully. Shape: {current_data.shape}")
            logger.info(f"Columns: {current_data.columns.tolist()}")
            
            # Initialize preprocessor with the file path
            preprocessor = DataPreprocessor(str(file_path))
            logger.info("Preprocessor initialized successfully")
            
            return {
                "message": "File uploaded successfully",
                "filename": file.filename,
                "rows": len(current_data),
                "columns": len(current_data.columns)
            }
        except UnicodeDecodeError:
            # Try with different encoding if utf-8 fails
            current_data = pd.read_csv(file_path, encoding='latin1')
            logger.info(f"File read successfully with latin1 encoding. Shape: {current_data.shape}")
            
            # Initialize preprocessor with the file path
            preprocessor = DataPreprocessor(str(file_path))
            logger.info("Preprocessor initialized successfully")
            
            return {
                "message": "File uploaded successfully",
                "filename": file.filename,
                "rows": len(current_data),
                "columns": len(current_data.columns)
            }
            
    except Exception as e:
        logger.error(f"Error in upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/preprocess")
async def preprocess_data(config: PreprocessingConfig):
    """Preprocess the uploaded data"""
    global current_data, preprocessed_data, preprocessor
    if current_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded")
    if preprocessor is None:
        raise HTTPException(status_code=400, detail="Preprocessor not initialized")
        
    try:
        logger.info(f"Starting preprocessing with config: {config.dict()}")
        
        # Create a copy of the data to avoid modifying the original
        df = current_data.copy()
        logger.info(f"Data loaded. Shape: {df.shape}")
        
        # Apply preprocessing steps
        if config.handle_missing:
            df = preprocessor.handle_missing_values(df, config.missing_method, config.missing_features)
            
        if config.handle_outliers:
            df = preprocessor.handle_outliers(df, config.outlier_method, config.outlier_threshold)
            
        if config.scale_features:
            df = preprocessor.scale_features(df, config.scaling_method, config.scaling_features)
            
        if config.encode_categorical:
            df = preprocessor.encode_categorical(df, config.encoding_method, config.encoding_features)
            
        if config.feature_selection:
            df = preprocessor.select_features(df, config.feature_selection_method, config.n_features)
            
        # Store the preprocessed data
        preprocessed_data = df
        logger.info(f"Data preprocessed. Shape: {df.shape}")
        
        # Get analysis results
        analysis = preprocessor.analyze_features(df)
        
        return {
            "message": "Data preprocessed successfully",
            "preprocessed_shape": df.shape,
            "analysis": analysis
        }
    except Exception as e:
        logger.error(f"Error in preprocessing step: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_models(config: TrainingConfig):
    """Train models with the specified configuration"""
    global preprocessed_data
    if preprocessed_data is None:
        raise HTTPException(status_code=400, detail="No preprocessed data available")
        
    try:
        logger.info(f"Starting training with config: {config.dict()}")
        
        # Train models
        results = model_trainer.train_models(
            preprocessed_data,
            config.algorithm_type,
            config.selected_models,
            config.test_size,
            config.random_state,
            config.optimize_hyperparameters
        )
        
        if not results or 'best_model' not in results or 'metrics' not in results:
            raise HTTPException(status_code=500, detail="Training failed: Invalid results format")
            
        # Ensure metrics are properly formatted
        best_metrics = results["metrics"]
        if not isinstance(best_metrics, dict):
            raise HTTPException(status_code=500, detail="Invalid metrics format")
            
        # Format the response to include all models' metrics
        response = {
            "status": "success",
            "message": "Models trained successfully",
            "data": {
                "best_model": {
                    "name": results["best_model"],
                    "metrics": {
                        "accuracy": float(best_metrics.get("accuracy", 0)),
                        "precision": float(best_metrics.get("precision", 0)),
                        "recall": float(best_metrics.get("recall", 0)),
                        "f1": float(best_metrics.get("f1", 0))
                    } if config.algorithm_type == "classification" else {
                        "mse": float(best_metrics.get("mse", 0)),
                        "rmse": float(best_metrics.get("rmse", 0)),
                        "r2": float(best_metrics.get("r2", 0))
                    }
                },
                "all_models": [
                    {
                        "name": model_name,
                        "metrics": {
                            "accuracy": float(metrics.get("accuracy", 0)),
                            "precision": float(metrics.get("precision", 0)),
                            "recall": float(metrics.get("recall", 0)),
                            "f1": float(metrics.get("f1", 0))
                        } if config.algorithm_type == "classification" else {
                            "mse": float(metrics.get("mse", 0)),
                            "rmse": float(metrics.get("rmse", 0)),
                            "r2": float(metrics.get("r2", 0))
                        }
                    }
                    for model_name, metrics in results["all_metrics"].items()
                ],
                "summary": {
                    "total_models": len(results["all_metrics"]),
                    "best_model_name": results["best_model"],
                    "best_model_score": float(best_metrics.get("f1", 0)) if config.algorithm_type == "classification" 
                                       else float(best_metrics.get("r2", 0))
                }
            }
        }
        
        logger.info(f"Training completed successfully. Response: {response}")
        return response
        
    except Exception as e:
        error_msg = f"Error in training: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": error_msg
            }
        )

@app.get("/models")
async def get_available_models():
    """Get list of available models by type"""
    try:
        models = {
            "regression": ["Linear Regression", "Random Forest", "XGBoost"],
            "classification": ["Logistic Regression", "Random Forest", "XGBoost"],
            "clustering": ["K-Means", "DBSCAN", "Hierarchical Clustering"]
        }
        return models
    except Exception as e:
        logger.error(f"Error in get_available_models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save-model")
async def save_model(model_name: str):
    """Save the best model to disk"""
    try:
        model_path = DATA_DIR / f"{model_name}.joblib"
        model_trainer.save_model(model_name, str(model_path))
        return {"message": f"Model saved successfully at {model_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load-model")
async def load_model(model_name: str):
    """Load a saved model"""
    try:
        model_path = DATA_DIR / f"{model_name}.joblib"
        model_trainer.load_model(model_name, str(model_path))
        return {"message": f"Model loaded successfully from {model_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data-info")
async def get_data_info():
    """Get information about the current dataset"""
    global current_data
    if current_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded")
        
    try:
        info = {
            "rows": len(current_data),
            "columns": len(current_data.columns),
            "column_types": current_data.dtypes.astype(str).to_dict(),
            "missing_values": current_data.isnull().sum().to_dict(),
            "numeric_columns": current_data.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": current_data.select_dtypes(include=['object', 'category']).columns.tolist()
        }
        return info
    except Exception as e:
        logger.error(f"Error in get_data_info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 