from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List, Dict, Any
import pandas as pd
import logging
from pathlib import Path
import os
import re
from backend.api.schemas import PreprocessingConfig, TrainingConfig, VisualizationConfig
from backend.core.preprocessor import DataPreprocessor
from backend.core.trainer import ModelTrainer
from backend.visualization import create_visualization
from backend.data.data_manager import DataManager
from sklearn.feature_selection import VarianceThreshold, SelectKBest, RFE, SelectFromModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Print to console
        logging.FileHandler('backend.log')  # Save to file
    ]
)
logger = logging.getLogger(__name__)

router = APIRouter()

# Create data directory if it doesn't exist
DATA_DIR = Path("backend/data")
DATA_DIR.mkdir(exist_ok=True, parents=True)

# Initialize data manager
data_manager = DataManager()

# Global variables
preprocessor = None
model_trainer = ModelTrainer()

def get_data():
    """Get the current data from data manager"""
    try:
        data = data_manager.get_current_data()
        if data is None:
            logger.error("No data available in data manager")
            raise HTTPException(status_code=400, detail="No data available. Please upload data first.")
        return data
    except ValueError as e:
        logger.error(f"Error getting data from data manager: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error getting data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper function for secure filename
def secure_filename(filename):
    """Generate a secure filename by removing potentially dangerous characters"""
    # Remove path components and dangerous characters
    filename = os.path.basename(filename)
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    # Remove other potentially dangerous characters
    filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    # Ensure filename is not empty and has reasonable length
    if not filename or filename == '.':
        filename = 'uploaded_file'
    return filename[:255]  # Limit filename length

# Dependency to get ModelTrainer instance
def get_trainer():
    return model_trainer

@router.post("/data/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a data file (CSV or Excel)"""
    logger.info(f"Received file upload request: {file.filename}")
    
    # Reset preprocessor
    global preprocessor
    preprocessor = None
    
    # Reset data manager
    try:
        data_manager.reset()
    except Exception as e:
        logger.error(f"Error resetting data manager: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to reset data manager")
    
    # Validate file extension first
    allowed_extensions = {'.csv', '.xlsx', '.xls'}
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        logger.error(f"Invalid file extension: {file_extension}")
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Allowed formats: {', '.join(allowed_extensions)}"
        )
    
    # Validate file size (e.g., max 100MB)
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        logger.error(f"File size exceeds limit: {len(content)} bytes")
        raise HTTPException(
            status_code=413,
            detail="File size exceeds maximum limit of 100MB"
        )
    
    try:
        # Create data directory if it doesn't exist
        DATA_DIR.mkdir(exist_ok=True, parents=True)
        
        # Generate safe filename to avoid path traversal
        safe_filename = secure_filename(file.filename)
        file_path = DATA_DIR / safe_filename
        
        logger.info(f"Processing file: {file.filename} -> {safe_filename}")
        
        # Save the uploaded file
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info(f"File saved to: {file_path}")
        
        # Read and validate the file
        df = None
        if file_extension == '.csv':
            try:
                logger.info("Attempting to read CSV file...")
                
                # Try reading with different encodings
                encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(
                            file_path,
                            encoding=encoding,
                            on_bad_lines='skip'
                        )
                        logger.info(f"Successfully read CSV with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        logger.warning(f"Failed to decode with {encoding}")
                        continue
                    except Exception as e:
                        logger.warning(f"Error reading with {encoding}: {str(e)}")
                        continue
                
                if df is None:
                    raise ValueError("Could not read CSV file with any encoding")
                    
            except Exception as e:
                logger.error(f"Failed to read CSV file: {str(e)}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Unable to read CSV file: {str(e)}"
                )
                
        elif file_extension in ['.xlsx', '.xls']:
            try:
                logger.info("Attempting to read Excel file...")
                df = pd.read_excel(file_path, engine='openpyxl')
                logger.info("Excel file read successfully")
                
            except Exception as e:
                logger.error(f"Failed to read Excel file: {str(e)}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Unable to read Excel file: {str(e)}"
                )
        
        # Validate data structure
        logger.info("Validating data structure...")
        if df is None:
            logger.error("DataFrame is None after reading file")
            raise HTTPException(
                status_code=400,
                detail="Failed to read file data"
            )
            
        if df.empty:
            logger.error("DataFrame is empty after reading file")
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty or contains no valid data"
            )
            
        if len(df.columns) == 0:
            logger.error("DataFrame has no columns")
            raise HTTPException(
                status_code=400,
                detail="The uploaded file has no valid columns"
            )
            
        # Set data in data manager
        try:
            logger.info("Setting data in data manager")
            data_manager.set_data(df)
            logger.info("Data set successfully in data manager")
        except Exception as e:
            logger.error(f"Failed to set data in data manager: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to set data: {str(e)}")
        
        # Initialize preprocessor
        try:
            logger.info("Initializing preprocessor...")
            preprocessor = DataPreprocessor()
            logger.info("Preprocessor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize preprocessor: {str(e)}")
            # Continue without preprocessor for now
            preprocessor = None
        
        logger.info("Preparing response...")
        response = {
            "message": "File uploaded and processed successfully",
            "filename": safe_filename,
            "original_filename": file.filename,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
        }
        logger.info("Response prepared successfully")
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in upload: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while processing the file: {str(e)}"
        )

def convert_numpy_types(obj):
    """Convert numpy types to Python native types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

@router.post("/data/preprocess")
async def process_step(config: dict):
    """Process data with given configuration"""
    try:
        # Get the current data
        df = get_data()
        if df is None:
            raise HTTPException(status_code=400, detail="No data available. Please upload data first.")
            
        # Create preprocessor and set data
        preprocessor = DataPreprocessor()
        preprocessor.set_data(df)
        
        # Process the step
        result = preprocessor.process_step(config)
        
        # Convert numpy types to Python native types
        result = convert_numpy_types(result)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in processing step: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/data/history")
async def get_processing_history():
    """Get processing history"""
    try:
        # Get the current data
        df = get_data()
        if df is None:
            raise HTTPException(status_code=400, detail="No data available. Please upload data first.")
            
        # Create preprocessor and set data
        preprocessor = DataPreprocessor()
        preprocessor.set_data(df)
        
        return preprocessor.processing_history
        
    except Exception as e:
        logger.error(f"Error getting processing history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/model/train")
async def train_models(config: TrainingConfig):
    """Train models based on the provided configuration"""
    try:
        logger.info(f"Starting training with config: {config.dict()}")
        
        # Get current data (features)
        features_data = get_data()
        if features_data is None or features_data.empty:
            raise HTTPException(status_code=400, detail="No data available for training")
            
        # Get target data if not clustering
        if config.algorithm_type != 'clustering':
            target_column = data_manager.get_target_column()
            if not target_column:
                raise HTTPException(status_code=400, detail="No target column set")
            
            # Get target data
            target_data = data_manager.get_target_data()
            if target_data is None:
                raise HTTPException(status_code=400, detail="Target data not available")
            
            # Combine features and target
            data = features_data.copy()
            data['target'] = target_data
        else:
            data = features_data
        
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Train models
        results = trainer.train_models(
            data=data,
            target=None if config.algorithm_type == 'clustering' else 'target',
            algorithm_type=config.algorithm_type,
            selected_models=config.selected_models,
            test_size=config.test_size,
            random_state=config.random_state,
            optimize_hyperparameters=config.optimize_hyperparameters,
            model_params=config.model_params
        )
        
        # Convert numpy types to Python native types
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        # Convert all numpy types in results
        results = convert_numpy_types(results)
        
        return {
            "status": "success",
            "message": "Models trained successfully",
            "data": results
        }
        
    except Exception as e:
        logger.error(f"Error in training step: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/model/save")
async def save_model(model_name: str, algorithm_type: str):
    """Save a trained model to disk"""
    try:
        # Create models directory if it doesn't exist
        MODELS_DIR = Path("backend/models")
        MODELS_DIR.mkdir(exist_ok=True, parents=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{algorithm_type}_{model_name}_{timestamp}.joblib"
        file_path = MODELS_DIR / filename
        
        # Save the model
        model_trainer.save_model(model_name, str(file_path))
        
        return {
            "status": "success",
            "message": f"Model saved successfully as {filename}",
            "file_path": str(file_path)
        }
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model/list")
async def get_available_models():
    """Get list of available models"""
    try:
        models = {
            "classification": [
                {"name": "Logistic Regression", "value": "Logistic Regression"},
                {"name": "Random Forest", "value": "Random Forest"},
                {"name": "XGBoost", "value": "XGBoost"},
                {"name": "SVM", "value": "SVM"}
            ],
            "regression": [
                {"name": "Linear Regression", "value": "Linear Regression"},
                {"name": "Random Forest", "value": "Random Forest"},
                {"name": "XGBoost", "value": "XGBoost"},
                {"name": "SVM", "value": "SVM"}
            ],
            "clustering": [
                {"name": "K-Means", "value": "K-Means"},
                {"name": "DBSCAN", "value": "DBSCAN"},
                {"name": "Hierarchical Clustering", "value": "Hierarchical Clustering"}
            ]
        }
        return models
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/data/info/{filename}")
async def get_data_info(filename: str):
    """Get information about a specific data file"""
    try:
        file_path = DATA_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
            
        if filename.endswith('.csv'):
            # Try different encodings
            encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
                    
            if df is None:
                raise HTTPException(status_code=400, detail="Unable to read CSV file with any encoding")
        else:
            df = pd.read_excel(file_path)
            
        # Get numerical columns statistics
        numerical_stats = {}
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            stats = df[col].describe()
            numerical_stats[col] = {
                "count": float(stats['count']),
                "mean": float(stats['mean']),
                "std": float(stats['std']),
                "min": float(stats['min']),
                "25%": float(stats['25%']),
                "50%": float(stats['50%']),
                "75%": float(stats['75%']),
                "max": float(stats['max'])
            }
            
        info = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_types": df.dtypes.astype(str).to_dict(),
            "non_null_counts": df.count().to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "numerical_stats": numerical_stats
        }
        return info
    except Exception as e:
        logger.error(f"Error getting data info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/visualization/create")
async def create_visualization_endpoint(config: VisualizationConfig):
    """Create visualization with given configuration"""
    try:
        logger.info(f"Received visualization request: {config.dict()}")
        
        # Get the current dataset
        file_path = DATA_DIR / config.filename
        logger.info(f"Looking for file: {file_path}")
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise HTTPException(status_code=404, detail="File not found")
            
        # Read the dataset
        if config.filename.endswith('.csv'):
            # Try different encodings
            encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    logger.info(f"Successfully read CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    logger.warning(f"Failed to decode with {encoding}")
                    continue
                    
            if df is None:
                logger.error("Could not read CSV file with any encoding")
                raise HTTPException(status_code=400, detail="Unable to read CSV file with any encoding")
        else:
            df = pd.read_excel(file_path)
            logger.info("Successfully read Excel file")
            
        # Create visualization
        logger.info("Creating visualization...")
        result = create_visualization(df, config.dict())
        logger.info("Visualization created successfully")
        
        return result
        
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/data/preprocess/set-target")
async def set_target_column(target_column: str):
    """Set target column and remove it from features"""
    logger.info(f"Setting target column: {target_column}")
    
    try:
        # Get current data
        current_data = data_manager.get_current_data()
        if current_data is None:
            logger.error("No data available")
            raise HTTPException(status_code=400, detail="No data available. Please upload data first.")
            
        # Clean target column name (remove '+' and spaces)
        target_column = target_column.replace('+', ' ').strip()
        logger.info(f"Cleaned target column name: {target_column}")
        
        # Find the exact column name in the data (handling whitespace)
        exact_column = None
        for col in current_data.columns:
            if col.strip() == target_column:
                exact_column = col
                break
                
        if exact_column is None:
            logger.error(f"Column {target_column} not found in data. Available columns: {current_data.columns.tolist()}")
            raise HTTPException(
                status_code=400, 
                detail=f"Column {target_column} not found in data. Available columns: {', '.join(current_data.columns.tolist())}"
            )
            
        # Store target column in data manager
        data_manager.set_target_column(exact_column)
        
        # Remove target column from features
        logger.info(f"Removing target column {exact_column} from features")
        features = current_data.drop(columns=[exact_column])
        
        # Update data in data manager
        logger.info("Updating data in data manager")
        data_manager.update_data(features)
        
        # Add to processing history
        data_manager.add_to_history({
            "step": "set_target",
            "target_column": exact_column,
            "remaining_features": features.columns.tolist()
        })
        
        logger.info(f"Target column {exact_column} set successfully")
        return {
            "message": f"Target column {exact_column} set successfully",
            "remaining_features": features.columns.tolist()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting target column: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/data/preprocess/drop-features")
async def drop_features(features: List[str]):
    """Drop specified features from the dataset"""
    try:
        logger.info(f"Dropping features: {features}")
        
        # Get current data
        current_data = data_manager.get_current_data()
        if current_data is None:
            raise HTTPException(status_code=400, detail="No data loaded")
            
        # Validate features exist
        missing_features = [f for f in features if f not in current_data.columns]
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Features not found in dataset: {', '.join(missing_features)}"
            )
            
        # Drop features
        logger.info(f"Dropping {len(features)} features")
        df = current_data.drop(columns=features)
        
        # Update data in data manager
        logger.info("Updating data in data manager")
        data_manager.update_data(df)
        
        # Add to history
        data_manager.add_to_history({
            'step': 'drop_features',
            'dropped_features': features,
            'remaining_features': df.columns.tolist()
        })
        
        logger.info(f"Successfully dropped {len(features)} features")
        return {
            "message": f"Successfully dropped {len(features)} features",
            "remaining_features": df.columns.tolist()
        }
    except Exception as e:
        logger.error(f"Error dropping features: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/data/preprocess/categorical-features")
async def get_categorical_features():
    """Get list of categorical features in the dataset"""
    logger.info("Getting categorical features")
    
    try:
        # Get current data
        current_data = data_manager.get_current_data()
        if current_data is None:
            logger.error("No data available")
            raise HTTPException(status_code=400, detail="No data available. Please upload data first.")
            
        # Identify categorical features
        categorical_features = current_data.select_dtypes(include=['object', 'category']).columns.tolist()
        logger.info(f"Found categorical features: {categorical_features}")
        
        return {
            "categorical_features": categorical_features
        }
    except Exception as e:
        logger.error(f"Error getting categorical features: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/data/preprocess/process-categorical")
async def process_categorical_features(config: Dict[str, str]):
    """Process categorical features with specified methods"""
    try:
        logger.info(f"Processing categorical features with config: {config}")
        
        # Get current data
        current_data = data_manager.get_current_data()
        if current_data is None:
            raise HTTPException(status_code=400, detail="No data loaded")
            
        # Process each categorical feature
        processed_features = []
        for feature, method in config.items():
            if feature not in current_data.columns:
                logger.warning(f"Feature {feature} not found in dataset")
                continue
                
            logger.info(f"Processing feature {feature} with method {method}")
            if method == "one_hot":
                # One-hot encoding
                dummies = pd.get_dummies(current_data[feature], prefix=feature)
                current_data = pd.concat([current_data.drop(columns=[feature]), dummies], axis=1)
                processed_features.append(f"{feature} (one-hot encoded)")
                logger.info(f"One-hot encoded {feature} into {len(dummies.columns)} columns")
            elif method == "label":
                # Label encoding
                current_data[feature] = current_data[feature].astype('category').cat.codes
                processed_features.append(f"{feature} (label encoded)")
                logger.info(f"Label encoded {feature}")
            elif method == "binary":
                # Binary encoding
                current_data[feature] = (current_data[feature] == current_data[feature].mode()[0]).astype(int)
                processed_features.append(f"{feature} (binary encoded)")
                logger.info(f"Binary encoded {feature}")
                
        # Update data in data manager
        logger.info("Updating data in data manager")
        data_manager.update_data(current_data)
        
        # Add to history
        data_manager.add_to_history({
            'step': 'process_categorical',
            'processed_features': processed_features,
            'remaining_features': current_data.columns.tolist()
        })
        
        logger.info("Categorical features processed successfully")
        return {
            "message": "Categorical features processed successfully",
            "processed_features": processed_features,
            "remaining_features": current_data.columns.tolist()
        }
    except Exception as e:
        logger.error(f"Error processing categorical features: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/data/preprocess/process-features")
async def process_features(config: Dict[str, Dict[str, str]]):
    """Process selected features with specified methods"""
    try:
        logger.info(f"Processing features with config: {config}")
        
        # Get current data
        current_data = data_manager.get_current_data()
        if current_data is None:
            raise HTTPException(status_code=400, detail="No data loaded")
            
        # Process each feature
        processed_features = []
        for feature, methods in config.items():
            if feature not in current_data.columns:
                logger.warning(f"Feature {feature} not found in dataset")
                continue
                
            logger.info(f"Processing feature {feature} with methods: {methods}")
            feature_log = []
            
            # Handle missing values
            if "missing" in methods:
                method = methods["missing"]
                if method == "mean":
                    current_data[feature].fillna(current_data[feature].mean(), inplace=True)
                    feature_log.append(f"Missing values filled with mean")
                elif method == "median":
                    current_data[feature].fillna(current_data[feature].median(), inplace=True)
                    feature_log.append(f"Missing values filled with median")
                elif method == "mode":
                    current_data[feature].fillna(current_data[feature].mode()[0], inplace=True)
                    feature_log.append(f"Missing values filled with mode")
                elif method == "drop":
                    current_data.dropna(subset=[feature], inplace=True)
                    feature_log.append(f"Rows with missing values dropped")
                    
            # Handle outliers
            if "outlier" in methods:
                method = methods["outlier"]
                if method == "iqr":
                    Q1 = current_data[feature].quantile(0.25)
                    Q3 = current_data[feature].quantile(0.75)
                    IQR = Q3 - Q1
                    current_data = current_data[
                        (current_data[feature] >= Q1 - 1.5 * IQR) & 
                        (current_data[feature] <= Q3 + 1.5 * IQR)
                    ]
                    feature_log.append(f"Outliers removed using IQR method")
                elif method == "zscore":
                    z_scores = np.abs((current_data[feature] - current_data[feature].mean()) / current_data[feature].std())
                    current_data = current_data[z_scores < 3]
                    feature_log.append(f"Outliers removed using Z-score method")
                    
            processed_features.append({
                "feature": feature,
                "methods": feature_log
            })
            
        # Update data in data manager
        logger.info("Updating data in data manager")
        data_manager.update_data(current_data)
        
        # Add to history
        data_manager.add_to_history({
            'step': 'process_features',
            'processed_features': processed_features,
            'remaining_features': current_data.columns.tolist()
        })
        
        logger.info("Features processed successfully")
        return {
            "message": "Features processed successfully",
            "processed_features": processed_features,
            "remaining_features": current_data.columns.tolist()
        }
    except Exception as e:
        logger.error(f"Error processing features: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/data/preprocess/scale-features")
async def scale_features(method: str):
    """Apply global feature scaling"""
    try:
        # Get current data from data manager
        current_data = data_manager.get_current_data()
        if current_data is None:
            raise HTTPException(status_code=400, detail="No data available. Please upload data first.")
            
        # Select numerical features
        numerical_features = current_data.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numerical_features) == 0:
            raise HTTPException(status_code=400, detail="No numerical features found in the dataset")
            
        # Create and apply scaler
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            raise HTTPException(status_code=400, detail="Invalid scaling method")
            
        # Apply scaling
        current_data[numerical_features] = scaler.fit_transform(current_data[numerical_features])
        
        # Update data in data manager
        data_manager.update_data(current_data)
        
        # Add to history
        data_manager.add_to_history({
            'step': 'scale_features',
            'method': method,
            'scaled_features': numerical_features.tolist()
        })
        
        return {
            "message": f"Features scaled using {method} scaling",
            "scaled_features": numerical_features.tolist()
        }
    except Exception as e:
        logger.error(f"Error scaling features: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/data/preprocess/final-info")
async def get_final_info():
    """Get information about the final preprocessed data"""
    try:
        # Get current data from data manager
        current_data = data_manager.get_current_data()
        if current_data is None:
            raise HTTPException(status_code=400, detail="No data available. Please upload data first.")
            
        # Get processing history
        history = data_manager.get_history()
        
        # Calculate statistics
        info = {
            "rows": len(current_data),
            "columns": len(current_data.columns),
            "column_names": current_data.columns.tolist(),
            "data_types": current_data.dtypes.astype(str).to_dict(),
            "memory_usage": f"{current_data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
            "missing_values": current_data.isnull().sum().to_dict(),
            "numerical_stats": {
                col: {
                    "mean": float(current_data[col].mean()),
                    "std": float(current_data[col].std()),
                    "min": float(current_data[col].min()),
                    "max": float(current_data[col].max())
                }
                for col in current_data.select_dtypes(include=['int64', 'float64']).columns
            },
            "processing_history": history
        }
        
        logger.info("Successfully generated final info")
        return info
    except Exception as e:
        logger.error(f"Error getting final info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def handle_non_json_values(obj):
    """Handle non-JSON compliant values like infinity and NaN"""
    if isinstance(obj, float):
        if np.isinf(obj):
            return None if obj > 0 else None
        if np.isnan(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {key: handle_non_json_values(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [handle_non_json_values(item) for item in obj]
    return obj

@router.get("/data/current")
async def get_current_data():
    """Get current data preview"""
    logger.info("Getting current data preview")
    
    try:
        data = data_manager.get_current_data()
        if data is None:
            logger.info("No data available")
            return {
                "status": "error",
                "message": "No data available. Please upload data first.",
                "data": None
            }
            
        # Convert DataFrame to dict and handle non-JSON values
        preview_data = data.head().to_dict(orient='records')
        preview_data = handle_non_json_values(preview_data)
        
        logger.info(f"Returning preview of {len(preview_data)} rows")
        
        return {
            "status": "success",
            "message": "Data retrieved successfully",
            "data": preview_data
        }
    except Exception as e:
        logger.error(f"Error getting current data: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "data": None
        } 