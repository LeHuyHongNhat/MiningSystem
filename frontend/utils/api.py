import requests
import json
import base64
import pandas as pd
import io
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

# API Configuration
API_BASE_URL = "http://127.0.0.1:8000/api"  # Update to match backend port and base path
API_CONFIG = {
    'base_url': API_BASE_URL,
    'timeout': 30,
    'max_retries': 3
}

def create_session():
    session = requests.Session()
    retry = Retry(
        total=API_CONFIG['max_retries'],
        backoff_factor=0.1,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def upload_file(contents, filename):
    """Upload file to backend"""
    try:
        # Decode the base64 content
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # Create a BytesIO object
        file_obj = io.BytesIO(decoded)
        
        # Create files dict for request
        files = {
            'file': (filename, file_obj, content_type)
        }
        
        # Send POST request to upload endpoint
        response = requests.post(f"{API_BASE_URL}/data/upload", files=files)
        response.raise_for_status()
        
        return response.json()
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise

def preprocess_step(config):
    """Preprocess data using backend API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/data/preprocess",
            json=config
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

def get_processing_history():
    """Get the processing history"""
    try:
        session = create_session()
        response = session.get(
            f"{API_BASE_URL}/data/history",
            timeout=API_CONFIG['timeout']
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise Exception(f"Error getting processing history: {str(e)}")

def get_current_data():
    """Get current data preview"""
    try:
        response = requests.get(f"{API_BASE_URL}/data/current")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error getting current data: {str(e)}")
        return None

def preprocess_data(config):
    """Send preprocessing configuration to the backend"""
    return preprocess_step(config)

def train_model(config):
    """Train model using backend API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/model/train",
            json=config
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

def get_available_models():
    """Get list of available models from the backend"""
    try:
        response = requests.get(f"{API_BASE_URL}/model/list")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise Exception(f"Error getting available models: {str(e)}")

def get_data_info(filename):
    """Get data information from backend"""
    try:
        response = requests.get(f"{API_BASE_URL}/data/info/{filename}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error getting data info: {str(e)}")
        raise

def create_visualization(config):
    """Create visualization using backend API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/visualization/create",
            json=config
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        raise

def parse_contents(contents, filename):
    """Parse uploaded file contents into pandas DataFrame"""
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # Create a BytesIO object from the decoded bytes
        file_obj = io.BytesIO(decoded)
        
        if 'csv' in filename:
            df = pd.read_csv(file_obj)
        elif 'xls' in filename:
            df = pd.read_excel(file_obj)
        else:
            raise ValueError("Unsupported file format")
            
        return df
    except Exception as e:
        logger.error(f"Error parsing file contents: {str(e)}")
        raise

def select_features(config):
    """Select features using backend API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/features/select",
            json=config
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error selecting features: {str(e)}")
        raise

def set_target_column(target_column: str):
    """Set target column and remove it from features"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/data/preprocess/set-target",
            params={"target_column": target_column}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error setting target column: {str(e)}")
        raise

def drop_features(features_to_drop: List[str]):
    """Drop selected features from the dataset"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/data/preprocess/drop-features",
            json=features_to_drop
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error dropping features: {str(e)}")
        raise

def get_categorical_features():
    """Get list of categorical features in the dataset"""
    try:
        response = requests.get(f"{API_BASE_URL}/data/preprocess/categorical-features")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error getting categorical features: {str(e)}")
        raise

def process_categorical_features(config: Dict[str, str]):
    """Process categorical features with specified methods"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/data/preprocess/process-categorical",
            json=config
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error processing categorical features: {str(e)}")
        raise

def process_features(config: Dict[str, Dict[str, str]]):
    """Process selected features with specified methods"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/data/preprocess/process-features",
            json=config
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error processing features: {str(e)}")
        raise

def scale_features(method: str):
    """Apply global feature scaling"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/data/preprocess/scale-features",
            params={"method": method}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error scaling features: {str(e)}")
        raise

def get_final_info():
    """Get information about the final preprocessed data"""
    try:
        response = requests.get(f"{API_BASE_URL}/data/preprocess/final-info")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error getting final info: {str(e)}")
        raise 