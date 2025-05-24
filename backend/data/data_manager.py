import pandas as pd
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self):
        """Initialize the data manager"""
        self.current_data = None
        self.target_column = None
        self.target_data = None
        self.processing_history = []
        logger.info("DataManager initialized")
        
    def reset(self):
        """Reset the data manager"""
        self.current_data = None
        self.target_column = None
        self.target_data = None
        self.processing_history = []
        logger.info("DataManager reset")
        
    def set_data(self, data: pd.DataFrame):
        """Set the current data"""
        if data is None:
            logger.error("Attempted to set None data")
            raise ValueError("Data cannot be None")
            
        if not isinstance(data, pd.DataFrame):
            logger.error(f"Invalid data type: {type(data)}")
            raise ValueError("Data must be a pandas DataFrame")
            
        if data.empty:
            logger.error("Attempted to set empty DataFrame")
            raise ValueError("Data cannot be empty")
            
        self.current_data = data.copy()
        self.processing_history = []
        logger.info(f"Data set with shape: {data.shape}")
        
    def get_current_data(self) -> pd.DataFrame:
        """Get the current data"""
        return self.current_data.copy() if self.current_data is not None else None
        
    def get_preprocessed_data(self) -> pd.DataFrame:
        """Get preprocessed data"""
        if self.current_data is None:
            logger.error("No data loaded")
            raise ValueError("No data loaded")
        return self.current_data
        
    def update_data(self, data: pd.DataFrame):
        """Update the current data"""
        if data is None:
            raise ValueError("DataFrame cannot be None")
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if data.empty:
            raise ValueError("DataFrame cannot be empty")
            
        logger.info(f"Updating data with shape: {data.shape}")
        self.current_data = data.copy()
        logger.info("Data updated successfully")
        
    def set_target_column(self, column: str):
        """Set the target column and store its data"""
        if self.current_data is None:
            raise ValueError("No data available")
            
        if column not in self.current_data.columns:
            raise ValueError(f"Column {column} not found in data")
            
        self.target_column = column
        self.target_data = self.current_data[column].copy()
        
    def get_target_column(self) -> str:
        """Get the target column name"""
        return self.target_column
        
    def get_target_data(self) -> pd.Series:
        """Get the target data"""
        return self.target_data.copy() if self.target_data is not None else None
        
    def add_to_history(self, step: dict):
        """Add a processing step to history"""
        if not isinstance(step, dict):
            logger.error(f"Invalid step type: {type(step)}")
            raise ValueError("Step must be a dictionary")
            
        self.processing_history.append(step)
        logger.info(f"Added step to history: {step}")
        
    def get_history(self) -> list:
        """Get the processing history"""
        return self.processing_history.copy()
        
    def clear_history(self):
        """Clear processing history"""
        self.processing_history = []
        logger.info("Processing history cleared") 