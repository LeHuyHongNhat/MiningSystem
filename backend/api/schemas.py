from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class PreprocessingConfig(BaseModel):
    """Configuration for data preprocessing"""
    target_column: str
    preprocessing_steps: List[str] = []
    drop_features: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None
    categorical_handling: str = "none"  # none, drop, encode
    encoding_method: str = "onehot"  # onehot, label, target
    handle_missing: bool = False
    missing_method: str = "none"
    missing_features: Optional[List[str]] = None
    handle_outliers: bool = False
    outlier_method: str = "none"
    outlier_threshold: float = 3.0
    scale_features: bool = False
    scaling_method: str = "none"
    scaling_features: Optional[List[str]] = None
    feature_selection: bool = False
    feature_selection_method: Optional[str] = None
    n_features: Optional[int] = None
    global_scaling: bool = False
    global_scaling_method: Optional[str] = None
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "target_column": "target",
                "preprocessing_steps": [],
                "drop_features": ["id", "unnecessary_column"],
                "categorical_features": ["country", "category"],
                "categorical_handling": "encode",
                "encoding_method": "onehot",
                "handle_missing": False,
                "missing_method": "none",
                "missing_features": ["Price", "Quantity"],
                "handle_outliers": False,
                "outlier_method": "none",
                "outlier_threshold": 3.0,
                "scale_features": False,
                "scaling_method": "none",
                "scaling_features": ["Price", "Quantity"],
                "feature_selection": True,
                "feature_selection_method": "variance",
                "n_features": 10,
                "global_scaling": True,
                "global_scaling_method": "standard"
            }
        },
        "protected_namespaces": ()
    }

class TrainingConfig(BaseModel):
    algorithm_type: str
    selected_models: List[str]
    test_size: Optional[float] = None
    random_state: Optional[int] = None
    optimize_hyperparameters: bool = False
    model_params: Optional[Dict[str, Dict[str, Any]]] = None
    
    model_config = {
        "protected_namespaces": ()
    }

class VisualizationConfig(BaseModel):
    plot_type: str
    features: List[str]
    filename: str
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "plot_type": "scatter_2d",
                "features": ["feature1", "feature2"],
                "filename": "data.csv"
            }
        },
        "protected_namespaces": ()
    } 