from frontend.callbacks.upload_callbacks import register_upload_callbacks
from frontend.callbacks.visualization_callbacks import register_visualization_callbacks
from frontend.callbacks.preprocessing_callbacks import register_preprocessing_callbacks
from frontend.callbacks.feature_selection_callbacks import register_feature_selection_callbacks
from frontend.callbacks.model_training_callbacks import register_model_training_callbacks

def register_all_callbacks(app):
    """Register all callbacks for the application"""
    register_upload_callbacks(app)
    register_visualization_callbacks(app)
    register_preprocessing_callbacks(app)
    register_feature_selection_callbacks(app)
    register_model_training_callbacks(app) 