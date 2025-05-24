import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
import joblib
import logging
from typing import Optional, Dict, Any, List
from sklearn.metrics import silhouette_score, silhouette_samples

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        """Initialize the model trainer"""
        self.models = {}
        self.best_model = None
        self.best_score = float('-inf')
        
    def get_available_models(self) -> dict:
        """Get list of available models by type"""
        try:
            models = {
                "regression": [
                    "Linear Regression",
                    "Random Forest",
                    "XGBoost",
                    "SVM"
                ],
                "classification": [
                    "Logistic Regression",
                    "Random Forest",
                    "XGBoost",
                    "SVM"
                ],
                "clustering": [
                    "K-Means",
                    "DBSCAN",
                    "Hierarchical Clustering"
                ]
            }
            return models
        except Exception as e:
            logger.error(f"Error in get_available_models: {str(e)}")
            raise
        
    def train_models(self, data: pd.DataFrame, target: Optional[str] = None, algorithm_type: str = 'classification',
                    selected_models: List[str] = None, test_size: Optional[float] = None,
                    random_state: Optional[int] = None, optimize_hyperparameters: bool = False,
                    model_params: Dict[str, Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train models based on the provided configuration
        
        Args:
            data (pd.DataFrame): Input data
            target (str, optional): Name of the target column. Required for supervised learning.
            algorithm_type (str): Type of algorithm ('classification', 'regression', or 'clustering')
            selected_models (List[str]): List of model names to train
            test_size (float, optional): Test set size for supervised learning
            random_state (int, optional): Random seed for reproducibility
            optimize_hyperparameters (bool): Whether to optimize hyperparameters
            model_params (Dict[str, Dict[str, Any]], optional): Model-specific parameters
            
        Returns:
            Dict[str, Any]: Training results including metrics and visualization data
        """
        try:
            # Validate input data
            if data is None or data.empty:
                raise ValueError("No data provided for training")
                
            # Validate algorithm type
            if algorithm_type not in ['classification', 'regression', 'clustering']:
                raise ValueError(f"Unsupported algorithm type: {algorithm_type}")
                
            # Validate selected models
            if not selected_models:
                raise ValueError("No models selected for training")
                
            # Prepare data
            if algorithm_type == 'clustering':
                X = data
                y = None
            else:
                # Check if target column exists
                if target not in data.columns:
                    raise ValueError(f"Target column '{target}' not found in the dataset. Please set a target column before training.")
                    
                # Check for missing values
                if data[target].isnull().any():
                    raise ValueError("Target column contains missing values. Please handle missing values before training.")
                    
                # Check for infinite values
                if np.isinf(data[target].values).any():
                    raise ValueError("Target column contains infinite values. Please handle infinite values before training.")
                    
                X = data.drop(target, axis=1)
                y = data[target]
                
                # For classification, ensure labels are consecutive integers starting from 0
                if algorithm_type == 'classification':
                    unique_labels = np.unique(y)
                    label_map = {label: i for i, label in enumerate(unique_labels)}
                    y = np.array([label_map[label] for label in y])
                
                # Validate features
                if X.empty:
                    raise ValueError("No features available for training")
                    
                # Check for missing values in features
                if X.isnull().any().any():
                    raise ValueError("Features contain missing values. Please handle missing values before training.")
                    
                # Check for infinite values in features
                if np.isinf(X.values).any():
                    raise ValueError("Features contain infinite values. Please handle infinite values before training.")
                
            # Split data for supervised learning
            if algorithm_type in ['regression', 'classification']:
                if test_size is None:
                    test_size = 0.2
                if random_state is None:
                    random_state = 42
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
            
            # Initialize models based on algorithm type
            if algorithm_type == 'regression':
                model_dict = {
                    'Linear Regression': LinearRegression(),
                    'Random Forest': RandomForestRegressor(random_state=random_state),
                    'XGBoost': XGBRegressor(random_state=random_state),
                    'SVM': SVR()
                }
            elif algorithm_type == 'classification':
                model_dict = {
                    'Logistic Regression': LogisticRegression(random_state=random_state),
                    'Random Forest': RandomForestClassifier(random_state=random_state),
                    'XGBoost': XGBClassifier(random_state=random_state),
                    'SVM': SVC(random_state=random_state)
                }
            else:  # clustering
                model_dict = {
                    'K-Means': KMeans(random_state=random_state),
                    'DBSCAN': DBSCAN(),
                    'Hierarchical Clustering': AgglomerativeClustering()
                }
            
            # Filter models based on selection
            models_to_train = {name: model for name, model in model_dict.items() if name in selected_models}
            
            # Train models
            results = {}
            all_models = []
            best_score = float('-inf')
            best_model_name = None
            
            for model_name, model in models_to_train.items():
                try:
                    # Set model parameters if provided
                    if model_params and model_name in model_params:
                        model.set_params(**model_params[model_name])
                    
                    # Train model
                    if algorithm_type == 'clustering':
                        model.fit(X)
                        labels = model.labels_
                        score = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else 0
                        
                        results[model_name] = {
                            'silhouette_score': score,
                            'n_clusters': len(np.unique(labels))
                        }
                        
                        # Store visualization data
                        if 'visualization_data' not in results:
                            results['visualization_data'] = {}
                        results['visualization_data']['labels'] = results['visualization_data'].get('labels', {})
                        results['visualization_data']['labels'][model_name] = labels.tolist()
                        
                        if model_name == 'K-Means':
                            results['visualization_data']['centroids'] = results['visualization_data'].get('centroids', {})
                            results['visualization_data']['centroids'][model_name] = model.cluster_centers_.tolist()
                        
                        # Calculate silhouette samples for visualization
                        if len(np.unique(labels)) > 1:
                            silhouette_samples = silhouette_samples(X, labels)
                            results['visualization_data']['silhouette_samples'] = results['visualization_data'].get('silhouette_samples', {})
                            results['visualization_data']['silhouette_samples'][model_name] = silhouette_samples.tolist()
                        
                    else:  # supervised learning
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        if algorithm_type == 'classification':
                            # Calculate metrics with zero_division=0 to avoid warnings
                            accuracy = accuracy_score(y_test, y_pred)
                            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                            
                            results[model_name] = {
                                'accuracy': accuracy,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1
                            }
                            
                            # Store confusion matrix for visualization
                            if 'visualization_data' not in results:
                                results['visualization_data'] = {}
                            results['visualization_data']['confusion_matrix'] = results['visualization_data'].get('confusion_matrix', {})
                            results['visualization_data']['confusion_matrix'][model_name] = confusion_matrix(y_test, y_pred).tolist()
                            
                            # Calculate and store feature importance
                            if 'visualization_data' not in results:
                                results['visualization_data'] = {}
                            results['visualization_data']['feature_importance'] = results['visualization_data'].get('feature_importance', {})
                            
                            # Get feature importance based on model type
                            if model_name == 'Random Forest':
                                importances = model.feature_importances_
                                feature_importance = dict(zip(X.columns, importances))
                            elif model_name == 'XGBoost':
                                importances = model.feature_importances_
                                feature_importance = dict(zip(X.columns, importances))
                            elif model_name == 'Logistic Regression':
                                # For logistic regression, use absolute coefficients as importance
                                importances = np.abs(model.coef_[0])
                                feature_importance = dict(zip(X.columns, importances))
                            elif model_name == 'SVM':
                                # For SVM, use absolute coefficients as importance
                                if hasattr(model, 'coef_'):
                                    importances = np.abs(model.coef_[0])
                                    feature_importance = dict(zip(X.columns, importances))
                                else:
                                    feature_importance = None
                            else:
                                feature_importance = None
                                
                            if feature_importance is not None:
                                # Sort features by importance
                                sorted_features = dict(sorted(feature_importance.items(), 
                                                            key=lambda x: x[1], 
                                                            reverse=True))
                                results['visualization_data']['feature_importance'][model_name] = sorted_features
                            
                            score = accuracy
                            
                        else:  # regression
                            # Calculate metrics
                            mse = mean_squared_error(y_test, y_pred)
                            rmse = np.sqrt(mse)
                            r2 = r2_score(y_test, y_pred)
                            
                            results[model_name] = {
                                'mse': mse,
                                'rmse': rmse,
                                'r2': r2
                            }
                            
                            # Store predictions for visualization
                            if 'visualization_data' not in results:
                                results['visualization_data'] = {}
                            results['visualization_data']['predictions'] = results['visualization_data'].get('predictions', {})
                            results['visualization_data']['predictions'][model_name] = y_pred.tolist()
                            results['visualization_data']['actual_values'] = results['visualization_data'].get('actual_values', {})
                            results['visualization_data']['actual_values'][model_name] = y_test.tolist()
                            
                            # Store feature importance if available
                            if hasattr(model, 'feature_importances_'):
                                results['visualization_data']['feature_importance'] = results['visualization_data'].get('feature_importance', {})
                                results['visualization_data']['feature_importance'][model_name] = dict(zip(X.columns, model.feature_importances_))
                            
                            score = -rmse  # Negative RMSE for maximization
                    
                    # Update best model
                    if score > best_score:
                        best_score = score
                        best_model_name = model_name
                    
                    # Add to all models list
                    all_models.append({
                        'name': model_name,
                        'metrics': results[model_name]
                    })
                    
                except Exception as e:
                    logger.error(f"Error training {model_name}: {str(e)}")
                    continue
            
            if not results:
                raise ValueError("No models were successfully trained")
            
            # Store best model and all models in results
            results['best_model'] = {
                'name': best_model_name,
                'metrics': results[best_model_name]
            }
            results['all_models'] = all_models
            
            return results
            
        except Exception as e:
            logger.error(f"Error in training step: {str(e)}")
            raise
            
    def save_model(self, model_name: str, file_path: str):
        """Save a trained model to disk"""
        try:
            if model_name in self.models:
                joblib.dump(self.models[model_name], file_path)
            else:
                raise ValueError(f"Model {model_name} not found")
        except Exception as e:
            logger.error(f"Error in save_model: {str(e)}")
            raise
            
    def load_model(self, model_name: str, file_path: str):
        """Load a saved model from disk"""
        try:
            self.models[model_name] = joblib.load(file_path)
        except Exception as e:
            logger.error(f"Error in load_model: {str(e)}")
            raise 