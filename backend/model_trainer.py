import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.models = {
            'classification': {
                'Logistic Regression': LogisticRegression(random_state=42),
                'Random Forest': RandomForestClassifier(random_state=42),
                'XGBoost': XGBClassifier(random_state=42)
            },
            'regression': {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(random_state=42),
                'XGBoost': XGBRegressor(random_state=42)
            }
        }
        
        self.param_grids = {
            'classification': {
                'Logistic Regression': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                },
                'Random Forest': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                },
                'XGBoost': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1]
                }
            },
            'regression': {
                'Linear Regression': {
                    'fit_intercept': [True, False]
                },
                'Random Forest': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                },
                'XGBoost': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1]
                }
            }
        }
        
        self.trained_models = {}
        self.best_model = None
        self.best_score = -np.inf
        
    def train_models(self, data, algorithm_type, selected_models, test_size=0.2, 
                    random_state=42, optimize_hyperparameters=True):
        """Train multiple models and return the best one"""
        try:
            logger.info(f"Starting model training with type: {algorithm_type}")
            
            if algorithm_type not in self.models:
                raise ValueError(f"Unsupported algorithm type: {algorithm_type}")
                
            # Split data into features and target
            X = data.drop('target', axis=1)
            y = data['target']
            
            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            results = {
                'models': {},
                'best_model': None,
                'metrics': {},
                'all_metrics': {}  # Store metrics for all models
            }
            
            # Train each selected model
            for model_name in selected_models:
                if model_name not in self.models[algorithm_type]:
                    logger.warning(f"Model {model_name} not found in {algorithm_type} models")
                    continue
                    
                logger.info(f"Training {model_name}")
                model = self.models[algorithm_type][model_name]
                
                try:
                    if optimize_hyperparameters:
                        # Perform grid search
                        grid_search = GridSearchCV(
                            model,
                            self.param_grids[algorithm_type][model_name],
                            cv=5,
                            scoring='accuracy' if algorithm_type == 'classification' else 'r2',
                            n_jobs=-1  # Use all available cores
                        )
                        grid_search.fit(X_train, y_train)
                        model = grid_search.best_estimator_
                        logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
                    
                    # Train the model
                    model.fit(X_train, y_train)
                    self.trained_models[model_name] = model
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    if algorithm_type == 'classification':
                        metrics = {
                            'accuracy': float(accuracy_score(y_test, y_pred)),
                            'precision': float(precision_score(y_test, y_pred, average='weighted')),
                            'recall': float(recall_score(y_test, y_pred, average='weighted')),
                            'f1': float(f1_score(y_test, y_pred, average='weighted'))
                        }
                    else:  # regression
                        metrics = {
                            'mse': float(mean_squared_error(y_test, y_pred)),
                            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                            'r2': float(r2_score(y_test, y_pred))
                        }
                    
                    # Store metrics for this model
                    results['models'][model_name] = metrics
                    results['all_metrics'][model_name] = metrics
                    
                    # Update best model
                    score = metrics['accuracy'] if algorithm_type == 'classification' else metrics['r2']
                    if score > self.best_score:
                        self.best_score = score
                        self.best_model = model
                        results['best_model'] = model_name
                        results['metrics'] = metrics
                        
                    logger.info(f"Model {model_name} metrics: {metrics}")
                        
                except Exception as e:
                    logger.error(f"Error training {model_name}: {str(e)}")
                    continue
            
            if not results['models']:
                raise ValueError("No models were successfully trained")
                
            logger.info(f"Training completed. Best model: {results['best_model']}")
            logger.info(f"All models metrics: {results['all_metrics']}")
            return results
            
        except Exception as e:
            logger.error(f"Error in train_models: {str(e)}")
            raise
            
    def predict(self, model_name, X):
        """Make predictions using a trained model"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found in trained models")
        return self.trained_models[model_name].predict(X)
        
    def save_model(self, model_name, filepath):
        """Save a trained model to disk"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found in trained models")
        import joblib
        joblib.dump(self.trained_models[model_name], filepath)
        
    def load_model(self, model_name, filepath):
        """Load a trained model from disk"""
        import joblib
        self.trained_models[model_name] = joblib.load(filepath) 