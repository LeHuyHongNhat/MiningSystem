import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, LogisticRegression
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        """Initialize the preprocessor"""
        self.scalers = {}
        self.encoders = {}
        self.data = None
        self.preprocessed_data = None
        self.processing_history = []
        
    def set_data(self, df: pd.DataFrame):
        """Set the data to preprocess"""
        self.data = df.copy()
        self.preprocessed_data = df.copy()
        self.processing_history = []
        
    def get_current_data(self) -> pd.DataFrame:
        """Get current state of data"""
        return self.preprocessed_data if self.preprocessed_data is not None else self.data
        
    def add_to_history(self, action: str, details: dict):
        """Add processing step to history"""
        self.processing_history.append({
            'action': action,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })
        
    def process_step(self, config: dict) -> dict:
        """Process a single step and return results"""
        try:
            current_data = self.get_current_data()
            if current_data is None:
                raise ValueError("No data available")
                
            df = current_data.copy()
            processed_features = []
            
            # Process based on step type
            if config.get('step_type') == 'drop_features':
                features = config.get('features', [])
                df = df.drop(columns=features)
                processed_features.extend([{
                    'name': feature,
                    'action': 'dropped',
                    'details': 'Feature removed from dataset'
                } for feature in features])
                
            elif config.get('step_type') == 'categorical':
                features = config.get('features', [])
                handling = config.get('handling', 'drop')
                if handling == 'drop':
                    df = df.drop(columns=features)
                    processed_features.extend([{
                        'name': feature,
                        'action': 'dropped',
                        'details': 'Categorical feature removed'
                    } for feature in features])
                elif handling == 'encode':
                    method = config.get('encoding_method', 'onehot')
                    df = self.encode_categorical(df, method, features)
                    processed_features.extend([{
                        'name': feature,
                        'action': 'encoded',
                        'details': f'Encoded using {method} method'
                    } for feature in features])
                    
            elif config.get('step_type') == 'preprocess':
                features = config.get('features', [])
                for feature in features:
                    if feature in df.columns:
                        # Handle missing values
                        if df[feature].isnull().any():
                            df[feature] = df[feature].fillna(df[feature].mean())
                            processed_features.append({
                                'name': feature,
                                'action': 'missing_values',
                                'details': 'Missing values filled with mean'
                            })
                            
                        # Handle outliers for numerical features
                        if df[feature].dtype in ['float64', 'int64']:
                            Q1 = df[feature].quantile(0.25)
                            Q3 = df[feature].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            df[feature] = df[feature].clip(lower=lower_bound, upper=upper_bound)
                            processed_features.append({
                                'name': feature,
                                'action': 'outliers',
                                'details': 'Outliers handled using IQR method'
                            })
                            
            elif config.get('step_type') == 'feature_selection':
                method = config.get('method')
                n_features = config.get('n_features')
                df = self.select_features(df, method, n_features)
                processed_features.append({
                    'name': 'feature_selection',
                    'action': 'selected',
                    'details': f'Applied {method} selection'
                })
                
            elif config.get('step_type') == 'scaling':
                method = config.get('method')
                df = self.scale_features(df, method)
                processed_features.append({
                    'name': 'global_scaling',
                    'action': 'scaled',
                    'details': f'Applied {method} scaling'
                })
                
            # Update preprocessed data
            self.preprocessed_data = df
            
            # Add to history
            self.add_to_history(config.get('step_type'), {
                'config': config,
                'processed_features': processed_features
            })
            
            # Prepare response
            result = {
                'preprocessed_data': df.head().to_dict('records'),
                'preprocessed_shape': df.shape,
                'column_types': df.dtypes.astype(str).to_dict(),
                'column_info': {
                    col: {
                        'non_null': int(df[col].count()),
                        'null': int(df[col].isnull().sum()),
                        'null_percentage': float(df[col].isnull().sum() / len(df) * 100)
                    } for col in df.columns
                },
                'numerical_stats': {
                    col: {
                        'count': float(df[col].count()),
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std()),
                        'min': float(df[col].min()),
                        '25%': float(df[col].quantile(0.25)),
                        '50%': float(df[col].quantile(0.50)),
                        '75%': float(df[col].quantile(0.75)),
                        'max': float(df[col].max())
                    } for col in df.select_dtypes(include=['float64', 'int64']).columns
                },
                'processed_features': processed_features
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in process_step: {str(e)}")
            raise
        
    def handle_missing_values(self, df: pd.DataFrame, method: str, features: list = None) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        if features is None:
            features = df.columns
            
        try:
            if method == 'remove_rows':
                df = df.dropna(subset=features)
            elif method == 'remove_features':
                df = df.drop(columns=features)
            else:
                imputer = None
                if method == 'mean':
                    imputer = SimpleImputer(strategy='mean')
                elif method == 'median':
                    imputer = SimpleImputer(strategy='median')
                elif method == 'mode':
                    imputer = SimpleImputer(strategy='most_frequent')
                elif method == 'knn':
                    imputer = KNNImputer(n_neighbors=5)
                    
                if imputer:
                    df[features] = imputer.fit_transform(df[features])
                    
            return df
        except Exception as e:
            logger.error(f"Error in handle_missing_values: {str(e)}")
            raise
            
    def handle_outliers(self, df: pd.DataFrame, method: str, threshold: float = 1.5) -> pd.DataFrame:
        """Handle outliers in the dataset"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if method == 'iqr':
                for col in numeric_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    
            elif method == 'zscore':
                for col in numeric_cols:
                    mean = df[col].mean()
                    std = df[col].std()
                    df[col] = df[col].clip(lower=mean - threshold * std, upper=mean + threshold * std)
                    
            elif method == 'isolation_forest':
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outliers = iso_forest.fit_predict(df[numeric_cols])
                df = df[outliers == 1]
                
            return df
        except Exception as e:
            logger.error(f"Error in handle_outliers: {str(e)}")
            raise
            
    def scale_features(self, df: pd.DataFrame, method: str, features: list = None) -> pd.DataFrame:
        """Scale features in the dataset"""
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns
            
        try:
            scaler = None
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
                
            if scaler:
                df[features] = scaler.fit_transform(df[features])
                self.scalers[method] = scaler
                
            return df
        except Exception as e:
            logger.error(f"Error in scale_features: {str(e)}")
            raise
            
    def encode_categorical(self, df: pd.DataFrame, method: str, features: list = None) -> pd.DataFrame:
        """Encode categorical features"""
        if features is None:
            features = df.select_dtypes(include=['object', 'category']).columns
            
        try:
            if method == 'label':
                for col in features:
                    df[col] = pd.Categorical(df[col]).codes
            elif method == 'onehot':
                df = pd.get_dummies(df, columns=features, prefix=features)
            elif method == 'target':
                # Implement target encoding if needed
                pass
                
            return df
        except Exception as e:
            logger.error(f"Error in encode_categorical: {str(e)}")
            raise
            
    def select_features(self, df: pd.DataFrame, method: str, n_features: int = None) -> pd.DataFrame:
        """Select features based on the specified method"""
        try:
            if method == 'variance':
                selector = VarianceThreshold(threshold=0.01)
                df = pd.DataFrame(selector.fit_transform(df), columns=df.columns[selector.get_support()])
                
            elif method == 'correlation':
                corr_matrix = df.corr().abs()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
                df = df.drop(columns=to_drop)
                
            elif method == 'mutual_info':
                if n_features is None:
                    n_features = len(df.columns) // 2
                selector = SelectKBest(score_func=mutual_info_regression, k=n_features)
                df = pd.DataFrame(selector.fit_transform(df, df['target']), 
                                columns=df.columns[selector.get_support()])
                
            elif method == 'rfe':
                if n_features is None:
                    n_features = len(df.columns) // 2
                estimator = LinearRegression()
                selector = RFE(estimator, n_features_to_select=n_features)
                df = pd.DataFrame(selector.fit_transform(df, df['target']),
                                columns=df.columns[selector.get_support()])
                
            return df
        except Exception as e:
            logger.error(f"Error in select_features: {str(e)}")
            raise
            
    def analyze_features(self, df: pd.DataFrame) -> dict:
        """Analyze features in the dataset"""
        try:
            analysis = {
                'numeric_summary': df.describe().to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'unique_values': {col: df[col].nunique() for col in df.columns},
                'correlation_matrix': df.corr().to_dict()
            }
            return analysis
        except Exception as e:
            logger.error(f"Error in analyze_features: {str(e)}")
            raise 