import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import IsolationForest
import logging
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.preprocessed_data = None
        self.feature_importance = None
        self.column_types = {}
        self.scalers = {}
        self.encoders = {}
        self.target_column = None
        self.feature_selector = None
        self.models = {
            'classification': {
                'Logistic Regression': LogisticRegression(),
                'Random Forest': RandomForestClassifier(),
                'XGBoost': XGBClassifier()
            },
            'regression': {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(),
                'XGBoost': XGBRegressor()
            }
        }
        
    def _detect_column_types(self):
        """Detect column types (categorical, numerical, datetime)"""
        if self.data is None:
            raise ValueError("No data loaded")
            
        for col in self.data.columns:
            if col == 'InvoiceDate':
                self.column_types[col] = 'datetime'
            elif col in ['Quantity', 'Price', 'TotalAmount']:
                self.column_types[col] = 'numerical'
            elif col in ['Invoice', 'StockCode', 'Description', 'Customer ID', 'Country']:
                self.column_types[col] = 'categorical'
                
    def _create_features(self, df):
        """Create additional features from the data"""
        # Convert InvoiceDate to datetime
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        # Extract time-based features
        df['InvoiceHour'] = df['InvoiceDate'].dt.hour
        df['InvoiceDay'] = df['InvoiceDate'].dt.day
        df['InvoiceMonth'] = df['InvoiceDate'].dt.month
        df['InvoiceYear'] = df['InvoiceDate'].dt.year
        df['InvoiceDayOfWeek'] = df['InvoiceDate'].dt.dayofweek
        
        # Calculate total amount
        df['TotalAmount'] = df['Quantity'] * df['Price']
        
        # Calculate items per invoice
        items_per_invoice = df.groupby('Invoice')['StockCode'].nunique().reset_index()
        items_per_invoice.columns = ['Invoice', 'ItemsPerInvoice']
        df = df.merge(items_per_invoice, on='Invoice', how='left')
        
        # Calculate customer purchase frequency
        customer_frequency = df.groupby('Customer ID')['Invoice'].nunique().reset_index()
        customer_frequency.columns = ['Customer ID', 'PurchaseFrequency']
        df = df.merge(customer_frequency, on='Customer ID', how='left')
        
        return df
                
    def clean_data(self, handle_missing=True, handle_outliers=True, 
                  scale_features=True, encode_categorical=True,
                  feature_selection=False, n_features=None,
                  target_column=None):
        """Clean and preprocess the data"""
        try:
            # Load data if not already loaded
            if self.data is None:
                if self.file_path.endswith('.csv'):
                    self.data = pd.read_csv(self.file_path)
                else:
                    self.data = pd.read_excel(self.file_path)
                    
            # Create additional features
            self.data = self._create_features(self.data)
            
            # Detect column types
            self._detect_column_types()
            
            # Make a copy of the data
            df = self.data.copy()
            
            # Handle missing values
            if handle_missing:
                for col in df.columns:
                    if df[col].isnull().any():
                        if self.column_types.get(col) == 'numerical':
                            # Use KNN imputation for numerical columns
                            imputer = KNNImputer(n_neighbors=5)
                            df[col] = imputer.fit_transform(df[[col]])
                        else:
                            # Use mode for categorical columns
                            df[col] = df[col].fillna(df[col].mode()[0])
                            
            # Handle outliers
            if handle_outliers:
                numerical_cols = [col for col, type_ in self.column_types.items() 
                                if type_ == 'numerical']
                for col in numerical_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df[col] = df[col].clip(lower_bound, upper_bound)
                        
            # Encode categorical variables
            if encode_categorical:
                categorical_cols = [col for col, type_ in self.column_types.items() 
                                  if type_ == 'categorical' and col != target_column]
                if categorical_cols:
                    self.encoders[target_column] = LabelEncoder()
                    for col in categorical_cols:
                        df[col] = self.encoders[target_column].fit_transform(df[col].astype(str))
                        
            # Scale features
            if scale_features:
                numerical_cols = [col for col, type_ in self.column_types.items() 
                                if type_ == 'numerical' and col != target_column]
                if numerical_cols:
                    self.scalers[target_column] = RobustScaler()
                    df[numerical_cols] = self.scalers[target_column].fit_transform(df[numerical_cols])
                    
            # Feature selection
            if feature_selection and n_features is not None and target_column:
                numerical_cols = [col for col, type_ in self.column_types.items() 
                                if type_ == 'numerical' and col != target_column]
                if numerical_cols:
                    selector = SelectKBest(score_func=f_classif, k=n_features)
                    df[numerical_cols] = selector.fit_transform(df[numerical_cols], df[target_column])
                    self.feature_importance = dict(zip(numerical_cols, 
                                                     selector.scores_))
                    
            self.preprocessed_data = df
            self.target_column = target_column
            return df
            
        except Exception as e:
            print(f"Error in clean_data: {str(e)}")  # Debug log
            raise
            
    def analyze_features(self):
        """Analyze features and return insights"""
        if self.preprocessed_data is None:
            raise ValueError("No preprocessed data available")
            
        try:
            analysis = {
                'correlations': self._analyze_correlations(),
                'distributions': self._analyze_distributions(),
                'outliers': self._analyze_outliers()
            }
            return analysis
        except Exception as e:
            print(f"Error in analyze_features: {str(e)}")  # Debug log
            raise
            
    def _analyze_correlations(self):
        """Analyze correlations between numerical features"""
        numerical_cols = [col for col, type_ in self.column_types.items() 
                         if type_ == 'numerical']
        if numerical_cols:
            return self.preprocessed_data[numerical_cols].corr().to_dict()
        return {}
        
    def _analyze_distributions(self):
        """Analyze distributions of numerical features"""
        numerical_cols = [col for col, type_ in self.column_types.items() 
                         if type_ == 'numerical']
        if numerical_cols:
            return {
                col: {
                    'mean': float(self.preprocessed_data[col].mean()),
                    'std': float(self.preprocessed_data[col].std()),
                    'min': float(self.preprocessed_data[col].min()),
                    'max': float(self.preprocessed_data[col].max())
                }
                for col in numerical_cols
            }
        return {}
        
    def _analyze_outliers(self):
        """Analyze outliers in numerical features"""
        numerical_cols = [col for col, type_ in self.column_types.items() 
                         if type_ == 'numerical']
        if numerical_cols:
            outliers = {}
            for col in numerical_cols:
                Q1 = self.preprocessed_data[col].quantile(0.25)
                Q3 = self.preprocessed_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers[col] = {
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'outlier_count': int(((self.preprocessed_data[col] < lower_bound) | 
                                        (self.preprocessed_data[col] > upper_bound)).sum())
                }
            return outliers
        return {}
        
    def get_features(self):
        """Get feature matrix X"""
        if self.preprocessed_data is None:
            raise ValueError("No preprocessed data available")
        if self.target_column:
            return self.preprocessed_data.drop(self.target_column, axis=1)
        return self.preprocessed_data
        
    def get_target(self):
        """Get target vector y"""
        if self.preprocessed_data is None or not self.target_column:
            raise ValueError("No preprocessed data available or no target column specified")
        return self.preprocessed_data[self.target_column]
        
    def load_data(self):
        """Load and validate the CSV file"""
        try:
            # Try to detect the file type and load accordingly
            if self.file_path.endswith('.csv'):
                self.data = pd.read_csv(self.file_path)
            elif self.file_path.endswith('.xlsx'):
                self.data = pd.read_excel(self.file_path)
            else:
                raise ValueError("Unsupported file format")
                
            # Detect column types
            self._detect_column_types()
            return True
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
            
    def prepare_features(self, target_column=None):
        """Prepare features for modeling"""
        if self.data is None:
            return False
            
        # Feature selection
        if target_column:
            self._select_features(target_column)
            
        # Prepare X and y
        if target_column:
            self.X = self.data.drop(columns=[target_column])
            self.y = self.data[target_column]
        else:
            self.X = self.data
            
        return True
        
    def _select_features(self, target_column, k=10):
        """Select most important features"""
        # For classification/regression
        if target_column in self.column_types and self.column_types[target_column] == 'categorical':
            selector = SelectKBest(f_classif, k=k)
        else:
            selector = SelectKBest(mutual_info_classif, k=k)
            
        # Fit selector
        selector.fit(self.data.drop(columns=[target_column]), self.data[target_column])
        
        # Get feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.data.drop(columns=[target_column]).columns,
            'importance': selector.scores_
        }).sort_values('importance', ascending=False)
        
    def get_train_test_split(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        if self.X is None or self.y is None:
            return None, None, None, None
            
        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
        
    def get_data_summary(self):
        """Get comprehensive statistical summary of the data"""
        if self.data is None:
            return None
            
        summary = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'column_types': self.column_types,
            'missing_values': self.data.isnull().sum().to_dict(),
            'numerical_summary': self.data.describe().to_dict(),
            'categorical_summary': {
                col: self.data[col].value_counts().to_dict()
                for col in self.column_types if self.column_types[col] == 'categorical'
            },
            'correlation_matrix': self.data[self.column_types['numerical']].corr().to_dict()
        }
        return summary 

    def handle_missing_values(self, df, method='mean', features=None):
        """Handle missing values in the dataset"""
        try:
            if features is None:
                features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if method == 'mean':
                imputer = SimpleImputer(strategy='mean')
            elif method == 'median':
                imputer = SimpleImputer(strategy='median')
            elif method == 'mode':
                imputer = SimpleImputer(strategy='most_frequent')
            else:
                raise ValueError(f"Unsupported missing value handling method: {method}")
            
            df[features] = imputer.fit_transform(df[features])
            logger.info(f"Missing values handled using {method} method")
            return df
        except Exception as e:
            logger.error(f"Error in handle_missing_values: {str(e)}")
            raise
            
    def handle_outliers(self, df, method='iqr', threshold=1.5, features=None):
        """Handle outliers in the dataset"""
        try:
            if features is None:
                features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if method == 'iqr':
                for feature in features:
                    Q1 = df[feature].quantile(0.25)
                    Q3 = df[feature].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    df[feature] = df[feature].clip(lower_bound, upper_bound)
            elif method == 'isolation_forest':
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                mask = iso_forest.fit_predict(df[features]) == 1
                df = df[mask]
            else:
                raise ValueError(f"Unsupported outlier handling method: {method}")
            
            logger.info(f"Outliers handled using {method} method")
            return df
        except Exception as e:
            logger.error(f"Error in handle_outliers: {str(e)}")
            raise
            
    def scale_features(self, df, method='standard', features=None):
        """Scale features in the dataset"""
        try:
            if features is None:
                features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unsupported scaling method: {method}")
            
            df[features] = scaler.fit_transform(df[features])
            self.scalers[method] = scaler
            logger.info(f"Features scaled using {method} method")
            return df
        except Exception as e:
            logger.error(f"Error in scale_features: {str(e)}")
            raise
            
    def encode_categorical(self, df, method='label', features=None):
        """Encode categorical features"""
        try:
            if features is None:
                features = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if method == 'label':
                for feature in features:
                    if feature not in self.encoders:
                        self.encoders[feature] = LabelEncoder()
                    df[feature] = self.encoders[feature].fit_transform(df[feature])
            elif method == 'onehot':
                encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                encoded = encoder.fit_transform(df[features])
                encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(features))
                df = pd.concat([df.drop(features, axis=1), encoded_df], axis=1)
            else:
                raise ValueError(f"Unsupported encoding method: {method}")
            
            logger.info(f"Categorical features encoded using {method} method")
            return df
        except Exception as e:
            logger.error(f"Error in encode_categorical: {str(e)}")
            raise
            
    def select_features(self, df, method='variance', n_features=None):
        """Select features based on specified method"""
        try:
            if method == 'variance':
                selector = VarianceThreshold(threshold=0.01)
                selector.fit(df)
                selected_features = df.columns[selector.get_support()].tolist()
            elif method == 'kbest':
                if n_features is None:
                    n_features = len(df.columns) // 2
                selector = SelectKBest(score_func=f_regression, k=n_features)
                selector.fit(df, df['target'])
                selected_features = df.columns[selector.get_support()].tolist()
            else:
                raise ValueError(f"Unsupported feature selection method: {method}")
            
            df = df[selected_features]
            self.feature_selector = selector
            logger.info(f"Features selected using {method} method")
            return df
        except Exception as e:
            logger.error(f"Error in select_features: {str(e)}")
            raise
            
    def analyze_features(self, df):
        """Analyze features in the dataset"""
        try:
            analysis = {
                "numeric_features": {
                    "count": len(df.select_dtypes(include=[np.number]).columns),
                    "names": df.select_dtypes(include=[np.number]).columns.tolist()
                },
                "categorical_features": {
                    "count": len(df.select_dtypes(include=['object', 'category']).columns),
                    "names": df.select_dtypes(include=['object', 'category']).columns.tolist()
                },
                "missing_values": df.isnull().sum().to_dict(),
                "basic_stats": df.describe().to_dict()
            }
            return analysis
        except Exception as e:
            logger.error(f"Error in analyze_features: {str(e)}")
            raise 