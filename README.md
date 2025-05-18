# Data Mining System

A comprehensive data mining system built with FastAPI, featuring advanced data preprocessing, multiple machine learning algorithms, and model explainability.

## Features

### Data Preprocessing

- Missing value handling with KNN imputation
- Outlier detection and treatment
- Feature scaling and normalization
- Categorical variable encoding
- Feature selection and dimensionality reduction
- Data quality analysis and visualization

### Machine Learning Algorithms

- Classification
  - Random Forest
  - XGBoost
  - Support Vector Machine
  - Neural Network
- Regression
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Random Forest
  - XGBoost
  - Support Vector Regression
  - Neural Network
- Clustering
  - K-means
  - DBSCAN
  - Hierarchical Clustering

### Model Evaluation

- Cross-validation
- Multiple evaluation metrics
- Hyperparameter optimization
- Model comparison
- SHAP-based model explanations

### API Endpoints

- `/upload` - Upload data files (CSV/Excel)
- `/preprocess` - Preprocess data with configurable options
- `/train` - Train models with specified configuration
- `/models` - List available models
- `/save-model` - Save trained models
- `/load-model` - Load saved models
- `/data-info` - Get dataset information

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd data-mining-system
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Start the FastAPI server:

```bash
uvicorn backend.main:app --reload
```

2. Access the API documentation:

- OpenAPI (Swagger) UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

3. Example API calls:

```python
# Upload data
import requests

files = {'file': open('data.csv', 'rb')}
response = requests.post('http://localhost:8000/upload', files=files)

# Preprocess data
preprocessing_config = {
    "handle_missing": True,
    "handle_outliers": True,
    "scale_features": True,
    "encode_categorical": True,
    "feature_selection": True,
    "n_features": 10
}
response = requests.post('http://localhost:8000/preprocess', json=preprocessing_config)

# Train models
training_config = {
    "algorithm_type": "classification",
    "selected_models": ["random_forest", "xgboost"],
    "test_size": 0.2,
    "random_state": 42,
    "optimize_hyperparameters": True
}
response = requests.post('http://localhost:8000/train', json=training_config)
```

## Project Structure

```
data-mining-system/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── data_preprocessing.py # Data preprocessing module
│   └── model_trainer.py     # Model training module
├── data/                    # Data storage directory
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

## Dependencies

- FastAPI - Web framework
- Pandas - Data manipulation
- NumPy - Numerical computing
- Scikit-learn - Machine learning
- XGBoost - Gradient boosting
- SHAP - Model explainability
- Uvicorn - ASGI server
- Python-multipart - File uploads
- Openpyxl - Excel file support
- Joblib - Model persistence
- Pydantic - Data validation
- Python-jose - JWT tokens
- Passlib - Password hashing
- Bcrypt - Password hashing
- Python-dotenv - Environment variables

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
