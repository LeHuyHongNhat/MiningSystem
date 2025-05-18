# Data Mining System

A comprehensive data mining system with FastAPI backend and modern frontend interface, featuring advanced data preprocessing, multiple machine learning algorithms, and model explainability.

## Demo

[Watch the demo video](https://youtu.be/DEYaQRODdPg)

## Features

- **Data Preprocessing**

  - Missing value handling, outlier detection
  - Feature scaling, encoding, and selection
  - Data quality analysis and visualization

- **Machine Learning**

  - Classification: Random Forest, XGBoost, SVM, Neural Network
  - Regression: Linear, Ridge, Lasso, Random Forest, XGBoost
  - Clustering: K-means, DBSCAN, Hierarchical

- **Model Evaluation**
  - Cross-validation and hyperparameter optimization
  - Multiple evaluation metrics
  - SHAP-based model explanations

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd data-mining-system
```

2. Create and activate virtual environment:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

### Backend

```bash
uvicorn backend.main:app --reload
```

Access API docs at: http://localhost:8000/docs

### Frontend

```bash
python frontend/app.py
```

Access the application at: http://127.0.0.1:8050/

## API Endpoints

- `/upload` - Upload data files (CSV/Excel)
- `/preprocess` - Preprocess data
- `/train` - Train models
- `/models` - List available models
- `/save-model` - Save trained models
- `/load-model` - Load saved models
- `/data-info` - Get dataset information

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
- Python-dotenv - Environment variables

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
