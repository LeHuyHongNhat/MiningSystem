import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
import json
import base64
import io
from datetime import datetime
import plotly.figure_factory as ff

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Data Mining System", className="text-center mb-4"),
            html.Hr()
        ])
    ]),
    
    # File Upload Section
    dbc.Row([
        dbc.Col([
            html.H3("Upload Data", className="mb-3"),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                multiple=False
            ),
            html.Div(id='upload-status'),
            dcc.Loading(
                id="loading-upload",
                type="circle",
                children=[html.Div(id="loading-upload-output")]
            )
        ])
    ]),
    
    # Data Information Section
    dbc.Row([
        dbc.Col([
            html.H3("Data Information", className="mb-3"),
            html.Div(id='data-info-content')
        ])
    ]),
    
    # Data Visualization Section
    dbc.Row([
        dbc.Col([
            html.H3("Data Visualization", className="mb-3"),
            html.Div(id='data-visualization')
        ])
    ]),
    
    # Preprocessing Section
    dbc.Row([
        dbc.Col([
            html.H3("Data Preprocessing", className="mb-3"),
            dbc.Card([
                dbc.CardBody([
                    # Target Column Selection
                    html.H5("Target Column", className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Target Column:"),
                            dcc.Dropdown(
                                id='target-column',
                                options=[],
                                value=None,
                                clearable=True
                            )
                        ])
                    ], className="mb-4"),
                    
                    # Missing Values Handling
                    html.H5("Missing Values", className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Handling Method:"),
                            dcc.Dropdown(
                                id='missing-handling-method',
                                options=[
                                    {'label': 'None', 'value': 'none'},
                                    {'label': 'Remove Rows', 'value': 'remove_rows'},
                                    {'label': 'Remove Features', 'value': 'remove_features'},
                                    {'label': 'Mean Imputation', 'value': 'mean'},
                                    {'label': 'Median Imputation', 'value': 'median'},
                                    {'label': 'Mode Imputation', 'value': 'mode'},
                                    {'label': 'KNN Imputation', 'value': 'knn'}
                                ],
                                value='none'
                            )
                        ]),
                        dbc.Col([
                            html.Label("Features to Handle:"),
                            dcc.Dropdown(
                                id='missing-features',
                                multi=True
                            )
                        ])
                    ], className="mb-4"),
                    
                    # Outlier Handling
                    html.H5("Outlier Detection", className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Method:"),
                            dcc.Dropdown(
                                id='outlier-method',
                                options=[
                                    {'label': 'None', 'value': 'none'},
                                    {'label': 'IQR Method', 'value': 'iqr'},
                                    {'label': 'Z-Score', 'value': 'zscore'},
                                    {'label': 'Isolation Forest', 'value': 'isolation_forest'}
                                ],
                                value='none'
                            )
                        ]),
                        dbc.Col([
                            html.Label("Threshold:"),
                            dcc.Slider(
                                id='outlier-threshold',
                                min=1,
                                max=5,
                                step=0.1,
                                value=1.5,
                                marks={i: str(i) for i in [1, 2, 3, 4, 5]},
                                disabled=True
                            )
                        ])
                    ], className="mb-4"),
                    
                    # Feature Scaling
                    html.H5("Feature Scaling", className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Method:"),
                            dcc.Dropdown(
                                id='scaling-method',
                                options=[
                                    {'label': 'None', 'value': 'none'},
                                    {'label': 'Standard Scaler', 'value': 'standard'},
                                    {'label': 'Min-Max Scaler', 'value': 'minmax'},
                                    {'label': 'Robust Scaler', 'value': 'robust'}
                                ],
                                value='none'
                            )
                        ]),
                        dbc.Col([
                            html.Label("Features to Scale:"),
                            dcc.Dropdown(
                                id='scaling-features',
                                multi=True
                            )
                        ])
                    ], className="mb-4"),
                    
                    # Categorical Encoding
                    html.H5("Categorical Encoding", className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Method:"),
                            dcc.Dropdown(
                                id='encoding-method',
                                options=[
                                    {'label': 'None', 'value': 'none'},
                                    {'label': 'Label Encoding', 'value': 'label'},
                                    {'label': 'One-Hot Encoding', 'value': 'onehot'},
                                    {'label': 'Target Encoding', 'value': 'target'}
                                ],
                                value='none'
                            )
                        ]),
                        dbc.Col([
                            html.Label("Features to Encode:"),
                            dcc.Dropdown(
                                id='encoding-features',
                                multi=True
                            )
                        ])
                    ], className="mb-4"),
                    
                    # Feature Selection
                    html.H5("Feature Selection", className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Method:"),
                            dcc.Dropdown(
                                id='feature-selection-method',
                                options=[
                                    {'label': 'None', 'value': 'none'},
                                    {'label': 'Variance Threshold', 'value': 'variance'},
                                    {'label': 'Correlation', 'value': 'correlation'},
                                    {'label': 'Mutual Information', 'value': 'mutual_info'},
                                    {'label': 'Recursive Feature Elimination', 'value': 'rfe'}
                                ],
                                value='none'
                            )
                        ]),
                        dbc.Col([
                            html.Label("Number of Features:"),
                            dcc.Input(
                                id='n-features',
                                type='number',
                                min=1,
                                value=10,
                                disabled=True
                            )
                        ])
                    ], className="mb-4"),
                    
                    # Preprocess Button
                    dbc.Button(
                        "Preprocess Data",
                        id="preprocess-button",
                        color="primary",
                        className="mt-3"
                    ),
                    dcc.Loading(
                        id="loading-preprocess",
                        type="circle",
                        children=[html.Div(id="preprocess-status")]
                    )
                ])
            ])
        ])
    ]),
    
    # Model Training Section
    dbc.Row([
        dbc.Col([
            html.H3("Model Training", className="mb-3"),
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Algorithm Type:"),
                            dcc.Dropdown(
                                id='algorithm-type',
                                options=[
                                    {'label': 'Regression', 'value': 'regression'},
                                    {'label': 'Classification', 'value': 'classification'},
                                    {'label': 'Clustering', 'value': 'clustering'}
                                ],
                                value='regression'
                            )
                        ]),
                        dbc.Col([
                            html.Label("Select Models:"),
                            dcc.Dropdown(
                                id='model-selection',
                                multi=True
                            )
                        ])
                    ], className="mb-4"),
                    
                    dbc.Button(
                        "Train Models",
                        id="train-button",
                        color="success",
                        className="mt-3"
                    ),
                    dcc.Loading(
                        id="loading-train",
                        type="circle",
                        children=[html.Div(id="training-status")]
                    )
                ])
            ])
        ])
    ]),
    
    # Results Section
    dbc.Row([
        dbc.Col([
            html.H3("Results", className="mb-3"),
            html.Div(id='results-content')
        ])
    ])
], fluid=True)

# Callbacks
@app.callback(
    [Output('outlier-threshold', 'disabled'),
     Output('n-features', 'disabled')],
    [Input('outlier-method', 'value'),
     Input('feature-selection-method', 'value')]
)
def toggle_inputs(outlier_method, feature_selection_method):
    return outlier_method == 'none', feature_selection_method == 'none'

@app.callback(
    [Output('upload-status', 'children'),
     Output('data-info-content', 'children'),
     Output('data-visualization', 'children'),
     Output('missing-features', 'options'),
     Output('scaling-features', 'options'),
     Output('encoding-features', 'options'),
     Output('target-column', 'options')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_output(contents, filename):
    if contents is None:
        return dash.no_update
        
    try:
        # Decode the file content
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # Read the file
        if filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            df = pd.read_excel(io.BytesIO(decoded))
            
        # Upload the file to backend
        files = {'file': (filename, decoded)}
        response = requests.post('http://localhost:8000/upload', files=files)
        
        if response.status_code == 200:
            # Get data information
            info_response = requests.get('http://localhost:8000/data-info')
            if info_response.status_code == 200:
                info = info_response.json()
                
                # Create data information display
                info_content = [
                    html.H4("Dataset Overview"),
                    html.P(f"Number of instances: {info['rows']}"),
                    html.P(f"Number of features: {info['columns']}"),
                    
                    html.H4("Feature Information"),
                    dbc.Table([
                        html.Thead(html.Tr([
                            html.Th("Feature"),
                            html.Th("Type"),
                            html.Th("Missing Values"),
                            html.Th("Unique Values")
                        ])),
                        html.Tbody([
                            html.Tr([
                                html.Td(col),
                                html.Td(info['column_types'][col]),
                                html.Td(info['missing_values'][col]),
                                html.Td(len(df[col].unique()))
                            ]) for col in df.columns
                        ])
                    ], bordered=True, hover=True)
                ]
                
                # Create visualizations
                viz_content = []
                
                # Numerical features distribution
                numerical_cols = info['numeric_columns']
                if numerical_cols:
                    for col in numerical_cols[:4]:  # Show first 4 numerical features
                        fig = px.histogram(df, x=col, title=f'Distribution of {col}')
                        viz_content.append(dcc.Graph(figure=fig))
                
                # Categorical features distribution
                categorical_cols = info['categorical_columns']
                if categorical_cols:
                    for col in categorical_cols[:4]:  # Show first 4 categorical features
                        value_counts = df[col].value_counts().reset_index()
                        value_counts.columns = ['Category', 'Count']
                        fig = px.bar(value_counts, 
                                   x='Category', 
                                   y='Count',
                                   title=f'Distribution of {col}')
                        viz_content.append(dcc.Graph(figure=fig))
                
                # Correlation matrix for numerical features
                if len(numerical_cols) > 1:
                    corr_matrix = df[numerical_cols].corr()
                    fig = px.imshow(corr_matrix,
                                  title='Correlation Matrix',
                                  labels=dict(color="Correlation"))
                    viz_content.append(dcc.Graph(figure=fig))
                
                # Create dropdown options
                feature_options = [{'label': col, 'value': col} for col in df.columns]
                
                return [
                    html.Div([
                        html.H4("Upload Successful!"),
                        html.P(f"File: {filename}"),
                        html.P(f"Shape: {df.shape}")
                    ]),
                    info_content,
                    viz_content,
                    feature_options,
                    feature_options,
                    feature_options,
                    feature_options
                ]
                
    except Exception as e:
        print(f"Error in update_output: {str(e)}")  # Debug log
        return [
            html.Div([
                html.H4("Upload Failed!"),
                html.P(f"Error: {str(e)}")
            ]),
            None, None, None, None, None, None
        ]

@app.callback(
    Output('preprocess-status', 'children'),
    [Input('preprocess-button', 'n_clicks')],
    [State('target-column', 'value'),
     State('missing-handling-method', 'value'),
     State('missing-features', 'value'),
     State('outlier-method', 'value'),
     State('outlier-threshold', 'value'),
     State('scaling-method', 'value'),
     State('scaling-features', 'value'),
     State('encoding-method', 'value'),
     State('encoding-features', 'value'),
     State('feature-selection-method', 'value'),
     State('n-features', 'value')]
)
def preprocess_data(n_clicks, target_column, missing_method, missing_features, 
                   outlier_method, outlier_threshold, scaling_method, scaling_features,
                   encoding_method, encoding_features, feature_selection_method, n_features):
    if n_clicks is None:
        return dash.no_update
        
    try:
        # Ensure we have valid string values for methods
        config = {
            'target_column': target_column,
            'handle_missing': missing_method != 'none',
            'missing_method': missing_method if missing_method != 'none' else 'mean',  # Default to mean if none
            'missing_features': missing_features or [],  # Empty list if None
            'handle_outliers': outlier_method != 'none',
            'outlier_method': outlier_method if outlier_method != 'none' else 'iqr',  # Default to iqr if none
            'outlier_threshold': outlier_threshold,
            'scale_features': scaling_method != 'none',
            'scaling_method': scaling_method if scaling_method != 'none' else 'standard',  # Default to standard if none
            'scaling_features': scaling_features or [],  # Empty list if None
            'encode_categorical': encoding_method != 'none',
            'encoding_method': encoding_method if encoding_method != 'none' else 'label',  # Default to label if none
            'encoding_features': encoding_features or [],  # Empty list if None
            'feature_selection': feature_selection_method != 'none',
            'feature_selection_method': feature_selection_method if feature_selection_method != 'none' else 'variance',  # Default to variance if none
            'n_features': n_features
        }
        
        # Remove any None values from the config
        config = {k: v for k, v in config.items() if v is not None}
        
        response = requests.post('http://localhost:8000/preprocess', json=config)
        
        if response.status_code == 200:
            result = response.json()
            return html.Div([
                html.H4("Preprocessing Successful!"),
                html.P(f"Preprocessed shape: {result['preprocessed_shape']}")
            ])
        else:
            return html.Div([
                html.H4("Preprocessing Failed!"),
                html.P(f"Error: {response.json()['detail']}")
            ])
            
    except Exception as e:
        return html.Div([
            html.H4("Preprocessing Failed!"),
            html.P(f"Error: {str(e)}")
        ])

@app.callback(
    [Output('model-selection', 'options'),
     Output('training-status', 'children')],
    [Input('algorithm-type', 'value'),
     Input('train-button', 'n_clicks')],
    [State('model-selection', 'value')]
)
def update_models_and_train(algorithm_type, n_clicks, selected_models):
    if algorithm_type is None:
        return dash.no_update, dash.no_update
        
    # Get available models
    response = requests.get('http://localhost:8000/models')
    if response.status_code == 200:
        models = response.json()
        model_options = [{'label': model, 'value': model} 
                        for model in models[algorithm_type]]
        
        if n_clicks is None:
            return model_options, dash.no_update
            
        # Train models
        try:
            config = {
                'algorithm_type': algorithm_type,
                'selected_models': selected_models,
                'test_size': 0.2,
                'random_state': 42,
                'optimize_hyperparameters': True
            }
            
            response = requests.post('http://localhost:8000/train', json=config)
            
            if response.status_code == 200:
                result = response.json()
                if result['status'] == 'success':
                    data = result['data']
                    return model_options, html.Div([
                        html.H4("Training Successful!"),
                        html.P(f"Best Model: {data['best_model']['name']}"),
                        html.P("Best Model Metrics:"),
                        html.Pre(json.dumps(data['best_model']['metrics'], indent=2)),
                        html.H5("All Models Performance:"),
                        html.Table([
                            html.Thead(html.Tr([
                                html.Th("Model"),
                                html.Th("Accuracy"),
                                html.Th("Precision"),
                                html.Th("Recall"),
                                html.Th("F1 Score")
                            ])),
                            html.Tbody([
                                html.Tr([
                                    html.Td(model['name']),
                                    html.Td(f"{model['metrics']['accuracy']:.4f}"),
                                    html.Td(f"{model['metrics']['precision']:.4f}"),
                                    html.Td(f"{model['metrics']['recall']:.4f}"),
                                    html.Td(f"{model['metrics']['f1']:.4f}")
                                ]) for model in data['all_models']
                            ])
                        ], className="table table-striped")
                    ])
                else:
                    return model_options, html.Div([
                        html.H4("Training Failed!"),
                        html.P(f"Error: {result['message']}")
                    ])
            else:
                error_detail = response.json().get('detail', {})
                error_message = error_detail.get('message', 'Unknown error') if isinstance(error_detail, dict) else str(error_detail)
                return model_options, html.Div([
                    html.H4("Training Failed!"),
                    html.P(f"Error: {error_message}")
                ])
                
        except Exception as e:
            return model_options, html.Div([
                html.H4("Training Failed!"),
                html.P(f"Error: {str(e)}")
            ])
            
    return [], None

if __name__ == '__main__':
    app.run_server(debug=True, port=8050) 