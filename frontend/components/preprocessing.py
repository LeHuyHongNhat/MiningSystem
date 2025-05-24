import dash_bootstrap_components as dbc
from dash import html, dcc

def create_preprocessing_section():
    """Create the data preprocessing section"""
    return [
        html.H3("Data Preprocessing", className="mb-3"),
        
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
        html.Button(
            "Set Target",
            id="set-target-button",
            className="btn btn-primary mb-3"
        ),
        html.Div(id="target-status"),
        
        # Drop Features Section
        html.H5("Drop Features", className="mb-3"),
        dbc.Row([
            dbc.Col([
                html.Label("Select Features to Drop:"),
                dcc.Dropdown(
                    id='drop-features',
                    multi=True,
                    options=[],
                    value=None
                )
            ])
        ], className="mb-4"),
        html.Button(
            "Drop Features",
            id="drop-features-button",
            className="btn btn-primary mb-3"
        ),
        html.Div(id="drop-features-status"),
        
        # Categorical Features Section
        html.H5("Process Categorical Features", className="mb-3"),
        dbc.Row([
            dbc.Col([
                html.Label("Select Categorical Features:"),
                dcc.Dropdown(
                    id='categorical-features',
                    multi=True,
                    options=[],
                    value=None
                )
            ]),
            dbc.Col([
                html.Label("Handling Method:"),
                dcc.Dropdown(
                    id='categorical-handling',
                    options=[
                        {'label': 'Drop', 'value': 'drop'},
                        {'label': 'Encode', 'value': 'encode'}
                    ],
                    value=None
                )
            ])
        ], className="mb-4"),
        html.Div(
            id='encoding-method-container',
            children=[
                html.Label("Encoding Method:"),
                dcc.Dropdown(
                    id='encoding-method',
                    options=[
                        {'label': 'One-Hot Encoding', 'value': 'one_hot'},
                        {'label': 'Label Encoding', 'value': 'label'},
                        {'label': 'Binary Encoding', 'value': 'binary'}
                    ],
                    value=None
                ),
            ],
            style={'display': 'none'}
        ),
        html.Button(
            "Process Categorical",
            id="process-categorical-button",
            className="btn btn-primary mb-3"
        ),
        html.Div(id="categorical-status"),
        
        # Feature Preprocessing Section
        html.H5("Feature Preprocessing", className="mb-3"),
        dbc.Row([
            dbc.Col([
                html.Label("Select Features to Process:"),
                dcc.Dropdown(
                    id='preprocessing-features',
                    multi=True,
                    options=[],
                    value=None
                )
            ]),
            dbc.Col([
                html.Label("Missing Value Handling:"),
                dcc.Dropdown(
                    id='missing-value-handling',
                    options=[
                        {'label': 'Mean', 'value': 'mean'},
                        {'label': 'Median', 'value': 'median'},
                        {'label': 'Mode', 'value': 'mode'},
                        {'label': 'Drop Rows', 'value': 'drop'}
                    ],
                    value=None
                )
            ]),
            dbc.Col([
                html.Label("Outlier Handling:"),
                dcc.Dropdown(
                    id='outlier-handling',
                    options=[
                        {'label': 'IQR Method', 'value': 'iqr'},
                        {'label': 'Z-Score Method', 'value': 'zscore'}
                    ],
                    value=None
                )
            ])
        ], className="mb-4"),
        html.Button(
            "Process Features",
            id="process-features-button",
            className="btn btn-primary mb-3"
        ),
        html.Div(id="features-status"),
        
        # Feature Selection Section
        html.H5("Feature Selection", className="mb-3"),
        dbc.Row([
            dbc.Col([
                html.Label("Selection Method:"),
                dcc.Dropdown(
                    id='feature-selection-method',
                    options=[
                        {'label': 'Variance Threshold', 'value': 'variance'},
                        {'label': 'Select K Best', 'value': 'kbest'},
                        {'label': 'Recursive Feature Elimination', 'value': 'rfe'},
                        {'label': 'Select From Model', 'value': 'model'}
                    ],
                    value=None
                )
            ]),
            dbc.Col([
                html.Label("Number of Features:"),
                dcc.Input(
                    id='n-features',
                    type='number',
                    min=1,
                    value=10
                )
            ])
        ], className="mb-4"),
        html.Button(
            "Select Features",
            id="select-features-button",
            className="btn btn-primary mb-3"
        ),
        html.Div(id="selection-status"),
        
        # Global Scaling Section
        html.H5("Global Feature Scaling", className="mb-3"),
        dbc.Row([
            dbc.Col([
                html.Label("Scaling Method:"),
                dcc.Dropdown(
                    id='global-scaling-method',
                    options=[
                        {'label': 'Standard Scaling', 'value': 'standard'},
                        {'label': 'Min-Max Scaling', 'value': 'minmax'},
                        {'label': 'Robust Scaling', 'value': 'robust'}
                    ],
                    value=None
                )
            ])
        ], className="mb-4"),
        html.Button(
            "Apply Scaling",
            id="apply-scaling-button",
            className="btn btn-primary mb-3"
        ),
        html.Div(id="scaling-status"),
        
        # Final Preprocessing Section
        html.H5("Final Preprocessing", className="mb-3"),
        html.Button(
            "Get Preprocessing Info",
            id="final-preprocess-button",
            className="btn btn-primary mb-3"
        ),
        html.Div(id="preprocess-status"),
        html.Div(id="preprocessed-data-info"),
        
        # Data Preview Section
        html.H5("Data Preview", className="mb-3"),
        html.Div(id="data-preview")
    ] 