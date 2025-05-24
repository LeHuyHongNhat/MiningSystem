from dash import html, dcc
import dash_bootstrap_components as dbc

def create_preprocessing_layout():
    return html.Div([
        # Target Column Section
        html.Div([
            html.H4("Set Target Column"),
            html.Div([
                html.Label("Select Target Column"),
                dcc.Dropdown(
                    id='target-column',
                    placeholder="Select target column"
                ),
            ], className="mb-3"),
            html.Button(
                "Set Target",
                id="set-target-button",
                className="btn btn-primary mb-3"
            ),
            html.Div(id="target-status"),
        ], className="mb-4"),
        
        # Drop Features Section
        html.Div([
            html.H4("Drop Features"),
            html.Div([
                html.Label("Select Features to Drop"),
                dcc.Dropdown(
                    id='drop-features',
                    multi=True,
                    placeholder="Select features to drop"
                ),
            ], className="mb-3"),
            html.Button(
                "Drop Features",
                id="drop-features-button",
                className="btn btn-primary mb-3"
            ),
            html.Div(id="drop-features-status"),
        ], className="mb-4"),
        
        # Categorical Features Section
        html.Div([
            html.H4("Process Categorical Features"),
            html.Div([
                html.Label("Select Categorical Features"),
                dcc.Dropdown(
                    id='categorical-features',
                    multi=True,
                    placeholder="Select categorical features"
                ),
            ], className="mb-3"),
            html.Div([
                html.Label("Handling Method"),
                dcc.Dropdown(
                    id='categorical-handling',
                    options=[
                        {'label': 'Drop', 'value': 'drop'},
                        {'label': 'Encode', 'value': 'encode'}
                    ],
                    placeholder="Select handling method"
                ),
            ], className="mb-3"),
            html.Div(
                id='encoding-method-container',
                children=[
                    html.Label("Encoding Method"),
                    dcc.Dropdown(
                        id='encoding-method',
                        options=[
                            {'label': 'One-Hot Encoding', 'value': 'one_hot'},
                            {'label': 'Label Encoding', 'value': 'label'},
                            {'label': 'Binary Encoding', 'value': 'binary'}
                        ],
                        placeholder="Select encoding method"
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
        ], className="mb-4"),
        
        # Feature Preprocessing Section
        html.Div([
            html.H4("Feature Preprocessing"),
            html.Div([
                html.Label("Select Features to Process"),
                dcc.Dropdown(
                    id='preprocessing-features',
                    multi=True,
                    placeholder="Select features to process"
                ),
            ], className="mb-3"),
            
            html.Div([
                html.Label("Missing Value Handling"),
                dcc.Dropdown(
                    id='missing-value-handling',
                    options=[
                        {'label': 'Mean', 'value': 'mean'},
                        {'label': 'Median', 'value': 'median'},
                        {'label': 'Mode', 'value': 'mode'},
                        {'label': 'Drop Rows', 'value': 'drop'}
                    ],
                    placeholder="Select method for handling missing values"
                ),
            ], className="mb-3"),
            
            html.Div([
                html.Label("Outlier Handling"),
                dcc.Dropdown(
                    id='outlier-handling',
                    options=[
                        {'label': 'IQR Method', 'value': 'iqr'},
                        {'label': 'Z-Score Method', 'value': 'zscore'}
                    ],
                    placeholder="Select method for handling outliers"
                ),
            ], className="mb-3"),
            
            html.Button(
                "Process Features",
                id="process-features-button",
                className="btn btn-primary mb-3"
            ),
            
            html.Div(id="features-status"),
        ], className="mb-4"),
        
        # Feature Selection Section
        html.Div([
            html.H4("Feature Selection"),
            html.Div([
                html.Label("Selection Method"),
                dcc.Dropdown(
                    id='feature-selection-method',
                    options=[
                        {'label': 'Variance Threshold', 'value': 'variance'},
                        {'label': 'Select K Best', 'value': 'kbest'},
                        {'label': 'Recursive Feature Elimination', 'value': 'rfe'},
                        {'label': 'Select From Model', 'value': 'model'}
                    ],
                    placeholder="Select feature selection method"
                ),
            ], className="mb-3"),
            html.Div([
                html.Label("Number of Features"),
                dcc.Input(
                    id='n-features',
                    type='number',
                    min=1,
                    placeholder="Enter number of features to select"
                ),
            ], className="mb-3"),
            html.Button(
                "Select Features",
                id="select-features-button",
                className="btn btn-primary mb-3"
            ),
            html.Div(id="selection-status"),
        ], className="mb-4"),
        
        # Global Scaling Section
        html.Div([
            html.H4("Global Feature Scaling"),
            html.Div([
                html.Label("Scaling Method"),
                dcc.Dropdown(
                    id='global-scaling-method',
                    options=[
                        {'label': 'Standard Scaling', 'value': 'standard'},
                        {'label': 'Min-Max Scaling', 'value': 'minmax'},
                        {'label': 'Robust Scaling', 'value': 'robust'}
                    ],
                    placeholder="Select scaling method"
                ),
            ], className="mb-3"),
            html.Button(
                "Apply Scaling",
                id="apply-scaling-button",
                className="btn btn-primary mb-3"
            ),
            html.Div(id="scaling-status"),
        ], className="mb-4"),
        
        # Final Preprocessing Section
        html.Div([
            html.H4("Final Preprocessing"),
            html.Button(
                "Get Preprocessing Info",
                id="final-preprocess-button",
                className="btn btn-primary mb-3"
            ),
            html.Div(id="preprocess-status"),
            html.Div(id="preprocessed-data-info"),
        ], className="mb-4"),
        
        # Data Preview Section
        html.Div([
            html.H4("Data Preview"),
            html.Div(id="data-preview"),
        ], className="mb-4"),
    ]) 