import dash_bootstrap_components as dbc
from dash import html, dcc

def create_preprocessing_section():
    """Create preprocessing section with individual feature controls"""
    return html.Div([
        html.H4("Data Preprocessing", className="mb-4"),
        
        # Data Preview Section
        dbc.Card([
            dbc.CardHeader("Data Preview"),
            dbc.CardBody([
                html.Div(id='data-preview')
            ])
        ], className="mb-4"),
        
        # Target Column Selection
        dbc.Card([
            dbc.CardHeader("Target Column"),
            dbc.CardBody([
                dcc.Dropdown(
                    id='target-column',
                    options=[],
                    placeholder="Select target column"
                ),
                dbc.Button(
                    "Set Target",
                    id='set-target-button',
                    color='primary',
                    className="mt-3"
                ),
                html.Div(id='target-status')
            ])
        ], className="mb-4"),
        
        # Feature Drop Selection
        dbc.Card([
            dbc.CardHeader("Select Features to Drop"),
            dbc.CardBody([
                dcc.Dropdown(
                    id='drop-features',
                    options=[],
                    placeholder="Select features to drop",
                    multi=True
                ),
                dbc.Button(
                    "Drop Features",
                    id='drop-features-button',
                    color='primary',
                    className="mt-3"
                ),
                html.Div(id='drop-features-status')
            ])
        ], className="mb-4"),
        
        # Categorical Data Handling
        dbc.Card([
            dbc.CardHeader("Categorical Data Handling"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Categorical Features"),
                        dcc.Dropdown(
                            id='categorical-features',
                            options=[],
                            placeholder="Select categorical features",
                            multi=True
                        )
                    ], width=6),
                    dbc.Col([
                        html.Label("Handling Method"),
                        dcc.Dropdown(
                            id='categorical-handling',
                            options=[
                                {'label': 'None', 'value': 'none'},
                                {'label': 'Drop', 'value': 'drop'},
                                {'label': 'Encode', 'value': 'encode'}
                            ],
                            value='none'
                        )
                    ], width=6)
                ], className="mb-3"),
                html.Div(
                    id='encoding-method-container',
                    style={'display': 'none'},
                    children=[
                        dbc.Row([
                            dbc.Col([
                                html.Label("Encoding Method"),
                                dcc.Dropdown(
                                    id='encoding-method',
                                    options=[
                                        {'label': 'One-Hot Encoding', 'value': 'onehot'},
                                        {'label': 'Label Encoding', 'value': 'label'},
                                        {'label': 'Target Encoding', 'value': 'target'}
                                    ],
                                    value='onehot'
                                )
                            ], width=12)
                        ], className="mb-3")
                    ]
                ),
                dbc.Button(
                    "Process Categorical",
                    id='process-categorical-button',
                    color='primary',
                    className="mt-3"
                ),
                html.Div(id='categorical-status')
            ])
        ], className="mb-4"),
        
        # Feature Selection
        dbc.Card([
            dbc.CardHeader("Select Features to Preprocess"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Features"),
                        dcc.Dropdown(
                            id='preprocessing-features',
                            options=[],
                            placeholder="Select features to preprocess",
                            multi=True
                        )
                    ], width=12)
                ], className="mb-3"),
                dbc.Row([
                    dbc.Col([
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
                        )
                    ], width=6),
                    dbc.Col([
                        html.Label("Outlier Handling"),
                        dcc.Dropdown(
                            id='outlier-handling',
                            options=[
                                {'label': 'IQR Method', 'value': 'iqr'},
                                {'label': 'Z-Score Method', 'value': 'zscore'}
                            ],
                            placeholder="Select method for handling outliers"
                        )
                    ], width=6)
                ], className="mb-3"),
                dbc.Button(
                    "Process Features",
                    id='process-features-button',
                    color='primary',
                    className="mt-3"
                ),
                html.Div(id='features-status')
            ])
        ], className="mb-4"),
        
        # Feature Selection
        dbc.Card([
            dbc.CardHeader("Feature Selection"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Method"),
                        dcc.Dropdown(
                            id='feature-selection-method',
                            options=[
                                {'label': 'None', 'value': 'none'},
                                {'label': 'Variance Threshold', 'value': 'variance'},
                                {'label': 'SelectKBest', 'value': 'selectkbest'},
                                {'label': 'RFE', 'value': 'rfe'},
                                {'label': 'L1-based', 'value': 'l1'}
                            ],
                            value='none'
                        )
                    ], width=6),
                    dbc.Col([
                        html.Label("Number of Features"),
                        dbc.Input(
                            id='n-features',
                            type='number',
                            min=1,
                            value=10
                        )
                    ], width=6)
                ]),
                dbc.Button(
                    "Select Features",
                    id='select-features-button',
                    color='primary',
                    className="mt-3"
                ),
                html.Div(id='selection-status')
            ])
        ], className="mb-4"),
        
        # Global Scaling Section
        dbc.Card([
            dbc.CardHeader("Global Feature Scaling"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Scaling Method"),
                        dcc.Dropdown(
                            id='global-scaling-method',
                            options=[
                                {'label': 'None', 'value': 'none'},
                                {'label': 'Standard Scaler', 'value': 'standard'},
                                {'label': 'Min-Max Scaler', 'value': 'minmax'},
                                {'label': 'Robust Scaler', 'value': 'robust'}
                            ],
                            value='none'
                        )
                    ], width=12)
                ]),
                dbc.Button(
                    "Apply Scaling",
                    id='apply-scaling-button',
                    color='primary',
                    className="mt-3"
                ),
                html.Div(id='scaling-status')
            ])
        ], className="mb-4"),
        
        # Final Preprocess Button
        dbc.Button(
            "Final Preprocess",
            id='final-preprocess-button',
            color='success',
            className="mb-4"
        ),
        
        # Preprocessing Status
        html.Div(id='preprocess-status'),
        
        # Preprocessed Data Information
        html.Div(id='preprocessed-data-info')
    ])

def create_layout():
    """Create the main layout of the application"""
    return html.Div([
        dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("Data Mining System", className="text-center my-4")
                ])
            ]),
            
            # Main Content with Tabs
            dbc.Tabs([
                # Upload Data & Information Tab
                dbc.Tab([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Upload Data", className="card-title"),
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
                            html.Div(id='upload-status')
                        ])
                    ], className="mb-4"),
                    
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Data Information", className="card-title"),
                            html.Div(id='data-info-content')
                        ])
                    ], className="mb-4")
                ], label="Upload Data & Information"),
                
                # Data Visualization Tab
                dbc.Tab([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Data Visualization", className="card-title"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Plot Type"),
                                    dcc.Dropdown(
                                        id='plot-type-selector',
                                        options=[],
                                        placeholder="Select plot type"
                                    )
                                ], width=6),
                                dbc.Col([
                                    html.Label("Feature 1"),
                                    dcc.Dropdown(
                                        id='feature-selector-1',
                                        options=[],
                                        placeholder="Select first feature"
                                    )
                                ], width=6)
                            ], className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Feature 2"),
                                    dcc.Dropdown(
                                        id='feature-selector-2',
                                        options=[],
                                        placeholder="Select second feature"
                                    )
                                ], width=6),
                                dbc.Col([
                                    html.Label("Feature 3"),
                                    dcc.Dropdown(
                                        id='feature-selector-3',
                                        options=[],
                                        placeholder="Select third feature"
                                    )
                                ], width=6)
                            ], className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button(
                                        "Show Visualization",
                                        id="show-visualization-button",
                                        color="primary",
                                        className="mt-3"
                                    )
                                ], width=12)
                            ]),
                            html.Div(id='visualization-display')
                        ])
                    ], className="mb-4")
                ], label="Data Visualization"),
                
                # Data Preprocessing Tab
                dbc.Tab([
                    dbc.Card([
                        dbc.CardBody([
                            create_preprocessing_section()
                        ])
                    ], className="mb-4")
                ], label="Data Preprocessing"),
                
                # Model Training Tab
                dbc.Tab([
                    dbc.Card([
                        dbc.CardBody([
                            create_training_layout()
                        ])
                    ], className="mb-4")
                ], label="Model Training")
            ], className="mb-4")
        ], fluid=True)
    ]) 

def create_training_layout():
    return html.Div([
        html.H2("Model Training"),
        
        # Algorithm selection
        html.Div([
            html.Label("Select Algorithm Type:"),
            dcc.Dropdown(
                id='algorithm-type',
                options=[
                    {'label': 'Classification', 'value': 'classification'},
                    {'label': 'Regression', 'value': 'regression'},
                    {'label': 'Clustering', 'value': 'clustering'}
                ],
                value=None
            )
        ], className="mb-4"),
        
        # Model selection
        html.Div([
            html.Label("Select Models:"),
            dcc.Dropdown(
                id='model-selection',
                options=[],
                value=[],
                multi=True
            )
        ], className="mb-4"),
        
        # Common parameters
        html.Div([
            html.Label("Test Size:"),
            dcc.Slider(
                id='test-size',
                min=0.1,
                max=0.5,
                step=0.05,
                value=0.2,
                marks={i/10: str(i/10) for i in range(1, 6)}
            ),
            html.Label("Random State:"),
            dcc.Input(
                id='random-state',
                type='number',
                value=42
            )
        ], id='common-params', className="mb-4"),
        
        # Clustering parameters
        html.Div([
            # K-Means parameters
            html.Div([
                html.H5("K-Means Parameters"),
                html.Label("Number of Clusters:"),
                dcc.Input(
                    id='kmeans-n-clusters',
                    type='number',
                    value=3,
                    min=2
                ),
                html.Label("Max Iterations:"),
                dcc.Input(
                    id='kmeans-max-iter',
                    type='number',
                    value=300,
                    min=100
                )
            ], id='kmeans-params', className="mb-4"),
            
            # DBSCAN parameters
            html.Div([
                html.H5("DBSCAN Parameters"),
                html.Label("Epsilon:"),
                dcc.Input(
                    id='dbscan-eps',
                    type='number',
                    value=0.5,
                    min=0.1,
                    step=0.1
                ),
                html.Label("Min Samples:"),
                dcc.Input(
                    id='dbscan-min-samples',
                    type='number',
                    value=5,
                    min=1
                )
            ], id='dbscan-params', className="mb-4"),
            
            # Hierarchical Clustering parameters
            html.Div([
                html.H5("Hierarchical Clustering Parameters"),
                html.Label("Number of Clusters:"),
                dcc.Input(
                    id='hierarchical-n-clusters',
                    type='number',
                    value=3,
                    min=2
                ),
                html.Label("Linkage Method:"),
                dcc.Dropdown(
                    id='hierarchical-linkage',
                    options=[
                        {'label': 'Ward', 'value': 'ward'},
                        {'label': 'Complete', 'value': 'complete'},
                        {'label': 'Average', 'value': 'average'},
                        {'label': 'Single', 'value': 'single'}
                    ],
                    value='ward'
                )
            ], id='hierarchical-params', className="mb-4")
        ], id='clustering-params', style={'display': 'none'}),
        
        # Training button
        html.Button(
            "Train Models",
            id='train-button',
            className="btn btn-primary mb-4"
        ),
        
        # Save model button
        html.Button(
            "Save Best Model",
            id='save-model-button',
            className="btn btn-success mb-4",
            style={'display': 'none'}
        ),
        
        # Status messages
        html.Div(id='save-model-status'),
        
        # Training results
        html.Div(id='training-results')
    ]) 