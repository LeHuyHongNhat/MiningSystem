from dash import html, dcc
import dash_bootstrap_components as dbc

def create_training_section():
    """Create the training section for model training"""
    return [
        html.H3("Model Training", className="mb-3"),
        
        # Algorithm Type Section
        html.H5("Algorithm Type", className="mb-3"),
        dbc.Row([
            dbc.Col([
                html.Label("Select Algorithm Type:"),
                dcc.Dropdown(
                    id="algorithm-type",
                    options=[
                        {"label": "Classification", "value": "classification"},
                        {"label": "Regression", "value": "regression"},
                        {"label": "Clustering", "value": "clustering"}
                    ],
                    value="classification"
                )
            ])
        ], className="mb-4"),
        
        # Model Selection Section
        html.H5("Model Selection", className="mb-3"),
        dbc.Row([
            dbc.Col([
                html.Label("Select Models:"),
                dcc.Dropdown(
                    id="model-selection",
                    options=[],
                    multi=True
                )
            ])
        ], className="mb-4"),
        
        # Training Parameters Section
        html.H5("Training Parameters", className="mb-3"),
        dbc.Row([
            dbc.Col([
                html.Label("Test Size:"),
                dcc.Slider(
                    id="test-size",
                    min=0.1,
                    max=0.5,
                    step=0.05,
                    marks={i/10: str(i/10) for i in range(1, 6)},
                    value=0.2
                )
            ]),
            dbc.Col([
                html.Label("Random State:"),
                dcc.Input(
                    id="random-state",
                    type="number",
                    value=42
                )
            ])
        ], className="mb-4"),
        
        # Hyperparameter Optimization Section
        html.H5("Hyperparameter Optimization", className="mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Checkbox(
                    id="optimize-hyperparameters",
                    label="Enable Hyperparameter Optimization",
                    value=False
                )
            ])
        ], className="mb-4"),
        
        # Train Button and Results
        dbc.Button("Train Models", id="train-button", color="primary", className="mt-3"),
        html.Div(id="training-status"),
        html.Div(id="training-results")
    ] 