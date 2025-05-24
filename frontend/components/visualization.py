import dash_bootstrap_components as dbc
from dash import html, dcc

def create_visualization_section():
    """Create the data visualization section"""
    return [
        html.H3("Data Visualization", className="mb-3"),
        dbc.Row([
            dbc.Col([
                html.H5("Available Visualizations", className="mb-3"),
                dcc.Dropdown(
                    id='visualization-selector',
                    options=[],
                    value=None,
                    clearable=True
                ),
                html.Div(id='visualization-display')
            ])
        ])
    ] 