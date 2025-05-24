import dash_bootstrap_components as dbc
from dash import html, dcc

def create_upload_section():
    """Create the file upload section"""
    return [
        # File Upload Section
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
        ),
        
        # Data Information Section
        html.H3("Data Information", className="mb-3 mt-4"),
        html.Div(id='data-info-content')
    ] 