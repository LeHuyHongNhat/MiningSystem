from dash import Input, Output, State, callback
import dash_bootstrap_components as dbc
from dash import html
import pandas as pd
import logging

from frontend.utils.api import upload_file, get_data_info, parse_contents

logger = logging.getLogger(__name__)

def register_upload_callbacks(app):
    @app.callback(
        [Output('upload-status', 'children'),
         Output('data-info-content', 'children'),
         Output('plot-type-selector', 'options'),
         Output('plot-type-selector', 'value'),
         Output('feature-selector-1', 'options'),
         Output('feature-selector-1', 'value'),
         Output('feature-selector-2', 'options'),
         Output('feature-selector-2', 'value'),
         Output('feature-selector-3', 'options'),
         Output('feature-selector-3', 'value'),
         Output('target-column', 'options'),
         Output('preprocessing-features', 'options', allow_duplicate=True),
         Output('preprocessing-features', 'value', allow_duplicate=True),
         Output('drop-features', 'options', allow_duplicate=True)],
        [Input('upload-data', 'contents')],
        [State('upload-data', 'filename'),
         State('preprocessing-features', 'value')],
        prevent_initial_call='initial_duplicate'
    )
    def update_output(contents, filename, current_preprocessing_values):
        """Handle file upload and update UI"""
        if contents is None:
            # Return empty options for all dropdowns
            empty_options = [{"label": "No data available", "value": ""}]
            return [
                html.Div([
                    html.H5("No file uploaded"),
                    html.P("Please upload a data file to begin.")
                ]),
                None,  # data-info-content
                empty_options,  # plot-type-selector options
                "",  # plot-type-selector value
                empty_options,  # feature-selector-1 options
                "",  # feature-selector-1 value
                empty_options,  # feature-selector-2 options
                "",  # feature-selector-2 value
                empty_options,  # feature-selector-3 options
                "",  # feature-selector-3 value
                empty_options,  # target-column options
                empty_options,  # preprocessing-features options
                [],  # preprocessing-features value
                empty_options  # drop-features options
            ]
            
        try:
            # Parse the uploaded file
            df = parse_contents(contents, filename)
            if df is None or df.empty:
                raise ValueError("Empty or invalid data file")
            
            # Upload to backend
            upload_result = upload_file(contents, filename)
            safe_filename = upload_result.get('filename')
            
            # Get data info from backend
            data_info = get_data_info(safe_filename)
            if not data_info:
                raise ValueError("Failed to get data information")
            
            # Create feature options
            feature_options = [{"label": col, "value": col} for col in df.columns]
            if not feature_options:
                raise ValueError("No features found in the data")
            
            # Create plot type options
            plot_options = [
                {"label": "Correlation Heatmap", "value": "heatmap"},
                {"label": "Distribution Plot", "value": "distribution"},
                {"label": "Box Plot", "value": "box"},
                {"label": "Violin Plot", "value": "violin"},
                {"label": "Scatter Plot (2D)", "value": "scatter_2d"},
                {"label": "Scatter Plot (3D)", "value": "scatter_3d"},
                {"label": "Bar Plot", "value": "bar"},
                {"label": "Pie Chart", "value": "pie"},
                {"label": "Line Plot", "value": "line"}
            ]
            
            # Create data info display
            info_content = create_data_info_display(data_info, df)
            
            return [
                html.Div([
                    html.H5("File uploaded successfully!"),
                    html.P(f"Filename: {filename}")
                ]),
                info_content,
                plot_options,
                "",  # plot-type-selector value
                feature_options,
                "",  # feature-selector-1 value
                feature_options,
                "",  # feature-selector-2 value
                feature_options,
                "",  # feature-selector-3 value
                feature_options,
                feature_options,  # preprocessing-features options
                current_preprocessing_values or [],  # preprocessing-features value
                feature_options  # drop-features options
            ]
                    
        except Exception as e:
            logger.error(f"Error in file upload: {str(e)}")
            # Return error message and empty options
            empty_options = [{"label": "No data available", "value": ""}]
            return [
                html.Div([
                    html.H5("Error uploading file!"),
                    html.P(str(e))
                ]),
                None,  # data-info-content
                empty_options,  # plot-type-selector options
                "",  # plot-type-selector value
                empty_options,  # feature-selector-1 options
                "",  # feature-selector-1 value
                empty_options,  # feature-selector-2 options
                "",  # feature-selector-2 value
                empty_options,  # feature-selector-3 options
                "",  # feature-selector-3 value
                empty_options,  # target-column options
                empty_options,  # preprocessing-features options
                [],  # preprocessing-features value
                empty_options  # drop-features options
            ]

def create_data_info_display(data_info, df):
    """Create the data information display"""
    return [
        html.H4("Dataset Overview", className="mb-4"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Total Rows", className="card-title text-center"),
                        html.H3(f"{data_info['rows']:,}", className="text-center text-primary")
                    ])
                ], className="mb-3")
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Total Columns", className="card-title text-center"),
                        html.H3(f"{data_info['columns']}", className="text-center text-primary")
                    ])
                ], className="mb-3")
            ], width=6)
        ], className="mb-4"),
        
        html.H4("Data Preview (First 10 Rows)", className="mb-3"),
        dbc.Card([
            dbc.CardBody([
                dbc.Table.from_dataframe(
                    df.head(10),
                    striped=True,
                    bordered=True,
                    hover=True,
                    responsive=True,
                    className="table table-striped table-hover"
                )
            ])
        ], className="mb-4"),
        
        html.H4("Column Information", className="mb-3"),
        dbc.Table([
            html.Thead(
                html.Tr([
                    html.Th("Column Name"),
                    html.Th("Data Type"),
                    html.Th("Non-Null Count"),
                    html.Th("Null Count"),
                    html.Th("Null Percentage")
                ])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(col),
                    html.Td(dtype),
                    html.Td(f"{data_info['non_null_counts'][col]:,}"),
                    html.Td(f"{data_info['null_counts'][col]:,}"),
                    html.Td(f"{(data_info['null_counts'][col] / data_info['rows'] * 100):.2f}%")
                ]) for col, dtype in data_info['column_types'].items()
            ])
        ], bordered=True, hover=True, responsive=True, striped=True, className="mb-4"),
        
        html.H4("Statistical Information for Numerical Features", className="mb-3"),
        dbc.Table([
            html.Thead(
                html.Tr([
                    html.Th("Feature"),
                    html.Th("Count"),
                    html.Th("Mean"),
                    html.Th("Std"),
                    html.Th("Min"),
                    html.Th("25%"),
                    html.Th("50%"),
                    html.Th("75%"),
                    html.Th("Max")
                ])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(feature),
                    html.Td(f"{stats['count']:,.0f}"),
                    html.Td(f"{stats['mean']:,.2f}"),
                    html.Td(f"{stats['std']:,.2f}"),
                    html.Td(f"{stats['min']:,.2f}"),
                    html.Td(f"{stats['25%']:,.2f}"),
                    html.Td(f"{stats['50%']:,.2f}"),
                    html.Td(f"{stats['75%']:,.2f}"),
                    html.Td(f"{stats['max']:,.2f}")
                ]) for feature, stats in data_info['numerical_stats'].items()
            ])
        ], bordered=True, hover=True, responsive=True, striped=True, className="mb-4"),
        
        html.H4("Data Quality Summary", className="mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Complete Columns", className="card-title text-center"),
                        html.H3(
                            f"{sum(1 for count in data_info['null_counts'].values() if count == 0)}",
                            className="text-center text-success"
                        )
                    ])
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Columns with Missing Values", className="card-title text-center"),
                        html.H3(
                            f"{sum(1 for count in data_info['null_counts'].values() if count > 0)}",
                            className="text-center text-warning"
                        )
                    ])
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Total Missing Values", className="card-title text-center"),
                        html.H3(
                            f"{sum(data_info['null_counts'].values()):,}",
                            className="text-center text-danger"
                        )
                    ])
                ])
            ], width=4)
        ])
    ] 