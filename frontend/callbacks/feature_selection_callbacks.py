from dash import Input, Output, State, callback
import dash_bootstrap_components as dbc
from dash import html
import pandas as pd
import logging

from frontend.utils.api import select_features, get_current_data

logger = logging.getLogger(__name__)

def register_feature_selection_callbacks(app):
    @app.callback(
        Output('feature-selection-status', 'children'),
        [Input('select-features-button', 'n_clicks')],
        [State('target-column', 'value'),
         State('feature-selection-method', 'value'),
         State('feature-selection-threshold', 'value')]
    )
    def select_features_callback(n_clicks, target_column, method, threshold):
        """Handle feature selection process"""
        if n_clicks is None:
            return None
            
        if not target_column:
            return html.Div([
                html.H5("Warning!", className="text-warning"),
                html.P("Please select a target column first.")
            ])
            
        if not method:
            return html.Div([
                html.H5("Warning!", className="text-warning"),
                html.P("Please select a feature selection method.")
            ])
            
        try:
            # Get current data info
            data_info = get_current_data()
            if not data_info:
                return html.Div([
                    html.H5("Error!", className="text-danger"),
                    html.P("No data available.")
                ])
                
            # Create config for feature selection
            config = {
                'step_type': 'feature_selection',
                'target_column': target_column,
                'method': method,
                'threshold': threshold
            }
            
            # Call feature selection API
            result = select_features(config)
            
            # Show selected features
            return html.Div([
                html.H5("Success!", className="text-success"),
                html.P(f"Selected {len(result['selected_features'])} features using {method} method"),
                html.H6("Selected Features:", className="mt-3"),
                html.Ul([html.Li(feature) for feature in result['selected_features']]),
                html.H6("Feature Importance Scores:", className="mt-3"),
                dbc.Table.from_dataframe(
                    pd.DataFrame(result['feature_importance']),
                    striped=True,
                    bordered=True,
                    hover=True,
                    responsive=True
                )
            ])
        except Exception as e:
            logger.error(f"Error selecting features: {str(e)}")
            return html.Div([
                html.H5("Error!", className="text-danger"),
                html.P(str(e))
            ]) 