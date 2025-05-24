from dash import Input, Output, State, callback, callback_context
import dash_bootstrap_components as dbc
from dash import html, dcc
import logging
from typing import List, Dict
import pandas as pd

from frontend.utils.api import (
    set_target_column,
    drop_features,
    get_categorical_features,
    process_categorical_features,
    process_features,
    scale_features,
    get_final_info,
    get_current_data
)

logger = logging.getLogger(__name__)

def register_preprocessing_callbacks(app):
    @app.callback(
        Output('target-status', 'children'),
        [Input('set-target-button', 'n_clicks')],
        [State('target-column', 'value')]
    )
    def handle_set_target(n_clicks, target_column):
        """Handle setting target column"""
        if n_clicks is None or not target_column:
            return None
            
        try:
            logger.info(f"Setting target column: {target_column}")
            result = set_target_column(target_column)
            logger.info(f"Target column set result: {result}")
            
            if not result or 'remaining_features' not in result:
                return html.Div([
                    html.H5("Error setting target column!"),
                    html.P("Invalid response from server")
                ], className="alert alert-danger")
                
            return html.Div([
                html.H5("Target column set successfully!"),
                html.P(f"Target column: {target_column}"),
                html.P(f"Remaining features: {', '.join(result['remaining_features'])}")
            ], className="alert alert-success")
        except Exception as e:
            logger.error(f"Error setting target column: {str(e)}")
            return html.Div([
                html.H5("Error setting target column!"),
                html.P(str(e))
            ], className="alert alert-danger")

    @app.callback(
        [Output('drop-features-status', 'children'),
         Output('drop-features', 'options'),
         Output('drop-features', 'value'),
         Output('target-column', 'options', allow_duplicate=True),
         Output('categorical-features', 'options', allow_duplicate=True),
         Output('categorical-features', 'value', allow_duplicate=True),
         Output('preprocessing-features', 'options', allow_duplicate=True),
         Output('preprocessing-features', 'value', allow_duplicate=True),
         Output('data-preview', 'children')],
        [Input('drop-features-button', 'n_clicks'),
         Input('process-categorical-button', 'n_clicks'),
         Input('set-target-button', 'n_clicks'),
         Input('upload-data', 'contents')],
        [State('drop-features', 'value'),
         State('target-column', 'value'),
         State('upload-data', 'filename')],
        prevent_initial_call='initial_duplicate'
    )
    def handle_features_update(drop_clicks, process_clicks, target_clicks, contents, features_to_drop, target_column, filename):
        """Handle updating features list and dropping features"""
        ctx = callback_context
        if not ctx.triggered:
            return [None, [], [], [], [], [], [], None, None]
            
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Get current data preview
        try:
            current_data = get_current_data()
            if current_data and 'data' in current_data:
                preview_df = pd.DataFrame(current_data['data'])
                preview_table = dbc.Table.from_dataframe(
                    preview_df.head(),
                    striped=True,
                    bordered=True,
                    hover=True,
                    responsive=True
                )
                # Get current features for preprocessing
                current_features = preview_df.columns.tolist()
                feature_options = [{"label": feature, "value": feature} for feature in current_features]
            else:
                preview_table = None
                feature_options = []
        except Exception as e:
            logger.error(f"Error getting data preview: {str(e)}")
            preview_table = None
            feature_options = []
        
        # Handle file upload
        if trigger_id == 'upload-data':
            return [None, feature_options, [], feature_options, feature_options, [], feature_options, None, preview_table]
        
        # Handle drop features
        if trigger_id == 'drop-features-button':
            if not features_to_drop:
                return [None, feature_options, [], feature_options, feature_options, [], feature_options, None, preview_table]
            try:
                logger.info(f"Dropping features: {features_to_drop}")
                result = drop_features(features_to_drop)
                logger.info(f"Drop features result: {result}")
                
                if not result or 'remaining_features' not in result:
                    return [
                        html.Div([
                            html.H5("Error dropping features!"),
                            html.P("Invalid response from server")
                        ], className="alert alert-danger"),
                        [], [], [], [], [], [], None, preview_table
                    ]
                
                # Create options for all dropdowns, excluding target column
                remaining_features = [f for f in result['remaining_features'] if f != target_column]
                feature_options = [{"label": feature, "value": feature} for feature in remaining_features]
                
                # Get updated data preview after dropping features
                try:
                    updated_data = get_current_data()
                    if updated_data and 'data' in updated_data:
                        updated_df = pd.DataFrame(updated_data['data'])
                        updated_preview = dbc.Table.from_dataframe(
                            updated_df.head(),
                            striped=True,
                            bordered=True,
                            hover=True,
                            responsive=True
                        )
                    else:
                        updated_preview = preview_table
                except Exception as e:
                    logger.error(f"Error getting updated preview: {str(e)}")
                    updated_preview = preview_table
                
                return [
                    html.Div([
                        html.H5("Features dropped successfully!"),
                        html.P(f"Dropped features: {', '.join(features_to_drop)}"),
                        html.P(f"Remaining features: {', '.join(remaining_features)}")
                    ], className="alert alert-success"),
                    feature_options,  # drop-features options
                    [],  # drop-features value (clear selection)
                    feature_options,  # target-column options
                    feature_options,  # categorical-features options
                    [],  # categorical-features value
                    feature_options,  # preprocessing-features options
                    None,  # preprocessing-features value
                    updated_preview  # Updated data preview
                ]
            except Exception as e:
                logger.error(f"Error dropping features: {str(e)}")
                return [
                    html.Div([
                        html.H5("Error dropping features!"),
                        html.P(str(e))
                    ], className="alert alert-danger"),
                    [], [], [], [], [], [], None, preview_table
                ]
        
        # Handle process categorical
        if trigger_id == 'process-categorical-button':
            try:
                # Get current data to get all features
                current_data = get_current_data()
                if current_data and 'data' in current_data:
                    df = pd.DataFrame(current_data['data'])
                    # Get all features except target column
                    available_features = [f for f in df.columns if f != target_column]
                    feature_options = [{"label": feature, "value": feature} for feature in available_features]
                    return [None, feature_options, [], feature_options, feature_options, [], feature_options, None, preview_table]
                else:
                    return [None, [], [], [], [], [], [], None, preview_table]
            except Exception as e:
                logger.error(f"Error getting features after categorical processing: {str(e)}")
                return [None, [], [], [], [], [], [], None, preview_table]
        
        # Handle target column change
        if trigger_id == 'set-target-button':
            try:
                # Get all features from current data
                current_data = get_current_data()
                if current_data and 'data' in current_data:
                    df = pd.DataFrame(current_data['data'])
                    # Filter out target column
                    available_features = [f for f in df.columns if f != target_column]
                    feature_options = [{"label": feature, "value": feature} for feature in available_features]
                    return [None, feature_options, [], feature_options, feature_options, [], feature_options, None, preview_table]
                else:
                    return [None, [], [], [], [], [], [], None, preview_table]
            except Exception as e:
                logger.error(f"Error updating features after target change: {str(e)}")
                return [None, [], [], [], [], [], [], None, preview_table]
        
        return [None, feature_options, [], feature_options, feature_options, [], feature_options, None, preview_table]

    @app.callback(
        Output('encoding-method-container', 'style'),
        [Input('categorical-handling', 'value')]
    )
    def toggle_encoding_method(handling_method):
        """Show/hide encoding method container based on handling method"""
        if handling_method == 'encode':
            return {'display': 'block'}
        return {'display': 'none'}

    @app.callback(
        [Output('categorical-status', 'children'),
         Output('data-preview', 'children', allow_duplicate=True),
         Output('preprocessing-features', 'options', allow_duplicate=True),
         Output('preprocessing-features', 'value', allow_duplicate=True)],
        [Input('process-categorical-button', 'n_clicks')],
        [State('categorical-features', 'value'),
         State('categorical-handling', 'value'),
         State('encoding-method', 'value')],
        prevent_initial_call='initial_duplicate'
    )
    def handle_process_categorical(n_clicks, features, handling_method, encoding_method):
        """Handle processing categorical features"""
        if n_clicks is None or not features or not handling_method:
            return None, None, [], None
            
        try:
            # Create config based on handling method
            if handling_method == 'encode':
                if not encoding_method:
                    return html.Div([
                        html.H5("Error processing categorical features!"),
                        html.P("Please select an encoding method")
                    ], className="alert alert-danger"), None, [], None
                config = {feature: encoding_method for feature in features}
            else:
                config = {feature: handling_method for feature in features}
                
            result = process_categorical_features(config)
            
            # Get updated data preview
            try:
                current_data = get_current_data()
                if current_data and 'data' in current_data:
                    preview_df = pd.DataFrame(current_data['data'])
                    preview_table = dbc.Table.from_dataframe(
                        preview_df.head(),
                        striped=True,
                        bordered=True,
                        hover=True,
                        responsive=True
                    )
                    # Create options for preprocessing features from remaining features
                    remaining_features = result.get('remaining_features', [])
                    logger.info(f"Updating preprocessing features with: {remaining_features}")
                    feature_options = [{"label": feature, "value": feature} for feature in remaining_features]
                else:
                    preview_table = None
                    feature_options = []
            except Exception as e:
                logger.error(f"Error getting updated preview: {str(e)}")
                preview_table = None
                feature_options = []
            
            # Create success message with processing details
            return html.Div([
                html.H5("Categorical features processed successfully!"),
                html.P("Processed features:"),
                html.Ul([html.Li(feature) for feature in result['processed_features']]),
                html.P(f"Remaining features: {', '.join(result['remaining_features'])}")
            ], className="alert alert-success"), preview_table, feature_options, None
            
        except Exception as e:
            logger.error(f"Error processing categorical features: {str(e)}")
            return html.Div([
                html.H5("Error processing categorical features!"),
                html.P(str(e))
            ], className="alert alert-danger"), None, [], None

    @app.callback(
        [Output('features-status', 'children'),
         Output('data-preview', 'children', allow_duplicate=True)],
        [Input('process-features-button', 'n_clicks')],
        [State('preprocessing-features', 'value'),
         State('missing-value-handling', 'value'),
         State('outlier-handling', 'value')],
        prevent_initial_call='initial_duplicate'
    )
    def handle_process_features(n_clicks, features, missing_method, outlier_method):
        """Handle processing features"""
        if n_clicks is None or not features:
            return None, None
            
        try:
            # Create config for each feature
            config = {}
            for feature in features:
                feature_config = {}
                if missing_method:
                    feature_config["missing"] = missing_method
                if outlier_method:
                    feature_config["outlier"] = outlier_method
                if feature_config:  # Only add feature if it has any methods
                    config[feature] = feature_config
            
            if not config:  # If no valid config was created
                return html.Div([
                    html.H5("Error processing features!"),
                    html.P("Please select at least one processing method (missing values or outliers)")
                ], className="alert alert-danger"), None
                
            logger.info(f"Processing features with config: {config}")
            result = process_features(config)
            
            # Get updated data preview
            try:
                current_data = get_current_data()
                if current_data and 'data' in current_data:
                    preview_df = pd.DataFrame(current_data['data'])
                    preview_table = dbc.Table.from_dataframe(
                        preview_df.head(),
                        striped=True,
                        bordered=True,
                        hover=True,
                        responsive=True
                    )
                else:
                    preview_table = None
            except Exception as e:
                logger.error(f"Error getting updated preview: {str(e)}")
                preview_table = None
            
            # Create success message with processing details
            return html.Div([
                html.H5("Features processed successfully!"),
                html.P("Processed features:"),
                html.Ul([
                    html.Li([
                        html.Strong(feature['feature']),
                        html.Ul([html.Li(method) for method in feature['methods']])
                    ]) for feature in result['processed_features']
                ]),
                html.P(f"Remaining features: {', '.join(result['remaining_features'])}")
            ], className="alert alert-success"), preview_table
            
        except Exception as e:
            logger.error(f"Error processing features: {str(e)}")
            return html.Div([
                html.H5("Error processing features!"),
                html.P(str(e))
            ], className="alert alert-danger"), None

    @app.callback(
        [Output('scaling-status', 'children'),
         Output('data-preview', 'children', allow_duplicate=True)],
        [Input('apply-scaling-button', 'n_clicks')],
        [State('global-scaling-method', 'value')],
        prevent_initial_call='initial_duplicate'
    )
    def handle_scale_features(n_clicks, method):
        """Handle scaling features"""
        if n_clicks is None or not method:
            return None, None
            
        try:
            # The method names from the dropdown already match the API expectations
            # No need for mapping since the dropdown values are 'standard', 'minmax', 'robust'
            logger.info(f"Applying {method} scaling to all numerical features")
            result = scale_features(method)
            logger.info(f"Scaling result: {result}")
            
            # Get updated data preview
            try:
                current_data = get_current_data()
                if current_data and 'data' in current_data:
                    preview_df = pd.DataFrame(current_data['data'])
                    preview_table = dbc.Table.from_dataframe(
                        preview_df.head(),
                        striped=True,
                        bordered=True,
                        hover=True,
                        responsive=True
                    )
                else:
                    preview_table = None
            except Exception as e:
                logger.error(f"Error getting updated preview: {str(e)}")
                preview_table = None
            
            return html.Div([
                html.H5("Features scaled successfully!"),
                html.P(f"Applied {method} scaling to all numerical features"),
                html.P("Scaled features:"),
                html.Ul([html.Li(feature) for feature in result.get('scaled_features', [])])
            ], className="alert alert-success"), preview_table
            
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            return html.Div([
                html.H5("Error scaling features!"),
                html.P(str(e))
            ], className="alert alert-danger"), None

    @app.callback(
        [Output('preprocess-status', 'children'),
         Output('preprocessed-data-info', 'children')],
        [Input('final-preprocess-button', 'n_clicks')]
    )
    def handle_final_preprocess(n_clicks):
        """Handle final preprocessing info"""
        if n_clicks is None:
            return None, None
            
        try:
            # Get final preprocessing info
            info = get_final_info()
            
            # Create info cards
            cards = [
                dbc.Card([
                    dbc.CardHeader("Dataset Information"),
                    dbc.CardBody([
                        html.P(f"Number of rows: {info['rows']}"),
                        html.P(f"Number of columns: {info['columns']}"),
                        html.P(f"Memory usage: {info['memory_usage']}")
                    ])
                ], className="mb-3"),
                
                dbc.Card([
                    dbc.CardHeader("Column Information"),
                    dbc.CardBody([
                        html.Table([
                            html.Thead([
                                html.Tr([
                                    html.Th("Column"),
                                    html.Th("Type"),
                                    html.Th("Missing Values")
                                ])
                            ]),
                            html.Tbody([
                                html.Tr([
                                    html.Td(col),
                                    html.Td(info['data_types'][col]),
                                    html.Td(info['missing_values'][col])
                                ])
                                for col in info['column_names']
                            ])
                        ], className="table table-striped")
                    ])
                ], className="mb-3"),
                
                dbc.Card([
                    dbc.CardHeader("Numerical Features Statistics"),
                    dbc.CardBody([
                        html.Table([
                            html.Thead([
                                html.Tr([
                                    html.Th("Feature"),
                                    html.Th("Mean"),
                                    html.Th("Std"),
                                    html.Th("Min"),
                                    html.Th("Max")
                                ])
                            ]),
                            html.Tbody([
                                html.Tr([
                                    html.Td(feature),
                                    html.Td(f"{stats['mean']:.2f}"),
                                    html.Td(f"{stats['std']:.2f}"),
                                    html.Td(f"{stats['min']:.2f}"),
                                    html.Td(f"{stats['max']:.2f}")
                                ])
                                for feature, stats in info['numerical_stats'].items()
                            ])
                        ], className="table table-striped")
                    ])
                ])
            ]
            
            return html.Div([
                html.H5("Success!", className="text-success"),
                html.P("Final preprocessing information retrieved successfully.")
            ]), html.Div(cards)
            
        except Exception as e:
            logger.error(f"Error getting final preprocessing info: {str(e)}")
            return html.Div([
                html.H5("Error!", className="text-danger"),
                html.P(str(e))
            ]), None

    @app.callback(
        [Output('preprocessing-features', 'options'),
         Output('preprocessing-features', 'value')],
        [Input('upload-data', 'contents')],
        [State('upload-data', 'filename')]
    )
    def update_preprocessing_features(contents, filename):
        """Update preprocessing features dropdown when data is loaded"""
        if contents is None:
            return [], None
        
        try:
            # Get current data
            current_data = get_current_data()
            if current_data and 'data' in current_data:
                df = pd.DataFrame(current_data['data'])
                # Create options for all features
                options = [{"label": col, "value": col} for col in df.columns]
                return options, None
            return [], None
        except Exception as e:
            logger.error(f"Error updating preprocessing features: {str(e)}")
            return [], None 