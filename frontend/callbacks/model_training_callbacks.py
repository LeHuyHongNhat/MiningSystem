from dash import Input, Output, State, html, dcc, callback, no_update, callback_context
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np
from ..utils.api import train_model, get_current_data, get_available_models
import json
import logging
import requests
import dash_table

logger = logging.getLogger(__name__)

def register_model_training_callbacks(app):
    @callback(
        [Output('model-selection', 'options'),
         Output('common-params', 'style'),
         Output('clustering-params', 'style'),
         Output('kmeans-params', 'style'),
         Output('dbscan-params', 'style'),
         Output('hierarchical-params', 'style')],
        [Input('algorithm-type', 'value'),
         Input('model-selection', 'value')]
    )
    def update_model_options_and_params(algorithm_type, selected_models):
        """Update available models and parameter visibility based on algorithm type and selected models"""
        ctx = callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
        
        # Initialize default styles
        common_style = {'display': 'none'}
        clustering_style = {'display': 'none'}
        kmeans_style = {'display': 'none'}
        dbscan_style = {'display': 'none'}
        hierarchical_style = {'display': 'none'}
        model_options = []
        
        if not algorithm_type:
            return model_options, common_style, clustering_style, kmeans_style, dbscan_style, hierarchical_style
        
        # Get available models from API
        models = get_available_models()
        if not models:
            return model_options, common_style, clustering_style, kmeans_style, dbscan_style, hierarchical_style
        
        # Update model options
        if algorithm_type in models:
            model_options = []
            for model in models[algorithm_type]:
                if isinstance(model, dict):
                    # Handle case where model is a dict with name/value
                    model_options.append({
                        'label': model['name'] if 'name' in model else model['value'],
                        'value': model['value']
                    })
                else:
                    # Handle case where model is a simple string
                    model_options.append({
                        'label': model,
                        'value': model
                    })
        
        # Update parameter visibility based on algorithm type
        if algorithm_type in ['classification', 'regression']:
            common_style = {'display': 'block'}
        elif algorithm_type == 'clustering':
            clustering_style = {'display': 'block'}
            
            # Update clustering parameter visibility if models are selected
            if selected_models:
                kmeans_style = {'display': 'block'} if 'K-Means' in selected_models else {'display': 'none'}
                dbscan_style = {'display': 'block'} if 'DBSCAN' in selected_models else {'display': 'none'}
                hierarchical_style = {'display': 'block'} if 'Hierarchical Clustering' in selected_models else {'display': 'none'}
        
        return model_options, common_style, clustering_style, kmeans_style, dbscan_style, hierarchical_style

    @callback(
        [Output('training-results', 'children'),
         Output('save-model-button', 'style')],
        [Input('train-button', 'n_clicks')],
        [State('algorithm-type', 'value'),
         State('model-selection', 'value'),
         State('test-size', 'value'),
         State('random-state', 'value'),
         State('kmeans-n-clusters', 'value'),
         State('kmeans-max-iter', 'value'),
         State('dbscan-eps', 'value'),
         State('dbscan-min-samples', 'value'),
         State('hierarchical-n-clusters', 'value'),
         State('hierarchical-linkage', 'value')],
        prevent_initial_call=True
    )
    def train_model_callback(n_clicks, algorithm_type, selected_models, test_size, random_state,
                            kmeans_n_clusters, kmeans_max_iter, dbscan_eps, dbscan_min_samples,
                            hierarchical_n_clusters, hierarchical_linkage):
        if not n_clicks:
            return [], {'display': 'none'}
        
        try:
            # Prepare training configuration
            config = {
                'algorithm_type': algorithm_type,
                'selected_models': selected_models,
                'test_size': test_size,
                'random_state': random_state,
                'model_params': {}
            }
            
            # Add model-specific parameters
            if algorithm_type == 'clustering':
                if 'K-Means' in selected_models:
                    config['model_params']['K-Means'] = {
                        'n_clusters': kmeans_n_clusters,
                        'max_iter': kmeans_max_iter
                    }
                if 'DBSCAN' in selected_models:
                    config['model_params']['DBSCAN'] = {
                        'eps': dbscan_eps,
                        'min_samples': dbscan_min_samples
                    }
                if 'Hierarchical Clustering' in selected_models:
                    config['model_params']['Hierarchical Clustering'] = {
                        'n_clusters': hierarchical_n_clusters,
                        'linkage': hierarchical_linkage
                    }
                
            # Train models
            response = train_model(config)
            if not response or 'data' not in response:
                raise ValueError("Invalid response from training API")
                
            results = response['data']
            visualization_data = results.get('visualization_data', {})
            
            # Create visualization components based on algorithm type
            visualization_components = []
            
            if algorithm_type == 'clustering':
                for model_name in selected_models:
                    if model_name in visualization_data.get('labels', {}):
                        # Create silhouette plot
                        silhouette_plot = dcc.Graph(
                            figure={
                                'data': [{
                                    'type': 'scatter',
                                    'x': visualization_data['silhouette_samples'].get(model_name, []),
                                    'y': list(range(len(visualization_data['silhouette_samples'].get(model_name, [])))),
                                    'mode': 'markers',
                                    'marker': {
                                        'color': visualization_data['labels'].get(model_name, []),
                                        'colorscale': 'Viridis'
                                    }
                                }],
                                'layout': {
                                    'title': f'Silhouette Plot - {model_name}',
                                    'xaxis': {'title': 'Silhouette Score'},
                                    'yaxis': {'title': 'Sample Index'}
                                }
                            }
                        )
                        
                        # Create cluster visualization (2D scatter plot)
                        if model_name == 'K-Means' and model_name in visualization_data.get('centroids', {}):
                            cluster_plot = dcc.Graph(
                                figure={
                                    'data': [
                                        {
                                            'type': 'scatter',
                                            'x': [x[0] for x in visualization_data['centroids'][model_name]],
                                            'y': [x[1] for x in visualization_data['centroids'][model_name]],
                                            'mode': 'markers',
                                            'marker': {
                                                'size': 15,
                                                'color': 'red',
                                                'symbol': 'star'
                                            },
                                            'name': 'Centroids'
                                        }
                                    ],
                                    'layout': {
                                        'title': f'Cluster Centers - {model_name}',
                                        'xaxis': {'title': 'Feature 1'},
                                        'yaxis': {'title': 'Feature 2'}
                                    }
                                }
                            )
                            visualization_components.append(cluster_plot)
                            
                        visualization_components.append(silhouette_plot)
                        
            elif algorithm_type == 'classification':
                for model_name in selected_models:
                    if model_name in visualization_data.get('confusion_matrix', {}):
                        # Create confusion matrix heatmap
                        confusion_matrix = visualization_data['confusion_matrix'][model_name]
                        confusion_plot = dcc.Graph(
                            figure={
                                'data': [{
                                    'type': 'heatmap',
                                    'z': confusion_matrix,
                                    'colorscale': 'Viridis'
                                }],
                                'layout': {
                                    'title': f'Confusion Matrix - {model_name}',
                                    'xaxis': {'title': 'Predicted'},
                                    'yaxis': {'title': 'Actual'}
                                }
                            }
                        )
                        
                        # Create feature importance plot
                        if model_name in visualization_data.get('feature_importance', {}):
                            importance = visualization_data['feature_importance'][model_name]
                            importance_plot = dcc.Graph(
                                figure={
                                    'data': [{
                                        'type': 'bar',
                                        'x': list(importance.keys()),
                                        'y': list(importance.values())
                                    }],
                                    'layout': {
                                        'title': f'Feature Importance - {model_name}',
                                        'xaxis': {'title': 'Features'},
                                        'yaxis': {'title': 'Importance'}
                                    }
                                }
                            )
                            visualization_components.append(importance_plot)
                            
                        visualization_components.append(confusion_plot)
                        
            elif algorithm_type == 'regression':
                for model_name in selected_models:
                    if model_name in visualization_data.get('predictions', {}):
                        # Create prediction vs actual plot
                        prediction_plot = dcc.Graph(
                            figure={
                                'data': [
                                    {
                                        'type': 'scatter',
                                        'x': visualization_data['actual_values'][model_name],
                                        'y': visualization_data['predictions'][model_name],
                                        'mode': 'markers',
                                        'name': 'Predictions'
                                    },
                                    {
                                        'type': 'scatter',
                                        'x': visualization_data['actual_values'][model_name],
                                        'y': visualization_data['actual_values'][model_name],
                                        'mode': 'lines',
                                        'name': 'Perfect Prediction',
                                        'line': {'color': 'red', 'dash': 'dash'}
                                    }
                                ],
                                'layout': {
                                    'title': f'Predictions vs Actual - {model_name}',
                                    'xaxis': {'title': 'Actual Values'},
                                    'yaxis': {'title': 'Predicted Values'}
                                }
                            }
                        )
                        
                        # Create feature importance plot
                        if model_name in visualization_data.get('feature_importance', {}):
                            importance = visualization_data['feature_importance'][model_name]
                            importance_plot = dcc.Graph(
                                figure={
                                    'data': [{
                                        'type': 'bar',
                                        'x': list(importance.keys()),
                                        'y': list(importance.values())
                                    }],
                                    'layout': {
                                        'title': f'Feature Importance - {model_name}',
                                        'xaxis': {'title': 'Features'},
                                        'yaxis': {'title': 'Importance'}
                                    }
                                }
                            )
                            visualization_components.append(importance_plot)
                            
                        visualization_components.append(prediction_plot)
                        
            # Create metrics table
            metrics_table = html.Div([
                html.H4('Training Metrics'),
                dash_table.DataTable(
                    data=[{
                        'Model': model['name'],
                        **model['metrics']
                    } for model in results['all_models']],
                    columns=[{'name': col, 'id': col} for col in ['Model'] + list(results['all_models'][0]['metrics'].keys())],
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'textAlign': 'left',
                        'padding': '10px',
                        'whiteSpace': 'normal',
                        'height': 'auto'
                    }
                )
            ])
            
            # Combine all components
            result_components = [
                html.Div([
                    html.H3('Training Results'),
                    html.H4(f"Best Model: {results['best_model']['name']}"),
                    metrics_table,
                    html.Div(visualization_components, style={'marginTop': '20px'})
                ])
            ]
            
            return result_components, {'display': 'block'}
            
        except Exception as e:
            logger.error(f"Error in train_model_callback: {str(e)}")
            return [html.Div(f"Error: {str(e)}", style={'color': 'red'})], {'display': 'none'}

    @callback(
        Output('save-model-status', 'children'),
        [Input('save-model-button', 'n_clicks')],
        prevent_initial_call=True
    )
    def save_model_callback(n_clicks):
        if not n_clicks:
            return ""
        
        try:
            # Call API to save model
            response = requests.post('http://127.0.0.1:8000/api/model/save')
            response.raise_for_status()
            
            return html.Div("Model saved successfully!", style={'color': 'green'})
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return html.Div(f"Error saving model: {str(e)}", style={'color': 'red'}) 