from dash import Input, Output, State, callback
import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.graph_objects as go
import json
import logging

from frontend.utils.api import create_visualization

logger = logging.getLogger(__name__)

def register_visualization_callbacks(app):
    @app.callback(
        [Output('feature-selector-2', 'style'),
         Output('feature-selector-3', 'style')],
        [Input('plot-type-selector', 'value')]
    )
    def update_feature_selectors_visibility(plot_type):
        """Show/hide feature selectors based on plot type"""
        if plot_type in ['scatter_3d']:
            return {'display': 'block'}, {'display': 'block'}
        elif plot_type in ['scatter_2d', 'box', 'violin', 'line']:
            return {'display': 'block'}, {'display': 'block'}
        else:
            return {'display': 'none'}, {'display': 'none'}

    @app.callback(
        [Output('feature-selector-2', 'placeholder'),
         Output('feature-selector-3', 'placeholder')],
        [Input('plot-type-selector', 'value')]
    )
    def update_feature_selectors_placeholders(plot_type):
        """Update feature selector placeholders based on plot type"""
        if plot_type == 'line':
            return 'Select Y-axis feature', 'Select animation feature (optional)'
        elif plot_type in ['scatter_2d', 'box', 'violin']:
            return 'Select second feature', 'Select third feature (optional)'
        elif plot_type == 'scatter_3d':
            return 'Select Y-axis feature', 'Select Z-axis feature'
        else:
            return 'Select feature', 'Select feature'

    @app.callback(
        Output('visualization-display', 'children'),
        [Input('show-visualization-button', 'n_clicks')],
        [State('plot-type-selector', 'value'),
         State('feature-selector-1', 'value'),
         State('feature-selector-2', 'value'),
         State('feature-selector-3', 'value'),
         State('upload-data', 'contents'),
         State('upload-data', 'filename')]
    )
    def update_visualization(n_clicks, plot_type, feature1, feature2, feature3, contents, filename):
        """Update visualization based on selected features and plot type"""
        if n_clicks is None:
            return None
            
        # Check if required fields are selected
        if not plot_type:
            return html.Div([
                html.H5("Please select a plot type!"),
                html.P("You need to select a plot type first.")
            ])
            
        if not contents or not filename:
            return html.Div([
                html.H5("Please upload a file!"),
                html.P("You need to upload a data file first.")
            ])
            
        # Define required features for each plot type
        plot_requirements = {
            'heatmap': {
                'required': 0,
                'optional': 0,
                'description': 'No features required'
            },
            'distribution': {
                'required': 1,
                'optional': 0,
                'description': 'One feature required'
            },
            'box': {
                'required': 1,
                'optional': 1,
                'description': 'One feature required, one optional for grouping'
            },
            'violin': {
                'required': 1,
                'optional': 1,
                'description': 'One feature required, one optional for grouping'
            },
            'scatter_2d': {
                'required': 2,
                'optional': 0,
                'description': 'Two features required (X and Y axes)'
            },
            'scatter_3d': {
                'required': 3,
                'optional': 0,
                'description': 'Three features required (X, Y, and Z axes)'
            },
            'bar': {
                'required': 1,
                'optional': 1,
                'description': 'One feature required, one optional for grouping'
            },
            'pie': {
                'required': 1,
                'optional': 0,
                'description': 'One feature required'
            },
            'line': {
                'required': 1,
                'optional': 1,
                'description': 'One feature required, one optional for animation'
            }
        }
        
        # Get requirements for selected plot type
        requirements = plot_requirements.get(plot_type)
        if not requirements:
            return html.Div([
                html.H5("Invalid plot type!"),
                html.P("Please select a valid plot type.")
            ])
        
        # Get selected features
        selected_features = [f for f in [feature1, feature2, feature3] if f is not None]
        
        # Special handling for heatmap
        if plot_type == 'heatmap':
            # For heatmap, we don't need any features
            selected_features = []
        
        # Validate number of features
        if len(selected_features) < requirements['required']:
            return html.Div([
                html.H5("Not enough features selected!"),
                html.P(f"{requirements['description']}")
            ])
        
        if len(selected_features) > (requirements['required'] + requirements['optional']):
            return html.Div([
                html.H5("Too many features selected!"),
                html.P(f"{requirements['description']}")
            ])
        
        # Check for duplicate features (skip for heatmap)
        if plot_type != 'heatmap' and len(selected_features) != len(set(selected_features)):
            return html.Div([
                html.H5("Duplicate features detected!"),
                html.P("Please select different features for each axis.")
            ])
            
        try:
            # Convert spaces to underscores in filename
            safe_filename = filename.replace(' ', '_')
            
            # Create visualization config
            config = {
                "plot_type": plot_type,
                "features": selected_features,
                "filename": safe_filename
            }
            
            # Get visualization from backend using API function
            result = create_visualization(config)
            
            # Parse the JSON string to create the figure
            fig = go.Figure(json.loads(result['figure']))
            
            # Update layout with better styling
            fig.update_layout(
                title={
                    'text': result.get('title', ''),
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 24, 'color': '#000000'}
                },
                xaxis_title=result.get('xaxis_title', ''),
                yaxis_title=result.get('yaxis_title', ''),
                template='plotly_white',
                height=800,
                margin=dict(l=50, r=50, t=100, b=50),
                paper_bgcolor='white',
                plot_bgcolor='white',
                font=dict(family="Arial, sans-serif", size=14, color='#000000'),
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.05,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1
                ),
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(0,0,0,0.1)',
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor='rgba(0,0,0,0.2)',
                    color='#000000'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(0,0,0,0.1)',
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor='rgba(0,0,0,0.2)',
                    color='#000000'
                )
            )
            
            # Update hover template based on plot type
            for trace in fig.data:
                if isinstance(trace, go.Heatmap):
                    trace.update(
                        hovertemplate="<b>%{x}</b><br>" +
                                     "Correlation: %{z:.2f}<br>" +
                                     "<extra></extra>"
                    )
                elif isinstance(trace, go.Bar):
                    trace.update(
                        hovertemplate="<b>%{x}</b><br>" +
                                     "Count: %{y}<br>" +
                                     "<extra></extra>"
                    )
                elif isinstance(trace, go.Pie):
                    trace.update(
                        hovertemplate="<b>%{label}</b><br>" +
                                     "Value: %{value}<br>" +
                                     "Percentage: %{percent}<br>" +
                                     "<extra></extra>"
                    )
                elif isinstance(trace, go.Scatter):
                    trace.update(
                        hovertemplate="<b>%{x}</b><br>" +
                                     "Value: %{y:.2f}<br>" +
                                     "<extra></extra>"
                    )
                elif isinstance(trace, go.Box):
                    trace.update(
                        hovertemplate="<b>%{x}</b><br>" +
                                     "Median: %{median:.2f}<br>" +
                                     "Q1: %{q1:.2f}<br>" +
                                     "Q3: %{q3:.2f}<br>" +
                                     "<extra></extra>"
                    )
                elif isinstance(trace, go.Violin):
                    trace.update(
                        hovertemplate="<b>%{x}</b><br>" +
                                     "Density: %{y:.2f}<br>" +
                                     "<extra></extra>"
                    )
            
            # Wrap the graph in a container for better presentation
            return html.Div([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(
                            figure=fig,
                            config={
                                'displayModeBar': True,
                                'displaylogo': False,
                                'modeBarButtonsToAdd': [
                                    'drawline',
                                    'drawopenpath',
                                    'drawclosedpath',
                                    'drawcircle',
                                    'drawrect',
                                    'eraseshape'
                                ]
                            },
                            style={
                                'height': '800px',
                                'backgroundColor': 'white'
                            }
                        )
                    ])
                ], className="shadow-lg")
            ], className="mt-4")
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return html.Div([
                html.H5("Error creating visualization!"),
                html.P(str(e))
            ]) 