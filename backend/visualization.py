import plotly.express as px
import plotly.graph_objects as go
import plotly
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
import logging
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set seaborn style for brighter colors
sns.set_style("whitegrid")
sns.set_palette("bright")  # Using bright color palette

# Define custom bright color sequences
BRIGHT_COLORS = [
    '#FF6B6B',  # Coral Red
    '#4ECDC4',  # Turquoise
    '#45B7D1',  # Sky Blue
    '#96CEB4',  # Sage Green
    '#FFEEAD',  # Cream Yellow
    '#D4A5A5',  # Dusty Rose
    '#9B59B6',  # Purple
    '#3498DB',  # Blue
    '#E67E22',  # Orange
    '#2ECC71'   # Green
]

def serialize_figure(fig: go.Figure) -> Dict[str, Any]:
    """
    Serialize a Plotly figure to JSON with error handling.
    
    Args:
        fig (go.Figure): The Plotly figure to serialize
        
    Returns:
        Dict[str, Any]: Dictionary containing the serialized figure and metadata
        
    Raises:
        ValueError: If serialization fails
    """
    try:
        # Convert figure to JSON string
        figure_json = fig.to_json()
        
        return {
            'figure': figure_json,  # Return JSON string
            'title': fig.layout.title.text if fig.layout.title else '',
            'xaxis_title': fig.layout.xaxis.title.text if fig.layout.xaxis.title else '',
            'yaxis_title': fig.layout.yaxis.title.text if fig.layout.yaxis.title else ''
        }
    except Exception as e:
        logger.error(f"Failed to serialize figure: {str(e)}")
        raise ValueError(f"Figure serialization failed: {str(e)}")

def create_visualization(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create visualization based on configuration.
    
    Args:
        df (pd.DataFrame): Input dataframe
        config (Dict[str, Any]): Visualization configuration
        
    Returns:
        Dict[str, Any]: Serialized figure data
        
    Raises:
        ValueError: If plot type is unknown or creation fails
    """
    if df.empty:
        raise ValueError("Input dataframe is empty")
        
    plot_type = config.get('plot_type')
    features = config.get('features', [])
    
    try:
        if plot_type == 'heatmap':
            return create_correlation_heatmap(df)
        elif plot_type == 'distribution':
            if not features:
                raise ValueError("Feature required for distribution plot")
            return create_distribution_plot(df, features[0])
        elif plot_type == 'box':
            if not features:
                raise ValueError("Feature required for box plot")
            return create_box_plot(df, features[0], features[1] if len(features) > 1 else None)
        elif plot_type == 'violin':
            if not features:
                raise ValueError("Feature required for violin plot")
            return create_violin_plot(df, features[0], features[1] if len(features) > 1 else None)
        elif plot_type == 'scatter_2d':
            if len(features) < 2:
                raise ValueError("Two features required for 2D scatter plot")
            return create_scatter_plot(df, features[0], features[1])
        elif plot_type == 'scatter_3d':
            if len(features) < 3:
                raise ValueError("Three features required for 3D scatter plot")
            return create_scatter_3d_plot(df, features[0], features[1], features[2])
        elif plot_type == 'bar':
            if not features:
                raise ValueError("Feature required for bar plot")
            return create_bar_plot(df, features[0])
        elif plot_type == 'pie':
            if not features:
                raise ValueError("Feature required for pie chart")
            return create_pie_chart(df, features[0])
        elif plot_type == 'line':
            if not features:
                raise ValueError("Feature required for line plot")
            return create_line_plot(df, features[0], features[1] if len(features) > 1 else None, 
                                  features[2] if len(features) > 2 else None)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
    except Exception as e:
        logger.error(f"Failed to create {plot_type} plot: {str(e)}")
        raise ValueError(f"Plot creation failed: {str(e)}")

def debug_heatmap_creation(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Debug function to test heatmap creation with minimal configuration.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        Dict[str, Any]: Debug information and test results
    """
    debug_info = {
        'plotly_version': plotly.__version__,
        'test_results': [],
        'errors': []
    }
    
    try:
        # Test 1: Basic heatmap without any extra properties
        numerical_df = df.select_dtypes(include=[np.number])
        if numerical_df.empty:
            debug_info['errors'].append("No numerical columns found in dataframe")
            return debug_info
            
        corr_matrix = numerical_df.corr()
        
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns
        ))
        debug_info['test_results'].append('Basic heatmap created successfully')
        
        # Test 2: Add colorscale
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdYlBu_r'
        ))
        debug_info['test_results'].append('Heatmap with colorscale created successfully')
        
        # Test 3: Add text
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdYlBu_r',
            text=[[f'{val:.2f}' for val in row] for row in corr_matrix.values],
            texttemplate='%{text}'
        ))
        debug_info['test_results'].append('Heatmap with text created successfully')
        
        # Test 4: Add colorbar
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdYlBu_r',
            text=[[f'{val:.2f}' for val in row] for row in corr_matrix.values],
            texttemplate='%{text}',
            showscale=True,
            colorbar=dict(title='Correlation')
        ))
        debug_info['test_results'].append('Heatmap with colorbar created successfully')
        
        return debug_info
        
    except Exception as e:
        debug_info['errors'].append(str(e))
        logger.error(f"Debug heatmap creation error: {str(e)}")
        return debug_info

def create_correlation_heatmap(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create correlation heatmap using plotly - Simplified version.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        Dict[str, Any]: Serialized figure data
        
    Raises:
        ValueError: If heatmap creation fails
    """
    try:
        # First run debug to identify any issues
        debug_info = debug_heatmap_creation(df)
        if debug_info['errors']:
            logger.error(f"Heatmap debug errors: {debug_info['errors']}")
            raise ValueError(f"Heatmap creation failed during debug: {debug_info['errors']}")
            
        # Select only numerical columns and handle missing values
        numerical_df = df.select_dtypes(include=[np.number])
        if numerical_df.empty:
            raise ValueError("No numerical columns found in dataframe")
            
        numerical_df = numerical_df.fillna(numerical_df.mean())
        
        # Calculate correlation matrix
        corr_matrix = numerical_df.corr()
        corr_matrix = corr_matrix.round(2)
        
        # Create text matrix for annotations
        text_matrix = [[f'{val:.2f}' for val in row] for row in corr_matrix.values]
        
        # Create heatmap with minimal configuration
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdYlBu_r',
            zmin=-1,
            zmax=1,
            text=text_matrix,
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            showscale=True,
            colorbar=dict(
                title='Correlation',
                titleside='right',
                titlefont=dict(size=14)
            ),
            hoverinfo='z',
            hovertemplate='Correlation: %{z:.2f}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Correlation Heatmap',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24, 'color': '#000000'}
            },
            xaxis_title='Features',
            yaxis_title='Features',
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=800,
            width=1000,
            xaxis={'tickangle': 45},
            yaxis={'tickangle': 0},
            margin=dict(l=50, r=50, t=100, b=50)
        )
        
        # Convert figure to JSON and return
        return serialize_figure(fig)
        
    except Exception as e:
        logger.error(f"Error in create_correlation_heatmap: {str(e)}")
        raise ValueError(f"Failed to create correlation heatmap: {str(e)}")

def create_distribution_plot(df: pd.DataFrame, feature: str) -> Dict[str, Any]:
    """
    Create distribution plot using plotly.
    
    Args:
        df (pd.DataFrame): Input dataframe
        feature (str): Feature to plot
        
    Returns:
        Dict[str, Any]: Serialized figure data
        
    Raises:
        ValueError: If distribution plot creation fails
    """
    try:
        if feature not in df.columns:
            raise ValueError(f"Feature {feature} not found in dataframe")
            
        # Create histogram with KDE
        fig = px.histogram(
            df,
            x=feature,
            marginal='box',
            title=f'Distribution of {feature}',
            color_discrete_sequence=BRIGHT_COLORS,
            opacity=0.8,
            nbins=50
        )
        
        # Add KDE curve using scipy
        kde = stats.gaussian_kde(df[feature].dropna())
        x_range = np.linspace(df[feature].min(), df[feature].max(), 100)
        y_range = kde(x_range)
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_range,
            mode='lines',
            name='KDE',
            line=dict(color=BRIGHT_COLORS[3], width=2)
        ))
        
        # Update layout
        fig.update_layout(
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=600,
            showlegend=True,
            hovermode='x unified'
        )
        
        # Update hover template for each trace type separately
        for trace in fig.data:
            if isinstance(trace, go.Histogram):
                trace.update(
                    hovertemplate="<b>%{x}</b><br>" +
                                 "Count: %{y}<br>" +
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
            elif isinstance(trace, go.Scatter):
                trace.update(
                    hovertemplate="<b>%{x}</b><br>" +
                                 "Density: %{y:.4f}<br>" +
                                 "<extra></extra>"
                )
        
        return serialize_figure(fig)
        
    except Exception as e:
        logger.error(f"Error in create_distribution_plot: {str(e)}")
        raise ValueError(f"Failed to create distribution plot: {str(e)}")

def create_box_plot(df: pd.DataFrame, feature: str, group_by: str = None) -> Dict[str, Any]:
    """Create box plot using seaborn"""
    if group_by:
        fig = px.box(
            df,
            x=group_by,
            y=feature,
            title=f'Box Plot of {feature} by {group_by}',
            color=group_by,
            color_discrete_sequence=BRIGHT_COLORS,
            points='all'
        )
    else:
        fig = px.box(
            df,
            y=feature,
            title=f'Box Plot of {feature}',
            color_discrete_sequence=BRIGHT_COLORS,
            points='all'
        )
    
    # Update layout
    fig.update_layout(
        template='plotly_white',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return {
        'figure': fig.to_json(),
        'title': f'Box Plot of {feature}',
        'xaxis_title': group_by if group_by else feature,
        'yaxis_title': feature
    }

def create_violin_plot(df: pd.DataFrame, feature: str, group_by: str = None) -> Dict[str, Any]:
    """Create violin plot using seaborn"""
    if group_by:
        fig = px.violin(
            df,
            x=group_by,
            y=feature,
            title=f'Violin Plot of {feature} by {group_by}',
            color=group_by,
            color_discrete_sequence=BRIGHT_COLORS,
            box=True,
            points='all'
        )
    else:
        fig = px.violin(
            df,
            y=feature,
            title=f'Violin Plot of {feature}',
            color_discrete_sequence=BRIGHT_COLORS,
            box=True,
            points='all'
        )
    
    # Update layout
    fig.update_layout(
        template='plotly_white',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return {
        'figure': fig.to_json(),
        'title': f'Violin Plot of {feature}',
        'xaxis_title': group_by if group_by else feature,
        'yaxis_title': feature
    }

def create_scatter_plot(df: pd.DataFrame, x: str, y: str) -> Dict[str, Any]:
    """Create scatter plot using plotly"""
    try:
        # Clean data by removing NaN values
        clean_df = df[[x, y]].dropna()
        
        if clean_df.empty:
            raise ValueError(f"No valid data points after removing NaN values for {x} and {y}")
            
        # Create scatter plot
        fig = px.scatter(
            clean_df,
            x=x,
            y=y,
            title=f'Scatter Plot: {x} vs {y}',
            color_discrete_sequence=BRIGHT_COLORS,
            opacity=0.7,
            marginal_x='histogram',
            marginal_y='histogram'
        )
        
        # Try to add trend line if possible
        try:
            # Calculate trend line
            x_data = clean_df[x].values
            y_data = clean_df[y].values
            
            # Check if data is valid for trend line
            if len(x_data) > 1 and len(y_data) > 1:
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                
                # Add trend line
                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=p(x_data),
                    mode='lines',
                    name='Trend Line',
                    line=dict(color=BRIGHT_COLORS[1], width=2)
                ))
        except Exception as e:
            logger.warning(f"Could not add trend line: {str(e)}")
        
        # Update layout
        fig.update_layout(
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=600,
            showlegend=True,
            hovermode='closest'
        )
        
        # Update hover template
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>" +
                         "Value: %{y:.2f}<br>" +
                         "<extra></extra>"
        )
        
        return serialize_figure(fig)
        
    except Exception as e:
        logger.error(f"Failed to create scatter_2d plot: {str(e)}")
        raise ValueError(f"Failed to create scatter plot: {str(e)}")

def create_scatter_3d_plot(df: pd.DataFrame, x: str, y: str, z: str) -> Dict[str, Any]:
    """Create 3D scatter plot"""
    fig = px.scatter_3d(
        df,
        x=x,
        y=y,
        z=z,
        title=f'3D Scatter Plot: {x} vs {y} vs {z}',
        color_discrete_sequence=BRIGHT_COLORS,
        opacity=0.7
    )
    
    # Update layout
    fig.update_layout(
        template='plotly_white',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return {
        'figure': fig.to_json(),
        'title': f'3D Scatter Plot: {x} vs {y} vs {z}',
        'xaxis_title': x,
        'yaxis_title': y
    }

def create_bar_plot(df: pd.DataFrame, feature: str) -> Dict[str, Any]:
    """Create bar plot using plotly"""
    try:
        value_counts = df[feature].value_counts()
        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=f'Bar Plot of {feature}',
            color=value_counts.values,
            color_continuous_scale=BRIGHT_COLORS
        )
        
        # Update layout
        fig.update_layout(
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=600,
            showlegend=True,
            hovermode='x unified'
        )
        
        # Update hover template
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>" +
                         "Count: %{y}<br>" +
                         "<extra></extra>"
        )
        
        return serialize_figure(fig)
        
    except Exception as e:
        logger.error(f"Failed to create bar plot: {str(e)}")
        raise ValueError(f"Failed to create bar plot: {str(e)}")

def create_pie_chart(df: pd.DataFrame, feature: str) -> Dict[str, Any]:
    """Create pie chart"""
    try:
        value_counts = df[feature].value_counts()
        fig = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            title=f'Pie Chart of {feature}',
            color_discrete_sequence=BRIGHT_COLORS,
            hole=0.4
        )
        
        # Update layout
        fig.update_layout(
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=600,
            showlegend=True,
            hovermode='closest'
        )
        
        # Update hover template
        fig.update_traces(
            hovertemplate="<b>%{label}</b><br>" +
                         "Value: %{value}<br>" +
                         "Percentage: %{percent}<br>" +
                         "<extra></extra>"
        )
        
        return serialize_figure(fig)
        
    except Exception as e:
        logger.error(f"Failed to create pie chart: {str(e)}")
        raise ValueError(f"Failed to create pie chart: {str(e)}")

def create_line_plot(df: pd.DataFrame, x: str, y: str = None, animation_column: str = None) -> Dict[str, Any]:
    """Create line plot with flexible animations using plotly express"""
    try:
        # Sort data by x-axis for smooth animation
        df = df.sort_values(by=x)
        
        # Create base figure with animation if animation_column is provided
        if animation_column and animation_column in df.columns:
            # Check if the column is categorical
            if df[animation_column].dtype == 'object' or df[animation_column].dtype.name == 'category':
                fig = px.line(
                    df,
                    x=x,
                    y=y if y else x,
                    color=animation_column,  # Use color for categories
                    title=f'<b>Line Plot: {y if y else x} over {x}</b>',
                    color_discrete_sequence=BRIGHT_COLORS,
                    markers=True,
                    template='plotly_white',
                    animation_frame=animation_column,  # Add animation
                    range_x=[df[x].min(), df[x].max()],  # Fix x-axis range for smooth animation
                    range_y=[df[y if y else x].min(), df[y if y else x].max()]  # Fix y-axis range
                )
                
                # Update animation settings
                fig.update_layout(
                    updatemenus=[{
                        'type': 'buttons',
                        'showactive': False,
                        'buttons': [
                            {
                                'label': 'Play',
                                'method': 'animate',
                                'args': [None, {
                                    'frame': {'duration': 500, 'redraw': True},
                                    'fromcurrent': True,
                                    'mode': 'immediate'
                                }]
                            },
                            {
                                'label': 'Pause',
                                'method': 'animate',
                                'args': [[None], {
                                    'frame': {'duration': 0, 'redraw': False},
                                    'mode': 'immediate',
                                    'transition': {'duration': 0}
                                }]
                            }
                        ]
                    }]
                )
            else:
                # If animation_column is not categorical, use it as y-axis
                fig = px.line(
                    df,
                    x=x,
                    y=animation_column,
                    title=f'<b>Line Plot: {animation_column} over {x}</b>',
                    color_discrete_sequence=BRIGHT_COLORS,
                    markers=True,
                    template='plotly_white'
                )
        else:
            # Create simple line plot without animation
            fig = px.line(
                df,
                x=x,
                y=y if y else x,
                title=f'<b>Line Plot: {y if y else x} over {x}</b>',
                color_discrete_sequence=BRIGHT_COLORS,
                markers=True,
                template='plotly_white'
            )
        
        # Update layout for better visualization
        fig.update_layout(
            title_x=0.5,
            title_font_size=24,
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
                zerolinecolor='rgba(0,0,0,0.2)'
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0,0,0,0.1)',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='rgba(0,0,0,0.2)'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=100, b=50),
            hovermode='x unified',
            height=600
        )
        
        # Update hover template
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>" +
                         "Value: %{y:.2f}<br>" +
                         "<extra></extra>"
        )
        
        return serialize_figure(fig)
        
    except Exception as e:
        logger.error(f"Failed to create line plot: {str(e)}")
        raise ValueError(f"Failed to create line plot: {str(e)}") 