import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def create_correlation_heatmap(df):
    """Create correlation heatmap for numerical features"""
    numerical_df = df.select_dtypes(include=['int64', 'float64'])
    corr_matrix = numerical_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1
    ))
    
    fig.update_layout(
        title="Correlation Heatmap",
        xaxis_title="Features",
        yaxis_title="Features",
        height=600
    )
    
    return fig

def create_distribution_plot(df, feature):
    """Create distribution plot for a feature"""
    fig = px.histogram(
        df,
        x=feature,
        marginal="box",
        title=f"Distribution of {feature}"
    )
    
    fig.update_layout(
        height=500,
        showlegend=False
    )
    
    return fig

def create_box_plot(df, feature, group_by=None):
    """Create box plot for a feature, optionally grouped by another feature"""
    if group_by:
        fig = px.box(
            df,
            x=group_by,
            y=feature,
            title=f"Box Plot of {feature} by {group_by}"
        )
    else:
        fig = px.box(
            df,
            y=feature,
            title=f"Box Plot of {feature}"
        )
    
    fig.update_layout(
        height=500,
        showlegend=False
    )
    
    return fig

def create_violin_plot(df, feature, group_by=None):
    """Create violin plot for a feature, optionally grouped by another feature"""
    if group_by:
        fig = px.violin(
            df,
            x=group_by,
            y=feature,
            title=f"Violin Plot of {feature} by {group_by}"
        )
    else:
        fig = px.violin(
            df,
            y=feature,
            title=f"Violin Plot of {feature}"
        )
    
    fig.update_layout(
        height=500,
        showlegend=False
    )
    
    return fig

def create_scatter_plot(df, x_feature, y_feature, color_feature=None):
    """Create scatter plot for two features, optionally colored by a third feature"""
    if color_feature:
        fig = px.scatter(
            df,
            x=x_feature,
            y=y_feature,
            color=color_feature,
            title=f"Scatter Plot of {y_feature} vs {x_feature}"
        )
    else:
        fig = px.scatter(
            df,
            x=x_feature,
            y=y_feature,
            title=f"Scatter Plot of {y_feature} vs {x_feature}"
        )
    
    fig.update_layout(
        height=500,
        showlegend=True
    )
    
    return fig

def create_scatter_3d_plot(df, x_feature, y_feature, z_feature, color_feature=None):
    """Create 3D scatter plot for three features, optionally colored by a fourth feature"""
    if color_feature:
        fig = px.scatter_3d(
            df,
            x=x_feature,
            y=y_feature,
            z=z_feature,
            color=color_feature,
            title=f"3D Scatter Plot of {x_feature}, {y_feature}, and {z_feature}"
        )
    else:
        fig = px.scatter_3d(
            df,
            x=x_feature,
            y=y_feature,
            z=z_feature,
            title=f"3D Scatter Plot of {x_feature}, {y_feature}, and {z_feature}"
        )
    
    fig.update_layout(
        height=600,
        showlegend=True
    )
    
    return fig

def create_bar_plot(df, feature, group_by=None):
    """Create bar plot for a feature, optionally grouped by another feature"""
    if group_by:
        fig = px.bar(
            df,
            x=group_by,
            y=feature,
            title=f"Bar Plot of {feature} by {group_by}"
        )
    else:
        fig = px.bar(
            df,
            x=feature,
            title=f"Bar Plot of {feature}"
        )
    
    fig.update_layout(
        height=500,
        showlegend=False
    )
    
    return fig

def create_pie_chart(df, feature):
    """Create pie chart for a feature"""
    value_counts = df[feature].value_counts()
    
    fig = px.pie(
        values=value_counts.values,
        names=value_counts.index,
        title=f"Pie Chart of {feature}"
    )
    
    fig.update_layout(
        height=500,
        showlegend=True
    )
    
    return fig

def create_line_plot(df, feature, group_by=None):
    """Create line plot for a feature, optionally grouped by another feature"""
    if group_by:
        fig = px.line(
            df,
            x=group_by,
            y=feature,
            title=f"Line Plot of {feature} by {group_by}"
        )
    else:
        fig = px.line(
            df,
            y=feature,
            title=f"Line Plot of {feature}"
        )
    
    fig.update_layout(
        height=500,
        showlegend=False
    )
    
    return fig

def create_feature_importance_plot(importance_dict):
    """Create feature importance plot"""
    fig = px.bar(
        x=list(importance_dict.keys()),
        y=list(importance_dict.values()),
        title="Feature Importance"
    )
    fig.update_layout(
        xaxis_title="Features",
        yaxis_title="Importance Score"
    )
    return fig

def create_confusion_matrix(cm, labels):
    """Create confusion matrix heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale="Viridis"
    ))
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual"
    )
    return fig

def create_roc_curve(fpr, tpr, auc):
    """Create ROC curve"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        name=f"ROC (AUC = {auc:.2f})"
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        name="Random",
        line=dict(dash="dash")
    ))
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )
    return fig

def create_residual_plot(y_true, y_pred):
    """Create residual plot"""
    residuals = y_true - y_pred
    fig = px.scatter(
        x=y_pred,
        y=residuals,
        title="Residual Plot"
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(
        xaxis_title="Predicted Values",
        yaxis_title="Residuals"
    )
    return fig 