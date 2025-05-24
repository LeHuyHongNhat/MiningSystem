import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import logging

from frontend.callbacks import register_all_callbacks
from frontend.components.layout import create_layout

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

# Set the title
app.title = "Data Mining System"

# Create the layout
app.layout = create_layout()

# Register all callbacks
register_all_callbacks(app)

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 8050))
    
    # Run the app
    app.run_server(
        debug=True,
        port=port,
        host='localhost'
    ) 