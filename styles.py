"""
Custom CSS styles for the Streamlit application.
"""

def load_css():
    """
    Returns the custom CSS styles as a string.
    """
    return """
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .risk-alert {
        background: #b02a37; /* deep red */
        border-left: 6px solid #7a1c24;
        color: #ffffff;       /* white text */
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: bold;
        box-shadow: 0 0 10px rgba(176, 42, 55, 0.4);
    }

    .success-alert {
        background: #157347; /* deep green */
        border-left: 6px solid #0f5132;
        color: #ffffff;       /* white text */
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: bold;
        box_shadow: 0 0 10px rgba(21, 115, 71, 0.4);
    }
    </style>
    """