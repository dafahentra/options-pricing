"""
Custom CSS styles for the Streamlit application.
"""

import streamlit as st

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

    /* NEW CSS TO HIDE STREAMLIT UI ELEMENTS */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    .stDecoration {display: none;}

    /* Hide "Made with Streamlit" */
    .css-hi6a2p {display: none;}
    .css-9s5bis {display: none;}
    .css-1dp5vir {display: none;}

    /* Hide hamburger menu */
    .css-14xtw13 {display: none;}

    /* Global Styles (Assuming COLORS are defined elsewhere or removed if not used) */
    .stApp {
        /* Example: background: linear-gradient(135deg, #2a2a30 0%, #0c1220 100%); */
        /* Example: color: #f0f2f6; */
    }

    /* Sidebar Styling (Assuming COLORS are defined elsewhere or removed if not used) */
    .css-1d391kg {
        /* Example: background: linear-gradient(180deg, #3a3a40 0%, #2a2a30 100%); */
        /* Example: border-right: 1px solid #4a4a50; */
    }

    .css-17eq0hr {
        background: transparent;
    }

    </style>
    """