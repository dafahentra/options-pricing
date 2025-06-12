"""
Custom CSS styles for the Streamlit application.
Optimized for Dark Mode with purple theme.
"""

from config import COLORS


def load_css():
    """
    Returns the custom CSS styles optimized for dark mode.
    """
    return f"""
    <style>
    /* ========== CSS VARIABLES ========== */
    :root {{
        --primary: {COLORS['primary']};
        --secondary: {COLORS['secondary']};
        --danger: {COLORS['danger']};
        --success: {COLORS['success']};
        --warning: {COLORS['warning']};
        --info: {COLORS['info']};
        --dark-bg: {COLORS['dark_bg']};
        --dark-card: {COLORS['dark_card']};
        --dark-border: {COLORS['dark_border']};
        --text-primary: {COLORS['text_primary']};
        --text-secondary: {COLORS['text_secondary']};
    }}
    
    /* ========== MAIN HEADER ========== */
    .main-header {{
        background: linear-gradient(135deg, {COLORS['gradient_start']} 0%, {COLORS['gradient_end']} 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(197, 132, 247, 0.3);
        border: 1px solid rgba(197, 132, 247, 0.2);
    }}
    
    .main-header h1 {{
        margin: 0;
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }}
    
    .main-header p {{
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.95;
        font-weight: 300;
    }}
    
    /* ========== ALERTS FOR DARK MODE ========== */
    .risk-alert {{
        background: linear-gradient(135deg, rgba(255, 83, 112, 0.2) 0%, rgba(255, 83, 112, 0.1) 100%);
        border-left: 6px solid var(--danger);
        color: var(--text-primary);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-weight: 600;
        box-shadow: 0 4px 20px rgba(255, 83, 112, 0.2);
        backdrop-filter: blur(10px);
        animation: slideIn 0.3s ease;
        border: 1px solid rgba(255, 83, 112, 0.3);
    }}
    
    .success-alert {{
        background: linear-gradient(135deg, rgba(195, 232, 141, 0.2) 0%, rgba(195, 232, 141, 0.1) 100%);
        border-left: 6px solid var(--success);
        color: var(--text-primary);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-weight: 600;
        box-shadow: 0 4px 20px rgba(195, 232, 141, 0.2);
        backdrop-filter: blur(10px);
        animation: slideIn 0.3s ease;
        border: 1px solid rgba(195, 232, 141, 0.3);
    }}
    
    .info-alert {{
        background: linear-gradient(135deg, rgba(130, 177, 255, 0.2) 0%, rgba(130, 177, 255, 0.1) 100%);
        border-left: 6px solid var(--info);
        color: var(--text-primary);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-weight: 600;
        box-shadow: 0 4px 20px rgba(130, 177, 255, 0.2);
        backdrop-filter: blur(10px);
        animation: slideIn 0.3s ease;
        border: 1px solid rgba(130, 177, 255, 0.3);
    }}
    
    .warning-alert {{
        background: linear-gradient(135deg, rgba(255, 203, 107, 0.2) 0%, rgba(255, 203, 107, 0.1) 100%);
        border-left: 6px solid var(--warning);
        color: var(--text-primary);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-weight: 600;
        box-shadow: 0 4px 20px rgba(255, 203, 107, 0.2);
        backdrop-filter: blur(10px);
        animation: slideIn 0.3s ease;
        border: 1px solid rgba(255, 203, 107, 0.3);
    }}
    
    /* ========== ANIMATIONS ========== */
    @keyframes slideIn {{
        from {{
            transform: translateX(-20px);
            opacity: 0;
        }}
        to {{
            transform: translateX(0);
            opacity: 1;
        }}
    }}
    
    @keyframes glow {{
        0%, 100% {{
            box-shadow: 0 0 20px rgba(197, 132, 247, 0.5);
        }}
        50% {{
            box-shadow: 0 0 30px rgba(197, 132, 247, 0.8);
        }}
    }}
    
    /* ========== ENHANCED COMPONENTS ========== */
    /* Enhanced Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: {COLORS['dark_card']};
        padding: 0.5rem;
        border-radius: 12px;
        border: 1px solid {COLORS['dark_border']};
    }}
    
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        color: {COLORS['text_secondary']} !important;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: rgba(197, 132, 247, 0.1);
        color: {COLORS['text_primary']} !important;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, var(--primary) 0%, {COLORS['gradient_end']} 100%);
        box-shadow: 0 4px 15px rgba(197, 132, 247, 0.4);
        color: white !important;
    }}
    
    .stTabs [aria-selected="true"] span {{
        color: white !important;
    }}
    
    /* Force white text on active tab - additional selectors for compatibility */
    .stTabs [aria-selected="true"] * {{
        color: white !important;
    }}
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        color: white !important;
    }}
    
    /* Enhanced Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, var(--primary) 0%, {COLORS['gradient_end']} 100%);
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 10px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(197, 132, 247, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(197, 132, 247, 0.5);
        animation: glow 2s ease-in-out infinite;
    }}
    
    /* Enhanced Sliders */
    .stSlider > div > div > div > div {{
        background-color: var(--primary) !important;
    }}
    
    .stSlider > div > div > div > div > div {{
        background-color: var(--primary) !important;
        box-shadow: 0 0 10px rgba(197, 132, 247, 0.5);
    }}
    
    /* Enhanced Metrics */
    [data-testid="metric-container"] {{
        background-color: {COLORS['dark_card']};
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        border: 1px solid {COLORS['dark_border']};
        transition: all 0.3s ease;
    }}
    
    [data-testid="metric-container"]:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(197, 132, 247, 0.2);
        border-color: var(--primary);
    }}
    
    [data-testid="metric-container"] [data-testid="metric-label"] {{
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.8rem;
    }}
    
    [data-testid="metric-container"] [data-testid="metric-value"] {{
        color: var(--primary);
        font-weight: 700;
    }}
    
    /* Enhanced DataFrames */
    .dataframe {{
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }}
    
    .dataframe thead th {{
        background-color: rgba(197, 132, 247, 0.2) !important;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }}
    
    .dataframe tbody tr:hover {{
        background-color: rgba(197, 132, 247, 0.1) !important;
    }}
    
    /* Enhanced Plotly Charts */
    .js-plotly-plot {{
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        border: 1px solid {COLORS['dark_border']};
    }}
    
    /* Footer Enhancement */
    div[style*="text-align: center"] {{
        background-color: {COLORS['dark_card']};
        border-radius: 12px;
        padding: 2rem !important;
        margin-top: 3rem;
        border: 1px solid {COLORS['dark_border']};
        box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.2);
    }}
    
    /* Hide Streamlit Elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .stDeployButton {{display: none;}}
    
    /* Enhanced Scrollbar */
    ::-webkit-scrollbar {{
        width: 10px;
        height: 10px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {COLORS['dark_bg']};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {COLORS['dark_border']};
        border-radius: 5px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: var(--primary);
    }}
    
    /* Responsive Design */
    @media (max-width: 768px) {{
        .main-header h1 {{
            font-size: 2rem;
        }}
        
        .stButton > button {{
            padding: 0.5rem 1rem;
            font-size: 0.9rem;
        }}
    }}
    </style>
    """