"""
Configuration file for Options Pricing Analytics
Contains all constants, default values, and configuration settings
"""

# Page Configuration
PAGE_CONFIG = {
    "page_title": "Options Pricing Analytics",
    "page_icon": "📈",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Default Market Parameters
DEFAULT_PARAMS = {
    "stock_price": 100.0,
    "strike_price": 100.0,
    "time_to_expiry": 0.25,
    "risk_free_rate": 5.0,  # in percentage
    "volatility": 20.0,      # in percentage
    "dividend_yield": 0.0    # in percentage
}

# Simulation Parameters
SIMULATION_PARAMS = {
    "mc_simulations_default": 100000,
    "mc_simulations_min": 10000,
    "mc_simulations_max": 200000,
    "mc_simulations_step": 10000,
    "binomial_steps_default": 100,
    "binomial_steps_min": 50,
    "binomial_steps_max": 300,
    "binomial_steps_step": 10
}

# Scenario Analysis Defaults
SCENARIO_DEFAULTS = {
    "spot_price_min_factor": 0.5,
    "spot_price_max_factor": 1.5,
    "volatility_min": 5.0,
    "volatility_max": 50.0,
    "default_steps": 50,
    "time_decay_default_days": 30
}

# IV Calculator Settings
IV_SETTINGS = {
    "max_iterations_default": 100,
    "max_iterations_min": 50,
    "max_iterations_max": 200,
    "tolerance_options": [1e-4, 1e-5, 1e-6],
    "tolerance_default_index": 2,
    "min_vol_default": 0.1,
    "max_vol_default": 500.0
}

# Volatility Surface Settings
VOL_SURFACE_SETTINGS = {
    "resolution_min": 5,
    "resolution_max": 15,
    "resolution_default": 10,
    "base_vol_min": 10,
    "base_vol_max": 50,
    "base_vol_default": 20
}

# Portfolio Analysis Settings
PORTFOLIO_SETTINGS = {
    "max_positions": 5,
    "position_types": ["Long Call", "Short Call", "Long Put", "Short Put"],
    "pnl_spot_min_factor": 0.5,
    "pnl_spot_max_factor": 1.5,
    "pnl_points": 100
}

# Color Schemes - Optimized for Dark Mode
COLORS = {
    "primary": "#c584f7",        # Your primary purple
    "secondary": "#64ffda",      # Bright cyan for contrast
    "tertiary": "#82aaff",       # Soft blue
    "danger": "#ff5370",         # Bright red for dark mode
    "success": "#c3e88d",        # Soft green
    "warning": "#ffcb6b",        # Warm yellow
    "info": "#82b1ff",           # Light blue
    "gradient_start": "#c584f7",  # Purple gradient
    "gradient_end": "#8b5cf6",    # Deeper purple
    "dark_bg": "#0e1117",        # Streamlit dark background
    "dark_card": "#1a1d26",      # Card background
    "dark_border": "#2d3139",    # Border color
    "text_primary": "#fafafa",   # Primary text
    "text_secondary": "#b8bcc8"  # Secondary text
}

# Chart Colors - High contrast for dark mode
CHART_COLORS = {
    "price_comparison": ["#ff5370", "#64ffda", "#82aaff"],
    "greeks": "#c584f7",
    "delta": "#ff5370",
    "gamma": "#64ffda",
    "vega": "#c584f7",
    "theta": "#ffcb6b",
    "rho": "#c3e88d",
    "distribution": "#c584f7",
    "payoff": "#64ffda",
    "plotly_template": "plotly_dark"  # Dark theme for Plotly
}

# Moneyness Thresholds
MONEYNESS_THRESHOLDS = {
    "itm": 1.05,
    "otm": 0.95
}

# Premium Thresholds
PREMIUM_THRESHOLDS = {
    "high": 5.0  # percentage of stock price
}

# Greeks Display Format
GREEKS_FORMAT = {
    "delta": {"decimals": 4, "symbol": "Δ"},
    "gamma": {"decimals": 4, "symbol": "Γ"},
    "theta": {"decimals": 4, "symbol": "Θ"},
    "vega": {"decimals": 4, "symbol": "ν"},
    "rho": {"decimals": 4, "symbol": "ρ"}
}

# Risk Metrics Settings
RISK_METRICS = {
    "var_confidence_levels": [95, 99],
    "histogram_bins": 50
}

# Tab Names
TABS = [
    "Pricing Models",
    "Greeks Analysis", 
    "Scenario Analysis",
    "Risk Metrics",
    "Implied Volatility",
    "Advanced Analytics"
]