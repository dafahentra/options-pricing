"""
Utility functions for Options Pricing Analytics
Contains helper functions for calculations and data processing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from models import OptionParameters, OptionType, BlackScholesModel
from config import MONEYNESS_THRESHOLDS, PREMIUM_THRESHOLDS


def calculate_moneyness(stock_price: float, strike_price: float) -> Tuple[float, str]:
    """
    Calculate moneyness ratio and status.
    
    Args:
        stock_price: Current stock price
        strike_price: Option strike price
        
    Returns:
        Tuple of (moneyness_ratio, status)
    """
    moneyness = stock_price / strike_price
    
    if moneyness > MONEYNESS_THRESHOLDS["itm"]:
        status = "ITM"
    elif moneyness < MONEYNESS_THRESHOLDS["otm"]:
        status = "OTM"
    else:
        status = "ATM"
    
    return moneyness, status


def calculate_time_value(option_price: float, stock_price: float, 
                        strike_price: float, option_type: OptionType) -> float:
    """
    Calculate time value of an option.
    
    Args:
        option_price: Current option price
        stock_price: Current stock price
        strike_price: Option strike price
        option_type: Type of option (CALL/PUT)
        
    Returns:
        Time value of the option
    """
    if option_type == OptionType.CALL:
        intrinsic_value = max(stock_price - strike_price, 0)
    else:
        intrinsic_value = max(strike_price - stock_price, 0)
    
    return option_price - intrinsic_value


def is_high_premium_option(option_price: float, stock_price: float) -> bool:
    """
    Check if option has high premium relative to stock price.
    
    Args:
        option_price: Current option price
        stock_price: Current stock price
        
    Returns:
        True if premium is high, False otherwise
    """
    if stock_price <= 0:
        return False
    
    premium_pct = (option_price / stock_price) * 100
    return premium_pct > PREMIUM_THRESHOLDS["high"]


def normalize_greeks(greeks_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize Greeks for spider chart visualization.
    
    Args:
        greeks_dict: Dictionary of Greek values
        
    Returns:
        Dictionary of normalized Greek values (0-100 scale)
    """
    normalized = {}
    
    for greek, value in greeks_dict.items():
        abs_value = abs(value)
        
        if greek == 'Delta':
            # Delta is already between 0 and 1 for absolute value
            normalized[greek] = abs_value * 100
        elif greek == 'Gamma':
            # Gamma typically small, scale up
            normalized[greek] = min(abs_value * 1000, 100)
        elif greek == 'Theta':
            # Theta scale based on daily theta
            normalized[greek] = min(abs(value / 365) * 100, 100)
        elif greek == 'Vega':
            # Vega scale based on 1% vol change
            normalized[greek] = min(abs_value, 100)
        elif greek == 'Rho':
            # Rho scale based on 1% rate change
            normalized[greek] = min(abs_value, 100)
    
    return normalized


def calculate_greeks_over_range(spot_range: np.ndarray, params: OptionParameters) -> Dict[str, List[float]]:
    """
    Calculate Greeks over a range of spot prices.
    
    Args:
        spot_range: Array of spot prices
        params: Base option parameters
        
    Returns:
        Dictionary containing lists of Greek values
    """
    greeks = {
        'delta': [],
        'gamma': [],
        'vega': [],
        'theta': [],
        'rho': []
    }
    
    for spot in spot_range:
        temp_params = OptionParameters(
            spot, params.K, params.T, params.r, 
            params.sigma, params.option_type, params.q
        )
        result = BlackScholesModel.calculate_all(temp_params)
        
        greeks['delta'].append(result.delta)
        greeks['gamma'].append(result.gamma)
        greeks['vega'].append(result.vega)
        greeks['theta'].append(result.theta)
        greeks['rho'].append(result.rho)
    
    return greeks


def calculate_scenario_pnl(option_prices: List[float], initial_price: float) -> np.ndarray:
    """
    Calculate P&L for scenario analysis.
    
    Args:
        option_prices: List of option prices in scenario
        initial_price: Initial option price
        
    Returns:
        Array of P&L values
    """
    return np.array(option_prices) - initial_price


def find_breakeven_points(x_values: np.ndarray, pnl_values: np.ndarray, 
                         tolerance: float = 0.01) -> List[float]:
    """
    Find breakeven points where P&L crosses zero.
    
    Args:
        x_values: X-axis values (e.g., spot prices)
        pnl_values: P&L values
        tolerance: Tolerance for zero crossing
        
    Returns:
        List of breakeven points
    """
    breakeven_points = []
    
    for i in range(len(pnl_values) - 1):
        if (pnl_values[i] <= 0 <= pnl_values[i + 1]) or \
           (pnl_values[i] >= 0 >= pnl_values[i + 1]):
            # Linear interpolation to find exact crossing point
            if pnl_values[i + 1] != pnl_values[i]:
                x_cross = x_values[i] + (x_values[i + 1] - x_values[i]) * \
                         (-pnl_values[i] / (pnl_values[i + 1] - pnl_values[i]))
                breakeven_points.append(x_cross)
    
    return breakeven_points


def calculate_portfolio_position(position_type: str, spot_price: float, 
                               strike: float, quantity: int, 
                               premium: float) -> float:
    """
    Calculate P&L for a single portfolio position.
    
    Args:
        position_type: Type of position (e.g., "Long Call")
        spot_price: Current spot price
        strike: Strike price
        quantity: Number of contracts
        premium: Premium paid/received
        
    Returns:
        P&L for the position
    """
    if 'Call' in position_type:
        intrinsic = max(spot_price - strike, 0)
    else:  # Put
        intrinsic = max(strike - spot_price, 0)
    
    if 'Long' in position_type:
        pnl = quantity * (intrinsic - premium)
    else:  # Short
        pnl = quantity * (premium - intrinsic)
    
    return pnl


def format_currency(value: float, decimals: int = 2) -> str:
    """Format value as currency string."""
    return f"${value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format value as percentage string."""
    return f"{value:.{decimals}f}%"


def calculate_risk_metrics(simulated_prices: np.ndarray, confidence_levels: List[int]) -> Dict[str, float]:
    """
    Calculate risk metrics from simulated prices.
    
    Args:
        simulated_prices: Array of simulated prices
        confidence_levels: List of confidence levels for VaR
        
    Returns:
        Dictionary of risk metrics
    """
    metrics = {}
    
    for level in confidence_levels:
        percentile = 100 - level
        var = np.percentile(simulated_prices, percentile)
        metrics[f'var_{level}'] = var
        
        # Expected Shortfall (Conditional VaR)
        shortfall_prices = simulated_prices[simulated_prices <= var]
        if len(shortfall_prices) > 0:
            metrics[f'es_{level}'] = shortfall_prices.mean()
        else:
            metrics[f'es_{level}'] = var
    
    return metrics


def create_pricing_dataframe(models: List[str], prices: List[float], 
                           base_price: float) -> pd.DataFrame:
    """
    Create DataFrame for pricing comparison.
    
    Args:
        models: List of model names
        prices: List of prices from each model
        base_price: Base price for comparison (usually Black-Scholes)
        
    Returns:
        DataFrame with pricing comparison
    """
    data = {
        'Model': models,
        'Price': prices,
        'Difference from BS': [p - base_price for p in prices],
        'Difference %': [((p - base_price) / base_price) * 100 if base_price != 0 else 0 
                        for p in prices]
    }
    
    return pd.DataFrame(data)


def generate_market_insights(vol_differential: float) -> Dict[str, str]:
    """
    Generate market insights based on volatility differential.
    
    Args:
        vol_differential: Difference between implied and historical volatility
        
    Returns:
        Dictionary with alert type and message
    """
    if vol_differential > 10:
        return {
            "type": "high",
            "title": "High Implied Volatility",
            "message": "Market expects significant price movement. Consider volatility strategies."
        }
    elif vol_differential < -10:
        return {
            "type": "low",
            "title": "Low Implied Volatility",
            "message": "Market expects minimal price movement. Options may be underpriced."
        }
    else:
        return {
            "type": "normal",
            "title": "Normal Implied Volatility",
            "message": "IV is close to historical volatility. Fair pricing indicated."
        }


def validate_market_price(market_price: float, stock_price: float, 
                         strike_price: float, option_type: OptionType) -> Tuple[bool, Optional[str]]:
    """
    Validate if market price is reasonable.
    
    Args:
        market_price: Observed market price
        stock_price: Current stock price
        strike_price: Option strike price
        option_type: Type of option
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if price is positive
    if market_price <= 0:
        return False, "Market price must be positive"
    
    # Calculate intrinsic value
    if option_type == OptionType.CALL:
        intrinsic = max(stock_price - strike_price, 0)
    else:
        intrinsic = max(strike_price - stock_price, 0)
    
    # Check if price is below intrinsic value
    if market_price < intrinsic:
        return False, "Market price is below intrinsic value"
    
    # Check if price is unreasonably high
    if market_price > stock_price * 0.5:  # More than 50% of stock price
        return False, "Market price seems unreasonably high"
    
    return True, None