"""
Reusable UI components for Options Pricing Analytics
Contains functions to create common UI elements
"""

import streamlit as st
import numpy as np
from typing import Dict, List, Optional, Tuple
from config import COLORS, GREEKS_FORMAT
import utils


def render_header():
    """Render the main application header."""
    st.markdown("""
        <div class="main-header">
            <h1>Options Pricing Analytics</h1>
            <p>Extensive Options Pricing, Greeks Analysis & Risk Management Platform</p>
        </div>
    """, unsafe_allow_html=True)


def render_metric_card(title: str, value: str, subtitle: Optional[str] = None):
    """Render a styled metric card."""
    st.markdown(f"""
        <div class="metric-card">
            <h4>{title}</h4>
            <h2>{value}</h2>
            {f'<p>{subtitle}</p>' if subtitle else ''}
        </div>
    """, unsafe_allow_html=True)


def render_alert(alert_type: str, title: str, message: str):
    """
    Render an alert message.
    
    Args:
        alert_type: Type of alert ('risk', 'success', 'info', 'warning')
        title: Alert title
        message: Alert message
    """
    class_map = {
        'risk': 'risk-alert',
        'success': 'success-alert',
        'info': 'info-alert',
        'warning': 'warning-alert'
    }
    
    alert_class = class_map.get(alert_type, 'info-alert')
    
    st.markdown(f"""
        <div class="{alert_class}">
            <strong>{title}</strong><br>
            {message}
        </div>
    """, unsafe_allow_html=True)


def render_greeks_overview(greeks_result):
    """Render Greeks overview metrics."""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            f"Delta ({GREEKS_FORMAT['delta']['symbol']})", 
            f"{greeks_result.delta:.{GREEKS_FORMAT['delta']['decimals']}f}",
            "Price sensitivity"
        )
    with col2:
        st.metric(
            f"Gamma ({GREEKS_FORMAT['gamma']['symbol']})", 
            f"{greeks_result.gamma:.{GREEKS_FORMAT['gamma']['decimals']}f}",
            "Delta sensitivity"
        )
    with col3:
        st.metric(
            f"Theta ({GREEKS_FORMAT['theta']['symbol']})", 
            f"{greeks_result.theta:.{GREEKS_FORMAT['theta']['decimals']}f}",
            "Time decay"
        )
    with col4:
        st.metric(
            f"Vega ({GREEKS_FORMAT['vega']['symbol']})", 
            f"{greeks_result.vega:.{GREEKS_FORMAT['vega']['decimals']}f}",
            "Vol sensitivity"
        )
    with col5:
        st.metric(
            f"Rho ({GREEKS_FORMAT['rho']['symbol']})", 
            f"{greeks_result.rho:.{GREEKS_FORMAT['rho']['decimals']}f}",
            "Rate sensitivity"
        )


def render_pricing_table(df_pricing):
    """Render formatted pricing comparison table."""
    st.subheader("Pricing Comparison Table")
    st.dataframe(
        df_pricing.style.format({
            'Price': '${:.4f}',
            'Difference from BS': '${:.4f}',
            'Difference %': '{:.2f}%'
        }).background_gradient(cmap='RdYlGn', subset=['Price']),
        use_container_width=True
    )


def render_key_metrics(moneyness: float, moneyness_status: str, 
                      time_value: float, sigma: float, 
                      premium_pct: float, is_high_premium: bool):
    """Render key option metrics."""
    st.subheader("Key Metrics")
    
    st.metric("Moneyness", f"{moneyness:.3f}", moneyness_status)
    st.metric("Time Value", utils.format_currency(time_value, 4))
    st.metric("Annualized Vol", utils.format_percentage(sigma * 100))
    
    # Trade recommendation
    if is_high_premium:
        render_alert('risk', 'High Premium Option', 
                    f'Premium is {premium_pct:.1f}% of stock price')
    else:
        render_alert('success', 'Reasonable Premium', 
                    'Option pricing appears fair')


def render_scenario_statistics(pnl: List[float], spot_prices: List[float]):
    """Render scenario analysis statistics."""
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.metric("Max Profit", utils.format_currency(max(pnl)))
    with col_b:
        st.metric("Max Loss", utils.format_currency(min(pnl)))
    with col_c:
        breakeven_points = utils.find_breakeven_points(spot_prices, pnl)
        if len(breakeven_points) > 0:
            st.metric("Breakeven", utils.format_currency(breakeven_points[0]))
        else:
            st.metric("Breakeven", "N/A")


def render_risk_metrics(metrics: Dict[str, float], stock_price: float):
    """Render risk metrics display."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "VaR (95%)", 
            utils.format_currency(metrics['var_95']), 
            utils.format_currency(metrics['var_95'] - stock_price)
        )
        st.metric(
            "Expected Shortfall (95%)", 
            utils.format_currency(metrics['es_95'])
        )
    
    with col2:
        st.metric(
            "VaR (99%)", 
            utils.format_currency(metrics['var_99']), 
            utils.format_currency(metrics['var_99'] - stock_price)
        )
        st.metric(
            "Expected Shortfall (99%)", 
            utils.format_currency(metrics['es_99'])
        )


def render_payoff_statistics(mean_payoff: float, std_payoff: float, 
                           max_payoff: float, prob_profit: float):
    """Render payoff statistics."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Mean Payoff", utils.format_currency(mean_payoff))
        st.metric("Payoff Volatility", utils.format_currency(std_payoff))
    
    with col2:
        st.metric("Max Payoff", utils.format_currency(max_payoff))
        st.metric("Probability of Profit", utils.format_percentage(prob_profit))


def render_iv_results(implied_vol: float, historical_vol: float, 
                     time_value: float, vol_differential: float):
    """Render implied volatility calculation results."""
    st.success(f"**Implied Volatility: {implied_vol*100:.2f}%**")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.metric("Historical Vol", utils.format_percentage(historical_vol * 100))
        st.metric("Time Value", utils.format_currency(time_value, 4))
    
    with col_b:
        st.metric("Implied Vol", utils.format_percentage(implied_vol * 100))
        st.metric("Vol Differential", f"{vol_differential:+.2f}%")


def render_iv_error(error_message: str):
    """Render IV calculation error message."""
    st.error(f"❌ **Error calculating IV:** {error_message}")
    st.info("💡 **Possible reasons:**")
    st.write("- Market price below intrinsic value")
    st.write("- Very short time to expiry")
    st.write("- Extreme market conditions")
    st.write("- Market price unreasonably high (>50% of stock price)")


def render_surface_statistics(iv_surface):
    """Render volatility surface statistics."""
    st.subheader("Surface Statistics")
    
    col_a, col_b, col_c, col_d = st.columns(4)
    
    with col_a:
        st.metric("Min IV", utils.format_percentage(iv_surface.min()))
    with col_b:
        st.metric("Max IV", utils.format_percentage(iv_surface.max()))
    with col_c:
        st.metric("Mean IV", utils.format_percentage(iv_surface.mean()))
    with col_d:
        st.metric("IV Range", utils.format_percentage(np.ptp(iv_surface)))


def render_portfolio_statistics(portfolio_pnl, spot_range, current_price):
    """Render portfolio analysis statistics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Max Profit", utils.format_currency(max(portfolio_pnl)))
    with col2:
        st.metric("Max Loss", utils.format_currency(min(portfolio_pnl)))
    with col3:
        current_pnl = float(np.interp(current_price, spot_range, portfolio_pnl))
        st.metric("Current P&L", utils.format_currency(current_pnl))
    with col4:
        breakeven_points = utils.find_breakeven_points(spot_range, portfolio_pnl)
        st.metric("Breakeven Points", f"{len(breakeven_points)}")


def render_footer():
    """Render application footer."""
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p>Options Analytics | Built with Streamlit</p>
            <p><em>Disclaimer: This tool is for educational purposes only. 
            Always consult with a financial advisor before making investment decisions.</em></p>
        </div>
    """, unsafe_allow_html=True)


def create_sidebar_inputs() -> Dict:
    """Create and return sidebar input values."""
    from config import DEFAULT_PARAMS, SIMULATION_PARAMS
    
    with st.sidebar:
        st.header("Option Configuration")
        
        # Basic Parameters
        with st.expander("Market Parameters", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                S = st.number_input(
                    "Stock Price ($)", 
                    min_value=0.01, 
                    value=DEFAULT_PARAMS["stock_price"], 
                    step=0.01
                )
                K = st.number_input(
                    "Strike Price ($)", 
                    min_value=0.01, 
                    value=DEFAULT_PARAMS["strike_price"], 
                    step=0.01
                )
                T = st.number_input(
                    "Time to Expiry (Years)", 
                    min_value=0.001, 
                    value=DEFAULT_PARAMS["time_to_expiry"], 
                    step=0.001
                )
            with col2:
                r = st.number_input(
                    "Risk-Free Rate (%)", 
                    min_value=0.0, 
                    value=DEFAULT_PARAMS["risk_free_rate"], 
                    step=0.1
                ) / 100
                sigma = st.number_input(
                    "Volatility (%)", 
                    min_value=0.1, 
                    value=DEFAULT_PARAMS["volatility"], 
                    step=0.1
                ) / 100
                q = st.number_input(
                    "Dividend Yield (%)", 
                    min_value=0.0, 
                    value=DEFAULT_PARAMS["dividend_yield"], 
                    step=0.1
                ) / 100
        
        # Option Configuration
        with st.expander("Option Settings", expanded=True):
            option_type_str = st.selectbox("Option Type", ["Call", "Put"])
            
            # Advanced settings
            st.subheader("Advanced Settings")
            mc_simulations = st.slider(
                "Monte Carlo Simulations", 
                SIMULATION_PARAMS["mc_simulations_min"],
                SIMULATION_PARAMS["mc_simulations_max"],
                SIMULATION_PARAMS["mc_simulations_default"],
                SIMULATION_PARAMS["mc_simulations_step"]
            )
            binomial_steps = st.slider(
                "Binomial Steps", 
                SIMULATION_PARAMS["binomial_steps_min"],
                SIMULATION_PARAMS["binomial_steps_max"],
                SIMULATION_PARAMS["binomial_steps_default"],
                SIMULATION_PARAMS["binomial_steps_step"]
            )
            american_style = st.checkbox("American Style Option", value=False)
            use_antithetic = st.checkbox("Use Antithetic Variates", value=True)
            use_control_variate = st.checkbox("Use Control Variates", value=True)
    
    return {
        "S": S,
        "K": K,
        "T": T,
        "r": r,
        "sigma": sigma,
        "q": q,
        "option_type_str": option_type_str,
        "mc_simulations": mc_simulations,
        "binomial_steps": binomial_steps,
        "american_style": american_style,
        "use_antithetic": use_antithetic,
        "use_control_variate": use_control_variate
    }