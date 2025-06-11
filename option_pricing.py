"""
Options Pricing Analytics - Main Application
Clean, modular implementation using separated components
Optimized for dark mode with purple theme
"""

import streamlit as st
import numpy as np
from datetime import datetime

# Import configurations
from config import PAGE_CONFIG, TABS, SCENARIO_DEFAULTS, IV_SETTINGS, VOL_SURFACE_SETTINGS, PORTFOLIO_SETTINGS

# Import models
from models import (
    OptionParameters, OptionType, BlackScholesModel, 
    MonteCarloModel, BinomialModel, ImpliedVolatilityCalculator
)

# Import utilities
import utils
import visualizations as viz
import components as comp
from styles import load_css

# Page configuration
st.set_page_config(**PAGE_CONFIG)

# Load custom CSS
st.markdown(load_css(), unsafe_allow_html=True)

# Render header
comp.render_header()

# Initialize session state
if 'pricing_history' not in st.session_state:
    st.session_state.pricing_history = []
if 'scenario_data' not in st.session_state:
    st.session_state.scenario_data = None

# Get sidebar inputs
inputs = comp.create_sidebar_inputs()

# Extract values
S = inputs["S"]
K = inputs["K"]
T = inputs["T"]
r = inputs["r"]
sigma = inputs["sigma"]
q = inputs["q"]
option_type = OptionType.CALL if inputs["option_type_str"] == "Call" else OptionType.PUT

# Create option parameters
try:
    params = OptionParameters(S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type, q=q)
except ValueError as e:
    st.error(f"❌ Parameter Error: {e}")
    st.stop()

# Create tabs
tabs = st.tabs(TABS)

# Tab 1: Pricing Models Comparison
with tabs[0]:
    st.header("Multi-Model Pricing Comparison")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Calculate prices using different models
        with st.spinner("Calculating option prices..."):
            # Black-Scholes
            bs_result = BlackScholesModel.calculate_all(params)
            
            # Monte Carlo
            mc_price, S_T = MonteCarloModel.price(
                params, 
                inputs["mc_simulations"], 
                inputs["use_antithetic"], 
                inputs["use_control_variate"]
            )
            
            # Binomial
            bin_price = BinomialModel.price(
                params, 
                inputs["binomial_steps"], 
                inputs["american_style"]
            )
        
        # Create comparison dataframe
        models = ['Black-Scholes', 'Monte Carlo', 'Binomial Tree']
        prices = [bs_result.price, mc_price, bin_price]
        df_pricing = utils.create_pricing_dataframe(models, prices, bs_result.price)
        
        # Display pricing table
        comp.render_pricing_table(df_pricing)
    
    with col2:
        # Calculate key metrics
        moneyness, moneyness_status = utils.calculate_moneyness(S, K)
        time_value = utils.calculate_time_value(bs_result.price, S, K, option_type)
        is_high_premium = utils.is_high_premium_option(bs_result.price, S)
        premium_pct = (bs_result.price / S) * 100 if S > 0 else 0
        
        # Display key metrics
        comp.render_key_metrics(
            moneyness, moneyness_status, time_value, 
            sigma, premium_pct, is_high_premium
        )
    
    # Pricing visualization
    st.subheader("Model Comparison Visualization")
    fig_pricing = viz.create_pricing_comparison_chart(models, prices)
    st.plotly_chart(fig_pricing, use_container_width=True)

# Tab 2: Greeks Analysis
with tabs[1]:
    st.header("Greeks Analysis Dashboard")
    
    # Greeks overview
    comp.render_greeks_overview(bs_result)
    
    # Greeks visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Greeks Spider Chart")
        
        # Prepare Greeks data
        greeks_raw = {
            'Delta': bs_result.delta,
            'Gamma': bs_result.gamma,
            'Theta': bs_result.theta,
            'Vega': bs_result.vega,
            'Rho': bs_result.rho
        }
        
        greeks_normalized = utils.normalize_greeks(greeks_raw)
        
        greeks_data = {
            'Greek': list(greeks_normalized.keys()),
            'Value': list(greeks_normalized.values()),
            'Actual': list(greeks_raw.values())
        }
        
        fig_spider = viz.create_greeks_spider_chart(greeks_data)
        st.plotly_chart(fig_spider, use_container_width=True)
    
    with col2:
        st.subheader("Delta-Gamma Profile")
        
        # Create spot price range
        spot_range = np.linspace(S * 0.7, S * 1.3, 50)
        greeks_range = utils.calculate_greeks_over_range(spot_range, params)
        
        fig_dg = viz.create_delta_gamma_profile(
            spot_range, 
            greeks_range['delta'], 
            greeks_range['gamma'], 
            S
        )
        st.plotly_chart(fig_dg, use_container_width=True)

# Tab 3: Scenario Analysis
with tabs[2]:
    st.header("Scenario Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Scenario Settings")
        
        scenario_type = st.selectbox("Analysis Type", [
            "Spot Price Movement", "Volatility Change", 
            "Time Decay", "Interest Rate Change"
        ])
        
        if scenario_type == "Spot Price Movement":
            min_spot = st.number_input(
                "Min Spot Price", 
                value=S * SCENARIO_DEFAULTS["spot_price_min_factor"], 
                step=0.01
            )
            max_spot = st.number_input(
                "Max Spot Price", 
                value=S * SCENARIO_DEFAULTS["spot_price_max_factor"], 
                step=0.01
            )
            steps = st.slider("Number of Steps", 20, 100, SCENARIO_DEFAULTS["default_steps"])
            
        elif scenario_type == "Volatility Change":
            min_vol = st.number_input(
                "Min Volatility (%)", 
                value=SCENARIO_DEFAULTS["volatility_min"], 
                step=0.1
            ) / 100
            max_vol = st.number_input(
                "Max Volatility (%)", 
                value=SCENARIO_DEFAULTS["volatility_max"], 
                step=0.1
            ) / 100
            steps = st.slider("Number of Steps", 20, 100, SCENARIO_DEFAULTS["default_steps"])
            
        elif scenario_type == "Time Decay":
            time_steps = st.slider(
                "Days to Analyze", 
                1, 365, 
                SCENARIO_DEFAULTS["time_decay_default_days"]
            )
            steps = time_steps
            
        run_scenario = st.button("Run Scenario Analysis", type="primary")
    
    with col2:
        if run_scenario:
            st.subheader(f"{scenario_type} Analysis Results")
            
            with st.spinner("Running scenario analysis..."):
                if scenario_type == "Spot Price Movement":
                    spot_prices = np.linspace(min_spot, max_spot, steps)
                    option_prices = []
                    
                    for spot in spot_prices:
                        temp_params = OptionParameters(spot, K, T, r, sigma, option_type, q)
                        result = BlackScholesModel.calculate_all(temp_params)
                        option_prices.append(result.price)
                    
                    # Calculate P&L
                    pnl = utils.calculate_scenario_pnl(option_prices, bs_result.price)
                    
                    # Create visualization
                    fig_scenario = viz.create_scenario_analysis_chart(
                        spot_prices, option_prices, pnl, 
                        "Stock Price ($)", S
                    )
                    st.plotly_chart(fig_scenario, use_container_width=True)
                    
                    # Display statistics
                    comp.render_scenario_statistics(pnl, spot_prices)

# Tab 4: Risk Metrics
with tabs[3]:
    st.header("Comprehensive Risk Analysis")
    
    # Monte Carlo risk analysis
    with st.spinner("Calculating risk metrics..."):
        mc_price, S_T = MonteCarloModel.price(params, inputs["mc_simulations"])
        
        # Calculate option payoffs
        if option_type == OptionType.CALL:
            payoffs = np.maximum(S_T - K, 0)
        else:
            payoffs = np.maximum(K - S_T, 0)
        
        # Calculate risk metrics
        risk_metrics = utils.calculate_risk_metrics(S_T, [95, 99])
        
        # Payoff statistics
        mean_payoff = np.mean(payoffs)
        std_payoff = np.std(payoffs)
        max_payoff = np.max(payoffs)
        prob_profit = np.mean(payoffs > bs_result.price) * 100
    
    # Display risk metrics
    col1, col2 = st.columns(2)
    
    with col1:
        comp.render_risk_metrics(risk_metrics, S)
    
    with col2:
        comp.render_payoff_statistics(mean_payoff, std_payoff, max_payoff, prob_profit)
    
    # Risk visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price Distribution at Expiry")
        
        vlines = [
            {"value": risk_metrics['var_95'], "color": "red", "text": f"VaR 95%: ${risk_metrics['var_95']:.2f}"},
            {"value": S, "color": "green", "text": f"Current: ${S:.2f}"}
        ]
        
        fig_dist = viz.create_distribution_histogram(
            S_T, "Simulated Stock Price Distribution", 
            "Stock Price ($)", vlines
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        st.subheader("Payoff Distribution")
        
        vlines = [
            {"value": bs_result.price, "color": "orange", "text": f"Premium: ${bs_result.price:.2f}"}
        ]
        
        fig_payoff = viz.create_distribution_histogram(
            payoffs, "Option Payoff Distribution", 
            "Payoff ($)", vlines
        )
        st.plotly_chart(fig_payoff, use_container_width=True)

# Tab 5: Implied Volatility
with tabs[4]:
    st.header("Implied Volatility Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("IV Calculator")
        
        market_price = st.number_input(
            "Market Price ($)", 
            min_value=0.01, 
            value=bs_result.price, 
            step=0.01,
            help="Enter the observed market price of the option"
        )
        
        # Advanced IV settings
        with st.expander("Advanced IV Settings"):
            max_iterations = st.slider(
                "Max Iterations", 
                IV_SETTINGS["max_iterations_min"],
                IV_SETTINGS["max_iterations_max"],
                IV_SETTINGS["max_iterations_default"]
            )
            tolerance = st.selectbox(
                "Tolerance", 
                IV_SETTINGS["tolerance_options"],
                index=IV_SETTINGS["tolerance_default_index"]
            )
            min_vol = st.number_input(
                "Min Volatility (%)", 
                value=IV_SETTINGS["min_vol_default"], 
                step=0.1
            ) / 100
            max_vol = st.number_input(
                "Max Volatility (%)", 
                value=IV_SETTINGS["max_vol_default"], 
                step=10.0
            ) / 100
        
        calculate_iv = st.button("Calculate Implied Volatility", type="primary")
        
        if calculate_iv:
            # Validate market price
            is_valid, error_msg = utils.validate_market_price(
                market_price, S, K, option_type
            )
            
            if not is_valid:
                comp.render_iv_error(error_msg)
            else:
                try:
                    with st.spinner("Calculating implied volatility..."):
                        implied_vol = ImpliedVolatilityCalculator.calculate(
                            market_price, params, max_iterations, 
                            tolerance, min_vol, max_vol
                        )
                        
                        # Calculate related metrics
                        if option_type == OptionType.CALL:
                            intrinsic = max(S - K, 0)
                        else:
                            intrinsic = max(K - S, 0)
                        
                        time_value = market_price - intrinsic
                        vol_diff = (implied_vol - sigma) * 100
                        
                        # Display results
                        comp.render_iv_results(
                            implied_vol, sigma, time_value, vol_diff
                        )
                        
                        # Market insights
                        st.subheader("Market Insights")
                        insights = utils.generate_market_insights(vol_diff)
                        
                        alert_type_map = {
                            'high': 'risk',
                            'low': 'info',
                            'normal': 'success'
                        }
                        
                        comp.render_alert(
                            alert_type_map[insights['type']],
                            insights['title'],
                            insights['message']
                        )
                        
                except Exception as e:
                    st.error(f"❌ **Unexpected error:** {str(e)}")
                    st.info("Please check your input parameters and try again.")
    
    with col2:
        st.subheader("Volatility Surface Analysis")
        
        # Surface generation options
        with st.expander("Surface Settings"):
            surface_points = st.slider(
                "Surface Resolution", 
                VOL_SURFACE_SETTINGS["resolution_min"],
                VOL_SURFACE_SETTINGS["resolution_max"],
                VOL_SURFACE_SETTINGS["resolution_default"]
            )
            vol_smile = st.checkbox("Include Volatility Smile", value=True)
            base_vol_surface = st.slider(
                "Base Volatility (%)", 
                VOL_SURFACE_SETTINGS["base_vol_min"],
                VOL_SURFACE_SETTINGS["base_vol_max"],
                VOL_SURFACE_SETTINGS["base_vol_default"]
            ) / 100
        
        # Generate volatility surface
        with st.spinner("Generating volatility surface..."):
            try:
                # Create strike and time ranges
                strikes = np.linspace(K * 0.7, K * 1.3, surface_points)
                times = np.linspace(0.05, 1.0, surface_points)
                
                # Generate IV surface
                iv_surface = ImpliedVolatilityCalculator.calculate_iv_surface(
                    S, strikes, times, r, option_type, q, 
                    base_vol_surface, vol_smile
                )
                
                # Create visualization
                fig_surface = viz.create_volatility_surface(strikes, times, iv_surface)
                st.plotly_chart(fig_surface, use_container_width=True)
                
                # Surface statistics
                comp.render_surface_statistics(iv_surface)
                
            except Exception as e:
                st.error(f"Error generating volatility surface: {str(e)}")

# Tab 6: Advanced Analytics
with tabs[5]:
    st.header("Advanced Portfolio Analytics")
    
    # Portfolio simulation
    st.subheader("Multi-Option Portfolio Analysis")
    
    with st.expander("Portfolio Builder", expanded=True):
        num_positions = st.selectbox(
            "Number of Positions", 
            list(range(1, PORTFOLIO_SETTINGS["max_positions"] + 1))
        )
        
        portfolio_data = []
        for i in range(num_positions):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                pos_type = st.selectbox(
                    f"Position {i+1} Type", 
                    PORTFOLIO_SETTINGS["position_types"], 
                    key=f"type_{i}"
                )
            with col2:
                pos_strike = st.number_input(
                    f"Strike {i+1}", 
                    value=K, 
                    key=f"strike_{i}"
                )
            with col3:
                pos_quantity = st.number_input(
                    f"Quantity {i+1}", 
                    value=1, 
                    key=f"qty_{i}"
                )
            with col4:
                pos_premium = st.number_input(
                    f"Premium {i+1}", 
                    value=bs_result.price, 
                    key=f"prem_{i}"
                )
            
            portfolio_data.append({
                'type': pos_type,
                'strike': pos_strike,
                'quantity': pos_quantity,
                'premium': pos_premium
            })
    
    if st.button("Analyze Portfolio"):
        # Calculate portfolio P&L
        spot_range = np.linspace(
            S * PORTFOLIO_SETTINGS["pnl_spot_min_factor"], 
            S * PORTFOLIO_SETTINGS["pnl_spot_max_factor"], 
            PORTFOLIO_SETTINGS["pnl_points"]
        )
        portfolio_pnl = np.zeros(len(spot_range))
        
        for i, current_spot in enumerate(spot_range):
            total_pnl = 0
            
            for pos in portfolio_data:
                pnl = utils.calculate_portfolio_position(
                    pos['type'], current_spot, pos['strike'],
                    pos['quantity'], pos['premium']
                )
                total_pnl += pnl
            
            portfolio_pnl[i] = total_pnl
        
        # Portfolio visualization
        fig_portfolio = viz.create_portfolio_pnl_chart(spot_range, portfolio_pnl, S)
        st.plotly_chart(fig_portfolio, use_container_width=True)
        
        # Portfolio statistics
        comp.render_portfolio_statistics(portfolio_pnl, spot_range, S)

# Footer
comp.render_footer()