import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import our enhanced models
from models import (
    OptionParameters, OptionType, ModelType, PricingResult,
    BlackScholesModel, MonteCarloModel, BinomialModel,
    ImpliedVolatilityCalculator
)

# Import custom styles
from styles import load_css # Tambahkan baris ini

# Page configuration
st.set_page_config(
    page_title="Options Pricing Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown(load_css(), unsafe_allow_html=True) # Ubah baris ini

# Main header
st.markdown("""
    <div class="main-header">
        <h1>Options Pricing Analytics</h1>
        <p>Extensive Options Pricing, Greeks Analysis & Risk Management Platform</p>
    </div>
""", unsafe_allow_html=True)

# Initialize session state
if 'pricing_history' not in st.session_state:
    st.session_state.pricing_history = []
if 'scenario_data' not in st.session_state:
    st.session_state.scenario_data = None

# Sidebar configuration
with st.sidebar:
    st.header("Option Configuration")
    
    # Basic Parameters
    with st.expander("Market Parameters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            S = st.number_input("Stock Price ($)", min_value=0.01, value=100.0, step=0.01)
            K = st.number_input("Strike Price ($)", min_value=0.01, value=100.0, step=0.01)
            T = st.number_input("Time to Expiry (Years)", min_value=0.001, value=0.25, step=0.001)
        with col2:
            r = st.number_input("Risk-Free Rate (%)", min_value=0.0, value=5.0, step=0.1) / 100
            sigma = st.number_input("Volatility (%)", min_value=0.1, value=20.0, step=0.1) / 100
            q = st.number_input("Dividend Yield (%)", min_value=0.0, value=0.0, step=0.1) / 100
    
    # Option Configuration
    with st.expander("Option Settings", expanded=True):
        option_type_str = st.selectbox("Option Type", ["Call", "Put"])
        option_type = OptionType.CALL if option_type_str == "Call" else OptionType.PUT
        
        # Advanced settings
        st.subheader("Advanced Settings")
        mc_simulations = st.slider("Monte Carlo Simulations", 10000, 200000, 100000, 10000)
        binomial_steps = st.slider("Binomial Steps", 50, 300, 100, 10)
        american_style = st.checkbox("American Style Option", value=False)
        use_antithetic = st.checkbox("Use Antithetic Variates", value=True)
        use_control_variate = st.checkbox("Use Control Variates", value=True)

# Create option parameters
try:
    params = OptionParameters(S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type, q=q)
except ValueError as e:
    st.error(f"❌ Parameter Error: {e}")
    st.stop()

# Main dashboard tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Pricing Models", "Greeks Analysis", "Scenario Analysis", 
    "Risk Metrics", "Implied Volatility", "Advanced Analytics"
])

# Tab 1: Pricing Models Comparison
with tab1:
    st.header("Multi-Model Pricing Comparison")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Calculate prices using different models
        with st.spinner("Calculating option prices..."):
            # Black-Scholes
            bs_result = BlackScholesModel.calculate_all(params)
            
            # Monte Carlo
            mc_price, S_T = MonteCarloModel.price(
                params, mc_simulations, use_antithetic, use_control_variate
            )
            
            # Binomial
            bin_price = BinomialModel.price(params, binomial_steps, american_style)
        
        # Create comparison dataframe
        pricing_data = {
            'Model': ['Black-Scholes', 'Monte Carlo', 'Binomial Tree'],
            'Price': [bs_result.price, mc_price, bin_price],
            'Difference from BS': [0, mc_price - bs_result.price, bin_price - bs_result.price],
            'Difference %': [0, ((mc_price - bs_result.price) / bs_result.price) * 100,
                           ((bin_price - bs_result.price) / bs_result.price) * 100]
        }
        
        df_pricing = pd.DataFrame(pricing_data)
        
        # Display pricing table
        cmap = 'RdYlGn'  # atau coba 'RdYlGn' untuk sinyal positif-negatif
        st.subheader("Pricing Comparison Table")
        st.dataframe(
            df_pricing.style.format({
                'Price': '${:.4f}',
                'Difference from BS': '${:.4f}',
                'Difference %': '{:.2f}%'
            }).background_gradient(cmap=cmap, subset=['Price']),
            use_container_width=True
        )
    
    with col2:
        # Key metrics cards
        st.subheader("Key Metrics")
        
        # Moneyness
        moneyness = S / K
        if moneyness > 1.05:
            moneyness_status = "ITM"
        elif moneyness < 0.95:
            moneyness_status = "OTM"
        else:
            moneyness_status = "ATM"
        
        st.metric("Moneyness", f"{moneyness:.3f}", moneyness_status)
        st.metric("Time Value", f"${bs_result.price - max(S-K if option_type==OptionType.CALL else K-S, 0):.4f}")
        st.metric("Annualized Vol", f"{sigma*100:.1f}%")
        
        # Quick trade recommendation
        if bs_result.price > 0:
            premium_pct = (bs_result.price / S) * 100
            if premium_pct > 5:
                st.markdown('<div class="risk-alert">High Premium Option</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-alert">Reasonable Premium</div>', unsafe_allow_html=True)
    
    # Pricing visualization
    st.subheader("Model Comparison Visualization")
    fig_pricing = go.Figure()
    
    models = ['Black-Scholes', 'Monte Carlo', 'Binomial']
    prices = [bs_result.price, mc_price, bin_price]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    fig_pricing.add_trace(go.Bar(
        x=models,
        y=prices,
        marker_color=colors,
        text=[f'${p:.4f}' for p in prices],
        textposition='auto',
    ))
    
    fig_pricing.update_layout(
        title="Option Price Comparison Across Models",
        yaxis_title="Option Price ($)",
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig_pricing, use_container_width=True)

# Tab 2: Greeks Analysis
with tab2:
    st.header("Greeks Analysis Dashboard")
    
    # Greeks overview
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Delta (Δ)", f"{bs_result.delta:.4f}", 
                 "Price sensitivity")
    with col2:
        st.metric("Gamma (Γ)", f"{bs_result.gamma:.4f}",
                 "Delta sensitivity")
    with col3:
        st.metric("Theta (Θ)", f"{bs_result.theta:.4f}",
                 "Time decay")
    with col4:
        st.metric("Vega (ν)", f"{bs_result.vega:.4f}",
                 "Vol sensitivity")
    with col5:
        st.metric("Rho (ρ)", f"{bs_result.rho:.4f}",
                 "Rate sensitivity")
    
    # Greeks visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Greeks Spider Chart")
        
        # --- MODIFIED CODE START ---
        # Normalize Greeks for spider chart
        # Find the absolute maximum value among the Greeks for scaling
        all_greeks_values = [
            abs(bs_result.delta), 
            abs(bs_result.gamma),
            abs(bs_result.theta), 
            abs(bs_result.vega), 
            abs(bs_result.rho)
        ]
        
        # Handle cases where all Greeks might be zero or very small to avoid division by zero
        # Add a small epsilon to max_abs_greek if it's zero
        max_abs_greek = max(all_greeks_values)
        if max_abs_greek == 0:
            max_abs_greek = 1 # Fallback to 1 to prevent division by zero

        # Apply normalization to each Greek value
        greeks_data = {
            'Greek': ['Delta', 'Gamma', 'Theta', 'Vega', 'Rho'],
            'Value': [
                bs_result.delta / max_abs_greek,
                bs_result.gamma / max_abs_greek,
                bs_result.theta / max_abs_greek, 
                bs_result.vega / max_abs_greek, 
                bs_result.rho / max_abs_greek
            ],
            'Actual': [bs_result.delta, bs_result.gamma, bs_result.theta,
                      bs_result.vega, bs_result.rho]
        }
        # --- MODIFIED CODE END ---
        
        fig_spider = go.Figure()
        
        fig_spider.add_trace(go.Scatterpolar(
            r=greeks_data['Value'],
            theta=greeks_data['Greek'],
            fill='toself',
            name='Greeks Profile',
            line_color='#667eea'
        ))
        
        fig_spider.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[-1, 1] # Range adjusted to accommodate normalized values
                )),
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig_spider, use_container_width=True)
    
    with col2:
        st.subheader("Delta-Gamma Profile")
        
        # Create spot price range for Greeks analysis
        spot_range = np.linspace(S * 0.7, S * 1.3, 50)
        deltas = []
        gammas = []
        
        for spot in spot_range:
            temp_params = OptionParameters(spot, K, T, r, sigma, option_type, q)
            temp_result = BlackScholesModel.calculate_all(temp_params)
            deltas.append(temp_result.delta)
            gammas.append(temp_result.gamma)
        
        fig_dg = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_dg.add_trace(
            go.Scatter(x=spot_range, y=deltas, name="Delta", line=dict(color='#FF6B6B')),
            secondary_y=False,
        )
        
        fig_dg.add_trace(
            go.Scatter(x=spot_range, y=gammas, name="Gamma", line=dict(color='#4ECDC4')),
            secondary_y=True,
        )
        
        fig_dg.add_vline(x=S, line_dash="dash", annotation_text="Current Price")
        
        fig_dg.update_yaxes(title_text="Delta", secondary_y=False)
        fig_dg.update_yaxes(title_text="Gamma", secondary_y=True)
        fig_dg.update_xaxes(title_text="Stock Price ($)")
        
        st.plotly_chart(fig_dg, use_container_width=True)

# Tab 3: Scenario Analysis
with tab3:
    st.header("Scenario Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Scenario Settings")
        
        scenario_type = st.selectbox("Analysis Type", [
            "Spot Price Movement", "Volatility Change", "Time Decay", "Interest Rate Change"
        ])
        
        if scenario_type == "Spot Price Movement":
            min_spot = st.number_input("Min Spot Price", value=S*0.5, step=0.01)
            max_spot = st.number_input("Max Spot Price", value=S*1.5, step=0.01)
            steps = st.slider("Number of Steps", 20, 100, 50)
            
        elif scenario_type == "Volatility Change":
            min_vol = st.number_input("Min Volatility (%)", value=5.0, step=0.1) / 100
            max_vol = st.number_input("Max Volatility (%)", value=50.0, step=0.1) / 100
            steps = st.slider("Number of Steps", 20, 100, 50)
            
        elif scenario_type == "Time Decay":
            time_steps = st.slider("Days to Analyze", 1, 365, 30)
            steps = time_steps
            
        run_scenario = st.button("Run Scenario Analysis", type="primary")
    
    with col2:
        if run_scenario:
            st.subheader(f"{scenario_type} Analysis Results")
            
            with st.spinner("Running scenario analysis..."):
                if scenario_type == "Spot Price Movement":
                    spot_prices = np.linspace(min_spot, max_spot, steps)
                    option_prices = []
                    deltas = []
                    
                    for spot in spot_prices:
                        temp_params = OptionParameters(spot, K, T, r, sigma, option_type, q)
                        result = BlackScholesModel.calculate_all(temp_params)
                        option_prices.append(result.price)
                        deltas.append(result.delta)
                    
                    # Create P&L analysis
                    initial_price = bs_result.price
                    pnl = np.array(option_prices) - initial_price
                    
                    fig_scenario = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=('Option Price vs Spot Price', 'P&L Analysis'),
                        vertical_spacing=0.1
                    )
                    
                    fig_scenario.add_trace(
                        go.Scatter(x=spot_prices, y=option_prices, name="Option Price",
                                 line=dict(color='#667eea', width=3)),
                        row=1, col=1
                    )
                    
                    fig_scenario.add_trace(
                        go.Scatter(x=spot_prices, y=pnl, name="P&L",
                                 line=dict(color='#FF6B6B', width=3),
                                 fill='tonexty'),
                        row=2, col=1
                    )
                    
                    fig_scenario.add_vline(x=S, line_dash="dash", annotation_text="Current Price")
                    fig_scenario.add_hline(y=0, line_dash="dash", row=2, col=1)
                    
                    fig_scenario.update_xaxes(title_text="Stock Price ($)", row=2, col=1)
                    fig_scenario.update_yaxes(title_text="Option Price ($)", row=1, col=1)
                    fig_scenario.update_yaxes(title_text="P&L ($)", row=2, col=1)
                    
                    st.plotly_chart(fig_scenario, use_container_width=True)
                    
                    # Scenario statistics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Max Profit", f"${max(pnl):.2f}")
                    with col_b:
                        st.metric("Max Loss", f"${min(pnl):.2f}")
                    with col_c:
                        breakeven_spots = spot_prices[np.abs(pnl) < 0.01]
                        if len(breakeven_spots) > 0:
                            st.metric("Breakeven", f"${breakeven_spots[0]:.2f}")
                        else:
                            st.metric("Breakeven", "N/A")

# Tab 4: Risk Metrics
with tab4:
    st.header("Comprehensive Risk Analysis")
    
    # Monte Carlo risk analysis
    with st.spinner("Calculating risk metrics..."):
        mc_price, S_T = MonteCarloModel.price(params, mc_simulations)
        
        # Calculate option payoffs at expiration
        if option_type == OptionType.CALL:
            payoffs = np.maximum(S_T - K, 0)
        else:
            payoffs = np.maximum(K - S_T, 0)
        
        # Risk metrics
        var_95 = np.percentile(S_T, 5)
        var_99 = np.percentile(S_T, 1)
        es_95 = S_T[S_T <= var_95].mean() if len(S_T[S_T <= var_95]) > 0 else var_95
        es_99 = S_T[S_T <= var_99].mean() if len(S_T[S_T <= var_99]) > 0 else var_99
        
        # Payoff statistics
        mean_payoff = np.mean(payoffs)
        std_payoff = np.std(payoffs)
        max_payoff = np.max(payoffs)
        prob_profit = np.mean(payoffs > bs_result.price) * 100
    
    # Risk metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("VaR (95%)", f"${var_95:.2f}", f"${var_95 - S:.2f}")
        st.metric("Expected Shortfall (95%)", f"${es_95:.2f}")
    
    with col2:
        st.metric("VaR (99%)", f"${var_99:.2f}", f"${var_99 - S:.2f}")
        st.metric("Expected Shortfall (99%)", f"${es_99:.2f}")
    
    with col3:
        st.metric("Mean Payoff", f"${mean_payoff:.2f}")
        st.metric("Payoff Volatility", f"${std_payoff:.2f}")
    
    with col4:
        st.metric("Max Payoff", f"${max_payoff:.2f}")
        st.metric("Probability of Profit", f"{prob_profit:.1f}%")
    
    # Risk visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price Distribution at Expiry")
        
        fig_dist = go.Figure()
        
        fig_dist.add_trace(go.Histogram(
            x=S_T,
            nbinsx=50,
            name="Price Distribution",
            marker_color='#667eea',
            opacity=0.7
        ))
        
        fig_dist.add_vline(x=var_95, line_dash="dash", line_color="red",
                          annotation_text=f"VaR 95%: ${var_95:.2f}")
        fig_dist.add_vline(x=S, line_dash="dash", line_color="green",
                          annotation_text=f"Current: ${S:.2f}")
        
        fig_dist.update_layout(
            title="Simulated Stock Price Distribution",
            xaxis_title="Stock Price ($)",
            yaxis_title="Frequency"
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        st.subheader("Payoff Distribution")
        
        fig_payoff = go.Figure()
        
        fig_payoff.add_trace(go.Histogram(
            x=payoffs,
            nbinsx=50,
            name="Payoff Distribution",
            marker_color='#4ECDC4',
            opacity=0.7
        ))
        
        fig_payoff.add_vline(x=bs_result.price, line_dash="dash", line_color="orange",
                           annotation_text=f"Premium: ${bs_result.price:.2f}")
        
        fig_payoff.update_layout(
            title="Option Payoff Distribution",
            xaxis_title="Payoff ($)",
            yaxis_title="Frequency"
        )
        
        st.plotly_chart(fig_payoff, use_container_width=True)

# Tab 5: Implied Volatility (FIXED VERSION)
with tab5:
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
            max_iterations = st.slider("Max Iterations", 50, 200, 100)
            tolerance = st.selectbox("Tolerance", [1e-4, 1e-5, 1e-6], index=2)
            min_vol = st.number_input("Min Volatility (%)", value=0.1, step=0.1) / 100
            max_vol = st.number_input("Max Volatility (%)", value=500.0, step=10.0) / 100
        
        calculate_iv = st.button("Calculate Implied Volatility", type="primary")
        
        if calculate_iv:
            try:
                with st.spinner("Calculating implied volatility..."):
                    # Use the enhanced IV calculator
                    implied_vol = ImpliedVolatilityCalculator.calculate(
                        market_price, params, max_iterations, tolerance, min_vol, max_vol
                    )
                    
                st.success(f"**Implied Volatility: {implied_vol*100:.2f}%**")
                
                # Detailed analysis
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.metric("Historical Vol", f"{sigma*100:.2f}%")
                    
                    # Calculate intrinsic value
                    if option_type == OptionType.CALL:
                        intrinsic = max(S - K, 0)
                    else:
                        intrinsic = max(K - S, 0)
                    
                    time_value = market_price - intrinsic
                    st.metric("Time Value", f"${time_value:.4f}")
                
                with col_b:
                    st.metric("Implied Vol", f"{implied_vol*100:.2f}%")
                    
                    # Vol differential
                    vol_diff = (implied_vol - sigma) * 100
                    st.metric("Vol Differential", f"{vol_diff:+.2f}%")
                
                # Market insights
                st.subheader("Market Insights")
                
                if vol_diff > 10:
                    st.markdown("""
                    <div class="risk-alert">
                        <strong>High Implied Volatility</strong><br>
                        Market expects significant price movement. Consider volatility strategies.
                    </div>
                    """, unsafe_allow_html=True)
                elif vol_diff < -10:
                    st.markdown("""
                    <div style="background: #0066cc; color: white; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                        <strong>Low Implied Volatility</strong><br>
                        Market expects minimal price movement. Options may be underpriced.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="success-alert">
                        <strong>Normal Implied Volatility</strong><br>
                        IV is close to historical volatility. Fair pricing indicated.
                    </div>
                    """, unsafe_allow_html=True)
                    
            except ValueError as e:
                st.error(f"❌ **Error calculating IV:** {str(e)}")
                st.info("💡 **Possible reasons:**")
                st.write("- Market price below intrinsic value")
                st.write("- Very short time to expiry")
                st.write("- Extreme market conditions")
                
            except Exception as e:
                st.error(f"❌ **Unexpected error:** {str(e)}")
                st.info("Please check your input parameters and try again.")
    
    with col2:
        st.subheader("Volatility Surface Analysis")
        
        # Surface generation options
        with st.expander("Surface Settings"):
            surface_points = st.slider("Surface Resolution", 5, 15, 10)
            vol_smile = st.checkbox("Include Volatility Smile", value=True)
            base_vol_surface = st.slider("Base Volatility (%)", 10, 50, 20) / 100
        
        # Generate volatility surface
        with st.spinner("Generating volatility surface..."):
            try:
                # Create strike and time ranges
                strikes = np.linspace(K * 0.7, K * 1.3, surface_points)
                times = np.linspace(0.05, 1.0, surface_points)
                
                # Generate IV surface using the enhanced method
                iv_surface = ImpliedVolatilityCalculator.calculate_iv_surface(
                    S, strikes, times, r, option_type, q, base_vol_surface, vol_smile
                )
                
                # Create 3D surface plot
                fig_surface = go.Figure(data=[go.Surface(
                    z=iv_surface,
                    x=strikes,
                    y=times,
                    colorscale='Viridis',
                    colorbar=dict(title=dict(text="Implied Vol (%)", side="right")),
                    showscale=True
                )])
                
                fig_surface.update_layout(
                    title='Implied Volatility Surface',
                    scene=dict(
                        xaxis_title='Strike Price ($)',
                        yaxis_title='Time to Expiry (Years)',
                        zaxis_title='Implied Volatility (%)',
                        camera=dict(eye=dict(x=1.2, y=1.2, z=0.8))
                    ),
                    height=500,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                
                st.plotly_chart(fig_surface, use_container_width=True)
                
                # Surface statistics
                st.subheader("Surface Statistics")
                
                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    st.metric("Min IV", f"{np.min(iv_surface):.1f}%")
                with col_b:
                    st.metric("Max IV", f"{np.max(iv_surface):.1f}%")
                with col_c:
                    st.metric("Mean IV", f"{np.mean(iv_surface):.1f}%")
                with col_d:
                    st.metric("IV Range", f"{np.ptp(iv_surface):.1f}%")
                
            except Exception as e:
                st.error(f"Error generating volatility surface: {str(e)}")
                st.info("Using simplified visualization...")
                
                # Fallback: Simple 2D heatmap
                fig_heatmap = px.imshow(
                    np.random.rand(10, 10) * 30 + 15,  # Dummy data
                    title="Volatility Heatmap (Simplified)",
                    color_continuous_scale='Viridis',
                    labels=dict(color="Implied Vol (%)")
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Volatility term structure
    st.subheader("Volatility Term Structure")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ATM volatility term structure
        try:
            times_ts = np.linspace(0.05, 2.0, 20)
            iv_term_structure = []
            
            for t in times_ts:
                if t > 0:
                    temp_params = OptionParameters(S, K, t, r, sigma, option_type, q)
                    temp_price = BlackScholesModel.price(temp_params)
                    
                    # Add some realistic term structure
                    market_price_ts = temp_price * (1 + 0.1 * np.exp(-2*t) + np.random.normal(0, 0.01))
                    
                    try:
                        iv_ts = ImpliedVolatilityCalculator.calculate(market_price_ts, temp_params)
                        iv_term_structure.append(iv_ts * 100)
                    except:
                        iv_term_structure.append(sigma * 100)
                else:
                    iv_term_structure.append(sigma * 100)
            
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=times_ts,
                y=iv_term_structure,
                mode='lines+markers',
                name='ATM Implied Volatility',
                line=dict(color='#667eea', width=3),
                marker=dict(size=6)
            ))
            
            fig_ts.add_hline(y=sigma*100, line_dash="dash", 
                           annotation_text=f"Historical Vol: {sigma*100:.1f}%")
            
            fig_ts.update_layout(
                title='At-The-Money Volatility Term Structure',
                xaxis_title='Time to Expiry (Years)',
                yaxis_title='Implied Volatility (%)',
                height=400
            )
            
            st.plotly_chart(fig_ts, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error generating term structure: {str(e)}")
    
    with col2:
        # Volatility smile/skew
        try:
            strikes_smile = np.linspace(K * 0.8, K * 1.2, 15)
            iv_smile = []
            
            for strike in strikes_smile:
                temp_params = OptionParameters(S, strike, T, r, sigma, option_type, q)
                temp_price = BlackScholesModel.price(temp_params)
                
                # Add volatility smile effect
                moneyness = np.log(S / strike)
                smile_effect = 0.1 * moneyness**2
                market_price_smile = temp_price * (1 + smile_effect + np.random.normal(0, 0.01))
                
                try:
                    iv_smile_val = ImpliedVolatilityCalculator.calculate(market_price_smile, temp_params)
                    iv_smile.append(iv_smile_val * 100)
                except:
                    iv_smile.append(sigma * 100)
            
            fig_smile = go.Figure()
            fig_smile.add_trace(go.Scatter(
                x=strikes_smile,
                y=iv_smile,
                mode='lines+markers',
                name='Volatility Smile',
                line=dict(color='#4ECDC4', width=3),
                marker=dict(size=6)
            ))
            
            fig_smile.add_vline(x=K, line_dash="dash", 
                              annotation_text=f"Strike: ${K:.2f}")
            fig_smile.add_vline(x=S, line_dash="dash", line_color="green",
                              annotation_text=f"Spot: ${S:.2f}")
            
            fig_smile.update_layout(
                title='Volatility Smile/Skew',
                xaxis_title='Strike Price ($)',
                yaxis_title='Implied Volatility (%)',
                height=400
            )
            
            st.plotly_chart(fig_smile, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error generating volatility smile: {str(e)}")

# Tab 6: Advanced Analytics
with tab6:
    st.header("Advanced Portfolio Analytics")
    
    # Portfolio simulation
    st.subheader("Multi-Option Portfolio Analysis")
    
    with st.expander("Portfolio Builder", expanded=True):
        num_positions = st.selectbox("Number of Positions", [1, 2, 3, 4, 5])
        
        portfolio_data = []
        for i in range(num_positions):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                pos_type = st.selectbox(f"Position {i+1} Type", ["Long Call", "Short Call", "Long Put", "Short Put"], key=f"type_{i}")
            with col2:
                pos_strike = st.number_input(f"Strike {i+1}", value=K, key=f"strike_{i}")
            with col3:
                pos_quantity = st.number_input(f"Quantity {i+1}", value=1, key=f"qty_{i}")
            with col4:
                pos_premium = st.number_input(f"Premium {i+1}", value=bs_result.price, key=f"prem_{i}")
            
            portfolio_data.append({
                'type': pos_type,
                'strike': pos_strike,
                'quantity': pos_quantity,
                'premium': pos_premium
            })
    
    if st.button("Analyze Portfolio"):
        # Calculate portfolio P&L
        spot_range = np.linspace(S * 0.5, S * 1.5, 100)
        portfolio_pnl = np.zeros(len(spot_range))
        
        for spot in range(len(spot_range)):
            current_spot = spot_range[spot]
            total_pnl = 0
            
            for pos in portfolio_data:
                strike = pos['strike']
                quantity = pos['quantity']
                premium = pos['premium']
                
                if 'Call' in pos['type']:
                    intrinsic = max(current_spot - strike, 0)
                else:
                    intrinsic = max(strike - current_spot, 0)
                
                if 'Long' in pos['type']:
                    pnl = quantity * (intrinsic - premium)
                else:
                    pnl = quantity * (premium - intrinsic)
                
                total_pnl += pnl
            
            portfolio_pnl[spot] = total_pnl
        
        # Portfolio visualization
        fig_portfolio = go.Figure()
        
        fig_portfolio.add_trace(go.Scatter(
            x=spot_range,
            y=portfolio_pnl,
            mode='lines',
            name='Portfolio P&L',
            line=dict(color='#667eea', width=3),
            fill='tonexty'
        ))
        
        fig_portfolio.add_hline(y=0, line_dash="dash")
        fig_portfolio.add_vline(x=S, line_dash="dash", annotation_text="Current Price")
        
        fig_portfolio.update_layout(
            title="Portfolio P&L Diagram",
            xaxis_title="Stock Price at Expiry ($)",
            yaxis_title="Profit/Loss ($)",
            height=500
        )
        
        st.plotly_chart(fig_portfolio, use_container_width=True)
        
        # Portfolio statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Max Profit", f"${np.max(portfolio_pnl):.2f}")
        with col2:
            st.metric("Max Loss", f"${np.min(portfolio_pnl):.2f}")
        with col3:
            current_pnl = np.interp(S, spot_range, portfolio_pnl)
            st.metric("Current P&L", f"${current_pnl:.2f}")
        with col4:
            breakeven_points = len(spot_range[np.abs(portfolio_pnl) < 1])
            st.metric("Breakeven Points", f"{breakeven_points}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Options Analytics | Built with Streamlit</p>
        <p><em>Disclaimer: This tool is for educational purposes only. Always consult with a financial advisor before making investment decisions.</em></p>
    </div>
""", unsafe_allow_html=True)