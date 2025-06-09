import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from models import black_scholes, monte_carlo_option
import plotly.graph_objects as go

st.set_page_config(
    page_title="Options Pricing",
    layout="wide"
)

st.title("Options Pricing and Risk Assessment Tool")
with st.expander("about this tool"):
    st.caption("The Option Pricing and Risk Assessment Tool helps you calculate the fair price of options and evaluate potential risks using two methods: the Black-Scholes Model for theoretical pricing and Monte Carlo Simulations to simulate future price outcomes. With this tool, you can easily calculate option prices for both Call and Put options, visualize the distribution of simulated asset prices at maturity, and analyze risk metrics like Value at Risk (VaR) and Expected Shortfall (ES). Simply input your parameters and explore interactive charts and insights into pricing and risk.")

st.sidebar.header("Option Parameters")
S = st.sidebar.number_input("Current Stock Price (S)", min_value=0.0, value=100.0)
K = st.sidebar.number_input("Strike Price (K)", min_value=0.0, value=100.0)
T = st.sidebar.number_input("Time to Maturity (T, in years)", min_value=0.01, value=1.0)
r = st.sidebar.number_input("Risk-Free Rate (r)", min_value=0.0, value=0.05, step=0.01)
sigma = st.sidebar.number_input("Volatility (σ)", min_value=0.01, value=0.2, step=0.01)
option_type = st.sidebar.selectbox("Option Type", ["call", "put"])
simulations = st.sidebar.slider("Monte Carlo Simulations", 1000, 50000, 10000)

# Black-Scholes Pricing
with st.container(border=True):
    c1,c2 = st.columns([10,1])
    c1.subheader("Black-Scholes Pricing")
    with c2.popover(":material/info:"):
        st.markdown("""
            The **Black-Scholes Model** is a widely-used mathematical formula for pricing European-style options. It calculates the fair value of a call or put option based on key factors like the current stock price, strike price, time to expiration, volatility, and the risk-free interest rate.

            The formula for the **call option price (C)** is:

            $$
            C = S_0 \cdot N(d_1) - K \cdot e^{-rT} \cdot N(d_2)
            $$
            For the **put option price (P)**:

            $$
            P = K \cdot e^{-rT} \cdot N(-d_2) - S_0 \cdot N(-d_1)
            $$

            Where:
            - $$ S_0 $$ is the current stock price
            - $$ K $$ is the strike price
            - $$ T $$ is the time to maturity (in years)
            - $$ r $$ is the risk-free interest rate
            - $$ \sigma $$ is the volatility of the stock
            - $$ N(x) $$ is the cumulative distribution function (CDF) of the standard normal distribution

            This model is useful for estimating the price of options in markets with efficient pricing and no dividends.
            """)
    bs_price = black_scholes(S, K, T, r, sigma, option_type)
    st.write(f"The {option_type.capitalize()} option price using Black-Scholes Model is: **${bs_price:.2f}**")

# Monte Carlo Simulation
with st.container(border=True):
    c1,c2 = st.columns([10,1])
    c1.subheader("Monte Carlo Simulation")
    with c2.popover(":material/info:"):
        st.markdown("""
            **Monte Carlo Simulation** is a powerful method for estimating option prices by simulating the potential future movements of the asset's price. By running many random simulations, this method helps visualize the range of possible outcomes and estimate the expected price of the option.

            The simulation uses the following formula for each simulated price path:

            $$
            S_T = S_0 \cdot e^{(r - 0.5 \sigma^2)T + \sigma \sqrt{T} \cdot Z}
            $$

            Where:
            - $$ S_0 $$ is the initial stock price
            - $$ r $$ is the risk-free rate
            - $$ \sigma $$ is the volatility
            - $$ T $$ is the time to maturity
            - $$ Z $$ is a random variable generated from a standard normal distribution ($$ Z \sim N(0, 1) $$)

            The simulation runs many iterations (e.g., 10,000 or more) to generate possible future prices, and the option price is then the average of the discounted payoffs at maturity.

            Monte Carlo simulations are especially useful when the Black-Scholes model is not applicable or when options have complex features like American-style options or early exercise possibilities.
            """)
    mc_price, S_T = monte_carlo_option(S, K, T, r, sigma, simulations, option_type)
    st.write(f"The {option_type.capitalize()} option price using Monte Carlo Simulation is: **${mc_price:.2f}**")

# Plot the Distribution of Simulated Prices
with st.container(border=True):
    st.subheader("Distribution of Simulated Prices at Maturity")
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=S_T,  # Simulated prices at maturity
            nbinsx=50,  # Number of bins
            marker_color="#c584f7", 
            marker_line_color="black",
            marker_line_width=1.2
        )
    )
    fig.update_layout(
        title="Distribution of Simulated Asset Prices",
        xaxis_title="Price",
        yaxis_title="Frequency",
        bargap=0.05,  # Space between bars
    )

    st.plotly_chart(fig)

# Value at Risk (VaR) and Expected Shortfall (ES)
with st.container(border=True):
    c1,c2 = st.columns([10,1])
    c1.subheader("Risk Assessment Metrics")
    with c2.popover(":material/info:"):
        st.markdown("""
            **Risk assessment** helps evaluate the potential downside of an option position. Two common risk metrics used are **Value at Risk (VaR)** and **Expected Shortfall (ES)**.

            - **Value at Risk (VaR)** is the worst potential loss over a specified time frame, at a given confidence level (e.g., 95% or 99%). It tells us, for example, "There is a 5% chance that the loss will exceed this amount."

            The formula for VaR at a confidence level $$ \\alpha $$ is:

            $$
            VaR = \mu + Z_{\\alpha} \cdot \sigma
            $$

            Where:
            - $$ \mu $$ is the expected return
            - $$ \sigma $$ is the standard deviation (volatility)
            - $$ Z_{\\alpha} $$ is the Z-score corresponding to the confidence level (e.g., for 95%, $$ Z_{\\alpha} \\approx 1.65 $$)

            - **Expected Shortfall (ES)** is the average loss that occurs when the loss exceeds the VaR threshold. It provides a more detailed view of the worst-case scenario, particularly when losses are extreme.

            These risk metrics are important for understanding the potential for significant losses, especially in volatile markets, and can help in decision-making and risk management.
            """)
    VaR_95 = np.percentile(S_T, 5)
    ES_95 = S_T[S_T <= VaR_95].mean()

    st.write(f"**Value at Risk (VaR 95%)**: ${VaR_95:.2f}")
    st.write(f"**Expected Shortfall (ES 95%)**: ${ES_95:.2f}")
