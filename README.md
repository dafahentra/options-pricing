# 💰 Option Pricing and Risk Assessment Tool 💰

## Overview

The **Option Pricing and Risk Assessment Tool** provides a simple yet powerful way to calculate the fair value of options and evaluate the associated risks using two methods:  
- **Black-Scholes Model**: A mathematical model used for pricing European-style options.
- **Monte Carlo Simulations**: A method used to simulate the potential future asset price paths and estimate the option price.

This tool also includes risk assessment metrics like **Value at Risk (VaR)** and **Expected Shortfall (ES)** to help evaluate the potential downside risk.

[![Watch the video](./image.png)](https://youtu.be/HZKh0vNGbHU?si=N9h_U9xDd868__Go)


## Features
- **Black-Scholes Pricing**: Calculate the price of **Call** and **Put** options.
- **Monte Carlo Simulation**: Simulate thousands of asset price paths to estimate option pricing.
- **Risk Metrics**: Evaluate risk using **Value at Risk (VaR)** and **Expected Shortfall (ES)**.
- **Interactive Visualizations**: Plot option price distributions and simulated price paths using interactive charts.

## Requirements

To run this project, you need the following Python packages:
- streamlit
- numpy
- scipy
- matplotlib
- plotly

You can install the required dependencies by running:
```bash
pip install -r requirements.txt
```
### `requirements.txt` file:
```
streamlit  
numpy  
scipy  
matplotlib  
plotly  
```

## How to Run

1. Clone the repository or download the source code to your local machine.
2. Navigate to the project directory in your terminal.
3. Run the Streamlit app:
```bash
streamlit run app.py
```
This will launch the app in your default web browser, where you can input parameters and explore the pricing models and risk metrics.

## Usage

1. **Black-Scholes Pricing**:  
   - Input parameters such as stock price, strike price, volatility, risk-free rate, and time to maturity.
   - Calculate the option prices for **Call** and **Put** options.
   
2. **Monte Carlo Simulation**:  
   - Run simulations to visualize the possible future asset prices at maturity.
   - View the distribution of simulated prices and the estimated option price based on these simulations.

3. **Risk Assessment**:  
   - View the **Value at Risk (VaR)** and **Expected Shortfall (ES)** based on simulated price paths to assess potential downside risks.

## Project Structure
```
Option_Pricing/  
│  
├── app.py              # Main Streamlit app  
├── models.py           # Functions for Black-Scholes and Monte Carlo calculations  
├── requirements.txt    # Required Python packages  
├── .gitignore          # Git ignore file for the project  
└── README.md           # This file  
```
## Contributing

Feel free to fork the repository and submit pull requests if you'd like to contribute. If you find any bugs or issues, please open an issue in the repository.

## License

This project is open-source and available under the MIT License. See the [LICENSE](LICENSE) file for more details.
