# Options Pricing Analytics 📈

A comprehensive web application for options pricing, Greeks analysis, and risk management built with Streamlit. Features multiple pricing models, advanced analytics, and a modern dark-mode interface optimized for financial professionals.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## 🌟 Features

### Pricing Models
- **Black-Scholes Model** - Classic analytical option pricing with full Greeks calculation
- **Monte Carlo Simulation** - Advanced simulation with variance reduction techniques (antithetic variates & control variates)
- **Binomial Tree Model** - Support for both European and American style options

### Greeks Analysis
- Real-time calculation of all major Greeks (Delta, Gamma, Theta, Vega, Rho)
- Interactive spider/radar chart visualization
- Delta-Gamma profile analysis
- Greeks sensitivity across spot price ranges

### Risk Analytics
- Value at Risk (VaR) calculations at 95% and 99% confidence levels
- Expected Shortfall (Conditional VaR)
- Monte Carlo risk simulation with customizable paths
- Probability of profit analysis
- Payoff distribution visualization

### Advanced Features
- **Implied Volatility Calculator** - Robust Newton-Raphson method with bisection fallback
- **Volatility Surface Generation** - 3D visualization with smile/skew effects
- **Scenario Analysis** - What-if analysis for spot price, volatility, time decay, and interest rates
- **Portfolio Builder** - Multi-option portfolio P&L analysis with up to 5 positions
- **Real-time Calculations** - All calculations update instantly with parameter changes

### UI/UX
- 🌙 Modern dark mode interface with purple accent theme
- 📱 Fully responsive design
- 📊 Interactive Plotly charts with custom styling
- ⚡ Optimized performance with efficient calculations
- 🎨 Smooth animations and transitions

## 🖼️ Screenshots

*[Add screenshots of your application here]*

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Clone the Repository
```bash
git clone https://github.com/dafahentra/options-pricing.git
cd options-pricing-analytics
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Requirements File
Create a `requirements.txt` file with:
```txt
streamlit>=1.28.0
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
plotly>=5.10.0
```

## 📘 Usage

### Running the Application
```bash
streamlit run option_pricing.py
```

The application will open in your default browser at `http://localhost:8501`

### Configuration
1. **Streamlit Config**: The app uses a custom theme defined in `.streamlit/config.toml`
2. **Default Parameters**: Modify default values in `config.py`
3. **Color Scheme**: Customize colors in the `COLORS` dictionary in `config.py`

### Basic Workflow
1. **Set Market Parameters**: Use the sidebar to input stock price, strike price, time to expiry, etc.
2. **Choose Pricing Model**: Compare prices across Black-Scholes, Monte Carlo, and Binomial models
3. **Analyze Greeks**: View real-time Greeks calculations and sensitivity profiles
4. **Run Scenarios**: Test various market conditions and their impact on option value
5. **Calculate Implied Volatility**: Input market prices to derive implied volatility
6. **Build Portfolios**: Combine multiple options to analyze complex strategies

## 📂 Project Structure

```
options-pricing-analytics/
│
├── .streamlit/
│   └── config.toml          # Streamlit theme configuration
│
├── option_pricing.py        # Main application file
├── models.py               # Option pricing models and calculations
├── components.py           # Reusable UI components
├── visualizations.py       # Plotly chart functions
├── utils.py               # Utility functions and helpers
├── config.py              # Configuration constants
├── styles.py              # Custom CSS styles
├── requirements.txt       # Python dependencies
├── LICENSE               # MIT License
└── README.md            # This file
```

### File Descriptions

- **option_pricing.py**: Main Streamlit application with tab-based interface
- **models.py**: Core financial models including Black-Scholes, Monte Carlo, Binomial, and IV calculator
- **components.py**: UI components for consistent rendering across the app
- **visualizations.py**: Plotly visualization functions with dark mode optimization
- **utils.py**: Helper functions for calculations, formatting, and data processing
- **config.py**: Centralized configuration for parameters, colors, and settings
- **styles.py**: Custom CSS for enhanced UI styling

## 🛠️ Technologies Used

- **[Streamlit](https://streamlit.io/)** - Web application framework
- **[NumPy](https://numpy.org/)** - Numerical computations
- **[Pandas](https://pandas.pydata.org/)** - Data manipulation
- **[SciPy](https://scipy.org/)** - Scientific computing (Normal distribution functions)
- **[Plotly](https://plotly.com/)** - Interactive visualizations

## 📊 Mathematical Models

### Black-Scholes Formula
The application implements the Black-Scholes-Merton model with dividend yield support:
- Call Option: C = S₀e^(-qT)N(d₁) - Ke^(-rT)N(d₂)
- Put Option: P = Ke^(-rT)N(-d₂) - S₀e^(-qT)N(-d₁)

### Monte Carlo Simulation
- Geometric Brownian Motion: S(T) = S₀ * exp((r - q - σ²/2)T + σ√T * Z)
- Variance reduction techniques for improved accuracy

### Binomial Tree
- Cox-Ross-Rubinstein parameterization
- Support for early exercise (American options)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add docstrings to all functions
- Maintain the existing code structure
- Test thoroughly before submitting PR
- Update documentation as needed

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Financial formulas based on John C. Hull's "Options, Futures, and Other Derivatives"
- UI design inspired by modern financial terminals
- Dark mode color scheme optimized for extended viewing sessions

## 📧 Contact

For questions, suggestions, or issues, please:
- Open an issue on GitHub
- Contact: [dapahentral@gmail.com]

---

**Disclaimer**: This application is for educational and analytical purposes only. Always consult with qualified financial advisors before making investment decisions. The creators assume no responsibility for financial losses incurred through the use of this tool.