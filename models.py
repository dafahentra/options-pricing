"""
Enhanced Option Pricing Models
=============================

A comprehensive collection of option pricing models and Greeks calculations
for financial derivatives analysis.

Author: Senior Developer
Version: 2.0
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Union
from enum import Enum
import warnings


class OptionType(Enum):
    """Enumeration for option types."""
    CALL = "call"
    PUT = "put"


class ModelType(Enum):
    """Enumeration for pricing model types."""
    BLACK_SCHOLES = "black_scholes"
    MONTE_CARLO = "monte_carlo"
    BINOMIAL = "binomial"


@dataclass
class OptionParameters:
    """
    Data class to hold option parameters with validation.
    
    Attributes:
        S: Current stock price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free rate
        sigma: Volatility
        option_type: Type of option (call/put)
        q: Dividend yield (default: 0.0)
    """
    S: float
    K: float
    T: float
    r: float
    sigma: float
    option_type: OptionType
    q: float = 0.0
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate option parameters."""
        if self.S <= 0:
            raise ValueError("Stock price must be positive")
        if self.K <= 0:
            raise ValueError("Strike price must be positive")
        if self.T <= 0:
            raise ValueError("Time to maturity must be positive")
        if self.sigma < 0:
            raise ValueError("Volatility cannot be negative")
        if not isinstance(self.option_type, OptionType):
            raise ValueError("option_type must be OptionType.CALL or OptionType.PUT")


@dataclass
class PricingResult:
    """
    Data class to hold pricing results and Greeks.
    
    Attributes:
        price: Option price
        delta: Price sensitivity to underlying price
        gamma: Delta sensitivity to underlying price
        theta: Price sensitivity to time decay
        vega: Price sensitivity to volatility
        rho: Price sensitivity to interest rate
        model_type: Type of model used for pricing
    """
    price: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    model_type: Optional[ModelType] = None


class BlackScholesModel:
    """
    Enhanced Black-Scholes option pricing model with Greeks calculations.
    
    This class provides comprehensive option pricing functionality including
    all major Greeks calculations and dividend adjustments.
    """
    
    @staticmethod
    def _calculate_d1_d2(params: OptionParameters) -> Tuple[float, float]:
        """Calculate d1 and d2 parameters for Black-Scholes formula."""
        d1 = (np.log(params.S / params.K) + 
              (params.r - params.q + 0.5 * params.sigma ** 2) * params.T) / \
             (params.sigma * np.sqrt(params.T))
        d2 = d1 - params.sigma * np.sqrt(params.T)
        return d1, d2
    
    @classmethod
    def price(cls, params: OptionParameters) -> float:
        """
        Calculate option price using Black-Scholes formula.
        
        Args:
            params: OptionParameters object containing all required parameters
            
        Returns:
            Option price
        """
        if params.T == 0:
            # Handle expiration case
            if params.option_type == OptionType.CALL:
                return max(params.S - params.K, 0)
            else:
                return max(params.K - params.S, 0)
        
        d1, d2 = cls._calculate_d1_d2(params)
        
        if params.option_type == OptionType.CALL:
            price = (params.S * np.exp(-params.q * params.T) * norm.cdf(d1) - 
                    params.K * np.exp(-params.r * params.T) * norm.cdf(d2))
        else:  # PUT
            price = (params.K * np.exp(-params.r * params.T) * norm.cdf(-d2) - 
                    params.S * np.exp(-params.q * params.T) * norm.cdf(-d1))
        
        return price
    
    @classmethod
    def delta(cls, params: OptionParameters) -> float:
        """Calculate Delta (price sensitivity to underlying price)."""
        if params.T == 0:
            if params.option_type == OptionType.CALL:
                return 1.0 if params.S > params.K else 0.0
            else:
                return -1.0 if params.S < params.K else 0.0
        
        d1, _ = cls._calculate_d1_d2(params)
        
        if params.option_type == OptionType.CALL:
            return np.exp(-params.q * params.T) * norm.cdf(d1)
        else:
            return -np.exp(-params.q * params.T) * norm.cdf(-d1)
    
    @classmethod
    def gamma(cls, params: OptionParameters) -> float:
        """Calculate Gamma (delta sensitivity to underlying price)."""
        if params.T == 0:
            return 0.0
        
        d1, _ = cls._calculate_d1_d2(params)
        return (np.exp(-params.q * params.T) * norm.pdf(d1)) / \
               (params.S * params.sigma * np.sqrt(params.T))
    
    @classmethod
    def theta(cls, params: OptionParameters) -> float:
        """Calculate Theta (price sensitivity to time decay)."""
        if params.T == 0:
            return 0.0
        
        d1, d2 = cls._calculate_d1_d2(params)
        
        term1 = -(params.S * np.exp(-params.q * params.T) * norm.pdf(d1) * 
                 params.sigma) / (2 * np.sqrt(params.T))
        
        if params.option_type == OptionType.CALL:
            term2 = params.q * params.S * np.exp(-params.q * params.T) * norm.cdf(d1)
            term3 = -params.r * params.K * np.exp(-params.r * params.T) * norm.cdf(d2)
            return term1 - term2 + term3
        else:
            term2 = -params.q * params.S * np.exp(-params.q * params.T) * norm.cdf(-d1)
            term3 = params.r * params.K * np.exp(-params.r * params.T) * norm.cdf(-d2)
            return term1 + term2 - term3
    
    @classmethod
    def vega(cls, params: OptionParameters) -> float:
        """Calculate Vega (price sensitivity to volatility)."""
        if params.T == 0:
            return 0.0
        
        d1, _ = cls._calculate_d1_d2(params)
        return params.S * np.exp(-params.q * params.T) * norm.pdf(d1) * np.sqrt(params.T)
    
    @classmethod
    def rho(cls, params: OptionParameters) -> float:
        """Calculate Rho (price sensitivity to interest rate)."""
        if params.T == 0:
            return 0.0
        
        _, d2 = cls._calculate_d1_d2(params)
        
        if params.option_type == OptionType.CALL:
            return params.K * params.T * np.exp(-params.r * params.T) * norm.cdf(d2)
        else:
            return -params.K * params.T * np.exp(-params.r * params.T) * norm.cdf(-d2)
    
    @classmethod
    def calculate_all(cls, params: OptionParameters) -> PricingResult:
        """Calculate option price and all Greeks."""
        return PricingResult(
            price=cls.price(params),
            delta=cls.delta(params),
            gamma=cls.gamma(params),
            theta=cls.theta(params),
            vega=cls.vega(params),
            rho=cls.rho(params),
            model_type=ModelType.BLACK_SCHOLES
        )


class MonteCarloModel:
    """
    Enhanced Monte Carlo option pricing model with variance reduction techniques.
    
    This class provides Monte Carlo simulation for option pricing with
    antithetic variates and control variates for improved accuracy.
    """
    
    @staticmethod
    def price(params: OptionParameters, 
              simulations: int = 100000,
              use_antithetic: bool = True,
              use_control_variate: bool = True,
              seed: Optional[int] = None) -> Tuple[float, np.ndarray]:
        """
        Calculate option price using Monte Carlo simulation.
        
        Args:
            params: OptionParameters object
            simulations: Number of simulation paths
            use_antithetic: Whether to use antithetic variates
            use_control_variate: Whether to use control variates
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (option_price, simulated_prices)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random numbers
        if use_antithetic:
            half_sims = simulations // 2
            Z = np.random.standard_normal(half_sims)
            Z = np.concatenate([Z, -Z])  # Antithetic variates
        else:
            Z = np.random.standard_normal(simulations)
        
        # Calculate terminal stock prices
        drift = (params.r - params.q - 0.5 * params.sigma ** 2) * params.T
        diffusion = params.sigma * np.sqrt(params.T) * Z
        S_T = params.S * np.exp(drift + diffusion)
        
        # Calculate payoffs
        if params.option_type == OptionType.CALL:
            payoffs = np.maximum(S_T - params.K, 0)
        else:
            payoffs = np.maximum(params.K - S_T, 0)
        
        # Apply control variate if requested
        if use_control_variate:
            # Use Black-Scholes as control variate
            bs_params = OptionParameters(
                S=params.S, K=params.K, T=params.T,
                r=params.r, sigma=params.sigma,
                option_type=params.option_type, q=params.q
            )
            bs_price = BlackScholesModel.price(bs_params)
            
            # Calculate control variate adjustment
            geometric_payoffs = params.S * np.exp(drift + diffusion)
            if params.option_type == OptionType.CALL:
                control_payoffs = np.maximum(geometric_payoffs - params.K, 0)
            else:
                control_payoffs = np.maximum(params.K - geometric_payoffs, 0)
            
            beta = np.cov(payoffs, control_payoffs)[0, 1] / np.var(control_payoffs)
            adjusted_payoffs = payoffs - beta * (control_payoffs - bs_price)
            option_price = np.exp(-params.r * params.T) * np.mean(adjusted_payoffs)
        else:
            option_price = np.exp(-params.r * params.T) * np.mean(payoffs)
        
        return option_price, S_T
    
    @staticmethod
    def calculate_greeks_fd(params: OptionParameters, 
                           simulations: int = 100000,
                           epsilon: float = 0.01) -> PricingResult:
        """
        Calculate Greeks using finite difference method with Monte Carlo.
        
        Args:
            params: OptionParameters object
            simulations: Number of simulation paths
            epsilon: Small change for finite difference
            
        Returns:
            PricingResult with price and Greeks
        """
        base_price, _ = MonteCarloModel.price(params, simulations)
        
        # Delta calculation
        params_up = OptionParameters(
            params.S + epsilon, params.K, params.T,
            params.r, params.sigma, params.option_type, params.q
        )
        params_down = OptionParameters(
            params.S - epsilon, params.K, params.T,
            params.r, params.sigma, params.option_type, params.q
        )
        price_up, _ = MonteCarloModel.price(params_up, simulations)
        price_down, _ = MonteCarloModel.price(params_down, simulations)
        delta = (price_up - price_down) / (2 * epsilon)
        
        # Gamma calculation
        gamma = (price_up - 2 * base_price + price_down) / (epsilon ** 2)
        
        # Vega calculation
        params_vega = OptionParameters(
            params.S, params.K, params.T,
            params.r, params.sigma + epsilon, params.option_type, params.q
        )
        price_vega, _ = MonteCarloModel.price(params_vega, simulations)
        vega = (price_vega - base_price) / epsilon
        
        return PricingResult(
            price=base_price,
            delta=delta,
            gamma=gamma,
            vega=vega,
            model_type=ModelType.MONTE_CARLO
        )


class BinomialModel:
    """
    Binomial tree model for American and European option pricing.
    """
    
    @staticmethod
    def price(params: OptionParameters, 
              steps: int = 100,
              american: bool = False) -> float:
        """
        Calculate option price using binomial tree model.
        
        Args:
            params: OptionParameters object
            steps: Number of time steps in the tree
            american: Whether to price American-style option
            
        Returns:
            Option price
        """
        dt = params.T / steps
        u = np.exp(params.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((params.r - params.q) * dt) - d) / (u - d)
        
        # Initialize asset prices at maturity
        S_values = np.zeros(steps + 1)
        for i in range(steps + 1):
            S_values[i] = params.S * (u ** (steps - i)) * (d ** i)
        
        # Initialize option values at maturity
        if params.option_type == OptionType.CALL:
            option_values = np.maximum(S_values - params.K, 0)
        else:
            option_values = np.maximum(params.K - S_values, 0)
        
        # Work backwards through the tree
        for j in range(steps - 1, -1, -1):
            for i in range(j + 1):
                # Calculate continuation value
                continuation = np.exp(-params.r * dt) * (
                    p * option_values[i] + (1 - p) * option_values[i + 1]
                )
                
                if american:
                    # Calculate intrinsic value for American option
                    current_price = params.S * (u ** (j - i)) * (d ** i)
                    if params.option_type == OptionType.CALL:
                        intrinsic = max(current_price - params.K, 0)
                    else:
                        intrinsic = max(params.K - current_price, 0)
                    
                    option_values[i] = max(continuation, intrinsic)
                else:
                    option_values[i] = continuation
        
        return option_values[0]


class ImpliedVolatilityCalculator:
    """
    Enhanced Calculator for implied volatility using Newton-Raphson method with robust error handling.
    """
    
    @staticmethod
    def calculate(market_price: float,
                  params: OptionParameters,
                  max_iterations: int = 100,
                  tolerance: float = 1e-6,
                  min_vol: float = 0.001,
                  max_vol: float = 5.0) -> float:
        """
        Calculate implied volatility from market price with robust error handling.
        
        Args:
            market_price: Observed market price
            params: OptionParameters (sigma will be ignored)
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            min_vol: Minimum volatility bound
            max_vol: Maximum volatility bound
            
        Returns:
            Implied volatility
            
        Raises:
            ValueError: If implied volatility cannot be calculated
        """
        # Input validation
        if market_price <= 0:
            raise ValueError("Market price must be positive")
        
        if params.T <= 0:
            raise ValueError("Time to expiry must be positive")
        
        # Check if option has any time value
        if params.option_type == OptionType.CALL:
            intrinsic_value = max(params.S - params.K, 0)
        else:
            intrinsic_value = max(params.K - params.S, 0)
        
        if market_price < intrinsic_value:
            raise ValueError("Market price is below intrinsic value")
        
        # If market price equals intrinsic value, volatility is essentially zero
        if abs(market_price - intrinsic_value) < tolerance:
            return min_vol
        
        # Initial guess using Brenner and Subrahmanyam approximation
        sigma = ImpliedVolatilityCalculator._initial_guess(market_price, params)
        sigma = max(min_vol, min(sigma, max_vol))
        
        # Newton-Raphson iteration with safeguards
        for iteration in range(max_iterations):
            # Create temporary parameters with current volatility
            temp_params = OptionParameters(
                S=params.S, K=params.K, T=params.T,
                r=params.r, sigma=sigma, option_type=params.option_type, q=params.q
            )
            
            try:
                # Calculate price and vega
                theoretical_price = BlackScholesModel.price(temp_params)
                vega = BlackScholesModel.vega(temp_params)
                
                # Check for convergence
                price_diff = theoretical_price - market_price
                if abs(price_diff) < tolerance:
                    return sigma
                
                # Check if vega is too small (near zero)
                if abs(vega) < 1e-10:
                    # Use alternative method or bisection fallback
                    return ImpliedVolatilityCalculator._bisection_method(
                        market_price, params, min_vol, max_vol, tolerance
                    )
                
                # Newton-Raphson update
                sigma_new = sigma - price_diff / vega
                
                # Apply bounds and dampening for stability
                sigma_new = max(min_vol, min(sigma_new, max_vol))
                
                # Add dampening factor to prevent oscillation
                if iteration > 10:
                    dampening = 0.5
                    sigma_new = sigma + dampening * (sigma_new - sigma)
                
                # Check for convergence in sigma
                if abs(sigma_new - sigma) < tolerance:
                    return sigma_new
                
                sigma = sigma_new
                
            except (ValueError, RuntimeError, OverflowError) as e:
                # If numerical issues occur, fall back to bisection method
                warnings.warn(f"Newton-Raphson failed at iteration {iteration}: {e}")
                return ImpliedVolatilityCalculator._bisection_method(
                    market_price, params, min_vol, max_vol, tolerance
                )
        
        # If we reach here, convergence failed
        warnings.warn(f"Implied volatility did not converge after {max_iterations} iterations")
        return sigma
    
    @staticmethod
    def _initial_guess(market_price: float, params: OptionParameters) -> float:
        """
        Generate initial guess for implied volatility using Brenner-Subrahmanyam approximation.
        """
        try:
            # Brenner and Subrahmanyam approximation
            sqrt_t = np.sqrt(params.T)
            if sqrt_t > 0:
                # For at-the-money options
                if abs(params.S - params.K) / params.S < 0.1:
                    return (market_price / params.S) * np.sqrt(2 * np.pi) / sqrt_t
                else:
                    # General approximation
                    return 2 * market_price / (params.S * sqrt_t)
            else:
                return 0.2  # Default fallback
        except:
            return 0.2  # Default fallback
    
    @staticmethod
    def _bisection_method(market_price: float, 
                         params: OptionParameters,
                         min_vol: float,
                         max_vol: float,
                         tolerance: float) -> float:
        """
        Bisection method as fallback for implied volatility calculation.
        """
        vol_low = min_vol
        vol_high = max_vol
        
        # Check bounds
        params_low = OptionParameters(
            params.S, params.K, params.T, params.r, vol_low, params.option_type, params.q
        )
        params_high = OptionParameters(
            params.S, params.K, params.T, params.r, vol_high, params.option_type, params.q
        )
        
        price_low = BlackScholesModel.price(params_low)
        price_high = BlackScholesModel.price(params_high)
        
        # Check if market price is within bounds
        if market_price < price_low:
            return vol_low
        if market_price > price_high:
            return vol_high
        
        # Bisection iteration
        for _ in range(100):  # Maximum iterations for bisection
            vol_mid = (vol_low + vol_high) / 2
            
            params_mid = OptionParameters(
                params.S, params.K, params.T, params.r, vol_mid, params.option_type, params.q
            )
            price_mid = BlackScholesModel.price(params_mid)
            
            if abs(price_mid - market_price) < tolerance:
                return vol_mid
            
            if price_mid < market_price:
                vol_low = vol_mid
            else:
                vol_high = vol_mid
            
            if abs(vol_high - vol_low) < tolerance:
                return (vol_low + vol_high) / 2
        
        return (vol_low + vol_high) / 2

    @staticmethod
    def calculate_iv_surface(S: float, K_range: np.ndarray, T_range: np.ndarray,
                           r: float, option_type: OptionType, q: float = 0.0,
                           base_vol: float = 0.2, vol_smile: bool = True) -> np.ndarray:
        """
        Generate a realistic implied volatility surface for visualization.
        
        Args:
            S: Current stock price
            K_range: Array of strike prices
            T_range: Array of time to expiry values
            r: Risk-free rate
            option_type: Option type
            q: Dividend yield
            base_vol: Base volatility level
            vol_smile: Whether to include volatility smile effect
            
        Returns:
            2D array of implied volatilities
        """
        iv_surface = np.zeros((len(T_range), len(K_range)))
        
        for i, T in enumerate(T_range):
            for j, K in enumerate(K_range):
                try:
                    # Calculate theoretical price with base volatility
                    params = OptionParameters(S, K, T, r, base_vol, option_type, q)
                    base_price = BlackScholesModel.price(params)
                    
                    # Add volatility smile/skew effects
                    if vol_smile:
                        # Moneyness effect (volatility smile)
                        moneyness = np.log(S / K)
                        smile_effect = 0.1 * moneyness**2  # Parabolic smile
                        
                        # Time effect (volatility term structure)
                        time_effect = 0.05 * np.exp(-2 * T)  # Higher vol for short-term
                        
                        vol_adjustment = smile_effect + time_effect
                        market_price = base_price * (1 + np.random.normal(0, 0.02))  # Add noise
                        
                        # Try to calculate IV
                        iv = ImpliedVolatilityCalculator.calculate(market_price, params)
                        iv_surface[i, j] = (iv + vol_adjustment) * 100
                    else:
                        iv_surface[i, j] = base_vol * 100
                        
                except Exception:
                    # Fallback to base volatility if calculation fails
                    iv_surface[i, j] = base_vol * 100
        
        return iv_surface


# Convenience functions for backward compatibility
def black_scholes(S: float, K: float, T: float, r: float, sigma: float, 
                 option_type: str = 'call', q: float = 0.0) -> float:
    """
    Legacy function for Black-Scholes pricing (backward compatibility).
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        sigma: Volatility
        option_type: 'call' or 'put'
        q: Dividend yield
        
    Returns:
        Option price
    """
    opt_type = OptionType.CALL if option_type.lower() == 'call' else OptionType.PUT
    params = OptionParameters(S, K, T, r, sigma, opt_type, q)
    return BlackScholesModel.price(params)


def monte_carlo_option(S: float, K: float, T: float, r: float, sigma: float,
                      simulations: int = 10000, option_type: str = 'call',
                      q: float = 0.0) -> Tuple[float, np.ndarray]:
    """
    Legacy function for Monte Carlo pricing (backward compatibility).
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        sigma: Volatility
        simulations: Number of simulations
        option_type: 'call' or 'put'
        q: Dividend yield
        
    Returns:
        Tuple of (option_price, simulated_prices)
    """
    opt_type = OptionType.CALL if option_type.lower() == 'call' else OptionType.PUT
    params = OptionParameters(S, K, T, r, sigma, opt_type, q)
    return MonteCarloModel.price(params, simulations)


if __name__ == "__main__":
    # Example usage
    params = OptionParameters(
        S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, 
        option_type=OptionType.CALL
    )
    
    # Black-Scholes pricing with Greeks
    bs_result = BlackScholesModel.calculate_all(params)
    print(f"Black-Scholes Price: ${bs_result.price:.2f}")
    print(f"Delta: {bs_result.delta:.4f}")
    print(f"Gamma: {bs_result.gamma:.4f}")
    print(f"Vega: {bs_result.vega:.4f}")
    
    # Monte Carlo pricing
    mc_price, _ = MonteCarloModel.price(params, simulations=100000)
    print(f"Monte Carlo Price: ${mc_price:.2f}")
    
    # Binomial pricing
    bin_price = BinomialModel.price(params, steps=100)
    print(f"Binomial Price: ${bin_price:.2f}")