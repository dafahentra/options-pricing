"""
Visualization functions for Options Pricing Analytics
Optimized for dark mode with enhanced styling
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from config import CHART_COLORS, COLORS


def apply_dark_theme(fig, transparent_bg=True):
    """Apply dark theme styling to any Plotly figure."""
    bg_color = 'rgba(0,0,0,0)' if transparent_bg else COLORS['dark_card']
    
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor=COLORS['dark_card'],
        paper_bgcolor=bg_color,
        font=dict(color=COLORS['text_primary'], size=12, family="system-ui"),
        title_font=dict(size=20, color=COLORS['text_primary'], family="system-ui"),
        hoverlabel=dict(
            bgcolor=COLORS['dark_card'],
            font_size=14,
            font_family="system-ui",
            font_color=COLORS['text_primary'],
            bordercolor=COLORS['primary']
        ),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    # Update grid styling
    fig.update_xaxes(
        gridcolor=COLORS['dark_border'],
        linecolor=COLORS['dark_border'],
        tickfont=dict(color=COLORS['text_secondary'])
    )
    fig.update_yaxes(
        gridcolor=COLORS['dark_border'],
        linecolor=COLORS['dark_border'],
        tickfont=dict(color=COLORS['text_secondary'])
    )
    
    return fig


def create_pricing_comparison_chart(models, prices):
    """Create bar chart comparing option prices across different models."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=models,
        y=prices,
        marker=dict(
            color=CHART_COLORS["price_comparison"],
            line=dict(color=COLORS['dark_border'], width=2)
        ),
        text=[f'${p:.4f}' for p in prices],
        textposition='outside',
        textfont=dict(color=COLORS['text_primary'], size=14, weight='bold')
    ))
    
    fig.update_layout(
        title="Option Price Comparison Across Models",
        yaxis_title="Option Price ($)",
        showlegend=False,
        height=400,
        yaxis=dict(range=[0, max(prices) * 1.15])  # Make room for text
    )
    
    return apply_dark_theme(fig)


def create_greeks_spider_chart(greeks_data):
    """Create spider/radar chart for Greeks visualization."""
    fig = go.Figure()
    
    # Add gradient effect with multiple traces
    for i in range(5, 0, -1):
        alpha = i / 5
        fig.add_trace(go.Scatterpolar(
            r=[v * alpha for v in greeks_data['Value']],
            theta=greeks_data['Greek'],
            fill='toself',
            fillcolor=f'rgba(197, 132, 247, {0.1 * alpha})',
            line=dict(color=f'rgba(197, 132, 247, {0.5 * alpha})', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Main trace
    fig.add_trace(go.Scatterpolar(
        r=greeks_data['Value'],
        theta=greeks_data['Greek'],
        fill='toself',
        fillcolor='rgba(197, 132, 247, 0.3)',
        name='Greeks Profile',
        line=dict(color=COLORS['primary'], width=3),
        marker=dict(size=8, color=COLORS['primary']),
        text=[f"{greek}: {actual:.4f}" for greek, actual in 
              zip(greeks_data['Greek'], greeks_data['Actual'])],
        hoverinfo='text+theta+r'
    ))
    
    fig.update_layout(
        polar=dict(
            bgcolor=COLORS['dark_card'],
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickmode='linear',
                tick0=0,
                dtick=20,
                gridcolor=COLORS['dark_border'],
                tickfont=dict(color=COLORS['text_secondary'])
            ),
            angularaxis=dict(
                gridcolor=COLORS['dark_border'],
                tickfont=dict(color=COLORS['text_primary'], size=14)
            )
        ),
        showlegend=False,
        height=400,
        title="Greeks Profile (Normalized)"
    )
    
    return apply_dark_theme(fig)


def create_delta_gamma_profile(spot_range, deltas, gammas, current_price):
    """Create dual-axis plot for Delta and Gamma profiles."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Delta trace with gradient fill
    fig.add_trace(
        go.Scatter(
            x=spot_range, 
            y=deltas, 
            name="Delta", 
            line=dict(color=CHART_COLORS["delta"], width=3),
            fill='tozeroy',
            fillcolor='rgba(255, 83, 112, 0.1)'
        ),
        secondary_y=False,
    )
    
    # Gamma trace
    fig.add_trace(
        go.Scatter(
            x=spot_range, 
            y=gammas, 
            name="Gamma", 
            line=dict(color=CHART_COLORS["gamma"], width=3),
            fill='tozeroy',
            fillcolor='rgba(100, 255, 218, 0.1)'
        ),
        secondary_y=True,
    )
    
    # Current price line with glow effect
    fig.add_vline(
        x=current_price, 
        line=dict(dash="dash", color=COLORS['primary'], width=2),
        annotation_text="Current Price",
        annotation_position="top",
        annotation=dict(font=dict(color=COLORS['primary'], size=14))
    )
    
    fig.update_yaxes(title_text="Delta", secondary_y=False, color=CHART_COLORS["delta"])
    fig.update_yaxes(title_text="Gamma", secondary_y=True, color=CHART_COLORS["gamma"])
    fig.update_xaxes(title_text="Stock Price ($)")
    fig.update_layout(height=400, title="Delta-Gamma Profile", legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor='rgba(26, 29, 38, 0.8)',
        bordercolor=COLORS['primary'],
        borderwidth=1
    ))
    
    return apply_dark_theme(fig)


def create_scenario_analysis_chart(x_values, y_values, pnl, x_label, current_value):
    """Create scenario analysis chart with price and P&L."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Option Price Analysis', 'P&L Analysis'),
        vertical_spacing=0.12,
        row_heights=[0.6, 0.4]
    )
    
    # Option price subplot with gradient
    fig.add_trace(
        go.Scatter(
            x=x_values, 
            y=y_values, 
            name="Option Price",
            line=dict(color=COLORS['primary'], width=3),
            fill='tozeroy',
            fillcolor='rgba(197, 132, 247, 0.1)'
        ),
        row=1, col=1
    )
    
    # P&L subplot with conditional coloring
    colors = np.where(pnl >= 0, 'rgba(195, 232, 141, 0.8)', 'rgba(255, 83, 112, 0.8)')
    
    fig.add_trace(
        go.Bar(
            x=x_values, 
            y=pnl, 
            name="P&L",
            marker_color=colors,
            marker_line_width=0
        ),
        row=2, col=1
    )
    
    # Add reference lines
    fig.add_vline(
        x=current_value, 
        line=dict(dash="dash", color=COLORS['secondary'], width=2),
        annotation_text="Current",
        annotation_position="top"
    )
    fig.add_hline(y=0, line_dash="solid", line_color=COLORS['text_secondary'], row=2, col=1)
    
    fig.update_xaxes(title_text=x_label, row=2, col=1)
    fig.update_yaxes(title_text="Option Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="P&L ($)", row=2, col=1)
    fig.update_layout(height=600, showlegend=False)
    
    # Style subplot titles
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=16, color=COLORS['text_primary'])
    
    return apply_dark_theme(fig)


def create_distribution_histogram(data, title, x_label, vlines=None):
    """Create histogram for distribution visualization."""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=50,
        name=title,
        marker=dict(
            color=CHART_COLORS["distribution"],
            line=dict(color=COLORS['dark_border'], width=1)
        ),
        opacity=0.8
    ))
    
    if vlines:
        for vline in vlines:
            fig.add_vline(
                x=vline["value"], 
                line=dict(
                    dash=vline.get("dash", "dash"), 
                    color=vline.get("color", "red"),
                    width=2
                ),
                annotation_text=vline.get("text", ""),
                annotation_position="top",
                annotation=dict(
                    font=dict(color=vline.get("color", "red"), size=12),
                    bgcolor=COLORS['dark_card'],
                    bordercolor=vline.get("color", "red"),
                    borderwidth=1
                )
            )
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="Frequency",
        height=400,
        bargap=0.05
    )
    
    return apply_dark_theme(fig)


def create_volatility_surface(strikes, times, iv_surface):
    """Create 3D volatility surface plot."""
    fig = go.Figure(data=[go.Surface(
        z=iv_surface,
        x=strikes,
        y=times,
        colorscale=[
            [0.0, COLORS['gradient_end']],
            [0.2, COLORS['primary']],
            [0.4, COLORS['secondary']],
            [0.6, COLORS['tertiary']],
            [0.8, COLORS['warning']],
            [1.0, COLORS['danger']]
        ],
        colorbar=dict(
            title=dict(text="Implied Vol (%)", side="right", font=dict(color=COLORS['text_primary'])),
            tickformat='.1f',
            tickfont=dict(color=COLORS['text_secondary']),
            bgcolor=COLORS['dark_card'],
            bordercolor=COLORS['dark_border'],
            borderwidth=1
        ),
        showscale=True,
        contours=dict(
            z=dict(
                show=True,
                usecolormap=True,
                highlightcolor=COLORS['secondary'],
                project=dict(z=True)
            )
        ),
        lighting=dict(
            ambient=0.5,
            diffuse=0.8,
            specular=0.2,
            fresnel=0.2
        )
    )])
    
    fig.update_layout(
        title='Implied Volatility Surface',
        scene=dict(
            xaxis=dict(
                title=dict(text='Strike Price ($)', font=dict(color=COLORS['text_primary'])),
                tickfont=dict(color=COLORS['text_secondary']),
                gridcolor=COLORS['dark_border'],
                backgroundcolor=COLORS['dark_card']
            ),
            yaxis=dict(
                title=dict(text='Time to Expiry (Years)', font=dict(color=COLORS['text_primary'])),
                tickfont=dict(color=COLORS['text_secondary']),
                gridcolor=COLORS['dark_border'],
                backgroundcolor=COLORS['dark_card']
            ),
            zaxis=dict(
                title=dict(text='Implied Volatility (%)', font=dict(color=COLORS['text_primary'])),
                tickfont=dict(color=COLORS['text_secondary']),
                gridcolor=COLORS['dark_border'],
                backgroundcolor=COLORS['dark_card']
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0),
                center=dict(x=0, y=0, z=-0.1)
            ),
            bgcolor=COLORS['dark_card']
        ),
        height=500,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return apply_dark_theme(fig, transparent_bg=False)


def create_volatility_term_structure(times, ivs, historical_vol):
    """Create volatility term structure chart."""
    fig = go.Figure()
    
    # Historical volatility line
    fig.add_trace(go.Scatter(
        x=times,
        y=[historical_vol*100]*len(times),
        mode='lines',
        name='Historical Volatility',
        line=dict(color=COLORS['text_secondary'], width=2, dash='dash'),
        showlegend=True
    ))
    
    # IV term structure with gradient
    fig.add_trace(go.Scatter(
        x=times,
        y=ivs,
        mode='lines+markers',
        name='ATM Implied Volatility',
        line=dict(color=COLORS['primary'], width=3),
        marker=dict(
            size=8, 
            color=COLORS['primary'],
            line=dict(color=COLORS['dark_border'], width=2)
        ),
        fill='tonexty',
        fillcolor='rgba(197, 132, 247, 0.1)'
    ))
    
    fig.update_layout(
        title='At-The-Money Volatility Term Structure',
        xaxis_title='Time to Expiry (Years)',
        yaxis_title='Implied Volatility (%)',
        height=400,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor='rgba(26, 29, 38, 0.8)',
            bordercolor=COLORS['primary'],
            borderwidth=1
        )
    )
    
    return apply_dark_theme(fig)


def create_volatility_smile(strikes, ivs, strike_price, spot_price):
    """Create volatility smile/skew chart."""
    fig = go.Figure()
    
    # Add gradient background for smile
    fig.add_trace(go.Scatter(
        x=strikes,
        y=ivs,
        mode='lines',
        line=dict(width=0),
        fill='tozeroy',
        fillcolor='rgba(100, 255, 218, 0.05)',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Main smile curve
    fig.add_trace(go.Scatter(
        x=strikes,
        y=ivs,
        mode='lines+markers',
        name='Volatility Smile',
        line=dict(color=COLORS['secondary'], width=3),
        marker=dict(
            size=8,
            color=COLORS['secondary'],
            line=dict(color=COLORS['dark_border'], width=2)
        )
    ))
    
    # Reference lines
    fig.add_vline(
        x=strike_price, 
        line=dict(dash="dash", color=COLORS['text_secondary'], width=2),
        annotation_text=f"Strike: ${strike_price:.2f}",
        annotation_position="top left"
    )
    
    fig.add_vline(
        x=spot_price, 
        line=dict(dash="dash", color=COLORS['success'], width=2),
        annotation_text=f"Spot: ${spot_price:.2f}",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title='Volatility Smile/Skew',
        xaxis_title='Strike Price ($)',
        yaxis_title='Implied Volatility (%)',
        height=400,
        showlegend=True
    )
    
    return apply_dark_theme(fig)


def create_portfolio_pnl_chart(spot_range, portfolio_pnl, current_price):
    """Create portfolio P&L diagram."""
    fig = go.Figure()
    
    # Create gradient fill based on P&L
    # Separate profit and loss regions
    profit_mask = portfolio_pnl >= 0
    loss_mask = portfolio_pnl < 0
    
    # Add shaded regions
    if any(profit_mask):
        fig.add_trace(go.Scatter(
            x=spot_range[profit_mask],
            y=portfolio_pnl[profit_mask],
            fill='tozeroy',
            fillcolor='rgba(195, 232, 141, 0.2)',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    if any(loss_mask):
        fig.add_trace(go.Scatter(
            x=spot_range[loss_mask],
            y=portfolio_pnl[loss_mask],
            fill='tozeroy',
            fillcolor='rgba(255, 83, 112, 0.2)',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Main P&L line
    fig.add_trace(go.Scatter(
        x=spot_range,
        y=portfolio_pnl,
        mode='lines',
        name='Portfolio P&L',
        line=dict(color=COLORS['primary'], width=4),
        hovertemplate='Price: $%{x:.2f}<br>P&L: $%{y:.2f}<extra></extra>'
    ))
    
    # Zero line
    fig.add_hline(
        y=0, 
        line=dict(color=COLORS['text_secondary'], width=2),
        annotation_text="Breakeven",
        annotation_position="right"
    )
    
    # Current price line
    fig.add_vline(
        x=current_price, 
        line=dict(dash="dash", color=COLORS['secondary'], width=3),
        annotation_text="Current Price",
        annotation_position="top",
        annotation=dict(
            font=dict(color=COLORS['secondary'], size=14),
            bgcolor=COLORS['dark_card'],
            bordercolor=COLORS['secondary'],
            borderwidth=1
        )
    )
    
    # Add max profit/loss annotations
    max_profit_idx = np.argmax(portfolio_pnl)
    max_loss_idx = np.argmin(portfolio_pnl)
    
    if portfolio_pnl[max_profit_idx] > 0:
        fig.add_annotation(
            x=spot_range[max_profit_idx],
            y=portfolio_pnl[max_profit_idx],
            text=f"Max Profit: ${portfolio_pnl[max_profit_idx]:.2f}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
            bgcolor=COLORS['success'],
            bordercolor=COLORS['success'],
            font=dict(color='white')
        )
    
    if portfolio_pnl[max_loss_idx] < 0:
        fig.add_annotation(
            x=spot_range[max_loss_idx],
            y=portfolio_pnl[max_loss_idx],
            text=f"Max Loss: ${portfolio_pnl[max_loss_idx]:.2f}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=40,
            bgcolor=COLORS['danger'],
            bordercolor=COLORS['danger'],
            font=dict(color='white')
        )
    
    fig.update_layout(
        title="Portfolio P&L Diagram",
        xaxis_title="Stock Price at Expiry ($)",
        yaxis_title="Profit/Loss ($)",
        height=500,
        showlegend=False,
        hovermode='x unified'
    )
    
    return apply_dark_theme(fig)