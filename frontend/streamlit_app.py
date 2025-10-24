import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import json

API_BASE =  "https://option-pricing-pinn.onrender.com"

st.set_page_config(
    page_title="PINN Option Pricing Research Platform", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 600;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .subheader {
        font-size: 1.4rem;
        font-weight: 500;
        color: #34495e;
        margin-bottom: 1rem;
        border-left: 4px solid #3498db;
        padding-left: 1rem;
    }
    .greek-card {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        padding: 1.2rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .greek-value {
        font-size: 1.6rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .greek-label {
        font-size: 0.9rem;
        opacity: 0.9;
        font-weight: 500;
    }
    .metric-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #3498db;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .dashboard-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .dashboard-card:hover {
        transform: translateY(-2px);
    }
    .dashboard-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .dashboard-label {
        font-size: 1rem;
        opacity: 0.9;
        font-weight: 500;
    }
    .risk-low {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
    }
    .risk-medium {
        background: linear-gradient(135deg, #FF9800 0%, #f57c00 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
    }
    .risk-high {
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
    }
    .quick-action-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        margin: 0.25rem;
        cursor: pointer;
        transition: all 0.3s;
    }
    .quick-action-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .financial-table {
        background: white;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .performance-indicator {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    .performance-excellent {
        background: #e8f5e8;
        color: #2e7d32;
        border: 1px solid #4caf50;
    }
    .performance-good {
        background: #e3f2fd;
        color: #1976d2;
        border: 1px solid #2196f3;
    }
    .performance-moderate {
        background: #fff3e0;
        color: #f57c00;
        border: 1px solid #ff9800;
    }
    .performance-poor {
        background: #ffebee;
        color: #c62828;
        border: 1px solid #f44336;
    }
</style>
""", unsafe_allow_html=True)

# Professional header
st.markdown('<h1 class="main-header">PINN Option Pricing Research Platform</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #7f8c8d; font-size: 1rem; margin-bottom: 2rem;">Physics-Informed Neural Networks for European & American Options</p>', unsafe_allow_html=True)

# Sidebar with input parameters
with st.sidebar:
    st.markdown("## Configuration")
    st.markdown("---")
    
    # Option Configuration
    st.markdown("### Option Parameters")
    option_style = st.selectbox("Option Style", ["european", "american"], 
                               help="European: Can only be exercised at maturity. American: Can be exercised anytime.")
    option_type = st.selectbox("Option Type", ["call", "put"], 
                              help="Call: Right to buy. Put: Right to sell.")
    
    # Market Parameters
    st.markdown("### Market Parameters")
    S = st.number_input("Spot Price (S)", value=100.0, min_value=0.0, step=1.0, 
                       help="Current price of the underlying asset")
    K = st.number_input("Strike Price (K)", value=100.0, min_value=0.0, step=1.0,
                       help="Price at which the option can be exercised")
    r = st.number_input("Risk-free Rate (r)", value=0.05, step=0.01, format="%.3f",
                       help="Risk-free interest rate")
    vol = st.number_input("Volatility (σ)", value=0.2, min_value=0.0001, step=0.01, format="%.3f",
                        help="Annualized volatility of the underlying asset")
    
    # Time Parameters
    st.markdown("### Time Parameters")
    t = st.number_input("Time to Maturity (t)", value=0.0, min_value=0.0, max_value=1.0, step=0.1, format="%.2f",
                       help="Time remaining until option expiration")
    T = st.number_input("Total Maturity (T)", value=1.0, min_value=0.0, step=0.1, format="%.2f",
                       help="Total time to expiration")
    
    # Surface Parameters
    st.markdown("### Surface Parameters")
    S_min = st.number_input("S Min", value=0.0, help="Minimum spot price for surface plots")
    S_max = st.number_input("S Max", value=200.0, help="Maximum spot price for surface plots")
    t_min = st.number_input("t Min", value=0.0, help="Minimum time for 3D surface plots")
    t_max = st.number_input("t Max", value=1.0, help="Maximum time for 3D surface plots")
    n_S = st.number_input("S Resolution", value=50, min_value=10, max_value=200, step=10,
                         help="Number of points along S axis")
    n_t = st.number_input("t Resolution", value=20, min_value=10, max_value=50, step=5,
                         help="Number of points along t axis")
    
    # Currency Settings
    st.markdown("### Currency Settings")
    currency_options = {
        "USD": {"symbol": "$", "name": "US Dollar"},
        "EUR": {"symbol": "€", "name": "Euro"},
        "GBP": {"symbol": "£", "name": "British Pound"},
        "JPY": {"symbol": "¥", "name": "Japanese Yen"},
        "INR": {"symbol": "₹", "name": "Indian Rupee"},
        "CAD": {"symbol": "C$", "name": "Canadian Dollar"},
        "AUD": {"symbol": "A$", "name": "Australian Dollar"},
        "CHF": {"symbol": "CHF", "name": "Swiss Franc"}
    }
    
    selected_currency = st.selectbox(
        "Currency", 
        options=list(currency_options.keys()),
        format_func=lambda x: f"{currency_options[x]['symbol']} {currency_options[x]['name']}",
        help="Select the currency for displaying prices and values"
    )
    
    currency_symbol = currency_options[selected_currency]["symbol"]
    
    # API Settings
    st.markdown("### API Settings")
    api_base = st.text_input("API Base URL", value=API_BASE, help="Backend API endpoint")

# Dashboard-style metrics cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f'''
    <div class="dashboard-card">
        <div class="dashboard-value">{currency_symbol}{S:.2f}</div>
        <div class="dashboard-label">Spot Price</div>
    </div>
    ''', unsafe_allow_html=True)

with col2:
    st.markdown(f'''
    <div class="dashboard-card">
        <div class="dashboard-value">{currency_symbol}{K:.2f}</div>
        <div class="dashboard-label">Strike Price</div>
    </div>
    ''', unsafe_allow_html=True)

with col3:
    st.markdown(f'''
    <div class="dashboard-card">
        <div class="dashboard-value">{vol:.1%}</div>
        <div class="dashboard-label">Volatility</div>
    </div>
    ''', unsafe_allow_html=True)

with col4:
    moneyness = S/K
    risk_level = "Low" if moneyness > 0.95 and moneyness < 1.05 else "Medium" if moneyness > 0.9 and moneyness < 1.1 else "High"
    risk_class = "risk-low" if risk_level == "Low" else "risk-medium" if risk_level == "Medium" else "risk-high"
    st.markdown(f'''
    <div class="dashboard-card">
        <div class="dashboard-value">{moneyness:.3f}</div>
        <div class="dashboard-label">Moneyness</div>
        <div class="{risk_class}" style="margin-top: 0.5rem;">{risk_level} Risk</div>
    </div>
    ''', unsafe_allow_html=True)


# Main tabs
pricing_tab, greeks_tab, surface2d_tab, surface3d_tab, convergence_tab, analysis_tab = st.tabs([
    "Price", "Greeks", "2D Surface", "3D Surface", "Convergence", "Analysis & Export"
])

with pricing_tab:
    st.markdown('<h2 class="subheader">Option Pricing</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("Compute PINN Price", type="primary", use_container_width=True):
            payload = {"S": S, "t": t, "r": r, "K": K, "volatility": vol, "T": T, 
                      "option_style": option_style, "option_type": option_type}
            try:
                with st.spinner("Computing PINN price..."):
                    resp = requests.post(f"{api_base}/price", json=payload, timeout=60)
                    resp.raise_for_status()
                    result = resp.json()
                
                # Professional pricing results display
                st.markdown("### Pricing Results")
                
                # Main pricing metrics in cards
                price_col1, price_col2, price_col3 = st.columns(3)
                
                with price_col1:
                    st.markdown(f'''
                    <div class="dashboard-card">
                        <div class="dashboard-value">{currency_symbol}{result['price']:.4f}</div>
                        <div class="dashboard-label">PINN Price</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Display Black-Scholes comparison for European options
                if result['bs_price'] is not None:
                    with price_col2:
                        st.markdown(f'''
                        <div class="dashboard-card">
                            <div class="dashboard-value">{currency_symbol}{result['bs_price']:.4f}</div>
                            <div class="dashboard-label">Black-Scholes</div>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    with price_col3:
                        st.markdown(f'''
                        <div class="dashboard-card">
                            <div class="dashboard-value">{currency_symbol}{result['error']:.4f}</div>
                            <div class="dashboard-label">Absolute Error</div>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    # Professional comparison table
                    st.markdown("### Detailed Comparison")
                    comparison_data = {
                        "Metric": ["PINN Price", "Black-Scholes Price", "Absolute Error"],
                        "Value": [f"{currency_symbol}{result['price']:.4f}", f"{currency_symbol}{result['bs_price']:.4f}", 
                                 f"{currency_symbol}{result['error']:.4f}"],
                        "Status": ["PINN Model", "Analytical", "Difference"]
                    }
                    
                    df_comparison = pd.DataFrame(comparison_data)
                    st.dataframe(df_comparison, use_container_width=True, hide_index=True)
                
                # Additional metrics
                st.markdown("### Additional Metrics")
                col_x, col_y, col_z = st.columns(3)
                with col_x:
                    st.metric("Moneyness", f"{S/K:.3f}")
                with col_y:
                    st.metric("Time Value", f"{T-t:.3f}")
                with col_z:
                    intrinsic_val = max(S-K, 0) if option_type == "call" else max(K-S, 0)
                    st.metric("Intrinsic Value", f"{currency_symbol}{intrinsic_val:.4f}")
                
                # Export functionality
                st.markdown("### Export Results")
                export_col1, export_col2 = st.columns(2)
                
                with export_col1:
                    # Create export data
                    export_data = {
                        "Parameter": ["Spot Price", "Strike Price", "Risk-free Rate", "Volatility", "Time to Maturity", "Option Style", "Option Type"],
                        "Value": [f"{currency_symbol}{S:.2f}", f"{currency_symbol}{K:.2f}", f"{r:.1%}", f"{vol:.1%}", f"{t:.2f}", option_style.title(), option_type.title()]
                    }
                    
                    if result['bs_price'] is not None:
                        export_data["Parameter"].extend(["PINN Price", "Black-Scholes Price", "Absolute Error"])
                        export_data["Value"].extend([f"{currency_symbol}{result['price']:.4f}", f"{currency_symbol}{result['bs_price']:.4f}", 
                                                   f"{currency_symbol}{result['error']:.4f}"])
                    
                    df_export = pd.DataFrame(export_data)
                    csv = df_export.to_csv(index=False)
                    
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"pinn_pricing_{option_style}_{option_type}_{S}_{K}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with export_col2:
                    # JSON export
                    json_data = {
                        "parameters": {
                            "S": S, "K": K, "r": r, "volatility": vol, "t": t, "T": T,
                            "option_style": option_style, "option_type": option_type
                        },
                        "results": {
                            "pinn_price": result['price'],
                            "bs_price": result.get('bs_price'),
                            "error": result.get('error')
                        },
                        "metadata": {
                            "timestamp": pd.Timestamp.now().isoformat(),
                            "model": "PINN",
                            "version": "1.0"
                        }
                    }
                    
                    import json
                    json_str = json.dumps(json_data, indent=2)
                    
                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name=f"pinn_pricing_{option_style}_{option_type}_{S}_{K}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                    
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col2:
        st.markdown("### Current Configuration")
        st.write(f"**Option Style:** {option_style.title()}")
        st.write(f"**Option Type:** {option_type.title()}")
        st.write(f"**Spot Price:** {currency_symbol}{S:.2f}")
        st.write(f"**Strike Price:** {currency_symbol}{K:.2f}")
        st.write(f"**Risk-free Rate:** {r:.1%}")
        st.write(f"**Volatility:** {vol:.1%}")
        st.write(f"**Time to Maturity:** {t:.2f}")
        st.write(f"**Currency:** {selected_currency}")

with greeks_tab:
    st.markdown('<h2 class="subheader">Option Greeks & Risk Sensitivities</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("Compute Greeks", type="primary", use_container_width=True):
            payload = {"S": S, "t": t, "r": r, "K": K, "volatility": vol, "T": T,
                      "option_style": option_style, "option_type": option_type}
            try:
                with st.spinner("Computing Greeks..."):
                    resp = requests.post(f"{api_base}/greeks", json=payload, timeout=60)
                    resp.raise_for_status()
                    greeks = resp.json()
                
                # Professional Greeks display with risk indicators
                st.markdown("### Greeks Analysis")
                
                # Greeks cards with risk indicators
                col_a, col_b, col_c, col_d, col_e = st.columns(5)
                
                # Delta risk assessment
                delta_risk = "High" if abs(greeks["delta"]) > 0.8 else "Medium" if abs(greeks["delta"]) > 0.5 else "Low"
                delta_class = "risk-high" if delta_risk == "High" else "risk-medium" if delta_risk == "Medium" else "risk-low"
                
                with col_a:
                    st.markdown(f'''
                    <div class="greek-card">
                        <div class="greek-label">Delta (Δ)</div>
                        <div class="greek-value">{greeks["delta"]:.4f}</div>
                        <div class="{delta_class}" style="margin-top: 0.5rem; font-size: 0.7rem;">{delta_risk}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Gamma risk assessment
                gamma_risk = "High" if greeks["gamma"] > 0.1 else "Medium" if greeks["gamma"] > 0.05 else "Low"
                gamma_class = "risk-high" if gamma_risk == "High" else "risk-medium" if gamma_risk == "Medium" else "risk-low"
                
                with col_b:
                    st.markdown(f'''
                    <div class="greek-card">
                        <div class="greek-label">Gamma (Γ)</div>
                        <div class="greek-value">{greeks["gamma"]:.4f}</div>
                        <div class="{gamma_class}" style="margin-top: 0.5rem; font-size: 0.7rem;">{gamma_risk}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Vega risk assessment
                vega_risk = "High" if greeks["vega"] > 50 else "Medium" if greeks["vega"] > 25 else "Low"
                vega_class = "risk-high" if vega_risk == "High" else "risk-medium" if vega_risk == "Medium" else "risk-low"
                
                with col_c:
                    st.markdown(f'''
                    <div class="greek-card">
                        <div class="greek-label">Vega (ν)</div>
                        <div class="greek-value">{greeks["vega"]:.4f}</div>
                        <div class="{vega_class}" style="margin-top: 0.5rem; font-size: 0.7rem;">{vega_risk}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Theta risk assessment
                theta_risk = "High" if abs(greeks["theta"]) > 10 else "Medium" if abs(greeks["theta"]) > 5 else "Low"
                theta_class = "risk-high" if theta_risk == "High" else "risk-medium" if theta_risk == "Medium" else "risk-low"
                
                with col_d:
                    st.markdown(f'''
                    <div class="greek-card">
                        <div class="greek-label">Theta (Θ)</div>
                        <div class="greek-value">{greeks["theta"]:.4f}</div>
                        <div class="{theta_class}" style="margin-top: 0.5rem; font-size: 0.7rem;">{theta_risk}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Rho risk assessment
                rho_risk = "High" if abs(greeks["rho"]) > 20 else "Medium" if abs(greeks["rho"]) > 10 else "Low"
                rho_class = "risk-high" if rho_risk == "High" else "risk-medium" if rho_risk == "Medium" else "risk-low"
                
                with col_e:
                    st.markdown(f'''
                    <div class="greek-card">
                        <div class="greek-label">Rho (ρ)</div>
                        <div class="greek-value">{greeks["rho"]:.4f}</div>
                        <div class="{rho_class}" style="margin-top: 0.5rem; font-size: 0.7rem;">{rho_risk}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Black-Scholes comparison for European options
                if greeks["bs_delta"] is not None:
                    st.markdown("### Black-Scholes Greeks Comparison")
                    
                    # Create comparison table
                    comparison_data = {
                        "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
                        "PINN": [greeks["delta"], greeks["gamma"], greeks["vega"], greeks["theta"], greeks["rho"]],
                        "Black-Scholes": [greeks["bs_delta"], greeks["bs_gamma"], greeks["bs_vega"], greeks["bs_theta"], greeks["bs_rho"]],
                        "Error": [
                            abs(greeks["delta"] - greeks["bs_delta"]),
                            abs(greeks["gamma"] - greeks["bs_gamma"]),
                            abs(greeks["vega"] - greeks["bs_vega"]),
                            abs(greeks["theta"] - greeks["bs_theta"]),
                            abs(greeks["rho"] - greeks["bs_rho"])
                        ]
                    }
                    
                    df_comparison = pd.DataFrame(comparison_data)
                    df_comparison["Error"] = df_comparison["Error"].round(6)
                    
                    st.dataframe(df_comparison, use_container_width=True)
                
                # Greeks explanation
                st.markdown("### Greeks Interpretation")
                col_x, col_y = st.columns(2)
                with col_x:
                    st.markdown("""
                    **Delta (Δ)**: Price sensitivity to underlying asset price  
                    **Gamma (Γ)**: Rate of change of delta  
                    **Vega (ν)**: Price sensitivity to volatility
                    """)
                with col_y:
                    st.markdown("""
                    **Theta (Θ)**: Price sensitivity to time decay  
                    **Rho (ρ)**: Price sensitivity to interest rate
                    """)
                    
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col2:
        st.markdown("### Risk Analysis")
        st.write("**Current Position:**")
        st.write(f"- Option Style: {option_style.title()}")
        st.write(f"- Option Type: {option_type.title()}")
        st.write(f"- Spot Price: ${S:.2f}")
        st.write(f"- Strike Price: ${K:.2f}")
        st.write(f"- Moneyness: {S/K:.3f}")
        
        # Risk indicators
        if S/K > 1.1:
            st.success("Deep ITM")
        elif S/K > 1.05:
            st.info("ITM")
        elif S/K > 0.95:
            st.warning("ATM")
        elif S/K > 0.9:
            st.info("OTM")
        else:
            st.error("Deep OTM")

with surface2d_tab:
    st.markdown('<h2 class="subheader">Pricing Curve vs Underlying Price</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("Generate Pricing Curve", type="primary", use_container_width=True):
            payload = {"t": t, "r": r, "K": K, "volatility": vol, "S_min": S_min, "S_max": S_max, "n_S": int(n_S),
                      "option_style": option_style, "option_type": option_type}
            try:
                with st.spinner("Generating pricing curve..."):
                    resp = requests.post(f"{api_base}/surface", json=payload, timeout=60)
                    resp.raise_for_status()
                    points = resp.json()
                    df = pd.DataFrame(points)
                
                # Create interactive plot with proper axis labels
                fig = px.line(df, x="S", y="price", 
                             title=f"{option_style.title()} {option_type.title()} Option Price vs Underlying Price",
                             labels={"S": "Underlying Price ($)", "price": "Option Price ($)"})
                fig.update_layout(height=500, showlegend=False)
                fig.add_vline(x=K, line_dash="dash", line_color="red", 
                             annotation_text=f"Strike: ${K}")
                
                # Update hover template for better interactivity
                fig.update_traces(hovertemplate="<b>Underlying Price:</b> $%{x:.2f}<br><b>Option Price:</b> $%{y:.4f}<extra></extra>")
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col2:
        st.markdown("### Surface Parameters")
        st.write(f"**Time to Maturity:** {t:.2f}")
        st.write(f"**Risk-free Rate:** {r:.1%}")
        st.write(f"**Volatility:** {vol:.1%}")
        st.write(f"**Strike Price:** ${K:.2f}")
        st.write(f"**S Range:** ${S_min:.0f} - ${S_max:.0f}")
        st.write(f"**Resolution:** {n_S} points")

with surface3d_tab:
    st.markdown('<h2 class="subheader">Interactive 3D Pricing Surface</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        surface_type = st.selectbox("Surface Type", ["Price", "Delta", "Gamma", "Vega", "Theta", "Rho"])
        
        if st.button("Generate 3D Surface", type="primary", use_container_width=True):
            payload = {
                "r": r, "K": K, "volatility": vol,
                "S_min": S_min, "S_max": S_max,
                "t_min": t_min, "t_max": t_max,
                "n_S": int(n_S), "n_t": int(n_t),
                "option_style": option_style, "option_type": option_type
            }
            try:
                with st.spinner("Generating 3D surface..."):
                    resp = requests.post(f"{api_base}/surface3d", json=payload, timeout=60)
                    resp.raise_for_status()
                    points = resp.json()
                    df = pd.DataFrame(points)
                
                # Create 3D surface plot with proper axis labels
                fig = go.Figure(data=[go.Surface(
                    x=df['S'].values.reshape(int(n_t), int(n_S)),
                    y=df['t'].values.reshape(int(n_t), int(n_S)),
                    z=df[surface_type.lower()].values.reshape(int(n_t), int(n_S)),
                    colorscale='Viridis',
                    name=surface_type,
                    hovertemplate="<b>Underlying Price:</b> $%{x:.2f}<br><b>Time to Maturity:</b> %{y:.2f}<br><b>" + surface_type + ":</b> %{z:.4f}<extra></extra>"
                )])
                
                fig.update_layout(
                    title=f'{option_style.title()} {option_type.title()} Option - 3D {surface_type} Surface',
                    scene=dict(
                        xaxis_title='Underlying Price ($)',
                        yaxis_title='Time to Maturity',
                        zaxis_title=f'{surface_type}',
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                    ),
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col2:
        st.markdown("### 3D Surface Parameters")
        st.write(f"**Risk-free Rate:** {r:.1%}")
        st.write(f"**Strike Price:** ${K:.2f}")
        st.write(f"**Volatility:** {vol:.1%}")
        st.write(f"**S Range:** ${S_min:.0f} - ${S_max:.0f}")
        st.write(f"**t Range:** {t_min:.2f} - {t_max:.2f}")
        st.write(f"**S Resolution:** {n_S} points")
        st.write(f"**t Resolution:** {n_t} points")
        st.write(f"**Total Points:** {n_S * n_t:,}")

with convergence_tab:
    st.markdown('<h2 class="subheader">PINN Training Convergence</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("Load Convergence Data", type="primary", use_container_width=True):
            try:
                with st.spinner("Loading convergence data..."):
                    resp = requests.get(f"{api_base}/convergence?option_style={option_style}&option_type={option_type}", timeout=60)
                    resp.raise_for_status()
                    conv_data = resp.json()
                
                if conv_data:
                    df = pd.DataFrame(conv_data)
                    
                    # Create convergence plot
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(x=df['epoch'], y=df['total'], 
                                           mode='lines', name='Total Loss', 
                                           line=dict(color='red', width=3)))
                    fig.add_trace(go.Scatter(x=df['epoch'], y=df['pde'], 
                                           mode='lines', name='PDE Loss', 
                                           line=dict(color='blue', width=2)))
                    fig.add_trace(go.Scatter(x=df['epoch'], y=df['terminal'], 
                                           mode='lines', name='Terminal Loss', 
                                           line=dict(color='green', width=2)))
                    fig.add_trace(go.Scatter(x=df['epoch'], y=df['left_bc'], 
                                           mode='lines', name='Left BC Loss', 
                                           line=dict(color='orange', width=2)))
                    fig.add_trace(go.Scatter(x=df['epoch'], y=df['right_bc'], 
                                           mode='lines', name='Right BC Loss', 
                                           line=dict(color='purple', width=2)))
                    
                    # Add exercise loss for American options
                    if 'exercise' in df.columns and df['exercise'].notna().any():
                        fig.add_trace(go.Scatter(x=df['epoch'], y=df['exercise'], 
                                               mode='lines', name='Exercise Loss', 
                                               line=dict(color='brown', width=2)))
                    
                    fig.update_layout(
                        title=f'{option_style.title()} {option_type.title()} Option - PINN Training Loss Convergence',
                        xaxis_title='Training Epoch',
                        yaxis_title='Loss Value',
                        yaxis_type="log",
                        height=500,
                        hovermode='x unified'
                    )
                    
                    # Update hover template
                    fig.update_traces(hovertemplate="<b>Epoch:</b> %{x}<br><b>Loss:</b> %{y:.6f}<extra></extra>")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Loss statistics
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.metric("Final Total Loss", f"{df['total'].iloc[-1]:.6f}")
                    with col_b:
                        st.metric("Final PDE Loss", f"{df['pde'].iloc[-1]:.6f}")
                    with col_c:
                        st.metric("Final Terminal Loss", f"{df['terminal'].iloc[-1]:.6f}")
                    with col_d:
                        st.metric("Training Epochs", f"{df['epoch'].iloc[-1]:,}")
                        
                    # Show exercise loss for American options
                    if 'exercise' in df.columns and df['exercise'].notna().any():
                        st.metric("Final Exercise Loss", f"{df['exercise'].iloc[-1]:.6f}")
                else:
                    st.info("No convergence data available.")
                    
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col2:
        st.markdown("### Model Information")
        st.write(f"**Option Style:** {option_style.title()}")
        st.write(f"**Option Type:** {option_type.title()}")
        st.write("**Loss Components:**")
        st.write("- PDE Loss: Physics constraint")
        st.write("- Terminal Loss: Payoff condition")
        st.write("- Boundary Loss: Boundary conditions")
        if option_style == "american":
            st.write("- Exercise Loss: Early exercise constraint")
        
        st.markdown("### Training Details")
        st.write("**Architecture:** 5→64→64→1")
        st.write("**Activation:** ReLU")
        st.write("**Optimizer:** Adam")
        st.write("**Learning Rate:** 1e-3")
        st.write("**Epochs:** 5,000")

with analysis_tab:
    st.markdown('<h2 class="subheader">PINN Simulation & Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("### Real-time PINN Training & Simulation")
    st.markdown("Configure parameter ranges and run live PINN simulations to generate fresh research data.")
    
    # Parameter range configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Price Parameters")
        S_min_range = st.number_input("Spot Price Min", value=50.0, min_value=0.0, step=1.0, help="Minimum spot price for simulation")
        S_max_range = st.number_input("Spot Price Max", value=150.0, min_value=0.0, step=1.0, help="Maximum spot price for simulation")
        
        st.markdown("#### Strike Parameters")
        K_min_range = st.number_input("Strike Price Min", value=80.0, min_value=0.0, step=1.0, help="Minimum strike price for simulation")
        K_max_range = st.number_input("Strike Price Max", value=120.0, min_value=0.0, step=1.0, help="Maximum strike price for simulation")
        
        st.markdown("#### Time Parameters")
        t_min_range = st.number_input("Time to Maturity Min", value=0.0, min_value=0.0, max_value=1.0, step=0.1, format="%.2f", help="Minimum time to maturity")
        t_max_range = st.number_input("Time to Maturity Max", value=1.0, min_value=0.0, max_value=1.0, step=0.1, format="%.2f", help="Maximum time to maturity")
    
    with col2:
        st.markdown("#### Rate Parameters")
        r_min_range = st.number_input("Risk-free Rate Min", value=0.01, min_value=0.0, step=0.01, format="%.3f", help="Minimum risk-free rate")
        r_max_range = st.number_input("Risk-free Rate Max", value=0.10, min_value=0.0, step=0.01, format="%.3f", help="Maximum risk-free rate")
        
        st.markdown("#### Volatility Parameters")
        vol_min_range = st.number_input("Volatility Min", value=0.1, min_value=0.001, step=0.01, format="%.3f", help="Minimum volatility")
        vol_max_range = st.number_input("Volatility Max", value=0.4, min_value=0.001, step=0.01, format="%.3f", help="Maximum volatility")
        
        st.markdown("#### Simulation Configuration")
        n_points = st.number_input("Number of Data Points", value=50, min_value=10, max_value=200, step=10, help="Total number of parameter combinations to simulate")
        simulation_type = st.selectbox("Simulation Type", ["random", "grid", "lhs"], 
                                      format_func=lambda x: {"random": "Random Sampling", "grid": "Grid Search", "lhs": "Latin Hypercube"}[x],
                                      help="Method for generating parameter combinations")
    
    # Simulation execution
    st.markdown("### Execute PINN Simulation")
    
    if st.button("Start PINN Simulation", type="primary", use_container_width=True):
        try:
            # Start simulation
            simulation_payload = {
                "S_min": S_min_range,
                "S_max": S_max_range,
                "K_min": K_min_range,
                "K_max": K_max_range,
                "t_min": t_min_range,
                "t_max": t_max_range,
                "r_min": r_min_range,
                "r_max": r_max_range,
                "vol_min": vol_min_range,
                "vol_max": vol_max_range,
                "n_points": n_points,
                "option_style": option_style,
                "option_type": option_type,
                "simulation_type": simulation_type
            }
            
            with st.spinner("Starting PINN simulation..."):
                resp = requests.post(f"{api_base}/simulate", json=simulation_payload, timeout=60)
                resp.raise_for_status()
                simulation_response = resp.json()
                simulation_id = simulation_response["simulation_id"]
            
            st.success(f"Simulation started! ID: {simulation_id}")
            
            # Monitor simulation progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            while True:
                try:
                    status_resp = requests.get(f"{api_base}/simulation/{simulation_id}", timeout=5)
                    status_resp.raise_for_status()
                    status_data = status_resp.json()
                    
                    progress = status_data["progress"]
                    status = status_data["status"]
                    
                    progress_bar.progress(progress)
                    status_text.text(f"Status: {status} ({progress:.1%})")
                    
                    if status == "completed":
                        st.success("Simulation completed successfully!")
                        results = status_data["results"]
                        break
                    elif status == "error":
                        st.error(f"Simulation failed: {status_data.get('error', 'Unknown error')}")
                        st.stop()
                    
                    time.sleep(2)  # Check every 2 seconds
                    
                except Exception as e:
                    st.warning(f"Error checking status: {e}")
                    time.sleep(2)
                    continue
            
            if results:
                df_results = pd.DataFrame(results)
                        
                # Display simulation results
                st.markdown("### Simulation Results")
                
                # Key metrics cards
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f'''
                    <div class="dashboard-card">
                        <div class="dashboard-value">{len(results)}</div>
                        <div class="dashboard-label">Simulations</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col2:
                    if "Absolute_Error" in df_results.columns:
                        mae = df_results["Absolute_Error"].mean()
                        st.markdown(f'''
                        <div class="dashboard-card">
                            <div class="dashboard-value">{currency_symbol}{mae:.4f}</div>
                            <div class="dashboard-label">MAE</div>
                        </div>
                        ''', unsafe_allow_html=True)
                
                with col3:
                    if "Training_Loss" in df_results.columns:
                        avg_loss = df_results["Training_Loss"].mean()
                        st.markdown(f'''
                        <div class="dashboard-card">
                            <div class="dashboard-value">{avg_loss:.6f}</div>
                            <div class="dashboard-label">Avg Loss</div>
                        </div>
                        ''', unsafe_allow_html=True)
                
                with col4:
                    if "Absolute_Error" in df_results.columns:
                        rmse = np.sqrt((df_results["Absolute_Error"]**2).mean())
                        st.markdown(f'''
                        <div class="dashboard-card">
                            <div class="dashboard-value">{currency_symbol}{rmse:.4f}</div>
                            <div class="dashboard-label">RMSE</div>
                        </div>
                        ''', unsafe_allow_html=True)
                
                # Statistical analysis
                st.markdown("### Statistical Analysis")
                
                if "Absolute_Error" in df_results.columns:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Error Distribution")
                        fig_error = px.histogram(df_results, x="Absolute_Error", 
                                               title="Distribution of Absolute Errors",
                                               labels={"Absolute_Error": "Absolute Error ($)", "count": "Frequency"},
                                               nbins=20)
                        st.plotly_chart(fig_error, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### Error vs Moneyness")
                        fig_moneyness = px.scatter(df_results, x="moneyness", y="Absolute_Error",
                                                 title="Absolute Error vs Moneyness",
                                                 labels={"moneyness": "Moneyness (S/K)", "Absolute_Error": "Absolute Error ($)"})
                        st.plotly_chart(fig_moneyness, use_container_width=True)
                
                # Greeks analysis
                st.markdown("### Greeks Analysis")
                
                greeks_cols = ["Delta", "Gamma", "Vega", "Theta", "Rho"]
                available_greeks = [col for col in greeks_cols if col in df_results.columns]
                
                if available_greeks:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Greeks Statistics")
                        greeks_stats = df_results[available_greeks].describe()
                        st.dataframe(greeks_stats, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### Greeks Correlation Matrix")
                        greeks_corr = df_results[available_greeks].corr()
                        fig_corr = px.imshow(greeks_corr, text_auto=True, aspect="auto",
                                           title="Greeks Correlation Matrix")
                        st.plotly_chart(fig_corr, use_container_width=True)
                
                # Detailed results table
                st.markdown("### Detailed Results")
                st.dataframe(df_results.round(6), use_container_width=True)
                        
                        # Export functionality
                st.markdown("### Export Simulation Data")
                
                export_col1, export_col2, export_col3 = st.columns(3)
                
                with export_col1:
                    # CSV Export
                    csv_data = df_results.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"pinn_simulation_{option_style}_{option_type}_{n_points}points.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with export_col2:
                    # Excel Export
                    import io
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df_results.to_excel(writer, sheet_name='Simulation_Results', index=False)
                        
                        # Add summary sheet
                        summary_data = {
                            'Metric': ['Total Simulations', 'MAE', 'Max Error', 'RMSE', 'Simulation Type', 'Training Loss'],
                            'Value': [len(results), 
                                     f"{df_results['Absolute_Error'].mean():.4f}" if "Absolute_Error" in df_results.columns else "N/A",
                                     f"{df_results['Absolute_Error'].max():.4f}" if "Absolute_Error" in df_results.columns else "N/A",
                                     f"{np.sqrt((df_results['Absolute_Error']**2).mean()):.4f}" if "Absolute_Error" in df_results.columns else "N/A",
                                     simulation_type,
                                     f"{df_results['Training_Loss'].mean():.6f}" if "Training_Loss" in df_results.columns else "N/A"]
                        }
                        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                    
                    excel_data = output.getvalue()
                    st.download_button(
                        label="Download Excel",
                        data=excel_data,
                        file_name=f"pinn_simulation_{option_style}_{option_type}_{n_points}points.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                
                with export_col3:
                    # JSON Export
                    simulation_report = {
                        "metadata": {
                            "timestamp": pd.Timestamp.now().isoformat(),
                            "simulation_id": simulation_id,
                            "simulation_type": simulation_type,
                            "total_points": len(results),
                            "option_style": option_style,
                            "option_type": option_type,
                            "currency": selected_currency
                        },
                        "parameters": {
                            "S_range": [S_min_range, S_max_range],
                            "K_range": [K_min_range, K_max_range],
                            "t_range": [t_min_range, t_max_range],
                            "r_range": [r_min_range, r_max_range],
                            "vol_range": [vol_min_range, vol_max_range]
                        },
                        "statistics": {
                            "mae": float(df_results["Absolute_Error"].mean()) if "Absolute_Error" in df_results.columns else None,
                            "max_error": float(df_results["Absolute_Error"].max()) if "Absolute_Error" in df_results.columns else None,
                            "rmse": float(np.sqrt((df_results["Absolute_Error"]**2).mean())) if "Absolute_Error" in df_results.columns else None,
                            "avg_training_loss": float(df_results["Training_Loss"].mean()) if "Training_Loss" in df_results.columns else None
                        },
                        "results": df_results.to_dict('records')
                    }
                    
                    json_str = json.dumps(simulation_report, indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name=f"pinn_simulation_{option_style}_{option_type}_{n_points}points.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                # Research insights
                st.markdown("### Research Insights")
                
                insights_col1, insights_col2 = st.columns(2)
                
                with insights_col1:
                    st.markdown("#### Model Performance")
                    if "Absolute_Error" in df_results.columns:
                        error_stats = df_results["Absolute_Error"].describe()
                        mae = df_results["Absolute_Error"].mean()
                        rmse = np.sqrt((df_results["Absolute_Error"]**2).mean())
                        st.write(f"**MAE:** {currency_symbol}{mae:.4f}")
                        st.write(f"**RMSE:** {currency_symbol}{rmse:.4f}")
                        st.write(f"**Median Error:** {currency_symbol}{error_stats['50%']:.4f}")
                        st.write(f"**Max Error:** {currency_symbol}{error_stats['max']:.4f}")
                
                with insights_col2:
                    st.markdown("#### Training Insights")
                    if "Training_Loss" in df_results.columns:
                        loss_stats = df_results["Training_Loss"].describe()
                        st.write(f"**Final Training Loss:** {loss_stats['mean']:.6f}")
                        st.write(f"**Loss Range:** {loss_stats['min']:.6f} - {loss_stats['max']:.6f}")
                    
                    if "moneyness" in df_results.columns and "Absolute_Error" in df_results.columns:
                        moneyness_corr = df_results[["moneyness", "Absolute_Error"]].corr().iloc[0, 1]
                        st.write(f"**Moneyness-Error Correlation:** {moneyness_corr:.3f}")
            
            else:
                st.error("No results generated. Please check your parameters and try again.")
                
        except Exception as e:
            st.error(f"Simulation failed: {e}")

# Professional footer
st.markdown("---")
st.markdown('<p style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">Powered by Physics-Informed Neural Networks (PINNs) | Research Platform</p>', unsafe_allow_html=True)
