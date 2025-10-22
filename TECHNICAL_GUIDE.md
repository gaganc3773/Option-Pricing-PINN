# Technical Implementation Guide - PINN Option Pricing Platform

##  **Backend Implementation Details**

### **Model Architecture**
```python
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.hiddenlayers = nn.Sequential(
            nn.Linear(5, 64),    # Input: [t, S, r, σ, K]
            nn.ReLU(),
            nn.Linear(64, 64),   # Hidden layer
            nn.ReLU(),
            nn.Linear(64, 1)     # Output: Option price
        )
    
    def forward(self, X):
        return self.hiddenlayers(X)
```

### **Model Loading System**
```python
_models: dict = {}
_device = torch.device("cpu")

def get_model(option_style: str, option_type: str) -> nn.Module:
    """Load model based on option style and type"""
    global _models
    model_key = f"{option_style}_{option_type}"
    
    if model_key not in _models:
        model = NN()
        try:
            model_path = f"models/{model_key}.pt"
            state = torch.load(model_path, map_location=_device)
            model.load_state_dict(state)
            model.eval()
            _models[model_key] = model
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail=f"Model {model_path} not found")
    
    return _models[model_key]
```

### **Black-Scholes Implementation**
```python
def bs_price_european(S: float, K: float, T: float, t: float, r: float, sigma: float, option_type: str = 'call') -> float:
    """Black-Scholes analytical solution for European options"""
    tau = T - t
    tau = max(tau, 1e-10)  # avoid zero division
    
    if tau <= 0:
        # At maturity
        if option_type == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r*tau) * norm.cdf(d2)
    else:  # put
        price = K * np.exp(-r*tau) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return float(price)
```

### **API Response Models**
```python
class PriceResponse(BaseModel):
    price: float
    bs_price: Optional[float] = None
    error: Optional[float] = None
    relative_error: Optional[float] = None

class GreeksResponse(BaseModel):
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    bs_delta: Optional[float] = None
    bs_gamma: Optional[float] = None
    bs_vega: Optional[float] = None
    bs_theta: Optional[float] = None
    bs_rho: Optional[float] = None
```

##  **Frontend Implementation Details**

### **Professional CSS Styling**
```css
.main-header {
    font-size: 2.2rem;
    font-weight: 600;
    color: #2c3e50;
    text-align: center;
    margin-bottom: 1.5rem;
    border-bottom: 2px solid #3498db;
    padding-bottom: 0.5rem;
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

.result-container {
    background: white;
    padding: 2rem;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
}
```

### **Sidebar Organization**
```python
# Sidebar with input parameters
with st.sidebar:
    st.markdown("## Configuration")
    
    # Option Configuration
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### Option Parameters")
    option_style = st.selectbox("Option Style", ["european", "american"])
    option_type = st.selectbox("Option Type", ["call", "put"])
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Market Parameters
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### Market Parameters")
    S = st.number_input("Spot Price (S)", value=100.0, min_value=0.0, step=1.0)
    K = st.number_input("Strike Price (K)", value=100.0, min_value=0.0, step=1.0)
    r = st.number_input("Risk-free Rate (r)", value=0.05, step=0.01, format="%.3f")
    vol = st.number_input("Volatility (σ)", value=0.2, min_value=0.0001, step=0.01, format="%.3f")
    st.markdown('</div>', unsafe_allow_html=True)
```

### **Interactive Plot Implementation**
```python
# 2D Surface with proper axis labels
fig = px.line(df, x="S", y="price", 
             title=f"{option_style.title()} {option_type.title()} Option Price vs Underlying Price",
             labels={"S": "Underlying Price ($)", "price": "Option Price ($)"})
fig.update_layout(height=500, showlegend=False)
fig.add_vline(x=K, line_dash="dash", line_color="red", 
             annotation_text=f"Strike: ${K}")

# Update hover template for better interactivity
fig.update_traces(hovertemplate="<b>Underlying Price:</b> $%{x:.2f}<br><b>Option Price:</b> $%{y:.4f}<extra></extra>")

# 3D Surface with proper axis labels
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
```

##  **Training Implementation**

### **Loss Function Implementation**
```python
def BSM_loss_european(model, X_f, X_T, X_lb, X_ub, T, S_max, option_type):
    """PINN Loss for European Options"""
    # PDE Residual
    X_f.requires_grad = True
    V_pred = model(X_f)
    
    t_f = X_f[:, 0:1]
    S_f = X_f[:, 1:2]
    r_f = X_f[:, 2:3]
    sigma_f = X_f[:, 3:4]
    
    grads = torch.autograd.grad(V_pred.sum(), X_f, create_graph=True)[0]
    V_t = grads[:, 0:1]
    V_S = grads[:, 1:2]
    V_SS = torch.autograd.grad(V_S.sum(), X_f, create_graph=True)[0][:, 1:2]
    
    # PDE: ∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0
    residual = V_t + 0.5 * sigma_f**2 * S_f**2 * V_SS + r_f * S_f * V_S - r_f * V_pred
    pde_loss = torch.mean(residual**2)
    
    # Terminal Condition
    S_T = X_T[:, 1:2]
    K_T = X_T[:, 4:5]
    
    if option_type == 'call':
        payoff = torch.clamp(S_T - K_T, min=0.0)
    else:
        payoff = torch.clamp(K_T - S_T, min=0.0)
    
    V_T_pred = model(X_T)
    terminal_loss = torch.mean((V_T_pred - payoff)**2)
    
    # Boundary Conditions
    V_lb_pred = model(X_lb)
    if option_type == 'call':
        left_bc_target = torch.zeros_like(V_lb_pred)
    else:
        K_lb = X_lb[:, 4:5]
        r_lb = X_lb[:, 2:3]
        left_bc_target = K_lb * torch.exp(-r_lb * (T - X_lb[:, 0:1]))
    left_bc_loss = torch.mean((V_lb_pred - left_bc_target)**2)
    
    # Right boundary
    t_ub = X_ub[:, 0:1]
    S_ub = X_ub[:, 1:2]
    r_ub = X_ub[:, 2:3]
    K_ub = X_ub[:, 4:5]
    V_ub_pred = model(X_ub)
    
    if option_type == 'call':
        right_bc_target = S_ub - K_ub * torch.exp(-r_ub * (T - t_ub))
    else:
        right_bc_target = torch.zeros_like(V_ub_pred)
    right_bc_loss = torch.mean((V_ub_pred - right_bc_target)**2)
    
    # Total Loss
    total_loss = (
        1.0 * pde_loss +
        10.0 * terminal_loss +
        10.0 * (left_bc_loss + right_bc_loss)
    )
    
    return total_loss, pde_loss.item(), terminal_loss.item(), left_bc_loss.item(), right_bc_loss.item()
```

### **American Options Loss Function**
```python
def BSM_loss_american(model, X_f, X_T, X_lb, X_ub, T, S_max, option_type):
    """PINN Loss for American Options with Early Exercise Constraint"""
    # Same as European PLUS early exercise penalty
    
    # ... (European loss components) ...
    
    # Early exercise penalty
    K_f = X_f[:, 4:5]
    payoff_all = torch.clamp(K_f - S_f, min=0.0) if option_type == 'put' else torch.clamp(S_f - K_f, min=0.0)
    exercise_loss = torch.mean(torch.relu(payoff_all - V_pred)**2)
    
    # Total Loss with exercise constraint
    total_loss = (
        1.0 * pde_loss +
        10.0 * terminal_loss +
        10.0 * (left_bc_loss + right_bc_loss) +
        10.0 * exercise_loss
    )
    
    return total_loss, pde_loss.item(), terminal_loss.item(), left_bc_loss.item(), right_bc_loss.item(), exercise_loss.item()
```

##  **Batch Analysis Implementation**

### **Frontend Batch Analysis**
```python
# Generate parameter combinations
S_values = np.linspace(S_range[0], S_range[1], n_points)
K_values = np.linspace(K_range[0], K_range[1], n_points)
vol_values = np.linspace(vol_range[0], vol_range[1], n_points)

results = []
for s in S_values:
    for k in K_values:
        for v in vol_values:
            payload = {"S": s, "t": t, "r": r, "K": k, "volatility": v, "T": T,
                      "option_style": option_style, "option_type": option_type}
            resp = requests.post(f"{api_base}/price", json=payload, timeout=5)
            if resp.status_code == 200:
                result = resp.json()
                results.append({
                    "S": s, "K": k, "volatility": v,
                    "PINN_Price": result["price"],
                    "BS_Price": result.get("bs_price"),
                    "Error": result.get("error"),
                    "Relative_Error": result.get("relative_error")
                })

# Export functionality
csv = df_results.to_csv(index=False)
st.download_button(
    label="Download CSV",
    data=csv,
    file_name=f"pinn_analysis_{option_style}_{option_type}.csv",
    mime="text/csv"
)
```

##  **Error Analysis Implementation**

### **Backend Error Calculation**
```python
# Calculate Black-Scholes price for European options
bs_price = None
error = None
relative_error = None

if req.option_style == "european":
    bs_price = bs_price_european(req.S, req.K, req.T, req.t, req.r, req.volatility, req.option_type)
    error = abs(pinn_price - bs_price)
    if bs_price != 0:
        relative_error = error / abs(bs_price)

return PriceResponse(
    price=pinn_price,
    bs_price=bs_price,
    error=error,
    relative_error=relative_error
)
```

### **Frontend Error Display**
```python
# Error analysis with color-coded indicators
if result['relative_error'] < 0.01:
    st.success(f" Excellent accuracy: {result['relative_error']:.2%} error")
elif result['relative_error'] < 0.05:
    st.info(f" Good accuracy: {result['relative_error']:.2%} error")
elif result['relative_error'] < 0.10:
    st.warning(f" Moderate accuracy: {result['relative_error']:.2%} error")
else:
    st.error(f" Poor accuracy: {result['relative_error']:.2%} error")
```

##  **Deployment Configuration**

### **Requirements.txt**
```txt
fastapi==0.115.0
uvicorn[standard]==0.30.6
pydantic==2.9.2
numpy>=1.24.0
torch>=2.5.0
scipy>=1.10.0
pandas>=2.0.0
pyDOE==0.3.8
scikit-learn>=1.3.0
matplotlib>=3.7.0
streamlit>=1.28.0
requests>=2.28.0
plotly>=5.0.0
```

### **Startup Scripts**
```bash
#!/bin/bash
# start_backend.sh
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

#!/bin/bash
# start_frontend.sh
streamlit run frontend/streamlit_app.py --server.headless true --server.port 8501
```

##  **Key Implementation Decisions**

1. **Model Architecture**: Simple 3-layer network (5→64→64→1) for computational efficiency
2. **Loss Weighting**: PDE=1.0, Terminal=10.0, Boundary=10.0, Exercise=10.0
3. **Training Data**: Latin Hypercube Sampling for better coverage
4. **Error Handling**: Comprehensive validation and graceful error messages
5. **UI Design**: Professional academic style without icons
6. **Export Format**: CSV for compatibility with research tools
7. **API Design**: RESTful with clear request/response models

##  **Future Implementation Notes**

### **Phase 3 Priorities**
1. **Sensitivity Analysis**: Add parameter sensitivity matrices
2. **Monte Carlo**: Implement Monte Carlo simulation for validation
3. **Real-time Data**: Integrate with financial data APIs
4. **Advanced Export**: Excel and JSON export formats

### **Performance Optimizations**
1. **Caching**: Cache frequently used calculations
2. **Parallel Processing**: Multi-threaded batch analysis
3. **GPU Support**: CUDA acceleration for training
4. **Database**: Store results and configurations

### **Research Features**
1. **Experiment Management**: Save/load parameter configurations
2. **Version Control**: Track model versions and changes
3. **Collaboration**: Multi-user access and sharing
4. **Documentation**: Integrated research notebook features

---

**Implementation Status**: Phase 1 & 2 Complete
**Next Development**: Phase 3 Advanced Features
**Context Preservation**: Complete technical documentation
