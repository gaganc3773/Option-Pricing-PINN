# PINN Option Pricing Research Platform - Complete Documentation

##  **Project Overview**

This is a comprehensive Physics-Informed Neural Network (PINN) based option pricing platform that supports both European and American options (call/put). The platform consists of a FastAPI backend and Streamlit frontend, providing professional-grade research tools for quantitative finance.

##  **Project Structure**

```
BSM_PINN/
├── backend/
│   ├── main.py                 # FastAPI backend with all endpoints
│   └── __init__.py
├── frontend/
│   └── streamlit_app.py        # Professional Streamlit frontend
├── models/                     # Trained PINN models
│   ├── european_call.pt
│   ├── european_put.pt
│   ├── american_call.pt
│   ├── american_put.pt
│   └── *_convergence.csv       # Training convergence data
├── train_models.py            # Training script for all models
├── BSM_PINN_2.ipynb           # Original Jupyter notebook
├── requirements.txt           # Python dependencies
└── README.md                  

##  **PINN Architecture & Training**

### **Neural Network Architecture**
- **Input Layer**: 5 neurons (t, S, r, σ, K)
- **Hidden Layers**: 2 layers of 64 neurons each
- **Output Layer**: 1 neuron (option price)
- **Activation**: ReLU
- **Optimizer**: Adam (lr=1e-3)
- **Training Epochs**: 5,000

### **Loss Functions**

#### **European Options**
```python
def BSM_loss_european(model, X_f, X_T, X_lb, X_ub, T, S_max, option_type):
    # PDE Residual: ∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0
    # Terminal Condition: V(T,S) = max(S-K, 0) for calls
    # Boundary Conditions: V(t,0) = 0, V(t,S_max) = S_max - K*exp(-r(T-t))
```

#### **American Options**
```python
def BSM_loss_american(model, X_f, X_T, X_lb, X_ub, T, S_max, option_type):
    # Same as European PLUS:
    # Early Exercise Constraint: V(t,S) ≥ max(S-K, 0) for calls
```

### **Training Data Generation**
- **Interior Points**: 10,000 points using Latin Hypercube Sampling
- **Terminal Points**: 1,000 points at maturity
- **Boundary Points**: 1,000 points each at S=0 and S=S_max
- **Parameter Ranges**:
  - S: [0, 200]
  - t: [0, 1]
  - r: [0.01, 0.1]
  - σ: [0.1, 0.5]
  - K: [50, 150]

### **Model Performance**
- **European Call**: RMSE: 4.07, R²: 0.990
- **European Put**: RMSE: 15.40, R²: 0.834
- **American Call**: Final Loss: 32.60
- **American Put**: Final Loss: 40.97

##  **Backend API (FastAPI)**

### **Core Endpoints**

#### **1. Price Prediction**
```http
POST /price
```
**Request Body:**
```json
{
  "S": 100.0,
  "t": 0.0,
  "r": 0.05,
  "K": 100.0,
  "volatility": 0.2,
  "T": 1.0,
  "option_style": "european",
  "option_type": "call"
}
```
**Response:**
```json
{
  "price": 4.5517,
  "bs_price": 4.5517,
  "error": 0.0001,
  "relative_error": 0.0002
}
```

#### **2. Greeks Calculation**
```http
POST /greeks
```
**Response:**
```json
{
  "delta": 0.5234,
  "gamma": 0.0123,
  "vega": 39.8765,
  "theta": -6.5432,
  "rho": 42.1098,
  "bs_delta": 0.5234,
  "bs_gamma": 0.0123,
  "bs_vega": 39.8765,
  "bs_theta": -6.5432,
  "bs_rho": 42.1098
}
```

#### **3. 2D Surface Plot**
```http
POST /surface
```

#### **4. 3D Surface Plot**
```http
POST /surface3d
```

#### **5. Convergence Data**
```http
GET /convergence?option_style=european&option_type=call
```

### **Black-Scholes Integration**
- **Analytical Pricing**: Implemented for European options
- **Greeks Calculation**: Full Black-Scholes Greeks
- **Error Analysis**: Automatic comparison with PINN predictions
- **Validation**: PINN vs analytical accuracy metrics

##  **Frontend (Streamlit)**

### **Professional UI Design**
- **Clean Interface**: No icons, professional color scheme
- **Sidebar Organization**: All inputs organized in logical sections
- **Responsive Layout**: Main content + sidebar configuration
- **Professional Styling**: Custom CSS with gradients and shadows

### **Tab Structure**

#### **1. Price Tab**
- PINN price calculation
- Black-Scholes comparison (European options)
- Error analysis with color-coded accuracy indicators
- Additional metrics: Moneyness, Time Value, Intrinsic Value

#### **2. Greeks Tab**
- All five Greeks calculation
- Black-Scholes comparison table
- Risk analysis indicators (ITM/OTM/ATM)
- Professional card-based display

#### **3. 2D Surface Tab**
- Price vs underlying price curves
- Interactive plots with proper axis labels
- Strike price reference lines
- Custom hover templates

#### **4. 3D Surface Tab**
- Interactive 3D surfaces for price and all Greeks
- Proper axis labeling: "Underlying Price ($)", "Time to Maturity", "Greek Name"
- Custom hover templates with descriptive labels
- Multiple surface types: Price, Delta, Gamma, Vega, Theta, Rho

#### **5. Convergence Tab**
- Real training convergence data from CSV files
- Loss component breakdown
- Exercise loss for American options
- Training statistics and model information

#### **6. Analysis & Export Tab** (NEW)
- Batch parameter analysis
- Error distribution plots
- CSV export functionality
- Summary statistics
- Future enhancement roadmap

### **Sidebar Configuration**
- **Option Parameters**: Style (European/American), Type (Call/Put)
- **Market Parameters**: S, K, r, σ with helpful tooltips
- **Time Parameters**: t, T
- **Surface Parameters**: Ranges and resolution settings
- **API Settings**: Backend configuration

## **Phase 1 & 2 Implementation Status**

### ** Completed Features**

#### **Phase 1.1: Model Comparison & Validation**
-  Black-Scholes analytical pricing integration
-  Error analysis with absolute and relative errors
-  Accuracy indicators (Excellent/Good/Moderate/Poor)
-  Greeks comparison tables
-  Model performance metrics

#### **Phase 1.2: Advanced Analytics**
-  Batch parameter analysis
-  Error distribution visualization
-  Summary statistics
-  CSV export functionality
-  Professional error analysis dashboard

#### **Phase 1.3: Enhanced Visualization**
-  Proper axis labels for all plots
-  Custom hover templates
-  Interactive 3D surfaces
-  Professional styling and layout

### ** In Progress**
-  Sensitivity analysis dashboard
-  Monte Carlo analysis
-  Real-time market data integration

### ** Next Phase Priorities**

#### **Phase 2.1: Experiment Management**
- Scenario builder (save/load configurations)
- Advanced batch processing
- Excel export functionality
- Research notebook integration

#### **Phase 2.2: Advanced PINN Analysis**
- Loss component analysis
- Model architecture comparison
- Hyperparameter optimization
- Uncertainty quantification

#### **Phase 2.3: Real-Time Market Integration**
- Live data feeds (Yahoo Finance, Alpha Vantage)
- Market validation
- Historical backtesting
- Portfolio analysis

##  **Technical Implementation Details**

### **Model Loading System**
```python
def get_model(option_style: str, option_type: str) -> nn.Module:
    model_key = f"{option_style}_{option_type}"
    if model_key not in _models:
        model = NN()
        model_path = f"models/{model_key}.pt"
        state = torch.load(model_path, map_location=_device)
        model.load_state_dict(state)
        model.eval()
        _models[model_key] = model
    return _models[model_key]
```

### **Black-Scholes Implementation**
```python
def bs_price_european(S, K, T, t, r, sigma, option_type):
    tau = T - t
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r*tau) * norm.cdf(d2)
    else:
        price = K * np.exp(-r*tau) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price
```

### **Error Analysis**
```python
if req.option_style == "european":
    bs_price = bs_price_european(req.S, req.K, req.T, req.t, req.r, req.volatility, req.option_type)
    error = abs(pinn_price - bs_price)
    if bs_price != 0:
        relative_error = error / abs(bs_price)
```

##  **Usage Examples**

### **Single Option Pricing**
1. Select option style and type in sidebar
2. Set market parameters (S, K, r, σ, t, T)
3. Click "Compute PINN Price"
4. View PINN price, Black-Scholes comparison, and error analysis

### **Batch Analysis**
1. Go to "Analysis & Export" tab
2. Set parameter ranges using sliders
3. Click "Run Batch Analysis"
4. View results table and error distribution
5. Download CSV for further analysis

### **3D Surface Analysis**
1. Go to "3D Surface" tab
2. Select surface type (Price, Delta, etc.)
3. Set resolution and ranges
4. Generate interactive 3D plot
5. Explore surface with proper axis labels

##  **Installation & Setup**

### **Requirements**
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

### **Running the Application**
```bash
# Start backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Start frontend
streamlit run frontend/streamlit_app.py --server.headless true --server.port 8501
```

### **Access Points**
- **Backend API**: http://localhost:8000
- **Frontend**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs

##  **Key Achievements**

1. **Professional Research Platform**: Clean, academic-style interface
2. **Multi-Model Support**: European/American, Call/Put combinations
3. **Black-Scholes Integration**: Analytical comparison and validation
4. **Advanced Analytics**: Batch analysis, error metrics, export functionality
5. **Enhanced Visualization**: Proper axis labels, interactive plots
6. **Research-Ready**: Export capabilities, statistical analysis

##  **Future Roadmap**

### **Phase 3: Advanced Research Capabilities**
- Ensemble methods and uncertainty quantification
- Financial engineering tools (option strategies, risk management)
- Research collaboration features

### **Phase 4: Advanced Visualization**
- Custom plot types (volatility smiles, term structures)
- Interactive dashboards
- Mobile responsiveness

### **Phase 5: Performance & Scalability**
- GPU acceleration
- Cloud deployment
- Microservices architecture
---
**Status**: Phase 1 & 2 Complete, Ready for Phase 3
**Next Focus**: Sensitivity analysis, Monte Carlo methods, real-time data integration
