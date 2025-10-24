from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional
import pandas as pd
from scipy.stats import norm
import time
from pyDOE import lhs

app = FastAPI(title="BSM PINN Pricing API")


class NN(nn.Module):
	def __init__(self):
		super(NN, self).__init__()
		self.hiddenlayers = nn.Sequential(
			nn.Linear(5, 64),
			nn.ReLU(),
			nn.Linear(64, 64),
			nn.ReLU(),
			nn.Linear(64, 1),
		)

	def forward(self, X):
		return self.hiddenlayers(X)


# Load model weights saved by training script
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


class PINNSimulationRequest(BaseModel):
	S_min: float
	S_max: float
	K_min: float
	K_max: float
	t_min: float
	t_max: float
	r_min: float
	r_max: float
	vol_min: float
	vol_max: float
	n_points: int = 100
	option_style: str = "european"
	option_type: str = "call"
	simulation_type: str = "random"  # "random", "grid", "lhs"


class PINNSimulationResponse(BaseModel):
	simulation_id: str
	status: str
	progress: float
	results: Optional[List[dict]] = None
	error: Optional[str] = None
	metadata: Optional[dict] = None


# Store simulation results
_simulation_results: dict = {}


def run_pinn_simulation(simulation_id: str, request: PINNSimulationRequest):
	"""Run PINN simulation for given parameters"""
	try:
		# Update simulation status
		_simulation_results[simulation_id] = {
			"status": "loading_model",
			"progress": 0.0,
			"results": None,
			"error": None
		}
		
		# Extract parameters from request
		S_min, S_max = request.S_min, request.S_max
		K_min, K_max = request.K_min, request.K_max
		t_min, t_max = request.t_min, request.t_max
		r_min, r_max = request.r_min, request.r_max
		vol_min, vol_max = request.vol_min, request.vol_max
		n_points = request.n_points
		option_style = request.option_style
		option_type = request.option_type
		simulation_type = request.simulation_type
		
		# Load pre-trained PINN model
		model = get_model(option_style, option_type)
		
		_simulation_results[simulation_id]["progress"] = 0.3
		
		# Generate test data
		if simulation_type == "random":
			S_test = np.random.uniform(S_min, S_max, n_points)
			K_test = np.random.uniform(K_min, K_max, n_points)
			t_test = np.random.uniform(t_min, t_max, n_points)
			r_test = np.random.uniform(r_min, r_max, n_points)
			vol_test = np.random.uniform(vol_min, vol_max, n_points)
		elif simulation_type == "grid":
			n_per_dim = int(n_points ** (1/5))
			S_test = np.linspace(S_min, S_max, n_per_dim)
			K_test = np.linspace(K_min, K_max, n_per_dim)
			t_test = np.linspace(t_min, t_max, n_per_dim)
			r_test = np.linspace(r_min, r_max, n_per_dim)
			vol_test = np.linspace(vol_min, vol_max, n_per_dim)
			
			S_mesh, K_mesh, t_mesh, r_mesh, vol_mesh = np.meshgrid(S_test, K_test, t_test, r_test, vol_test)
			S_test = S_mesh.flatten()[:n_points]
			K_test = K_mesh.flatten()[:n_points]
			t_test = t_mesh.flatten()[:n_points]
			r_test = r_mesh.flatten()[:n_points]
			vol_test = vol_mesh.flatten()[:n_points]
		else:  # lhs
			sample = lhs(5, samples=n_points)
			S_test = sample[:, 0] * (S_max - S_min) + S_min
			K_test = sample[:, 1] * (K_max - K_min) + K_min
			t_test = sample[:, 2] * (t_max - t_min) + t_min
			r_test = sample[:, 3] * (r_max - r_min) + r_min
			vol_test = sample[:, 4] * (vol_max - vol_min) + vol_min
		
		_simulation_results[simulation_id]["progress"] = 0.7
		
		# Generate predictions
		results = []
		X_test = np.column_stack([t_test, S_test, r_test, vol_test, K_test])
		X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
		
		with torch.no_grad():
			pinn_predictions = model(X_test_tensor).numpy().flatten()
		
		# Calculate Black-Scholes prices for comparison
		for i in range(n_points):
			bs_price = None
			if option_style == "european":
				bs_price = bs_price_european(S_test[i], K_test[i], 1.0, t_test[i], r_test[i], vol_test[i], option_type)
			
			# Calculate Greeks using autograd
			X_grad = torch.tensor([[t_test[i], S_test[i], r_test[i], vol_test[i], K_test[i]]], 
								  dtype=torch.float32, requires_grad=True)
			V = model(X_grad)
			
			grads = torch.autograd.grad(V.sum(), X_grad, create_graph=True)[0]
			dV_dS = grads[0, 1]
			dV_dt = grads[0, 0]
			dV_dr = grads[0, 2]
			dV_dsigma = grads[0, 3]
			
			# Second derivative for gamma
			d2V_dS = torch.autograd.grad(dV_dS, X_grad, retain_graph=True)[0][0, 1]
			
			result = {
				"S": float(S_test[i]),
				"K": float(K_test[i]),
				"t": float(t_test[i]),
				"r": float(r_test[i]),
				"volatility": float(vol_test[i]),
				"moneyness": float(S_test[i] / K_test[i]),
				"PINN_Price": float(pinn_predictions[i]),
				"BS_Price": float(bs_price) if bs_price is not None else None,
				"Absolute_Error": float(abs(pinn_predictions[i] - bs_price)) if bs_price is not None else None,
				"Intrinsic_Value": float(max(S_test[i] - K_test[i], 0)) if option_type == "call" else float(max(K_test[i] - S_test[i], 0)),
				"Time_Value": float(pinn_predictions[i] - max(S_test[i] - K_test[i], 0)) if option_type == "call" else float(pinn_predictions[i] - max(K_test[i] - S_test[i], 0)),
				"Delta": float(dV_dS),
				"Gamma": float(d2V_dS),
				"Vega": float(dV_dsigma),
				"Theta": float(-dV_dt),
				"Rho": float(dV_dr),
				"Option_Style": option_style,
				"Option_Type": option_type,
				"Model_Used": f"Pre-trained {option_style.title()} {option_type.title()} PINN Model",
				"Simulation_Type": simulation_type
			}
			results.append(result)
		
		# Update simulation results with metadata
		_simulation_results[simulation_id] = {
			"status": "completed",
			"progress": 1.0,
			"results": results,
			"error": None,
			"metadata": {
				"simulation_id": simulation_id,
				"option_style": option_style,
				"option_type": option_type,
				"model_used": f"Pre-trained {option_style.title()} {option_type.title()} PINN Model",
				"simulation_type": simulation_type,
				"total_points": n_points,
				"parameter_ranges": {
					"S_range": [S_min, S_max],
					"K_range": [K_min, K_max],
					"t_range": [t_min, t_max],
					"r_range": [r_min, r_max],
					"vol_range": [vol_min, vol_max]
				},
				"model_file": f"models/{option_style}_{option_type}.pt"
			}
		}
		
	except Exception as e:
		_simulation_results[simulation_id] = {
			"status": "error",
			"progress": 0.0,
			"results": None,
			"error": str(e)
		}


@app.post("/simulate", response_model=PINNSimulationResponse)
async def start_simulation(request: PINNSimulationRequest):
	"""Start a new PINN simulation"""
	simulation_id = f"sim_{int(time.time())}"
	
	# Validate inputs
	if request.option_style not in ["european", "american"]:
		raise HTTPException(status_code=400, detail="option_style must be 'european' or 'american'")
	if request.option_type not in ["call", "put"]:
		raise HTTPException(status_code=400, detail="option_type must be 'call' or 'put'")
	if request.simulation_type not in ["random", "grid", "lhs"]:
		raise HTTPException(status_code=400, detail="simulation_type must be 'random', 'grid', or 'lhs'")
	
	# Start simulation in background
	import threading
	thread = threading.Thread(target=run_pinn_simulation, args=(simulation_id, request))
	thread.start()
	
	return PINNSimulationResponse(
		simulation_id=simulation_id,
		status="started",
		progress=0.0
	)


@app.get("/simulation/{simulation_id}", response_model=PINNSimulationResponse)
async def get_simulation_status(simulation_id: str):
	"""Get the status of a simulation"""
	if simulation_id not in _simulation_results:
		raise HTTPException(status_code=404, detail="Simulation not found")
	
	result = _simulation_results[simulation_id]
	return PINNSimulationResponse(
		simulation_id=simulation_id,
		status=result["status"],
		progress=result["progress"],
		results=result["results"],
		error=result["error"],
		metadata=result.get("metadata")
	)


class PriceRequest(BaseModel):
	S: float
	t: float
	r: float
	K: float
	volatility: float
	T: float = 1.0
	option_style: str = "european"  # "european" or "american"
	option_type: str = "call"  # "call" or "put"


class PriceResponse(BaseModel):
	price: float
	bs_price: Optional[float] = None
	error: Optional[float] = None
	relative_error: Optional[float] = None


def predict_price(model: nn.Module, S: float, t: float, r: float, K: float, sigma: float, T: float) -> float:
	x = torch.tensor([[t, S, r, sigma, K]], dtype=torch.float32)
	with torch.no_grad():
		v = model(x).item()
	return float(v)

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

def bs_greeks_european(S: float, K: float, T: float, t: float, r: float, sigma: float, option_type: str = 'call') -> dict:
    """Black-Scholes Greeks for European options"""
    tau = T - t
    tau = max(tau, 1e-10)
    
    if tau <= 0:
        # At maturity, Greeks are not well-defined
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    
    # Common terms
    sqrt_tau = np.sqrt(tau)
    exp_neg_rt = np.exp(-r*tau)
    
    # Delta
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
    
    # Gamma (same for both)
    gamma = norm.pdf(d1) / (S * sigma * sqrt_tau)
    
    # Vega (same for both)
    vega = S * norm.pdf(d1) * sqrt_tau
    
    # Theta
    if option_type == 'call':
        theta = -(S * norm.pdf(d1) * sigma) / (2 * sqrt_tau) - r * K * exp_neg_rt * norm.cdf(d2)
    else:
        theta = -(S * norm.pdf(d1) * sigma) / (2 * sqrt_tau) + r * K * exp_neg_rt * norm.cdf(-d2)
    
    # Rho
    if option_type == 'call':
        rho = K * tau * exp_neg_rt * norm.cdf(d2)
    else:
        rho = -K * tau * exp_neg_rt * norm.cdf(-d2)
    
    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho)
    }


@app.post("/price", response_model=PriceResponse)
async def price(req: PriceRequest):
	# Validate inputs
	if req.option_style not in ["european", "american"]:
		raise HTTPException(status_code=400, detail="option_style must be 'european' or 'american'")
	if req.option_type not in ["call", "put"]:
		raise HTTPException(status_code=400, detail="option_type must be 'call' or 'put'")
	
	model = get_model(req.option_style, req.option_type)
	pinn_price = predict_price(model, req.S, req.t, req.r, req.K, req.volatility, req.T)
	
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


@app.post("/greeks", response_model=GreeksResponse)
async def greeks(req: PriceRequest):
	# Validate inputs
	if req.option_style not in ["european", "american"]:
		raise HTTPException(status_code=400, detail="option_style must be 'european' or 'american'")
	if req.option_type not in ["call", "put"]:
		raise HTTPException(status_code=400, detail="option_type must be 'call' or 'put'")
	
	model = get_model(req.option_style, req.option_type)
	# Autograd for derivatives
	x = torch.tensor([[req.t, req.S, req.r, req.volatility, req.K]], dtype=torch.float32, requires_grad=True)
	v = model(x)
	grads = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
	# V_S
	dV_dS = grads[0, 1]
	# V_SS
	d2V_dS = torch.autograd.grad(dV_dS, x, retain_graph=True)[0][0, 1]
	# V_sigma
	dV_dsigma = grads[0, 3]
	# V_t
	dV_dt = grads[0, 0]
	# V_r
	dV_dr = grads[0, 2]

	delta = float(dV_dS.detach().cpu().item())
	gamma = float(d2V_dS.detach().cpu().item())
	vega = float(dV_dsigma.detach().cpu().item())
	theta = float((-dV_dt).detach().cpu().item())  # conventional sign
	rho = float(dV_dr.detach().cpu().item())

	# Calculate Black-Scholes Greeks for European options
	bs_greeks = None
	if req.option_style == "european":
		bs_greeks = bs_greeks_european(req.S, req.K, req.T, req.t, req.r, req.volatility, req.option_type)

	return GreeksResponse(
		delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho,
		bs_delta=bs_greeks["delta"] if bs_greeks else None,
		bs_gamma=bs_greeks["gamma"] if bs_greeks else None,
		bs_vega=bs_greeks["vega"] if bs_greeks else None,
		bs_theta=bs_greeks["theta"] if bs_greeks else None,
		bs_rho=bs_greeks["rho"] if bs_greeks else None
	)


class SurfaceRequest(BaseModel):
	t: float
	r: float
	K: float
	volatility: float
	S_min: float = 0.0
	S_max: float = 200.0
	n_S: int = 50
	option_style: str = "european"
	option_type: str = "call"


class SurfacePoint(BaseModel):
	S: float
	price: float


@app.post("/surface", response_model=List[SurfacePoint])
async def surface(req: SurfaceRequest):
	# Validate inputs
	if req.option_style not in ["european", "american"]:
		raise HTTPException(status_code=400, detail="option_style must be 'european' or 'american'")
	if req.option_type not in ["call", "put"]:
		raise HTTPException(status_code=400, detail="option_type must be 'call' or 'put'")
	
	model = get_model(req.option_style, req.option_type)
	S_values = np.linspace(req.S_min, req.S_max, req.n_S)
	t = np.full_like(S_values, req.t)
	r = np.full_like(S_values, req.r)
	sigma = np.full_like(S_values, req.volatility)
	K = np.full_like(S_values, req.K)
	X = np.stack([t, S_values, r, sigma, K], axis=1)
	X_tensor = torch.tensor(X, dtype=torch.float32)
	with torch.no_grad():
		V = model(X_tensor).squeeze(1).cpu().numpy()
	return [SurfacePoint(S=float(s), price=float(p)) for s, p in zip(S_values, V)]


class ConvergencePoint(BaseModel):
	epoch: int
	total: float
	pde: float
	terminal: float
	left_bc: float
	right_bc: float
	exercise: Optional[float] = None  # For American options


@app.get("/convergence", response_model=List[ConvergencePoint])
async def convergence(option_style: str = "european", option_type: str = "call"):
	"""Get convergence data for a specific model"""
	# Validate inputs
	if option_style not in ["european", "american"]:
		raise HTTPException(status_code=400, detail="option_style must be 'european' or 'american'")
	if option_type not in ["call", "put"]:
		raise HTTPException(status_code=400, detail="option_type must be 'call' or 'put'")
	
	try:
		# Load convergence data from CSV
		model_key = f"{option_style}_{option_type}"
		df = pd.read_csv(f"backend/models/{model_key}_convergence.csv")
		
		convergence_data = []
		for _, row in df.iterrows():
			convergence_data.append(ConvergencePoint(
				epoch=int(row['epoch']),
				total=float(row['total']),
				pde=float(row['pde']),
				terminal=float(row['terminal']),
				left_bc=float(row['left_bc']),
				right_bc=float(row['right_bc']),
				exercise=float(row['exercise']) if 'exercise' in row and row['exercise'] > 0 else None
			))
		
		return convergence_data
		
	except FileNotFoundError:
		raise HTTPException(status_code=404, detail=f"Convergence data for {option_style} {option_type} not found")


class Surface3DRequest(BaseModel):
	r: float
	K: float
	volatility: float
	S_min: float = 0.0
	S_max: float = 200.0
	t_min: float = 0.0
	t_max: float = 1.0
	n_S: int = 20
	n_t: int = 20
	option_style: str = "european"
	option_type: str = "call"


class Surface3DPoint(BaseModel):
	S: float
	t: float
	price: float
	delta: float
	gamma: float
	vega: float
	theta: float
	rho: float


@app.post("/surface3d", response_model=List[Surface3DPoint])
async def surface3d(req: Surface3DRequest):
	# Validate inputs
	if req.option_style not in ["european", "american"]:
		raise HTTPException(status_code=400, detail="option_style must be 'european' or 'american'")
	if req.option_type not in ["call", "put"]:
		raise HTTPException(status_code=400, detail="option_type must be 'call' or 'put'")
	
	model = get_model(req.option_style, req.option_type)
	
	# Create meshgrid for S and t
	S_values = np.linspace(req.S_min, req.S_max, req.n_S)
	t_values = np.linspace(req.t_min, req.t_max, req.n_t)
	S_mesh, t_mesh = np.meshgrid(S_values, t_values)
	
	# Flatten for batch processing
	S_flat = S_mesh.flatten()
	t_flat = t_mesh.flatten()
	r_flat = np.full_like(S_flat, req.r)
	sigma_flat = np.full_like(S_flat, req.volatility)
	K_flat = np.full_like(S_flat, req.K)
	
	# Create input tensor
	X = np.stack([t_flat, S_flat, r_flat, sigma_flat, K_flat], axis=1)
	X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)
	
	# Get prices and Greeks
	with torch.no_grad():
		prices = model(X_tensor).squeeze(1).cpu().numpy()
	
	# Calculate Greeks using autograd
	model.eval()
	X_tensor_grad = torch.tensor(X, dtype=torch.float32, requires_grad=True)
	V = model(X_tensor_grad)
	
	# Calculate gradients
	grads = torch.autograd.grad(V.sum(), X_tensor_grad, create_graph=True)[0]
	dV_dS = grads[:, 1]
	dV_dt = grads[:, 0]
	dV_dr = grads[:, 2]
	dV_dsigma = grads[:, 3]
	
	# Second derivative for gamma
	d2V_dS = torch.autograd.grad(dV_dS.sum(), X_tensor_grad, retain_graph=True)[0][:, 1]
	
	# Convert to numpy
	delta = dV_dS.detach().cpu().numpy()
	gamma = d2V_dS.detach().cpu().numpy()
	vega = dV_dsigma.detach().cpu().numpy()
	theta = (-dV_dt).detach().cpu().numpy()  # conventional sign
	rho = dV_dr.detach().cpu().numpy()
	
	# Create response
	result = []
	for i in range(len(S_flat)):
		result.append(Surface3DPoint(
			S=float(S_flat[i]),
			t=float(t_flat[i]),
			price=float(prices[i]),
			delta=float(delta[i]),
			gamma=float(gamma[i]),
			vega=float(vega[i]),
			theta=float(theta[i]),
			rho=float(rho[i])
		))
	
	return result
