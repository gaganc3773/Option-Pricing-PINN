import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm
from pyDOE import lhs
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import os

# Neural Network Architecture
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.hiddenlayers = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, X):
        return self.hiddenlayers(X)

# European Option Loss Function
def BSM_loss_european(model, X_f, X_T, X_lb, X_ub, T, S_max,
                     lambda_pde=1.0, lambda_T=10.0, lambda_bc=10.0,
                     option_type='call'):
    """
    PINN Loss for European Call or Put Options under BSM.
    """
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

    # PDE Residual: ∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0
    residual = V_t + 0.5 * sigma_f**2 * S_f**2 * V_SS + r_f * S_f * V_S - r_f * V_pred
    pde_loss = torch.mean(residual**2)

    # Terminal Condition
    S_T = X_T[:, 1:2]
    K_T = X_T[:, 4:5]

    if option_type == 'call':
        payoff = torch.clamp(S_T - K_T, min=0.0)
    elif option_type == 'put':
        payoff = torch.clamp(K_T - S_T, min=0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    V_T_pred = model(X_T)
    terminal_loss = torch.mean((V_T_pred - payoff)**2)

    # Boundary Conditions
    # Left boundary: S = 0
    V_lb_pred = model(X_lb)
    if option_type == 'call':
        left_bc_target = torch.zeros_like(V_lb_pred)
    else:  # put
        K_lb = X_lb[:, 4:5]
        r_lb = X_lb[:, 2:3]
        left_bc_target = K_lb * torch.exp(-r_lb * (T - X_lb[:, 0:1]))
    left_bc_loss = torch.mean((V_lb_pred - left_bc_target)**2)

    # Right boundary: S = S_max
    t_ub = X_ub[:, 0:1]
    S_ub = X_ub[:, 1:2]
    r_ub = X_ub[:, 2:3]
    K_ub = X_ub[:, 4:5]
    V_ub_pred = model(X_ub)

    if option_type == 'call':
        right_bc_target = S_ub - K_ub * torch.exp(-r_ub * (T - t_ub))
    else:  # put
        right_bc_target = torch.zeros_like(V_ub_pred)
    right_bc_loss = torch.mean((V_ub_pred - right_bc_target)**2)

    # Total Loss
    total_loss = (
        lambda_pde * pde_loss +
        lambda_T * terminal_loss +
        lambda_bc * (left_bc_loss + right_bc_loss)
    )

    return total_loss, pde_loss.item(), terminal_loss.item(), left_bc_loss.item(), right_bc_loss.item()

# American Option Loss Function
def BSM_loss_american(model, X_f, X_T, X_lb, X_ub, T, S_max,
                      lambda_pde=1.0, lambda_T=10.0, lambda_bc=10.0, lambda_ex=10.0,
                      option_type='put'):
    """
    PINN Loss for American Options (default: Put).
    Adds early exercise constraint: V >= payoff(S).
    """
    # PDE Residual (same as European)
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

    # Early exercise penalty
    K_f = X_f[:, 4:5]  # Use K from interior points, not terminal points
    payoff_all = torch.clamp(K_f - S_f, min=0.0) if option_type == 'put' else torch.clamp(S_f - K_f, min=0.0)
    exercise_loss = torch.mean(torch.relu(payoff_all - V_pred)**2)

    # Total Loss
    total_loss = (
        lambda_pde * pde_loss +
        lambda_T * terminal_loss +
        lambda_bc * (left_bc_loss + right_bc_loss) +
        lambda_ex * exercise_loss
    )

    return total_loss, pde_loss.item(), terminal_loss.item(), left_bc_loss.item(), right_bc_loss.item(), exercise_loss.item()

# Black-Scholes analytical solution for European options
def bs_price_european(S, K, T, t, r, sigma, option_type='call'):
    tau = T - t
    tau = np.maximum(tau, 1e-10)  # avoid zero division
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r*tau) * norm.cdf(d2)
    else:  # put
        price = K * np.exp(-r*tau) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    price[S <= 0] = 0.0  # handle S=0
    return price

# Training function
def train_model(model, loss_fn, X_f, X_T, X_lb, X_ub, T_max, S_max, 
                option_type, epochs=5000, print_every=500):
    """Train a PINN model"""
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    
    for epoch in range(1, epochs+1):
        optimizer.zero_grad()
        
        if 'american' in loss_fn.__name__:
            total_loss, pde_loss, terminal_loss, left_bc_loss, right_bc_loss, exercise_loss = loss_fn(
                model, X_f, X_T, X_lb, X_ub, T=T_max, S_max=S_max,
                lambda_pde=1.0, lambda_T=10.0, lambda_bc=10.0, lambda_ex=10.0,
                option_type=option_type
            )
            losses.append([epoch, total_loss.item(), pde_loss, terminal_loss, left_bc_loss, right_bc_loss, exercise_loss])
        else:
            total_loss, pde_loss, terminal_loss, left_bc_loss, right_bc_loss = loss_fn(
                model, X_f, X_T, X_lb, X_ub, T=T_max, S_max=S_max,
                lambda_pde=1.0, lambda_T=10.0, lambda_bc=10.0,
                option_type=option_type
            )
            losses.append([epoch, total_loss.item(), pde_loss, terminal_loss, left_bc_loss, right_bc_loss, 0.0])
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % print_every == 0 or epoch == 1:
            print(f"Epoch {epoch:04d} | Total Loss: {total_loss.item():.6f} | "
                  f"PDE: {pde_loss:.6f}, Terminal: {terminal_loss:.6f}, "
                  f"Left BC: {left_bc_loss:.6f}, Right BC: {right_bc_loss:.6f}")
    
    return losses

# Generate training data
def generate_training_data(S_min=0.0, S_max=200.0, T_min=0.0, T_max=1.0,
                         r_range=[0.01, 0.1], sigma_range=[0.1, 0.5], K_range=[50, 150],
                         N_f=10000, N_T=1000, N_lb=1000, N_ub=1000):
    """Generate training data for PINN"""
    
    # PDE interior points
    lhs_samples = lhs(5, N_f)
    t_f = T_min + (T_max - T_min) * lhs_samples[:, 0:1]
    S_f = S_min + (S_max - S_min) * lhs_samples[:, 1:2]
    r_f = r_range[0] + (r_range[1] - r_range[0]) * lhs_samples[:, 2:3]
    sigma_f = sigma_range[0] + (sigma_range[1] - sigma_range[0]) * lhs_samples[:, 3:4]
    K_f = K_range[0] + (K_range[1] - K_range[0]) * lhs_samples[:, 4:5]
    X_f = torch.tensor(np.hstack([t_f, S_f, r_f, sigma_f, K_f]), dtype=torch.float32)
    
    # Terminal points
    S_T = np.random.uniform(S_min, S_max, (N_T, 1))
    r_T = np.random.uniform(r_range[0], r_range[1], (N_T, 1))
    sigma_T = np.random.uniform(sigma_range[0], sigma_range[1], (N_T, 1))
    K_T = np.random.uniform(K_range[0], K_range[1], (N_T, 1))
    t_T = T_max * np.ones((N_T, 1))
    X_T = torch.tensor(np.hstack([t_T, S_T, r_T, sigma_T, K_T]), dtype=torch.float32)
    
    # Left boundary points
    t_lb = np.random.uniform(T_min, T_max, (N_lb, 1))
    S_lb = np.zeros((N_lb, 1))
    r_lb = np.random.uniform(r_range[0], r_range[1], (N_lb, 1))
    sigma_lb = np.random.uniform(sigma_range[0], sigma_range[1], (N_lb, 1))
    K_lb = np.random.uniform(K_range[0], K_range[1], (N_lb, 1))
    X_lb = torch.tensor(np.hstack([t_lb, S_lb, r_lb, sigma_lb, K_lb]), dtype=torch.float32)
    
    # Right boundary points
    t_ub = np.random.uniform(T_min, T_max, (N_ub, 1))
    S_ub = S_max * np.ones((N_ub, 1))
    r_ub = np.random.uniform(r_range[0], r_range[1], (N_ub, 1))
    sigma_ub = np.random.uniform(sigma_range[0], sigma_range[1], (N_ub, 1))
    K_ub = np.random.uniform(K_range[0], K_range[1], (N_ub, 1))
    X_ub = torch.tensor(np.hstack([t_ub, S_ub, r_ub, sigma_ub, K_ub]), dtype=torch.float32)
    
    return X_f, X_T, X_lb, X_ub

# Test function
def test_model(model, S_min=0.0, S_max=200.0, T_min=0.0, T_max=1.0,
              r_range=[0.01, 0.1], sigma_range=[0.1, 0.5], K_range=[50, 150],
              n_S_test=50, n_t_test=50, n_r_test=3, n_sigma_test=3, n_K_test=3,
              option_type='call'):
    """Test model performance"""
    
    S_test = np.linspace(S_min, S_max, n_S_test)
    t_test = np.linspace(T_min, T_max, n_t_test)
    r_test = np.linspace(r_range[0], r_range[1], n_r_test)
    sigma_test = np.linspace(sigma_range[0], sigma_range[1], n_sigma_test)
    K_test = np.linspace(K_range[0], K_range[1], n_K_test)
    
    # Create meshgrid
    mesh = np.array(np.meshgrid(t_test, S_test, r_test, sigma_test, K_test, indexing='ij'))
    X_test = mesh.reshape(5, -1).T
    X_test = torch.tensor(X_test, dtype=torch.float32)
    
    model.eval()
    with torch.no_grad():
        V_pred_test = model(X_test).cpu().numpy()
    
    X_test_np = X_test.numpy()
    S_test_vals = X_test_np[:, 1]
    t_test_vals = X_test_np[:, 0]
    r_test_vals = X_test_np[:, 2]
    sigma_test_vals = X_test_np[:, 3]
    K_test_vals = X_test_np[:, 4]
    
    V_true_test = bs_price_european(S_test_vals, K_test_vals, T_max, t_test_vals, r_test_vals, sigma_test_vals, option_type)
    
    rmse = np.sqrt(mean_squared_error(V_true_test, V_pred_test))
    r2 = r2_score(V_true_test, V_pred_test)
    
    return rmse, r2, S_test_vals, V_true_test, V_pred_test

def main():
    """Main training function"""
    print("Starting PINN training for European and American options...")
    
    # Generate training data
    print("Generating training data...")
    X_f, X_T, X_lb, X_ub = generate_training_data()
    
    # Training parameters
    S_min, S_max = 0.0, 200.0
    T_min, T_max = 0.0, 1.0
    epochs = 5000
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Train all combinations
    combinations = [
        ('european', 'call', BSM_loss_european),
        ('european', 'put', BSM_loss_european),
        ('american', 'call', BSM_loss_american),
        ('american', 'put', BSM_loss_american)
    ]
    
    best_models = {}
    convergence_data = {}
    
    for option_style, option_type, loss_fn in combinations:
        print(f"\n{'='*60}")
        print(f"Training {option_style.upper()} {option_type.upper()} option model...")
        print(f"{'='*60}")
        
        # Initialize model
        model = NN()
        
        # Train model
        losses = train_model(model, loss_fn, X_f, X_T, X_lb, X_ub, T_max, S_max, option_type, epochs)
        
        # Test model (only for European options as we have analytical solutions)
        if option_style == 'european':
            rmse, r2, S_test_vals, V_true_test, V_pred_test = test_model(
                model, option_type=option_type
            )
            print(f"\nTest Results:")
            print(f"RMSE: {rmse:.6f}")
            print(f"R² Score: {r2:.6f}")
        
        # Save model
        model_name = f"{option_style}_{option_type}"
        model_path = f"models/{model_name}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        # Store convergence data
        convergence_data[model_name] = losses
        
        # Store best model info
        best_models[model_name] = {
            'path': model_path,
            'final_loss': losses[-1][1],
            'rmse': rmse if option_style == 'european' else None,
            'r2': r2 if option_style == 'european' else None
        }
    
    # Save convergence data
    print(f"\n{'='*60}")
    print("Saving convergence data...")
    print(f"{'='*60}")
    
    for model_name, losses in convergence_data.items():
        df = pd.DataFrame(losses, columns=['epoch', 'total', 'pde', 'terminal', 'left_bc', 'right_bc', 'exercise'])
        df.to_csv(f"models/{model_name}_convergence.csv", index=False)
        print(f"Convergence data saved for {model_name}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    
    for model_name, info in best_models.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Final Loss: {info['final_loss']:.6f}")
        if info['rmse'] is not None:
            print(f"  RMSE: {info['rmse']:.6f}")
            print(f"  R² Score: {info['r2']:.6f}")
        print(f"  Model Path: {info['path']}")
    
    print(f"\n{'='*60}")
    print("Training completed successfully!")
    print("All models saved in 'models/' directory")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
