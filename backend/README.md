# PINN Option Pricing Backend API

This is the backend API for the PINN (Physics-Informed Neural Networks) Option Pricing application, designed to be deployed on Railway.

## Features

- FastAPI-based REST API
- PINN models for European and American options (calls and puts)
- Real-time option pricing and Greeks calculation
- 2D and 3D surface generation
- Training convergence analysis
- Batch simulation capabilities

## API Endpoints

- `POST /price` - Calculate option price
- `POST /greeks` - Calculate option Greeks
- `POST /surface` - Generate 2D pricing surface
- `POST /surface3d` - Generate 3D pricing surface with Greeks
- `GET /convergence` - Get training convergence data
- `POST /simulate` - Run batch PINN simulations
- `GET /simulation/{id}` - Get simulation status and results

## Deployment

This backend is configured for Railway deployment with:
- `Procfile` for web process
- `railway.json` for build configuration
- `requirements.txt` for Python dependencies

## Model Files

The backend includes pre-trained PINN models:
- `models/european_call.pt` - European call option model
- `models/european_put.pt` - European put option model
- `models/american_call.pt` - American call option model
- `models/american_put.pt` - American put option model
- Corresponding convergence data in CSV format

## Usage

The API expects JSON requests with option parameters (S, K, r, volatility, t, T, option_style, option_type) and returns structured responses with pricing data and Greeks.
