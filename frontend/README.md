# PINN Option Pricing Frontend

This is the Streamlit frontend for the PINN (Physics-Informed Neural Networks) Option Pricing application.

## Features

- Interactive web interface for option pricing
- Real-time PINN model predictions
- Greeks calculation and visualization
- 2D and 3D surface plots
- Training convergence analysis
- Batch simulation capabilities
- Export functionality (CSV, Excel, JSON)
- Multi-currency support
- Professional dashboard design

## Pages

The application includes multiple tabs:
- **Price**: Single option pricing with Black-Scholes comparison
- **Greeks**: Risk sensitivities calculation and visualization
- **2D Surface**: Pricing curves vs underlying price
- **3D Surface**: Interactive 3D surfaces for prices and Greeks
- **Convergence**: Training loss convergence analysis
- **Analysis & Export**: Batch simulations and data export

## Configuration

The frontend automatically detects the backend API URL based on environment:
- Railway deployment: Uses `RAILWAY_PUBLIC_DOMAIN`
- Render deployment: Uses `RENDER_EXTERNAL_URL`
- Heroku deployment: Uses `HEROKU_APP_NAME`
- Local development: Uses `http://localhost:8000`

## Deployment

This frontend is configured for Streamlit Cloud deployment with:
- `requirements.txt` for Python dependencies
- `.streamlit/config.toml` for Streamlit configuration
- Auto-detection of backend API endpoints

## Usage

1. Configure option parameters in the sidebar
2. Select the appropriate tab for your analysis
3. Click the action buttons to generate results
4. Export data in various formats as needed

The frontend communicates with the backend API to provide real-time option pricing and analysis capabilities.
