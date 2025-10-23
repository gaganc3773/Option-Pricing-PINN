# PINN Option Pricing Research Platform - Deployment Guide

##  Quick Deployment Options

### Option 1: Railway (Recommended)
1. Push your code to GitHub
2. Go to [railway.app](https://railway.app)
3. Connect your GitHub account
4. Select your repository
5. Railway will auto-deploy both frontend and backend
6. Get your live URL from Railway dashboard

### Option 2: Streamlit Cloud + Railway
1. Deploy backend on Railway (see above)
2. Update API_BASE in streamlit_app.py to your Railway URL
3. Go to [share.streamlit.io](https://share.streamlit.io)
4. Connect GitHub and deploy frontend
5. Set main file: `frontend/streamlit_app.py`

### Option 3: Render
1. Push to GitHub
2. Go to [render.com](https://render.com)
3. Create two services:
   - Backend: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   - Frontend: `streamlit run frontend/streamlit_app.py --server.port $PORT --server.address 0.0.0.0`

##  Project Structure
```
Option_Pricing_Tool/
├── backend/
│   └── main.py          # FastAPI backend
├── frontend/
│   └── streamlit_app.py # Streamlit frontend
├── models/              # Trained PINN models
├── requirements.txt     # Python dependencies
├── Procfile            # For Railway/Heroku
├── railway.json        # Railway configuration
└── README.md
```

##  Environment Variables
The app auto-detects deployment environment:
- Railway: Uses RAILWAY_PUBLIC_DOMAIN
- Render: Uses RENDER_EXTERNAL_URL  
- Heroku: Uses HEROKU_APP_NAME
- Local: Uses localhost:8000

##  Features
- Physics-Informed Neural Networks for option pricing
- European & American options (calls & puts)
- Real-time Greeks calculation
- Interactive 3D surfaces
- Multi-currency support
- Professional research interface

##  Perfect for:
- Research presentations
- Academic demonstrations
- Internship portfolios
- Educational purposes
