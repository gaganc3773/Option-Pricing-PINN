#!/usr/bin/env python3
"""
Local development script for PINN Option Pricing Application
This script helps run both backend and frontend locally for development.
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def run_backend():
    """Run the FastAPI backend server"""
    print("🚀 Starting Backend Server...")
    backend_dir = Path(__file__).parent / "backend"
    os.chdir(backend_dir)
    
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"
        ], check=True)
    except KeyboardInterrupt:
        print("\n⏹️  Backend server stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Backend server failed to start: {e}")

def run_frontend():
    """Run the Streamlit frontend"""
    print("🎨 Starting Frontend Server...")
    frontend_dir = Path(__file__).parent / "frontend"
    os.chdir(frontend_dir)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501", "--server.address", "0.0.0.0"
        ], check=True)
    except KeyboardInterrupt:
        print("\n⏹️  Frontend server stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Frontend server failed to start: {e}")

def install_dependencies():
    """Install dependencies for both backend and frontend"""
    print("📦 Installing dependencies...")
    
    # Install backend dependencies
    backend_dir = Path(__file__).parent / "backend"
    if backend_dir.exists():
        print("Installing backend dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      cwd=backend_dir, check=True)
    
    # Install frontend dependencies
    frontend_dir = Path(__file__).parent / "frontend"
    if frontend_dir.exists():
        print("Installing frontend dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      cwd=frontend_dir, check=True)
    
    print("✅ Dependencies installed successfully!")

def main():
    """Main function to handle command line arguments"""
    if len(sys.argv) < 2:
        print("""
PINN Option Pricing - Local Development Script

Usage:
    python run_local.py [command]

Commands:
    install    - Install dependencies for both backend and frontend
    backend    - Run only the backend server
    frontend   - Run only the frontend server
    both       - Run both backend and frontend (requires two terminals)

Examples:
    python run_local.py install
    python run_local.py backend
    python run_local.py frontend
        """)
        return
    
    command = sys.argv[1].lower()
    
    if command == "install":
        install_dependencies()
    elif command == "backend":
        run_backend()
    elif command == "frontend":
        run_frontend()
    elif command == "both":
        print("""
⚠️  To run both backend and frontend simultaneously, you need two terminals:

Terminal 1 (Backend):
    python run_local.py backend

Terminal 2 (Frontend):
    python run_local.py frontend

Or use the individual commands:
    python run_local.py backend
    python run_local.py frontend
        """)
    else:
        print(f"❌ Unknown command: {command}")
        print("Available commands: install, backend, frontend, both")

if __name__ == "__main__":
    main()
