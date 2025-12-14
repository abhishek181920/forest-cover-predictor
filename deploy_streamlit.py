#!/usr/bin/env python3
"""
Deployment script for the Forest Cover Type Prediction Streamlit App
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False

def run_streamlit_app():
    """Run the Streamlit application"""
    print("Starting Streamlit application...")
    try:
        # Change to the project directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        # Run the Streamlit app
        subprocess.check_call([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit app: {e}")
        return False
    except KeyboardInterrupt:
        print("Streamlit app stopped by user.")
        return True

def main():
    """Main deployment function"""
    print("=" * 50)
    print("Forest Cover Type Prediction - Streamlit Deployment")
    print("=" * 50)
    
    # Check if Streamlit is installed
    try:
        import streamlit
        print(f"Streamlit version: {streamlit.__version__}")
    except ImportError:
        print("Streamlit not found. Installing requirements...")
        if not install_requirements():
            print("Failed to install requirements. Exiting.")
            return False
    
    # Run the Streamlit app
    return run_streamlit_app()

if __name__ == "__main__":
    success = main()
    if success:
        print("Deployment completed successfully!")
    else:
        print("Deployment failed!")
        sys.exit(1)