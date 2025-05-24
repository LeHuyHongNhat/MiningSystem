import subprocess
import sys
import os
import time
import webbrowser
from threading import Thread

def run_backend():
    """Run the backend server"""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # Explicitly set port 5000 for backend
    subprocess.run([sys.executable, "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "5000"])

def run_frontend():
    """Run the frontend server"""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([sys.executable, "-m", "frontend.app"])

def open_browser():
    """Open the browser after a short delay"""
    time.sleep(2)  # Wait for servers to start
    webbrowser.open('http://localhost:8050')

if __name__ == "__main__":
    print("Starting backend server on port 5000...")
    # Start backend server in a separate thread
    backend_thread = Thread(target=run_backend)
    backend_thread.daemon = True
    backend_thread.start()
    
    print("Starting frontend server on port 8050...")
    # Start frontend server in a separate thread
    frontend_thread = Thread(target=run_frontend)
    frontend_thread.daemon = True
    frontend_thread.start()
    
    print("Opening browser...")
    # Open browser
    open_browser()
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        sys.exit(0) 