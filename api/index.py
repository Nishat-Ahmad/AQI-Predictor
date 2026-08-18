import os
import sys

# Add project root to sys.path for Vercel serverless environment
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.main import app

# Expose FastAPI application instance for Vercel ASGI handler
app = app
