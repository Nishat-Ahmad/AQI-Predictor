import os
import sys

# Add project root to sys.path for Vercel serverless environment
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

from dotenv import load_dotenv
load_dotenv()

from backend.main import app

# Expose FastAPI application instance for Vercel ASGI handler
app = app
