# src/dashboard/main.py

from fastapi import FastAPI, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from contextlib import asynccontextmanager
import logging
import uvicorn
import os

# Import dashboard specific config, routers, auth dependency
from app.core.config import settings
from app.routers import auth
from app.routers.auth import require_dashboard_user # Import the dependency

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure Jinja2 Templates for the dashboard app
templates = Jinja2Templates(directory=settings.TEMPLATES_DIR)
logging.info(f"Dashboard Templates directory: {settings.TEMPLATES_DIR}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure static/templates directories exist for dashboard app
    if settings.STATIC_DIR and not os.path.exists(settings.STATIC_DIR):
        os.makedirs(settings.STATIC_DIR)
        logging.info(f"Created static directory: {settings.STATIC_DIR}")
    if settings.TEMPLATES_DIR and not os.path.exists(settings.TEMPLATES_DIR):
        os.makedirs(settings.TEMPLATES_DIR)
        logging.info(f"Created templates directory: {settings.TEMPLATES_DIR}")

    # No model loading or DB setup needed for the dashboard app itself
    logging.info("Dashboard application startup complete.")
    yield
    logging.info("Dashboard application shutdown.")

# Create FastAPI app instance for the dashboard
app = FastAPI(
    title=settings.APP_NAME,
    description="Web Dashboard for WIDS Monitoring",
    lifespan=lifespan
)

# Mount Static Files (optional, for custom CSS/JS)
if os.path.exists(settings.STATIC_DIR):
    app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")
    logging.info(f"Mounted dashboard static directory: {settings.STATIC_DIR}")

# Include Authentication Router (/login, /logout, /token)
app.include_router(auth.router)

# --- Protected Dashboard Route ---
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_dashboard_ui(
    request: Request,
    current_user: dict = Depends(require_dashboard_user) # Protect this route
):
    """Serves the main dashboard HTML, requires dashboard authentication."""
    template_name = "dashboard.html"
    # Pass user and backend API URL to the template context
    context = {
        "request": request,
        "user": current_user,
        "wids_api_base_url": settings.WIDS_API_BASE_URL # Make API URL available to JS
    }
    logging.info(f"Serving dashboard UI for user: {current_user.get('username')}")
    template_path = os.path.join(settings.TEMPLATES_DIR, template_name)
    if not os.path.exists(template_path):
         logging.error(f"Dashboard template not found: {template_path}")
         return HTMLResponse(content="500 Error: Dashboard template missing.", status_code=500)
    return templates.TemplateResponse(template_name, context)

# --- Root endpoint (optional health check) ---
# This is separate from the UI route above
@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "OK", "app": settings.APP_NAME}

# --- Main execution block ---
if __name__ == "__main__":
    logging.info(f"Starting Uvicorn server for {settings.APP_NAME}...")
    # Note: Run this from the 'src/dashboard' directory or adjust Python path
    uvicorn.run(
        "main:app", # Reference the app instance within this file
        host="0.0.0.0",
        port=8050,      # Run dashboard on a DIFFERENT port than the backend API
        reload=True,
        log_level="info"
    )

