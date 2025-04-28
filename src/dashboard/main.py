# src/dashboard/main.py

from fastapi.middleware.cors import CORSMiddleware
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
from app.routers import auth # Import auth router
from app.routers.auth import require_dashboard_user # Import the dependency

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



# Configure Jinja2 Templates
templates = Jinja2Templates(directory=settings.TEMPLATES_DIR)
logging.info(f"Dashboard Templates directory: {settings.TEMPLATES_DIR}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure static/templates directories exist
    if settings.STATIC_DIR and not os.path.exists(settings.STATIC_DIR): os.makedirs(settings.STATIC_DIR)
    if settings.TEMPLATES_DIR and not os.path.exists(settings.TEMPLATES_DIR): os.makedirs(settings.TEMPLATES_DIR)
    logging.info("Dashboard application startup complete.")
    yield
    logging.info("Dashboard application shutdown.")

# Create FastAPI app instance for the dashboard
app = FastAPI(
    title=settings.APP_NAME,
    description="Web Dashboard for WIDS Monitoring",
    lifespan=lifespan
)

# Mount Static Files (for custom.css)
if os.path.exists(settings.STATIC_DIR):
    app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")
    logging.info(f"Mounted dashboard static directory: {settings.STATIC_DIR}")

# Include Authentication Router (/login, /logout, /token)
app.include_router(auth.router)

origins = ["http://localhost", "http://localhost:8080", "http://127.0.0.1:8050", "null"]

app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
# --- Page Routes ---

@app.get("/", response_class=HTMLResponse, name="home", tags=["Pages"])
async def read_home_page(
    request: Request,
    current_user: dict = Depends(require_dashboard_user) # Protect this route
):
    """Serves the Home/Monitoring page."""
    template_name = "home.html"
    context = {
        "request": request,
        "user": current_user,
        "wids_api_base_url": settings.WIDS_API_BASE_URL,
        "active_page": "home" # Pass active page for navigation styling
    }
    logging.info(f"Serving home page for user: {current_user.get('username')}")
    template_path = os.path.join(settings.TEMPLATES_DIR, template_name)
    if not os.path.exists(template_path):
         logging.error(f"Home template not found: {template_path}")
         return HTMLResponse(content="500 Error: Home template missing.", status_code=500)
    return templates.TemplateResponse(template_name, context)

@app.get("/events", response_class=HTMLResponse, name="events", tags=["Pages"])
async def read_events_page(
    request: Request,
    current_user: dict = Depends(require_dashboard_user) # Protect this route
):
    """Serves the Events & Labeling page."""
    template_name = "events.html"
    context = {
        "request": request,
        "user": current_user,
        "wids_api_base_url": settings.WIDS_API_BASE_URL,
         "active_page": "events" # Pass active page for navigation styling
    }
    logging.info(f"Serving events page for user: {current_user.get('username')}")
    template_path = os.path.join(settings.TEMPLATES_DIR, template_name)
    if not os.path.exists(template_path):
         logging.error(f"Events template not found: {template_path}")
         return HTMLResponse(content="500 Error: Events template missing.", status_code=500)
    return templates.TemplateResponse(template_name, context)

# --- NEW Settings Page Route ---
@app.get("/settings", response_class=HTMLResponse, name="settings", tags=["Pages"])
async def read_settings_page(request: Request, current_user: dict = Depends(require_dashboard_user)):
    """Serves the Settings page."""
    context = {
        "request": request,
        "user": current_user,
        "settings": settings, # Pass the settings object
        "active_page": "settings"
    }
    logging.info(f"Serving settings page for user: {current_user.get('username')}")
    template_name = "settings.html"
    template_path = os.path.join(settings.TEMPLATES_DIR, template_name)
    if not os.path.exists(template_path):
         logging.error(f"Settings template not found: {template_path}")
         return HTMLResponse(content="500 Error: Settings template missing.", status_code=500)
    return templates.TemplateResponse(template_name, context)
# --- End NEW ---

# --- Health Check ---
@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "OK", "app": settings.APP_NAME}

# --- Main execution block ---
if __name__ == "__main__":
    logging.info(f"Starting Uvicorn server for {settings.APP_NAME}...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,      # Run dashboard on DIFFERENT port (e.g., 8050)
        reload=True,
        log_level="info"
    )
