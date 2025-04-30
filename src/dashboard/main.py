# src/dashboard/main.py

from fastapi import FastAPI, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from contextlib import asynccontextmanager
import logging
import uvicorn
import os
from typing import Optional # Import Optional

# Import dashboard specific config, routers, auth dependency
from app.core.config import settings
from app.routers import auth
# Import BOTH auth dependencies
from app.routers.auth import require_dashboard_user, get_current_dashboard_user

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

# Create FastAPI app instance
app = FastAPI(title=settings.APP_NAME, description="Web Dashboard for WIDS Monitoring", lifespan=lifespan)

# Mount Static Files
if os.path.exists(settings.STATIC_DIR):
    app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")
    logging.info(f"Mounted dashboard static directory: {settings.STATIC_DIR}")

# Include Authentication Router (Handles /login, /logout, /token)
app.include_router(auth.router)

# --- Page Routes ---

# --- Public Home Page ---
@app.get("/", response_class=HTMLResponse, name="public_home", tags=["Pages"])
async def read_public_home_page(
    request: Request,
    # Use the optional dependency to check if user is logged in
    current_user: Optional[dict] = Depends(get_current_dashboard_user)
):
    """Serves the public landing page."""
    context = {
        "request": request,
        "user": current_user, # Will be None if not logged in
        "wids_api_base_url": settings.WIDS_API_BASE_URL,
        "active_page": "public_home"
    }
    return templates.TemplateResponse("public_home.html", context)

# --- Admin Monitoring Page (Admin Home) ---
@app.get("/monitoring", response_class=HTMLResponse, name="monitoring", tags=["Pages"])
async def read_monitoring_page(
    request: Request,
    current_user: dict = Depends(require_dashboard_user) # Require login
):
    """Serves the main Admin Monitoring & Feedback page."""
    context = {"request": request, "user": current_user, "wids_api_base_url": settings.WIDS_API_BASE_URL, "active_page": "monitoring", "settings": settings}
    return templates.TemplateResponse("monitoring.html", context)

# --- Other Admin Pages ---
@app.get("/events", response_class=HTMLResponse, name="events", tags=["Pages"])
async def read_events_log_page(request: Request, current_user: dict = Depends(require_dashboard_user)): # Require login
    """Serves the Events Log page (all historical events)."""
    context = {"request": request, "user": current_user, "wids_api_base_url": settings.WIDS_API_BASE_URL, "active_page": "events"}
    return templates.TemplateResponse("events_log.html", context)

@app.get("/classification", response_class=HTMLResponse, name="classification", tags=["Pages"])
async def read_classification_mgmt_page(request: Request, current_user: dict = Depends(require_dashboard_user)): # Require login
    """Serves the Attack Classification Management page."""
    context = {"request": request, "user": current_user, "wids_api_base_url": settings.WIDS_API_BASE_URL, "active_page": "classification"}
    return templates.TemplateResponse("classification_mgmt.html", context)

@app.get("/mitigation", response_class=HTMLResponse, name="mitigation", tags=["Pages"])
async def read_mitigation_page(request: Request, current_user: dict = Depends(require_dashboard_user)): # Require login
    """Serves the Mitigation placeholder page."""
    context = {"request": request, "user": current_user, "wids_api_base_url": settings.WIDS_API_BASE_URL, "active_page": "mitigation"}
    return templates.TemplateResponse("mitigation.html", context)

@app.get("/settings", response_class=HTMLResponse, name="settings", tags=["Pages"])
async def read_settings_page(request: Request, current_user: dict = Depends(require_dashboard_user)): # Require login
    """Serves the Settings page."""
    context = {"request": request, "user": current_user, "settings": settings, "active_page": "settings"}
    return templates.TemplateResponse("settings.html", context)

# --- Health Check ---
@app.get("/health", tags=["Health"])
async def health_check(): return {"status": "OK", "app": settings.APP_NAME}

# --- Main execution block ---
if __name__ == "__main__":
    logging.info(f"Starting Uvicorn server for {settings.APP_NAME}...")
    uvicorn.run( "main:app", host="0.0.0.0", port=8050, reload=True, log_level="info")

