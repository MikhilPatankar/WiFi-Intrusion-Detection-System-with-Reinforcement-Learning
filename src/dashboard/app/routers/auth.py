# src/dashboard/app/routers/auth.py

from fastapi import APIRouter, Depends, HTTPException, status, Request, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from typing import Optional
from ..services import auth_service # Use dashboard auth service
from ..core.config import settings
import logging
from datetime import timedelta

# Setup logger for this router
log = logging.getLogger(__name__)

router = APIRouter(tags=["Dashboard Authentication"])
templates = Jinja2Templates(directory=settings.TEMPLATES_DIR)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token", auto_error=False)

# --- Login Page Route ---
@router.get("/login", response_class=HTMLResponse, include_in_schema=False, name="login_page")
async def login_page(request: Request, error: Optional[str] = None):
    return templates.TemplateResponse("login.html", {"request": request, "error": error})

# --- Token Generation Route ---
@router.post("/token", include_in_schema=False, name="login_for_token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    log.info(f"Login attempt for username: '{form_data.username}'")
    user = auth_service.get_dashboard_user(form_data.username)
    password_match = user and auth_service.verify_password(form_data.password, user["hashed_password"])
    log.info(f"User found: {bool(user)}, Password match: {password_match}, Disabled: {user.get('disabled', True) if user else 'N/A'}")

    if not user or not password_match or user["disabled"]:
        log.warning(f"Login failed for '{form_data.username}'.")
        # Redirect back to login page with error query parameter
        login_url = router.url_path_for("login_page") + "?error=invalid" # url_for works for routes in *this* router
        return RedirectResponse(url=login_url, status_code=status.HTTP_303_SEE_OTHER)

    log.info(f"Login successful for user: '{user['username']}'")
    access_token_expires = timedelta(minutes=settings.AUTH_ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth_service.create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )

    # --- FIX: Hardcode the redirect URL to the monitoring page path ---
    redirect_url = "/monitoring" # The route defined in main.py
    # --- End FIX ---
    response = RedirectResponse(url=redirect_url, status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(
        key="dashboard_access_token",
        value=f"Bearer {access_token}",
        httponly=True,
        max_age=int(access_token_expires.total_seconds()),
    )
    return response

# --- Logout Route ---
@router.get("/logout", include_in_schema=False, name="logout_and_redirect")
async def logout_and_redirect(request: Request):
    log.info("Logout requested.")
    login_url = request.url_for("public_home") # url_for works here
    response = RedirectResponse(url=login_url, status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie(key="dashboard_access_token")
    return response

# --- Dependencies (get_current_dashboard_user, require_dashboard_user) ---
# (Keep these functions exactly as they were)
async def get_current_dashboard_user(request: Request) -> Optional[dict]:
    token = request.cookies.get("dashboard_access_token");
    if not token or not token.startswith("Bearer "): 
        return None
    token = token.split("Bearer ")[1]
    username = auth_service.decode_access_token(token)
    if not username: 
        return None
    user = auth_service.get_dashboard_user(username)
    return user if user and not user["disabled"] else None
async def require_dashboard_user(request: Request) -> dict:
    user = await get_current_dashboard_user(request)
    if user is None:
        logging.warning("Unauth access attempt -> redirect")
        login_url = f"{request.url_for('login_page')}?error=expired"
        raise HTTPException(status_code=status.HTTP_307_TEMPORARY_REDIRECT, headers={"Location": login_url})
    return user

