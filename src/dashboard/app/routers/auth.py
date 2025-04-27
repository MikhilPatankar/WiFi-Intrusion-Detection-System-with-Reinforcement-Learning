
# --- File: src/dashboard/app/routers/auth.py ---

from fastapi import APIRouter, Depends, HTTPException, status, Request, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from typing import Optional
from ..services import auth_service # Use dashboard auth service
from ..core.config import settings
import logging
from datetime import timedelta

router = APIRouter(tags=["Dashboard Authentication"])
templates = Jinja2Templates(directory=settings.TEMPLATES_DIR)

# --- Login Page Route ---
@router.get("/login", response_class=HTMLResponse, include_in_schema=False)
async def login_page(request: Request, error: Optional[str] = None):
    """Serves the HTML login page for the dashboard."""
    return templates.TemplateResponse("login.html", {"request": request, "error": error})

# --- Token Generation Route ---
@router.post("/token", include_in_schema=False)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Processes dashboard login form, returns JWT via cookie."""
    user = auth_service.get_dashboard_user(form_data.username) # Use dashboard user check
    if not user or not auth_service.verify_password(form_data.password, user["hashed_password"]) or user["disabled"]:
        return RedirectResponse(url="/login?error=invalid", status_code=status.HTTP_303_SEE_OTHER)

    access_token_expires = timedelta(minutes=settings.AUTH_ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth_service.create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )

    response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER) # Redirect to dashboard root
    response.set_cookie(
        key="dashboard_access_token", # Use a distinct cookie name
        value=f"Bearer {access_token}",
        httponly=True,
        max_age=int(access_token_expires.total_seconds()),
        # secure=True, # Enable in production with HTTPS
        # samesite="lax"
    )
    return response

# --- Logout Route ---
@router.get("/logout", include_in_schema=False)
async def logout_and_redirect():
    """Clears the dashboard access token cookie."""
    response = RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie(key="dashboard_access_token")
    return response

# --- Dependency to get current dashboard user ---
async def get_current_dashboard_user(request: Request) -> Optional[dict]:
    """Verifies dashboard token from cookie, returns user or None."""
    token = request.cookies.get("dashboard_access_token") # Check specific cookie
    if not token or not token.startswith("Bearer "):
        return None
    token = token.split("Bearer ")[1]

    username = auth_service.decode_access_token(token)
    if not username:
        return None

    user = auth_service.get_dashboard_user(username)
    if not user or user["disabled"]:
        return None
    return user

async def require_dashboard_user(request: Request) -> dict:
    """Dependency that requires a valid dashboard user, redirects if none."""
    user = await get_current_dashboard_user(request)
    if user is None:
        logging.warning("Unauthenticated dashboard access attempt -> redirect")
        # Redirect using HTTPException
        raise HTTPException(
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            headers={"Location": "/login?error=expired"},
        )
    return user
