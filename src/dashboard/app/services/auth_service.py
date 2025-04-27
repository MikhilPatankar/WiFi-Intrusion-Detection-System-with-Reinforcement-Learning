# --- File: src/dashboard/app/services/auth_service.py ---

from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt # Using python-jose
from pydantic import BaseModel, ValidationError
from ..core.config import settings # Import dashboard settings

# --- Password Hashing ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

# --- JWT Token Handling ---
class TokenData(BaseModel):
    username: Optional[str] = None

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.AUTH_ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.AUTH_SECRET_KEY, algorithm=settings.AUTH_ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, settings.AUTH_SECRET_KEY, algorithms=[settings.AUTH_ALGORITHM])
        username: Optional[str] = payload.get("sub")
        if username is None:
            return None
        # Optional: Validate payload structure with TokenData model
        # token_data = TokenData(**payload)
        return username
    except (JWTError, ValidationError):
        # Catches decode errors, expired tokens, validation errors
        return None

# --- Dummy User Database (Dashboard Admin) ---
# Replace with a real user source if needed
DASHBOARD_USERS_DB = {
    "dashadmin": {
        "username": "dashadmin",
        "full_name": "Dashboard Admin",
        # Hash for "dashboardpassword"
        "hashed_password": "$2b$12$FsqcccgbiRjcHhR2KxJxbuiy3F8p6pLVer2oUvSjVwnQKepTB2m2u",
        "disabled": False,
    }
}

def get_dashboard_user(username: str) -> Optional[dict]:
    """Retrieves dashboard user data."""
    if username in DASHBOARD_USERS_DB:
        return DASHBOARD_USERS_DB[username]
    return None
