"""
ECG-RAMBA Security Module
=========================
JWT Authentication and Password Hashing utilities.
"""

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

from app.core.config import settings


# =============================================================================
# Password Hashing - Using bcrypt directly (passlib has issues with password length)
# =============================================================================
import bcrypt

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a hashed password."""
    # Encode and truncate to 72 bytes (bcrypt limit)
    password_bytes = plain_password.encode('utf-8')[:72]
    hashed_bytes = hashed_password.encode('utf-8')
    return bcrypt.checkpw(password_bytes, hashed_bytes)


def get_password_hash(password: str) -> str:
    """Hash a password using bcrypt."""
    # Encode and truncate to 72 bytes (bcrypt limit)
    password_bytes = password.encode('utf-8')[:72]
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')


# =============================================================================
# JWT Token Models
# =============================================================================
class Token(BaseModel):
    """OAuth2 token response model."""
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Data extracted from JWT token."""
    username: Optional[str] = None
    user_id: Optional[int] = None
    role: Optional[str] = None


# =============================================================================
# JWT Token Utilities
# =============================================================================
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Dictionary with claims to encode
        expires_delta: Optional custom expiration time
        
    Returns:
        Encoded JWT string
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    
    return encoded_jwt


def decode_token(token: str) -> Optional[TokenData]:
    """
    Decode and validate a JWT token.
    
    Args:
        token: JWT string to decode
        
    Returns:
        TokenData if valid, None otherwise
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        user_id: int = payload.get("user_id")
        role: str = payload.get("role", "user")
        
        if username is None:
            return None
            
        return TokenData(username=username, user_id=user_id, role=role)
        
    except JWTError:
        return None


# =============================================================================
# FastAPI OAuth2 Setup
# =============================================================================
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token", auto_error=False)


async def get_current_user(token: Optional[str] = Depends(oauth2_scheme)) -> Optional[TokenData]:
    """
    Dependency to get current user from JWT token.
    Returns None if no token or invalid token (for optional auth).
    
    Usage:
        @router.get("/protected")
        async def protected_route(user: TokenData = Depends(get_current_user)):
            if user is None:
                # Anonymous access
            else:
                # Authenticated access
    """
    if token is None:
        return None
    
    return decode_token(token)


async def require_auth(token: str = Depends(oauth2_scheme)) -> TokenData:
    """
    Dependency that REQUIRES valid authentication.
    Raises 401 if no token or invalid token.
    
    Usage:
        @router.get("/protected")
        async def protected_route(user: TokenData = Depends(require_auth)):
            # Only authenticated users reach here
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if token is None:
        raise credentials_exception
    
    token_data = decode_token(token)
    if token_data is None:
        raise credentials_exception
    
    return token_data


async def require_admin(user: TokenData = Depends(require_auth)) -> TokenData:
    """
    Dependency that requires admin role.
    
    Usage:
        @router.delete("/admin-only")
        async def admin_route(user: TokenData = Depends(require_admin)):
            # Only admins reach here
    """
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return user
