from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from src.config import SUPPORT_API_KEY


api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


def require_api_key(api_key: str = Security(api_key_header)) -> None:
    """
    Enforces API key auth only when SUPPORT_API_KEY is configured.

    This keeps local development and tests simple while allowing production
    deployments to turn on a basic security boundary through environment config.
    """
    if not SUPPORT_API_KEY:
        return

    if api_key != SUPPORT_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
