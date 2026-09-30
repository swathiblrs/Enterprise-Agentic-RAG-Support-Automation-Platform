import base64
import hashlib
import hmac
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from src.config import (
    AUTH_DEMO_ADMIN_PASSWORD,
    AUTH_DEMO_ADMIN_USERNAME,
    AUTH_DEMO_AGENT_PASSWORD,
    AUTH_DEMO_AGENT_USERNAME,
    JWT_EXPIRATION_MINUTES,
    JWT_SECRET_KEY,
    SUPPORT_API_KEY,
)


api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)

ROLE_VIEWER = "viewer"
ROLE_SUPPORT_AGENT = "support_agent"
ROLE_ADMIN = "admin"


DEMO_USERS = {
    AUTH_DEMO_ADMIN_USERNAME: {
        "password": AUTH_DEMO_ADMIN_PASSWORD,
        "role": ROLE_ADMIN,
    },
    AUTH_DEMO_AGENT_USERNAME: {
        "password": AUTH_DEMO_AGENT_PASSWORD,
        "role": ROLE_SUPPORT_AGENT,
    },
}


ROLE_HIERARCHY = {
    ROLE_VIEWER: 1,
    ROLE_SUPPORT_AGENT: 2,
    ROLE_ADMIN: 3,
}


def authenticate_user(username: str, password: str) -> Optional[Dict]:
    user = DEMO_USERS.get(username)

    if not user or not hmac.compare_digest(user["password"], password):
        return None

    return {
        "sub": username,
        "role": user["role"],
    }


def create_access_token(subject: str, role: str) -> str:
    if not JWT_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="JWT authentication is not configured.",
        )

    expires_at = datetime.utcnow() + timedelta(minutes=JWT_EXPIRATION_MINUTES)
    payload = {
        "sub": subject,
        "role": role,
        "exp": int(expires_at.timestamp()),
    }

    return encode_jwt(payload)


def encode_jwt(payload: Dict) -> str:
    header = {
        "alg": "HS256",
        "typ": "JWT",
    }

    header_segment = base64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_segment = base64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_segment}.{payload_segment}".encode("utf-8")
    signature = hmac.new(
        JWT_SECRET_KEY.encode("utf-8"),
        signing_input,
        hashlib.sha256,
    ).digest()

    return f"{header_segment}.{payload_segment}.{base64url_encode(signature)}"


def decode_jwt(token: str) -> Dict:
    try:
        header_segment, payload_segment, signature_segment = token.split(".")
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token.",
        ) from error

    signing_input = f"{header_segment}.{payload_segment}".encode("utf-8")
    expected_signature = hmac.new(
        JWT_SECRET_KEY.encode("utf-8"),
        signing_input,
        hashlib.sha256,
    ).digest()

    actual_signature = base64url_decode(signature_segment)

    if not hmac.compare_digest(actual_signature, expected_signature):
        raise_auth_error("Invalid bearer token signature.")

    payload = json.loads(base64url_decode(payload_segment))

    if payload.get("exp", 0) < int(datetime.utcnow().timestamp()):
        raise_auth_error("Bearer token has expired.")

    return payload


def base64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("utf-8")


def base64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def require_auth(
    allowed_roles: Optional[List[str]] = None,
):
    def dependency(
        api_key: Optional[str] = Security(api_key_header),
        credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    ) -> Dict:
        if not SUPPORT_API_KEY and not JWT_SECRET_KEY:
            return {
                "sub": "local-dev",
                "role": ROLE_ADMIN,
                "auth_type": "disabled",
            }

        if SUPPORT_API_KEY and api_key == SUPPORT_API_KEY:
            return {
                "sub": "api-key-client",
                "role": ROLE_ADMIN,
                "auth_type": "api_key",
            }

        if JWT_SECRET_KEY and credentials:
            claims = decode_jwt(credentials.credentials)
            ensure_role_allowed(claims.get("role", ROLE_VIEWER), allowed_roles)
            return {
                **claims,
                "auth_type": "jwt",
            }

        raise_auth_error("Invalid or missing credentials.")

    return dependency


def ensure_role_allowed(actual_role: str, allowed_roles: Optional[List[str]]) -> None:
    if not allowed_roles:
        return

    actual_level = ROLE_HIERARCHY.get(actual_role, 0)
    required_level = min(ROLE_HIERARCHY.get(role, 99) for role in allowed_roles)

    if actual_level < required_level:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient role permissions.",
        )


def raise_auth_error(detail: str) -> None:
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
    )
