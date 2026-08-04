"""Authentication: JWT bearer tokens over PBKDF2 password hashes.

Users live in the APP_USERS env var (name:salt$hash:role, comma-separated)
so the whole mechanism works without a database. This is the single-node
stand-in for a company SSO: the API contract (Bearer token, role claim)
is the same, only the identity provider is simplified.

CLI:
    python -m src.auth <password>   # print a salt$hash for APP_USERS
"""

import hashlib
import hmac
import secrets
import time

from .config import APP_USERS, JWT_SECRET, JWT_TTL_HOURS

_ITERATIONS = 200_000


def hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt), _ITERATIONS
    )
    return f"{salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    salt, _, _ = stored.partition("$")
    try:
        candidate = hash_password(password, salt)
    except ValueError:
        return False
    return hmac.compare_digest(candidate, stored)


def authenticate(username: str, password: str) -> dict | None:
    """Return {user, role} on valid credentials, else None."""
    entry = APP_USERS.get(username)
    if not entry or not verify_password(password, entry["hash"]):
        return None
    return {"user": username, "role": entry["role"]}


def create_token(user: str, role: str) -> str:
    import jwt

    if not JWT_SECRET:
        raise RuntimeError("JWT_SECRET is not set; cannot issue tokens.")
    payload = {"sub": user, "role": role,
               "exp": int(time.time()) + JWT_TTL_HOURS * 3600}
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def decode_token(token: str) -> dict | None:
    """Return {user, role} for a valid unexpired token, else None."""
    import jwt

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        return None
    return {"user": payload["sub"], "role": payload.get("role", "guest")}


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python -m src.auth <password>")
        sys.exit(1)
    print(hash_password(sys.argv[1]))
