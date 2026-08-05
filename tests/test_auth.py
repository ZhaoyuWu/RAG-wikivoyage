"""Unit tests for auth (PBKDF2 + JWT) and the sliding-window rate limiter."""

import os
import time

# 32+ bytes so PyJWT does not warn about a short HMAC key (RFC 7518 §3.2).
os.environ.setdefault("JWT_SECRET", "test-secret-for-unit-tests-0123456789")

from src import ratelimit
from src.auth import (
    create_token,
    decode_token,
    hash_password,
    verify_password,
)


def test_password_roundtrip():
    stored = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", stored)
    assert not verify_password("wrong password", stored)


def test_hash_is_salted():
    assert hash_password("same") != hash_password("same")


def test_verify_rejects_garbage_hash():
    assert not verify_password("anything", "not-a-valid-hash")


def test_token_roundtrip():
    token = create_token("alice", "admin")
    ident = decode_token(token)
    assert ident == {"user": "alice", "role": "admin"}


def test_tampered_token_rejected():
    token = create_token("alice", "admin")
    assert decode_token(token + "x") is None
    assert decode_token("") is None


def test_expired_token_rejected(monkeypatch):
    import src.auth as auth

    monkeypatch.setattr(auth, "JWT_TTL_HOURS", -1)
    token = auth.create_token("alice", "admin")
    assert auth.decode_token(token) is None


def test_ratelimit_blocks_then_frees():
    key = f"test:{time.monotonic()}"
    for _ in range(3):
        assert ratelimit.check(key, 3, window_s=0.3) is None
    retry = ratelimit.check(key, 3, window_s=0.3)
    assert retry is not None and retry > 0
    time.sleep(0.35)
    assert ratelimit.check(key, 3, window_s=0.3) is None


def test_ratelimit_keys_are_isolated():
    """Login throttling must be per-IP: one exhausted key must not block a
    different one, or one attacker could lock out every user (DoS)."""
    a, b = f"login:{time.monotonic()}:1.1.1.1", f"login:{time.monotonic()}:2.2.2.2"
    for _ in range(10):
        ratelimit.check(a, 10, window_s=5)
    assert ratelimit.check(a, 10, window_s=5) is not None   # attacker blocked
    assert ratelimit.check(b, 10, window_s=5) is None       # victim unaffected
