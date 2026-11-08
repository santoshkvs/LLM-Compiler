"""Sign and verify tables. HMAC-SHA256 is built in; Ed25519 needs ``cryptography``."""

from __future__ import annotations

import hashlib
import hmac
from typing import Any, Dict, Optional, Tuple, Union

from .errors import SignatureError

HMAC = "hmac-sha256"
ED25519 = "ed25519"

KeyLike = Union[str, bytes]


def _message(content_hash: str) -> bytes:
    return b"decisionforge-table:" + content_hash.encode("ascii")


def _key_bytes(key: KeyLike) -> bytes:
    return key if isinstance(key, bytes) else key.encode("utf-8")


def _require_crypto():
    try:
        from cryptography.hazmat.primitives.asymmetric import ed25519  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise SignatureError("Ed25519 needs the 'cryptography' package: pip install decisionforge[crypto]") from exc


def generate_ed25519_keypair() -> Tuple[str, str]:
    """Return ``(private_key_hex, public_key_hex)``."""
    _require_crypto()
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519

    priv = ed25519.Ed25519PrivateKey.generate()
    priv_raw = priv.private_bytes(
        serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption()
    )
    pub_raw = priv.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return priv_raw.hex(), pub_raw.hex()


def sign_table(table, key: KeyLike, *, alg: str = HMAC, key_id: Optional[str] = None) -> Dict[str, Any]:
    """Sign ``table.content_hash()``; stores and returns the signature dict.

    For ``hmac-sha256`` ``key`` is a shared secret. For ``ed25519`` it is the
    private key as 64 hex characters.
    """
    msg = _message(table.content_hash())
    if alg == HMAC:
        value = hmac.new(_key_bytes(key), msg, hashlib.sha256).hexdigest()
    elif alg == ED25519:
        _require_crypto()
        from cryptography.hazmat.primitives.asymmetric import ed25519

        priv = ed25519.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(key if isinstance(key, str) else key.decode()))
        value = priv.sign(msg).hex()
    else:
        raise SignatureError(f"unknown signature algorithm {alg!r}")
    sig = {"alg": alg, "key_id": key_id, "value": value}
    table.signature = sig
    return sig


def verify_table_signature(table, key: KeyLike) -> bool:
    """True iff ``table.signature`` is valid for the table's current content."""
    sig = table.signature
    if not sig:
        return False
    msg = _message(table.content_hash())
    alg = sig.get("alg")
    try:
        if alg == HMAC:
            expected = hmac.new(_key_bytes(key), msg, hashlib.sha256).hexdigest()
            return hmac.compare_digest(expected, sig.get("value", ""))
        if alg == ED25519:
            _require_crypto()
            from cryptography.exceptions import InvalidSignature
            from cryptography.hazmat.primitives.asymmetric import ed25519

            pub = ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(key if isinstance(key, str) else key.decode()))
            try:
                pub.verify(bytes.fromhex(sig["value"]), msg)
                return True
            except InvalidSignature:
                return False
    except (ValueError, KeyError):
        return False
    raise SignatureError(f"unknown signature algorithm {alg!r}")
