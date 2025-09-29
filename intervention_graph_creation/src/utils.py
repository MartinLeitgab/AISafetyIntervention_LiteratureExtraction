from hashlib import sha1


def short_id(s: str) -> str:
    """Deterministic 10-char hex ID (lowercase)."""
    norm = " ".join(s.strip().split()).lower()
    return sha1(norm.encode("utf-8")).hexdigest()[:10]
