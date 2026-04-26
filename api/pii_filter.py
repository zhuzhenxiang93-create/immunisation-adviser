"""
pii_filter.py — Detect and redact PII from user queries.

Patterns covered (NZ context):
  - NHI number      : 3 letters + 4 digits (e.g. ABC1234) or new format (3L+2D+2L+1D)
  - NZ phone number : 02x xxxxxxx, 0800 xxx xxx, +64 ...
  - Email address   : standard email pattern
  - Date of birth   : DD/MM/YYYY, DD-MM-YYYY, born on ...
  - Full name hint  : "patient [Name]", "caller [Name]" — flagged but not redacted
"""

from __future__ import annotations

import re

_PII_PATTERNS: list[tuple[str, str, str]] = [
    # (name, regex, replacement)
    ("NHI",       r"\b[A-Za-z]{3}\d{4}\b",                          "[NHI REDACTED]"),
    ("NHI_new",   r"\b[A-Za-z]{3}\d{2}[A-Za-z]{2}\d\b",            "[NHI REDACTED]"),
    ("email",     r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+","[EMAIL REDACTED]"),
    ("nz_phone",  r"(\+64|0)[2-9][\d\s\-]{7,11}",                  "[PHONE REDACTED]"),
    ("dob_slash", r"\b\d{1,2}/\d{1,2}/\d{4}\b",                    "[DOB REDACTED]"),
    ("dob_dash",  r"\b\d{1,2}-\d{1,2}-\d{4}\b",                    "[DOB REDACTED]"),
    ("dob_text",  r"\bborn\s+on\s+[\w\s,]+\d{4}\b",                "[DOB REDACTED]"),
]

# Patterns that flag the query but do NOT redact (requires human review)
_FLAG_PATTERNS: list[tuple[str, str]] = [
    ("patient_name", r"\b(patient|caller|client)\s+[A-Z][a-z]+\s+[A-Z][a-z]+"),
    ("mr_mrs",       r"\b(Mr|Mrs|Ms|Dr)\.\s+[A-Z][a-z]+"),
]


def scan(text: str) -> dict:
    """
    Scan text for PII.

    Returns:
        {
            "has_pii": bool,
            "types":   list[str],   # which PII types were found
            "redacted": str,        # text with PII replaced
        }
    """
    types: list[str] = []
    redacted = text

    for name, pattern, replacement in _PII_PATTERNS:
        if re.search(pattern, redacted, re.IGNORECASE):
            types.append(name)
            redacted = re.sub(pattern, replacement, redacted, flags=re.IGNORECASE)

    for name, pattern in _FLAG_PATTERNS:
        if re.search(pattern, text):
            types.append(name)

    return {
        "has_pii": len(types) > 0,
        "types":   types,
        "redacted": redacted,
    }
