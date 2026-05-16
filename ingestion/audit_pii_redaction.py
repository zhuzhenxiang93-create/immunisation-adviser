"""
ingestion/audit_pii_redaction.py — PII redaction audit for Contact Lens transcripts.

Verifies that Amazon Connect Contact Lens PII redaction is active on every file
and scans for residual PII patterns that Contact Lens does NOT natively cover:
  - NZ NHI numbers (old: ABC1234, new: ABC12DE)
  - NZ phone numbers (0[2-9]XXXXXXXX, +64 XXXXXXXXX)
  - Email addresses
  - Dates of birth (slash/dash/text formats)

Outputs:
  - Console summary (counts, gaps, examples)
  - data/pii_audit_report.json   (full machine-readable results)

Usage:
  python -m ingestion.audit_pii_redaction
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

EXTRACTED_DIR = Path("data/output20260400/extracted")
REPORT_PATH   = Path("data/pii_audit_report.json")

# ── NHI patterns ──────────────────────────────────────────────────────────────
# Old format: 3 uppercase alpha + 4 digits  e.g. ABC1234
# New format: 3 uppercase alpha + 2 digits + 2 uppercase alpha  e.g. ABC12DE
_NHI_OLD = re.compile(r"\b[A-Z]{3}\d{4}\b")
_NHI_NEW = re.compile(r"\b[A-Z]{3}\d{2}[A-Z]{2}\b")

# ── NZ phone (not already masked as [PHONE]) ──────────────────────────────────
_PHONE = re.compile(
    r"(?<!\[)"                           # not preceded by [  (avoid matching "[PHONE]")
    r"(\+64[\s\-]?[2-9][\d\s\-]{7,11}"  # international
    r"|0[2-9][\d\s\-]{7,11})"           # local
)

# ── Email (not already masked as [EMAIL]) ────────────────────────────────────
_EMAIL = re.compile(r"(?<!\[)[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

# ── Date of birth patterns ────────────────────────────────────────────────────
_DOB_SLASH = re.compile(r"\b\d{1,2}/\d{1,2}/(?:19|20)\d{2}\b")
_DOB_DASH  = re.compile(r"\b\d{1,2}-\d{1,2}-(?:19|20)\d{2}\b")
_DOB_TEXT  = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?\s*,?\s*(?:19|20)\d{2}\b",
    re.IGNORECASE,
)


def _get_transcript_text(data: dict) -> str:
    """Join all Transcript Content fields into one string for pattern scanning."""
    return " ".join(
        turn.get("Content", "") for turn in data.get("Transcript", [])
    )


def _check_redaction_active(data: dict) -> tuple[bool, str]:
    """Return (is_redacted, mask_mode)."""
    meta = data.get("ContentMetadata", {})
    output = meta.get("Output", "")
    mask = (
        meta.get("RedactionTypesMetadata", {})
            .get("PII", {})
            .get("RedactionMaskMode", "unknown")
    )
    return output == "Redacted", mask


def _scan_residual(text: str) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    for label, pattern in [
        ("nhi_old",  _NHI_OLD),
        ("nhi_new",  _NHI_NEW),
        ("phone",    _PHONE),
        ("email",    _EMAIL),
        ("dob_slash", _DOB_SLASH),
        ("dob_dash",  _DOB_DASH),
        ("dob_text",  _DOB_TEXT),
    ]:
        found = pattern.findall(text)
        if found:
            # Flatten groups in PHONE regex
            hits[label] = [m if isinstance(m, str) else m[0] for m in found]
    return hits


def audit() -> dict[str, Any]:
    files = sorted(EXTRACTED_DIR.glob("*_redacted_*.json"))
    if not files:
        print(f"[audit] No files found in {EXTRACTED_DIR}", file=sys.stderr)
        return {}

    total = len(files)
    redacted_count   = 0
    mask_modes: dict[str, int] = {}
    residual_hits: dict[str, int] = {}
    problem_files: list[dict] = []

    # Collect entity type coverage from first file only (same across all)
    entity_types_covered: list[str] = []

    for i, path in enumerate(files):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[audit] Cannot read {path.name}: {e}", file=sys.stderr)
            continue

        is_redacted, mask_mode = _check_redaction_active(data)

        if is_redacted:
            redacted_count += 1
        mask_modes[mask_mode] = mask_modes.get(mask_mode, 0) + 1

        if not entity_types_covered:
            entity_types_covered = (
                data.get("ContentMetadata", {})
                    .get("RedactionTypesMetadata", {})
                    .get("PII", {})
                    .get("RedactionEntitiesRequested", [])
            )

        text = _get_transcript_text(data)
        hits = _scan_residual(text)

        for label, matches in hits.items():
            residual_hits[label] = residual_hits.get(label, 0) + len(matches)

        if hits:
            problem_files.append({
                "file": path.name,
                "hits": hits,
            })

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{total}] scanned …")

    # ── Summary ───────────────────────────────────────────────────────────────
    report = {
        "total_files":           total,
        "redaction_active":      redacted_count,
        "redaction_pct":         round(100 * redacted_count / total, 1) if total else 0,
        "mask_modes":            mask_modes,
        "entity_types_covered":  entity_types_covered,
        "residual_hits_total":   residual_hits,
        "files_with_residual":   len(problem_files),
        "problem_files":         problem_files[:50],  # cap at 50 for readability
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def _print_summary(r: dict) -> None:
    print("\n========== PII REDACTION AUDIT ==========")
    print(f"Files scanned          : {r['total_files']}")
    print(f"Redaction active       : {r['redaction_active']} ({r['redaction_pct']}%)")
    print(f"Mask modes             : {r['mask_modes']}")
    print()
    print("Entity types Contact Lens covers:")
    for et in r.get("entity_types_covered", []):
        print(f"  {et}")
    print()
    print("Residual PII scan (patterns NOT in AWS entity list or slipping through):")
    hits = r.get("residual_hits_total", {})
    if hits:
        for label, count in hits.items():
            print(f"  {label:15s}: {count} occurrence(s)")
    else:
        print("  None found.")
    print()
    print(f"Files with residual hits: {r['files_with_residual']}")
    if r.get("problem_files"):
        print("\nFirst 5 files with residual hits:")
        for pf in r["problem_files"][:5]:
            print(f"  {pf['file']}")
            for label, matches in pf["hits"].items():
                print(f"    {label}: {matches[:3]}")
    print(f"\nFull report saved to: {REPORT_PATH}")
    print("=========================================\n")


if __name__ == "__main__":
    r = audit()
    _print_summary(r)
