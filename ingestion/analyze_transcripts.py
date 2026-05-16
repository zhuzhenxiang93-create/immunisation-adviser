"""
analyze_transcripts.py — Process IMAC Amazon Connect Contact Lens transcripts.

Reads all _redacted_ JSON files from data/output20260400/extracted/, extracts
the caller's main clinical question, classifies it, and produces:
  - data/transcript_analysis.json   (per-call summary)
  - data/transcript_questions.json  (curated eval questions from real calls)
  - data/volume_patterns.json       (aggregated volume patterns — time of day,
                                     day of week, topic frequency)

Note: transcript content is used for analysis and evaluation only.
      It is NOT included in the RAG retrieval knowledge base.

Usage:
    python -m ingestion.analyze_transcripts
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.classifier import classify_query

TRANSCRIPTS_DIR = "data/output20260400/extracted"
ANALYSIS_OUT    = "data/transcript_analysis.json"
QUESTIONS_OUT   = "data/transcript_questions.json"
VOLUME_OUT      = "data/volume_patterns.json"

# NZ Standard Time offset (UTC+12). April = post-daylight-saving.
_NZ_TZ = timezone(timedelta(hours=12))

_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Filename timestamp pattern: ...redacted_2026-04-15T00_14_14Z.json
_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2})T(\d{2})_(\d{2})_(\d{2})Z")

CLINICAL_KWS = [
    "vaccine", "vaccin", "dose", "immunis", "immuniz", "flu", "hpv", "mmr",
    "covid", "contraindic", "pregnant", "allerg", "schedule", "catch", "booster",
    "injection", "rotavirus", "meningococcal", "pneumococcal", "hepatitis",
    "bcg", "tdap", "varicella", "chickenpox", "tetanus", "pertussis",
    "whooping", "polio", "rsv", "funded", "pharmac", "eligib", "side effect",
    "adverse", "anaphylax", "reaction", "storage", "cold chain", "interval",
    "third dose", "second dose", "first dose", "shingrix", "zostavax",
    "bexsero", "nimenrix", "gardasil", "infanrix", "rotarix", "hib",
]

_PII = re.compile(r"\[(NAME|ADDRESS|PHONE|EMAIL|NHI)\]")
# Residual phone patterns that Contact Lens occasionally misses (post-hoc safety net)
_RESIDUAL_PHONE = re.compile(r"\b(\+64[\s\-]?[2-9][\d\s\-]{7,11}|0[2-9]\d{7,9})\b")


def _parse_timestamp(filename: str) -> dict | None:
    """Extract NZ local hour and day-of-week from a Contact Lens filename."""
    m = _TS_RE.search(filename)
    if not m:
        return None
    date_str, hh, mm, ss = m.groups()
    utc_dt = datetime(
        int(date_str[:4]), int(date_str[5:7]), int(date_str[8:10]),
        int(hh), int(mm), int(ss),
        tzinfo=timezone.utc,
    )
    nz_dt = utc_dt.astimezone(_NZ_TZ)
    return {
        "date":        nz_dt.strftime("%Y-%m-%d"),
        "hour_nz":     nz_dt.hour,          # 0-23 NZ local time
        "day_of_week": _DAYS[nz_dt.weekday()],
    }


def _clean(text: str) -> str:
    text = _PII.sub("[REDACTED]", text)
    text = _RESIDUAL_PHONE.sub("[PHONE]", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_question(transcript: list[dict], participants: dict[str, str]) -> str | None:
    """Return the first substantive clinical question from the CUSTOMER."""
    for entry in transcript:
        role = participants.get(entry["ParticipantId"], "")
        if role != "CUSTOMER":
            continue
        content = entry["Content"]
        if len(content) < 60:
            continue
        lower = content.lower()
        if any(kw in lower for kw in CLINICAL_KWS):
            return _clean(content[:600])
    return None


def _extract_agent_answer(transcript: list[dict], participants: dict[str, str],
                           question_idx: int) -> str:
    """Collect agent turns that follow the question, up to 400 chars."""
    agent_parts: list[str] = []
    collecting = False
    char_count = 0
    for i, entry in enumerate(transcript):
        role = participants.get(entry["ParticipantId"], "")
        if not collecting and i >= question_idx and role == "AGENT":
            collecting = True
        if collecting and role == "AGENT":
            part = entry["Content"].strip()
            if len(part) > 15:
                agent_parts.append(part)
                char_count += len(part)
            if char_count >= 400:
                break
    return " ".join(agent_parts)[:500]


def analyze(transcripts_dir: str = TRANSCRIPTS_DIR) -> tuple[list[dict], list[dict]]:
    folder = Path(transcripts_dir)
    files  = sorted(f for f in folder.iterdir() if f.suffix == ".json")
    print(f"Processing {len(files)} transcript files…")

    analyses: list[dict] = []
    questions: list[dict] = []
    seen_questions: set[str] = set()
    q_id = 1

    for i, fpath in enumerate(files, 1):
        if i % 200 == 0:
            print(f"  {i}/{len(files)}")

        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)

        participants = {p["ParticipantId"]: p["ParticipantRole"]
                        for p in data.get("Participants", [])}
        transcript   = data.get("Transcript", [])
        categories   = data.get("Categories", {}).get("MatchedCategories", [])

        question_text = _extract_question(transcript, participants)
        if not question_text:
            continue

        dedup_key = question_text[:100].lower()
        if dedup_key in seen_questions:
            continue
        seen_questions.add(dedup_key)

        clf = classify_query(question_text)

        # Find question position in transcript for agent answer extraction
        q_pos = 0
        for j, entry in enumerate(transcript):
            if entry["Content"] in question_text[:50]:
                q_pos = j
                break

        agent_answer = _extract_agent_answer(transcript, participants, q_pos)

        ts = _parse_timestamp(fpath.name)
        record: dict = {
            "file":              fpath.name,
            "timestamp":         ts,
            "question":          question_text,
            "agent_response":    agent_answer,
            "contact_lens_cats": categories,
            "classification":    clf,
        }
        analyses.append(record)

        # Build evaluation question if it looks clinically specific enough
        is_specific = (
            clf["vaccine_type"] != ["unknown"]
            or clf["query_type"] != ["general"]
            or clf["clinical_scenario"]
        )
        if is_specific and len(questions) < 30:
            questions.append({
                "id":                   f"T{q_id:03d}",
                "source":               "real_transcript",
                "query":                question_text,
                "category":             (clf["query_type"][0]
                                         if clf.get("query_type")
                                         else "general"),
                "vaccine_type":         clf["vaccine_type"],
                "clinical_scenario":    clf["clinical_scenario"],
                "caller_type":          clf["caller_type"],
                "patient_age_group":    clf["patient_age_group"],
                "urgency":              clf["urgency"],
                "agent_response_hint":  agent_answer[:300] if agent_answer else "",
                "expected_confidence":  "high",
            })
            q_id += 1

    return analyses, questions


def build_volume_patterns(analyses: list[dict]) -> dict:
    """
    Build aggregated volume patterns from classified call records.
    All timestamps are in NZ local time (UTC+12).
    """
    hourly:    Counter = Counter()   # hour 0-23
    daily:     Counter = Counter()   # Monday … Sunday
    by_date:   Counter = Counter()   # YYYY-MM-DD
    vaccine:   Counter = Counter()
    qtype:     Counter = Counter()
    scenario:  Counter = Counter()
    caller:    Counter = Counter()
    urgency:   Counter = Counter()

    for a in analyses:
        ts  = a.get("timestamp") or {}
        clf = a["classification"]

        if ts.get("hour_nz") is not None:
            hourly[ts["hour_nz"]] += 1
        if ts.get("day_of_week"):
            daily[ts["day_of_week"]] += 1
        if ts.get("date"):
            by_date[ts["date"]] += 1

        for v in clf.get("vaccine_type", []):
            vaccine[v] += 1
        for q in clf.get("query_type", []):
            qtype[q] += 1
        for s in clf.get("clinical_scenario", []):
            scenario[s] += 1
        caller[clf.get("caller_type", "unknown")] += 1
        urgency[clf.get("urgency", "routine")] += 1

    # Build ordered hourly dict (00:00 … 23:00)
    hourly_ordered = {f"{h:02d}:00": hourly.get(h, 0) for h in range(24)}

    # Build ordered day-of-week dict (Mon … Sun)
    dow_ordered = {d: daily.get(d, 0) for d in _DAYS}

    return {
        "total_classified_calls": len(analyses),
        "hourly_nz":    hourly_ordered,
        "day_of_week":  dow_ordered,
        "daily_volume": dict(sorted(by_date.items())),
        "top_vaccine_types":      dict(vaccine.most_common(15)),
        "top_query_types":        dict(qtype.most_common(15)),
        "top_clinical_scenarios": dict(scenario.most_common(15)),
        "caller_types":           dict(caller.most_common()),
        "urgency":                dict(urgency.most_common()),
    }


def print_summary(analyses: list[dict]) -> None:
    vaccine_counter: Counter = Counter()
    qtype_counter:   Counter = Counter()
    scenario_counter: Counter = Counter()
    caller_counter:  Counter = Counter()

    for a in analyses:
        clf = a["classification"]
        for v in clf["vaccine_type"]:
            vaccine_counter[v] += 1
        for q in clf["query_type"]:
            qtype_counter[q] += 1
        for s in clf["clinical_scenario"]:
            scenario_counter[s] += 1
        caller_counter[clf["caller_type"]] += 1

    print(f"\n{'='*60}")
    print(f"Total classified calls: {len(analyses)}")
    print(f"\nTop vaccine types:")
    for v, c in vaccine_counter.most_common(10):
        print(f"  {c:4d}  {v}")
    print(f"\nTop query types:")
    for q, c in qtype_counter.most_common(10):
        print(f"  {c:4d}  {q}")
    print(f"\nClinical scenarios:")
    for s, c in scenario_counter.most_common():
        print(f"  {c:4d}  {s}")
    print(f"\nCaller types:")
    for r, c in caller_counter.most_common():
        print(f"  {c:4d}  {r}")


if __name__ == "__main__":
    analyses, questions = analyze()

    Path(ANALYSIS_OUT).parent.mkdir(parents=True, exist_ok=True)

    with open(ANALYSIS_OUT, "w", encoding="utf-8") as f:
        json.dump(analyses, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(analyses)} call analyses → {ANALYSIS_OUT}")

    with open(QUESTIONS_OUT, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(questions)} transcript questions → {QUESTIONS_OUT}")

    volume = build_volume_patterns(analyses)
    with open(VOLUME_OUT, "w", encoding="utf-8") as f:
        json.dump(volume, f, ensure_ascii=False, indent=2)
    print(f"Saved volume patterns → {VOLUME_OUT}")

    print_summary(analyses)
