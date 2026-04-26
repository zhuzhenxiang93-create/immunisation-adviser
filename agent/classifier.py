"""
classifier.py — Rule-based query classification for immunisation queries.

Six classification dimensions:
  - vaccine_type     : which vaccine(s) the query is about
  - query_type       : what information is being sought (contraindication, dosage, etc.)
  - clinical_scenario: patient situation (pregnancy, immunocompromised, etc.)
  - caller_type      : inferred caller role (nurse, GP, parent, etc.)
  - patient_age_group: inferred patient age group (infant, child, adult, etc.)
  - urgency          : emergency / urgent / routine

Rule-based (no LLM call) — zero latency and zero cost.
"""

from __future__ import annotations

import re


def _match(text: str, patterns: list[str]) -> bool:
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


# ── Vaccine type ──────────────────────────────────────────────────────────────

_VACCINE_RULES: list[tuple[str, list[str]]] = [
    ("influenza",       ["influenza", r"\bflu\b", "flucelvax", "afluria", "fluad"]),
    ("MMR",             [r"\bmmr\b", "measles", "mumps", "rubella"]),
    ("MMRV",            [r"\bmmrv\b", "priorix tetra"]),
    ("varicella",       ["varicella", "chickenpox", r"\bvzv\b", "zostavax", "shingrix"]),
    ("COVID-19",        ["covid", "coronavirus", r"sars.cov", "comirnaty", "spikevax", "nuvaxovid"]),
    ("BCG",             [r"\bbcg\b", "tuberculosis", r"\btb\b"]),
    ("Tdap/Td",         [r"\btdap\b", r"\bdtp\b", r"\btd\b", "tetanus", "diphtheria", "pertussis", "whooping cough", "boostrix", "infanrix"]),
    ("HPV",             [r"\bhpv\b", "human papillomavirus", "gardasil", "cervarix"]),
    ("hepatitis B",     [r"hep\w*[\s\-]*b\b", "hepatitis b", r"\bhbv\b", "engerix", "hbvaxpro"]),
    ("hepatitis A",     [r"hep\w*[\s\-]*a\b", "hepatitis a", r"\bhav\b", "havrix", "avaxim"]),
    ("rotavirus",       ["rotavirus", "rotarix", "rotateq"]),
    ("pneumococcal",    ["pneumococcal", r"\bpcv\b", "prevenar", "synflorix"]),
    ("meningococcal",   ["meningococcal", r"men\s*[abcwy]", "bexsero", "nimenrix", "menveo"]),
    ("polio",           ["polio", r"\bipv\b", r"\bopv\b", "poliovirus"]),
    ("Hib",             [r"\bhib\b", "haemophilus influenzae type b"]),
    ("RSV",             [r"\brsv\b", "respiratory syncytial", "nirsevimab", "beyfortus"]),
    ("zoster",          ["zoster", "shingles", "herpes zoster"]),
]


# ── Query type ────────────────────────────────────────────────────────────────

_QUERY_TYPE_RULES: list[tuple[str, list[str]]] = [
    ("contraindication", ["contraindic", "precaution", "not suitable", "should.*avoid", r"\bcannot\b.*vaccin", "unsafe.*vaccin"]),
    ("schedule",         ["schedule", "when to give", "what age", "timing", "recommended age", "how often", "interval between", "next dose"]),
    ("dosage",           [r"\bdose\b", r"\bdoses\b", "dosage", "how much", r"\bml\b", "microgram", "volume"]),
    ("eligibility",      ["eligible", "funded", r"\bfree\b", "subsidis", "pharmac", "who can", "who should"]),
    ("storage",          ["stor", "cold chain", "refrigerat", "freez", "temperature", r"\bthaw\b", "fridge", "handling"]),
    ("catch-up",         ["catch.?up", "missed", "overdue", "behind schedule", "never vaccinated", "incomplete", "lapsed"]),
    ("adverse event",    ["adverse", "side effect", r"\breaction\b", r"\baefi\b", "after vaccination", "following.*vaccine", "symptom"]),
    ("co-administration",["co.?admin", "same time", "together with", "simultaneous", "combination", "other vaccine"]),
    ("administration",   ["administer", "inject", "route", "subcutaneous", "intramuscular", r"\boral\b", "intranasal", "technique"]),
    ("efficacy",         ["effective", "efficacy", "protection", "how well", "work.*against", "prevent"]),
]


# ── Clinical scenario ─────────────────────────────────────────────────────────

_SCENARIO_RULES: list[tuple[str, list[str]]] = [
    ("pregnancy",          ["pregnan", "breastfeed", "lactation", "trimester", "antenatal", "postnatal", "maternal"]),
    ("immunocompromised",  ["immunocompromis", "immunosuppress", r"\bhiv\b", "transplant", "chemotherapy", "biologics", "immunodeficien", "hiv positive", "low cd4"]),
    ("allergy",            ["allerg", "anaphylax", "egg allerg", "latex", "hypersensitiv"]),
    ("premature infant",   ["premature", "preterm", "prem baby", "early birth"]),
    ("travel",             ["travel", "overseas", "international", r"\babroad\b"]),
    ("outbreak",           ["outbreak", "exposure", "contact with", "post.?exposure"]),
    ("delayed schedule",   ["delayed", "late start", "catch.?up", "missed dose", "overdue"]),
]


# ── Caller type ───────────────────────────────────────────────────────────────

_CALLER_RULES: list[tuple[str, list[str]]] = [
    ("GP",          [r"\bgp\b", "general practitioner", "family doctor", "family physician"]),
    ("nurse",       [r"\bnurse\b", "nursing", "practice nurse", "immunisation nurse", r"\brn\b"]),
    ("midwife",     ["midwife", "midwifery", r"\bLMC\b"]),
    ("pharmacist",  ["pharmacist", "pharmacy", "chemist"]),
    ("parent",      ["parent", r"\bmother\b", r"\bfather\b", "my child", "my baby", "my son", "my daughter", "my infant"]),
    ("patient",     [r"\bmyself\b", "for me", "i have been", "i am getting"]),
]


# ── Patient age group ─────────────────────────────────────────────────────────

_AGE_RULES: list[tuple[str, list[str]]] = [
    ("neonate",    ["newborn", "neonate", r"\bneonatal\b", "0.*week", "first.*week"]),
    ("infant",     [r"\binfant\b", r"\bbaby\b", "6 week", "3 month", "under.*1.*year", "under 12 month", r"\b[2-9] month"]),
    ("child",      [r"\bchild\b", "toddler", "12 month", r"\b[1-9] year.?old\b", "under 10", "preschool", "school.?age"]),
    ("adolescent", ["adolescent", "teenager", "teen", r"\b1[1-9].?year", "secondary school", "year [7-9]"]),
    ("adult",      [r"\badult\b", r"\b[2-5][0-9].?year", "working age", "18 and over"]),
    ("elderly",    ["elderly", "older adult", r"\b6[5-9]\b", r"\b[7-9][0-9].?year", "aged care", "rest home", "65 and over", "older people"]),
    ("pregnant",   ["pregnan", "expectant"]),
]


# ── Urgency level ─────────────────────────────────────────────────────────────

_URGENCY_RULES: list[tuple[str, list[str]]] = [
    ("emergency", ["anaphylaxis", "collapsed", "not breathing", "unconscious", "severe reaction", "call 111"]),
    ("urgent",    ["post.?exposure", "just given", "wrong dose", "needlestick", "exposure to", "just happened", "accidentally given"]),
]


# ── Public API ────────────────────────────────────────────────────────────────

def _classify_rule_based(query: str) -> dict:
    vaccines  = [l for l, p in _VACCINE_RULES   if _match(query, p)]
    qtypes    = [l for l, p in _QUERY_TYPE_RULES if _match(query, p)]
    scenarios = [l for l, p in _SCENARIO_RULES  if _match(query, p)]
    callers   = [l for l, p in _CALLER_RULES    if _match(query, p)]
    ages      = [l for l, p in _AGE_RULES       if _match(query, p)]

    urgency = "routine"
    for level, patterns in _URGENCY_RULES:
        if _match(query, patterns):
            urgency = level
            break

    return {
        "vaccine_type":      vaccines  or ["unknown"],
        "query_type":        qtypes    or ["general"],
        "clinical_scenario": scenarios,
        "caller_type":       callers[0] if callers else "unknown",
        "patient_age_group": ages[0]    if ages    else "unknown",
        "urgency":           urgency,
    }


def _needs_fallback(result: dict) -> bool:
    """True when rule-based found nothing useful — likely colloquial input."""
    return (
        result["vaccine_type"]      == ["unknown"]
        and result["query_type"]    == ["general"]
        and not result["clinical_scenario"]
        and result["caller_type"]   == "unknown"
        and result["patient_age_group"] == "unknown"
    )


_LLM_SYSTEM = """You are a clinical query classifier for IMAC (NZ immunisation advisory centre).
Extract classification dimensions from the immunisation query. Return ONLY valid JSON, no markdown.

Schema:
{
  "vaccine_type": ["influenza"|"MMR"|"MMRV"|"varicella"|"COVID-19"|"BCG"|"Tdap/Td"|"HPV"|"hepatitis B"|"hepatitis A"|"rotavirus"|"pneumococcal"|"meningococcal"|"polio"|"Hib"|"RSV"|"zoster"|"unknown"],
  "query_type": ["contraindication"|"schedule"|"dosage"|"eligibility"|"storage"|"catch-up"|"adverse event"|"co-administration"|"administration"|"efficacy"|"general"],
  "clinical_scenario": ["pregnancy"|"immunocompromised"|"allergy"|"premature infant"|"travel"|"outbreak"|"delayed schedule"],
  "caller_type": "GP"|"nurse"|"midwife"|"pharmacist"|"parent"|"patient"|"unknown",
  "patient_age_group": "neonate"|"infant"|"child"|"adolescent"|"adult"|"elderly"|"pregnant"|"unknown",
  "urgency": "emergency"|"urgent"|"routine"
}

Rules: vaccine_type and query_type are arrays; clinical_scenario is an array (may be empty);
caller_type, patient_age_group, urgency are single strings."""


def _classify_with_llm(query: str) -> dict:
    """GPT-4o fallback classifier. Raises on failure so caller can catch."""
    import json
    from config.azure_config import get_openai_client, get_chat_model

    client = get_openai_client()
    resp = client.chat.completions.create(
        model=get_chat_model(),
        messages=[
            {"role": "system", "content": _LLM_SYSTEM},
            {"role": "user",   "content": f"Classify: {query}"},
        ],
        temperature=0,
        max_tokens=200,
    )
    raw = resp.choices[0].message.content.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def classify_query(query: str) -> dict:
    """
    Classify a query across six dimensions.

    Rule-based first (zero cost). If the result is fully empty — all unknown,
    no scenario — the query is likely colloquial, so GPT-4o is called as a
    fallback. If GPT-4o fails for any reason, the rule-based result is kept.

    Returns:
        {
            "vaccine_type":      list[str],
            "query_type":        list[str],
            "clinical_scenario": list[str],
            "caller_type":       str,
            "patient_age_group": str,
            "urgency":           str,
        }
    """
    result = _classify_rule_based(query)
    if _needs_fallback(result):
        try:
            result = _classify_with_llm(query)
        except Exception:
            pass  # silently keep rule-based result
    return result
