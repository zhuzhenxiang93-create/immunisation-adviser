"""
query_normalizer.py — Synonym expansion for clinical queries before retrieval.

Built from actual vocabulary in:
  - data/immunisation_rag_chunks(1).csv  (3,633 knowledge base chunks)
  - data/transcript_questions.json       (real IMAC caller language)
  - evaluation/question_set.json
  - evaluation/transcript_question_set.json

Strategy: APPEND expanded terms rather than replace, so the original
query is preserved and BM25 gets additional signal.
"""
from __future__ import annotations

import re

# ── Synonym map ───────────────────────────────────────────────────────────────
# Each entry: (regex_pattern, terms_to_append)
# Patterns use word boundaries (\b) to avoid partial matches.

_SYNONYMS: list[tuple[str, str]] = [

    # ── Lay names → medical names ─────────────────────────────────────────────
    (r"\bflu\b",                  "influenza"),
    (r"\bchicken\s*pox\b",        "varicella"),
    (r"\bchickenpox\b",           "varicella"),
    (r"\bshingles\b",             "zoster herpes zoster Shingrix Zostavax"),
    (r"\bwhooping\s*cough\b",     "pertussis"),
    (r"\bgerman\s*measles\b",     "rubella MMR"),
    (r"\bslapped\s*cheek\b",      "parvovirus B19"),
    (r"\bmeningitis\b",           "meningococcal meningitis"),
    (r"\bpneumonia\b",            "pneumococcal pneumonia PCV"),
    (r"\btuberculosis\b",         "BCG tuberculosis TB"),
    (r"\btb\b",                   "tuberculosis BCG"),

    # ── Vaccine brand names → generic / type ─────────────────────────────────
    (r"\bgardasil\b",             "HPV human papillomavirus"),
    (r"\bgardasil\s*9\b",         "HPV9 human papillomavirus"),
    (r"\brotarix\b",              "rotavirus"),
    (r"\brotorris\b",             "rotavirus Rotarix"),          # caller mispronunciation
    (r"\brotateq\b",              "rotavirus"),
    (r"\binfanrix\b",             "DTaP diphtheria tetanus pertussis"),
    (r"\bbexsero\b",              "meningococcal B MenB"),
    (r"\bnimenrix\b",             "meningococcal ACWY MenACWY"),
    (r"\bmenquadfi\b",            "meningococcal ACWY MenACWY"),
    (r"\bshingrix\b",             "zoster herpes zoster"),
    (r"\bzostavax\b",             "zoster herpes zoster"),
    (r"\bhavrix\b",               "hepatitis A"),
    (r"\bengerix\b",              "hepatitis B"),
    (r"\btwinrix\b",              "hepatitis A hepatitis B"),
    (r"\bboostrix\b",             "Tdap pertussis tetanus diphtheria booster"),
    (r"\bprevenar\b",             "pneumococcal PCV13"),
    (r"\bpneumovax\b",            "pneumococcal 23PPV"),
    (r"\bsynflorix\b",            "pneumococcal PCV"),
    (r"\bfluzone\b",              "influenza"),
    (r"\bfluarix\b",              "influenza"),
    (r"\bfluad\b",                "influenza adjuvanted"),
    (r"\bflumist\b",              "influenza live attenuated nasal"),
    (r"\binfluvac\b",             "influenza"),
    (r"\bsynagis\b",              "RSV palivizumab respiratory syncytial virus"),
    (r"\bbeyfortus\b",            "RSV nirsevimab respiratory syncytial virus"),
    (r"\bvivaxim\b",              "hepatitis A typhoid"),
    (r"\bavaxim\b",               "hepatitis A"),

    # ── Abbreviations → full terms ────────────────────────────────────────────
    (r"\bmmr\b",                  "measles mumps rubella MMR"),
    (r"\bmmrv\b",                 "measles mumps rubella varicella MMRV"),
    (r"\bhpv\b",                  "human papillomavirus HPV"),
    (r"\bbcg\b",                  "BCG bacille calmette guerin tuberculosis"),
    (r"\bdtap\b",                 "diphtheria tetanus pertussis DTaP"),
    (r"\btdap\b",                 "tetanus diphtheria pertussis Tdap"),
    (r"\bdtp\b",                  "diphtheria tetanus pertussis"),
    (r"\bipv\b",                  "inactivated polio vaccine poliomyelitis"),
    (r"\bhib\b",                  "haemophilus influenzae type b Hib"),
    (r"\bpcv\b",                  "pneumococcal conjugate vaccine PCV"),
    (r"\bpcv13\b",                "pneumococcal PCV13 Prevenar"),
    (r"\b23ppv\b",                "pneumococcal 23PPV Pneumovax"),
    (r"\bmen\s*b\b",              "meningococcal B MenB Bexsero"),
    (r"\bmen\s*acwy\b",           "meningococcal ACWY Nimenrix"),
    (r"\bmenb\b",                 "meningococcal B MenB Bexsero"),
    (r"\bmenacwy\b",              "meningococcal ACWY Nimenrix"),
    (r"\bhep\s*a\b",              "hepatitis A"),
    (r"\bhep\s*b\b",              "hepatitis B"),
    (r"\brsv\b",                  "respiratory syncytial virus RSV"),
    (r"\bgbs\b",                  "Guillain-Barré syndrome"),
    (r"\bhsct\b",                 "haematopoietic stem cell transplant"),
    (r"\bsot\b",                  "solid organ transplant"),
    (r"\bcovid\b",                "COVID-19 coronavirus"),
    (r"\bcoronavirus\b",          "COVID-19"),
    (r"\bcon\s*13\b",             "PCV13 pneumococcal Prevenar"),  # caller abbreviation

    # ── Clinical conditions (lay → medical) ───────────────────────────────────
    (r"\blupus\b",                "systemic lupus erythematosus SLE immunosuppressed"),
    (r"\brheumatoid\b",           "rheumatoid arthritis DMARD immunosuppressed"),
    (r"\bcrohn\b",                "inflammatory bowel disease immunosuppressed"),
    (r"\begg\s*allergy\b",        "egg allergy influenza contraindication"),
    (r"\banaphylaxis\b",          "anaphylaxis anaphylactic reaction contraindication"),
    (r"\banaphylactic\b",         "anaphylaxis contraindication"),
    (r"\bone\s*kidney\b",         "renal chronic kidney disease CKD"),
    (r"\bkidney\s*disease\b",     "chronic kidney disease CKD renal"),
    (r"\bno\s*spleen\b",          "asplenia splenectomy functional asplenia"),
    (r"\bsplenectomy\b",          "asplenia functional asplenia"),
    (r"\bhiv\b",                  "HIV immunocompromised CD4"),
    (r"\bimmunocompromised\b",    "immunocompromised immunosuppressed"),
    (r"\bimmunosuppressed\b",     "immunocompromised immunosuppressed"),
    (r"\bdiabetes\b",             "diabetes mellitus chronic condition"),
    (r"\bpregnant\b",             "pregnancy pregnant"),
    (r"\bpregnancy\b",            "pregnancy pregnant"),
    (r"\bbreastfeed\b",           "breastfeeding lactating"),
    (r"\bnewborn\b",              "neonate newborn infant"),
    (r"\bbaby\b",                 "infant baby"),
    (r"\btoddler\b",              "toddler child"),
    (r"\bpremature\b",            "premature preterm infant corrected age"),
    (r"\bpreterm\b",              "premature preterm infant corrected age"),
    (r"\belderly\b",              "elderly older adult"),

    # ── Clinical procedures / lay terms ──────────────────────────────────────
    (r"\bjab\b",                  "vaccine injection"),
    (r"\bshot\b",                 "vaccine injection"),
    (r"\bboost(?:er)?\b",         "booster dose"),
    (r"\bcatch[\s\-]?up\b",       "catch-up schedule delayed immunisation"),
    (r"\bside\s*effects?\b",      "adverse event adverse reaction"),
    (r"\breaction\b",             "adverse reaction adverse event"),
    (r"\bfridge\b",               "refrigerator cold chain storage temperature"),
    (r"\bstorage\b",              "storage cold chain temperature refrigerator"),
    (r"\bstore\b",                "storage cold chain temperature"),
    (r"\bfunded\b",               "funded funding eligibility PHARMAC schedule"),
    (r"\bfree\b",                 "funded free of charge PHARMAC"),
    (r"\bschedule\b",             "schedule immunisation programme"),
    (r"\binterval\b",             "interval minimum interval dose schedule"),
    (r"\bwound\b",                "wound tetanus-prone wound prophylaxis"),
    (r"\btravel\b",               "travel vaccine international travel"),
    (r"\boccupational\b",         "occupational risk healthcare worker"),
    (r"\bhealthcare\s*worker\b",  "healthcare worker occupational risk"),
    (r"\bco[\s\-]?admin\b",       "co-administration simultaneous vaccine"),
    (r"\bsame\s*time\b",          "co-administration simultaneous vaccine"),

    # ── Misspellings → correct ────────────────────────────────────────────────
    (r"\binfuenza\b",             "influenza"),
    (r"\binfluena\b",             "influenza"),
    (r"\bvaccin(?!e|ation|ated|ator|ology)\b", "vaccine"),
    (r"\bimmunisation\b",         "immunisation immunization"),
    (r"\bimmunization\b",         "immunisation immunization"),
    (r"\bcontraindication\b",     "contraindication contraindicated"),
]

# Pre-compile all patterns once at import time
_COMPILED: list[tuple[re.Pattern, str]] = [
    (re.compile(pat, re.IGNORECASE), repl)
    for pat, repl in _SYNONYMS
]


def normalize(query: str) -> str:
    """
    Expand synonyms in a query before retrieval.

    Appends expanded terms to the original query so:
    - Original intent and wording is preserved
    - BM25 gets additional matching signal
    - FAISS embedding is enriched with more context
    """
    added: list[str] = []
    q_lower = query.lower()

    for pattern, replacement in _COMPILED:
        if pattern.search(query):
            for term in replacement.split():
                if term.lower() not in q_lower and term not in added:
                    added.append(term)

    if added:
        return query + " " + " ".join(added)
    return query


if __name__ == "__main__":
    tests = [
        ("flu jab safe in pregnancy?",               "俚语 + 口语"),
        ("chicken pox storage fridge temp",           "俗名 + 口语"),
        ("hep b catch up for toddler",                "缩写 + 口语"),
        ("infuenza side effects",                     "拼写错误"),
        ("BCG contraindications newborn",             "缩写"),
        ("funded?",                                   "极短查询"),
        ("MMR booster after splenectomy",             "缩写 + 医学术语"),
        ("Rotorris second dose",                      "caller 发音错误"),
        ("shingles vaccine for someone on DMARDs",    "俗名 + 临床缩写"),
        ("one kidney can they get flu vaccine",       "lay 描述"),
        ("Men B for baby with no spleen",             "缩写 + lay 术语"),
        ("Is Bexsero funded?",                        "品牌名"),
        ("covid booster and flu shot same time",      "俚语 + 共同接种"),
    ]
    print(f"{'Query':<50} {'Added terms'}")
    print("-" * 100)
    for q, note in tests:
        result = normalize(q)
        added = result[len(q):].strip() if result != q else "(no change)"
        print(f"{q:<50} {added}   [{note}]")
