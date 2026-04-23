"""
scrape_handbook.py — Scrape NZ Immunisation Handbook using Playwright (JS rendering).
Produces data/chunks_raw.json ready for embed_and_index.py.

Usage:
    conda activate immunisation-adviser
    cd D:/714/hackthon/immunisation-adviser
    python -m ingestion.scrape_handbook
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from ingestion.chunk_documents import chunk_document, extract_sections

BASE_URL = "https://www.tewhatuora.govt.nz"
HANDBOOK_BASE = f"{BASE_URL}/for-health-professionals/clinical-guidance/immunisation-handbook"
OUTPUT_PATH = Path("data/chunks_raw.json")
SOURCE_NAME = "NZ Immunisation Handbook"

HANDBOOK_PAGES = [
    ("National Immunisation Schedule",                           "national-immunisation-schedule"),
    ("Funded vaccines for special groups",                       "funded-vaccines-for-special-groups"),
    ("Anaphylaxis response and management",                      "anaphylaxis-responsemanagement"),
    ("Introduction",                                             "introduction"),
    ("Chapter 1 - General immunisation principles",              "1-general-immunisation-principles"),
    ("Chapter 2 - Processes for safe immunisation",              "2-processes-for-safe-immunisation"),
    ("Chapter 3 - Vaccination questions and addressing concerns","3-vaccination-questions-and-addressing-concerns"),
    ("Chapter 4 - Immunisation of special groups",               "4-immunisation-of-special-groups"),
    ("Chapter 5 - COVID-19",                                     "5-coronavirus-disease-covid-19"),
    ("Chapter 6 - Diphtheria",                                   "6-diphtheria"),
    ("Chapter 7 - Haemophilus influenzae type b (Hib)",          "7-haemophilus-inuenzae-type-b-hib-disease"),
    ("Chapter 8 - Hepatitis A",                                  "8-hepatitis-a"),
    ("Chapter 9 - Hepatitis B",                                  "9-hepatitis-b"),
    ("Chapter 10 - Human papillomavirus (HPV)",                  "10-human-papillomavirus"),
    ("Chapter 11 - Influenza",                                   "11-influenza"),
    ("Chapter 12 - Measles",                                     "12-measles"),
    ("Chapter 13 - Meningococcal disease",                       "13-meningococcal-disease"),
    ("Chapter 14 - Mpox",                                        "14-mpox"),
    ("Chapter 15 - Mumps",                                       "15-mumps"),
    ("Chapter 16 - Pertussis (whooping cough)",                  "16-pertussis-whooping-cough"),
    ("Chapter 17 - Pneumococcal disease",                        "17-pneumococcal-disease"),
    ("Chapter 18 - Poliomyelitis",                               "18-poliomyelitis"),
    ("Chapter 19 - Respiratory syncytial virus",                 "19-respiratory-syncytial-virus"),
    ("Chapter 20 - Rotavirus",                                   "20-rotavirus"),
    ("Chapter 21 - Rubella",                                     "21-rubella"),
    ("Chapter 22 - Tetanus",                                     "22-tetanus"),
    ("Chapter 23 - Tuberculosis",                                "23-tuberculosis"),
    ("Chapter 24 - Varicella (chickenpox)",                      "24-varicella-chickenpox"),
    ("Chapter 25 - Zoster (herpes zoster/shingles)",             "25-zoster-herpes-zostershingles"),
    ("Appendix 2 - Planning immunisation catch-ups",             "appendix-2-planning-immunisation-catch-ups"),
    ("Appendix 6 - Passive immunisation",                        "appendix-6-passive-immunisation"),
    ("Appendix 7 - Vaccine preparation and disposal",            "appendix-7-vaccine-presentation-preparation-disposal-and-needle-stick-recommendations"),
    ("Glossary of vaccine brand names",                          "glossary-of-vaccine-brand-names-and-abbreviations"),
]


def extract_text_from_html(html: str) -> str:
    from bs4 import BeautifulSoup, Tag
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["nav", "footer", "script", "style", "header", "aside", "noscript"]):
        tag.decompose()

    main = (soup.find("main") or
            soup.find("div", {"id": "main-content"}) or
            soup.find("article") or
            soup.body)

    if not isinstance(main, Tag):
        return ""

    lines = []
    for elem in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th"]):
        text = elem.get_text(separator=" ", strip=True)
        if text and len(text) > 15:
            lines.append(text)

    return "\n\n".join(lines)


def scrape_all() -> list[dict]:
    from playwright.sync_api import sync_playwright

    all_chunks: list[dict] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        )
        page = context.new_page()

        for chapter_title, slug in HANDBOOK_PAGES:
            url = f"{HANDBOOK_BASE}/{slug}"
            print(f"Scraping: {chapter_title}")

            try:
                page.goto(url, wait_until="networkidle", timeout=30000)
                # Wait for main content to appear
                page.wait_for_timeout(2000)
                html = page.content()
            except Exception as e:
                print(f"  [WARN] {e}")
                time.sleep(2)
                continue

            text = extract_text_from_html(html)

            if len(text) < 100:
                print(f"  → Skipped (only {len(text)} chars)")
                time.sleep(1)
                continue

            sections = extract_sections(text)
            page_chunks: list[dict] = []

            for sec in sections:
                body = sec.get("body", "").strip()
                if not body:
                    continue
                chunks = chunk_document(
                    text=body,
                    source_name=SOURCE_NAME,
                    url=url,
                    chapter=chapter_title,
                    section=sec.get("heading", ""),
                )
                page_chunks.extend(chunks)

            print(f"  -> {len(page_chunks)} chunks")
            all_chunks.extend(page_chunks)
            time.sleep(1.5)

        browser.close()

    return all_chunks


if __name__ == "__main__":
    print(f"Scraping {len(HANDBOOK_PAGES)} handbook pages with Playwright...\n")
    chunks = scrape_all()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"\nDone - {len(chunks)} chunks saved to {OUTPUT_PATH}")
    print("Next: python -m ingestion.embed_and_index data/chunks_raw.json --local")
