"""
pdf_parser.py — Improved PDF Parser

Original issues:
1. Only extracted raw text — no structured field parsing
2. No fallback if PyPDF2 fails
3. No cleaning of extracted text

Improvements:
1. Extracts raw text ✓ (kept)
2. Also attempts to parse structured CTI fields from text
   (title, timestamps, tags, references, IOC indicators)
   → These feed into FeatureExtractor for the 19 paper features
3. Added pdfplumber as a better fallback (handles more PDF types)
4. Text cleaning (removes junk characters, excess whitespace)
"""

import re
from datetime import datetime

# Try best PDF library first, fall back gracefully
try:
    import pdfplumber
    _HAS_PDFPLUMBER = True
except ImportError:
    _HAS_PDFPLUMBER = False

try:
    import PyPDF2
    _HAS_PYPDF2 = True
except ImportError:
    _HAS_PYPDF2 = False


# ── Raw text extraction ───────────────────────────────────────────────

def extract_pdf_text(path):
    """
    Extract raw text from a PDF file.
    Tries pdfplumber first (better accuracy), falls back to PyPDF2.

    Returns:
        str: cleaned raw text
    """
    text = ""

    # Method 1: pdfplumber (better for structured PDFs / threat reports)
    if _HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if text.strip():
                return _clean_text(text)
        except Exception as e:
            print(f"[pdf_parser] pdfplumber failed: {e} — trying PyPDF2")

    # Method 2: PyPDF2 fallback
    if _HAS_PYPDF2:
        try:
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if text.strip():
                return _clean_text(text)
        except Exception as e:
            print(f"[pdf_parser] PyPDF2 failed: {e}")

    if not text.strip():
        print(f"[pdf_parser] ⚠ Could not extract text from: {path}")

    return _clean_text(text)


# ── Structured field extraction ───────────────────────────────────────

def extract_structured_fields(text):
    """
    Attempt to extract structured CTI fields from raw text.
    These fields power the 19-feature extraction in FeatureExtractor.

    This is a best-effort heuristic parser — real structured data
    would come from CyberTotal/AlienVault API responses (JSON).

    Returns:
        dict: structured fields matching FeatureExtractor expectations
    """
    fields = {
        "tags":                [],
        "references":          [],
        "malware_families":    [],
        "targeted_countries":  [],
        "industries":          [],
        "created":             None,
        "modified":            None,
        "export_count":        0,
        "subscriber_count":    0,
        "indicators_count":    0,
    }

    if not text:
        return fields

    # --- Tags: lines starting with #word or "Tags:" sections ---
    hashtag_tags = re.findall(r"#([A-Za-z][A-Za-z0-9_-]+)", text)
    tag_section  = re.findall(
        r"(?:Tags?|Labels?)\s*[:\-]\s*([^\n]+)", text, re.IGNORECASE)
    for line in tag_section:
        hashtag_tags += [t.strip() for t in re.split(r"[,;|]", line) if t.strip()]
    fields["tags"] = list(set(hashtag_tags))[:20]

    # --- References: URLs ---
    urls = re.findall(r"https?://[^\s\)\"\'<>]+", text)
    fields["references"] = list(set(urls))[:30]

    # --- Malware families: known names + "malware:" labels ---
    known_malware = [
        "ryuk", "emotet", "cobalt strike", "mimikatz", "wannacry",
        "notpetya", "lockbit", "revil", "darkside", "conti", "trickbot",
        "qakbot", "formbook", "remcos", "njrat", "asyncrat",
    ]
    text_lower = text.lower()
    found_malware = [m for m in known_malware if m in text_lower]

    # Also look for "malware family: X" patterns
    family_matches = re.findall(
        r"(?:malware\s+famil(?:y|ies)|malware)\s*[:\-]\s*([^\n,\.]{3,40})",
        text, re.IGNORECASE)
    found_malware += [m.strip() for m in family_matches]
    fields["malware_families"] = list(set(found_malware))[:10]

    # --- Targeted countries ---
    country_names = [
        "united states", "china", "russia", "iran", "north korea",
        "ukraine", "germany", "france", "india", "uk", "taiwan",
        "south korea", "japan", "israel", "brazil", "australia",
    ]
    found_countries = [c for c in country_names if c in text_lower]

    # Also grab 2-letter country codes after "targeting" / "target:"
    target_matches = re.findall(
        r"(?:target(?:ing|s|ed)|victim)\s+[:\-]?\s*([A-Z]{2,3})\b",
        text)
    found_countries += [m.lower() for m in target_matches]
    fields["targeted_countries"] = list(set(found_countries))[:20]

    # --- Industries ---
    industry_keywords = {
        "healthcare":             ["hospital", "medical", "healthcare", "clinic", "patient"],
        "finance":                ["bank", "financial", "finance", "payment", "swift"],
        "government":             ["government", "agency", "federal", "ministry", "cisa"],
        "technology":             ["software", "technology", "it", "cloud", "server"],
        "critical infrastructure":["infrastructure", "energy", "utility", "grid", "ics"],
        "education":              ["university", "school", "education", "academic"],
        "telecommunications":     ["telecom", "carrier", "isp", "5g", "routing"],
    }
    found_industries = []
    for industry, keywords in industry_keywords.items():
        if any(kw in text_lower for kw in keywords):
            found_industries.append(industry)
    fields["industries"] = found_industries

    # --- Timestamps ---
    # ISO format: 2023-04-12T08:23:00Z
    iso_dates = re.findall(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})?",
        text)
    # Human format: April 12, 2023 or 12/04/2023
    human_dates = re.findall(
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b",
        text, re.IGNORECASE)

    all_dates = iso_dates
    if not all_dates and human_dates:
        # Convert human dates to ISO approximation
        for d in human_dates[:2]:
            try:
                parsed = datetime.strptime(d.strip(), "%B %d, %Y")
                all_dates.append(parsed.strftime("%Y-%m-%dT00:00:00Z"))
            except Exception:
                pass

    if all_dates:
        fields["created"]  = all_dates[0]
        fields["modified"] = all_dates[-1]

    # --- IOC count estimate ---
    ip_count   = len(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text))
    hash_count = len(re.findall(r"\b[0-9a-fA-F]{32,64}\b", text))
    cve_count  = len(re.findall(r"CVE-\d{4}-\d+", text))
    dom_count  = len(re.findall(r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,6}\b", text))
    fields["indicators_count"] = ip_count + hash_count + cve_count + dom_count

    return fields


# ── Full parse (text + structured) ───────────────────────────────────

def parse_pdf(path):
    """
    Full PDF parse: returns both raw text and structured fields.
    Use this instead of extract_pdf_text() when you need features.

    Returns:
        tuple: (text: str, structured: dict)
    """
    text       = extract_pdf_text(path)
    structured = extract_structured_fields(text)
    return text, structured


# ── Internal helpers ──────────────────────────────────────────────────

def _clean_text(text):
    """Remove junk characters and normalize whitespace."""
    if not text:
        return ""
    # Remove null bytes and control characters (except newline/tab)
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", " ", text)
    # Collapse multiple spaces (but keep newlines for summary splitting)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
