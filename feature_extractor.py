"""
feature_extractor.py — Improved to match all 19 paper features from Table 3.

Paper features implemented:
1.  word_count                  ✓
2.  attack_ids_count            ✓ (MITRE ATT&CK IDs like T1566.001)
3.  tags_count                  ✓ (from structured field)
4.  export_count                ✓ (from structured field)
5.  targeted_countries_count    ✓ (from structured field)
6.  industries_count            ✓ (from structured field)
7.  subscriber_count            ✓ (from structured field)
8.  malware_families_count      ✓ (from structured field)
9.  indicator_type_counts.5     ✓ (file hash IoC count)
10. noun_density                ✓ (via NLTK POS, paper uses BERT-NER)
11. days_since_modified         ✓ (from structured field)
12. verb_density                ✓ (via NLTK POS, paper uses BERT-NER)
13. indicators_count            ✓
14. indicator_type_counts.2     ✓ (domain IoC count)
15. indicator_type_counts.10    ✓ (CVE IoC count)
16. indicator_type_counts.1     ✓ (IP IoC count)
17. indicator_count             ✓ (alias of indicators_count)
18. days_since_created          ✓ (from structured field)
19. references_count            ✓ (from structured field)

NOTE: The paper's Security Lexical Density Transformer uses BERT-NER
for noun/verb density. We use NLTK POS tagging as a practical
approximation (BERT-NER requires the full MITRE+Wikipedia trained model).
"""

import re
import nltk
from datetime import datetime, timezone


class FeatureExtractor:

    def __init__(self):
        nltk.download("punkt",                          quiet=True)
        nltk.download("punkt_tab",                     quiet=True)
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, text, structured=None):
        """
        Extract all 19 paper features from raw CTI text.

        Args:
            text (str): Raw CTI text (from PDF or threat report body).
            structured (dict | None): Parsed structured fields from the
                CTI platform (tags, references, IoCs, timestamps, …).
                When None, we fall back to text-only extraction for the
                structured fields (best-effort via regex).

        Returns:
            dict: All 19 features aligned with Table 3 of the paper.
        """
        if not text:
            return self._empty_features()

        # --- text-level features (always available) ---
        tokens    = nltk.word_tokenize(text)
        pos_tags  = nltk.pos_tag(tokens, lang="eng") if tokens else []
        n_tokens  = len(tokens) if tokens else 1

        noun_count = sum(1 for _, pos in pos_tags if pos.startswith("NN"))
        verb_count = sum(1 for _, pos in pos_tags if pos.startswith("VB"))

        # --- IoC regex extraction from raw text ---
        ip_matches   = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text)
        hash_matches = re.findall(
            r"\b[0-9a-fA-F]{32,64}\b", text)             # MD5/SHA1/SHA256
        domain_matches = re.findall(
            r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b", text)
        cve_matches  = re.findall(r"CVE-\d{4}-\d+", text)
        attack_ids   = re.findall(
            r"\bT\d{4}(?:\.\d{3})?\b", text)             # T1566 / T1566.001

        # Country codes / names (simple heuristic — 2-letter uppercase words)
        country_mentions = re.findall(r"\b[A-Z]{2,3}\b", text)
        country_est = min(len(set(country_mentions)), 50)

        total_indicators = (len(ip_matches) + len(hash_matches)
                            + len(domain_matches) + len(cve_matches))

        # --- structured field overrides (if available) ---
        s = structured or {}

        tags_count               = len(s.get("tags", []))
        export_count             = int(s.get("export_count", 0))
        targeted_countries_count = len(s.get("targeted_countries", [])) or country_est
        industries_count         = len(s.get("industries", []))
        subscriber_count         = int(s.get("subscriber_count", 0))
        malware_families_count   = len(s.get("malware_families", []))
        references_count         = len(s.get("references", []))
        indicators_count         = int(s.get("indicators_count", total_indicators))

        # Temporal features
        days_since_created  = self._days_since(s.get("created"))
        days_since_modified = self._days_since(s.get("modified"))

        features = {
            # --- text features ---
            "word_count":               len(tokens),
            "noun_density":             noun_count / n_tokens,
            "verb_density":             verb_count / n_tokens,

            # --- IoC type counts (match paper's indicator_type_counts.*) ---
            "indicator_type_counts.1":  len(ip_matches),       # IP
            "indicator_type_counts.2":  len(domain_matches),   # Domain
            "indicator_type_counts.5":  len(hash_matches),     # File hash
            "indicator_type_counts.10": len(cve_matches),      # CVE
            "indicators_count":         indicators_count,
            "indicator_count":          indicators_count,       # paper alias

            # --- MITRE ATT&CK IDs ---
            "attack_ids_count":         len(attack_ids),

            # --- structured / metadata features ---
            "tags_count":               tags_count,
            "export_count":             export_count,
            "targeted_countries_count": targeted_countries_count,
            "industries_count":         industries_count,
            "subscriber_count":         subscriber_count,
            "malware_families_count":   malware_families_count,
            "references_count":         references_count,

            # --- temporal features ---
            "days_since_created":       days_since_created,
            "days_since_modified":      days_since_modified,
        }

        return features

    def delta_features(self, f_new, f_old):
        """
        Compute |Δf_k| = |f_k(x_new) − f_k(x_old)| for all 19 features.
        Used as input to the IICM Random Forest classifier.

        Returns:
            list[float]: 19-element feature-difference vector.
        """
        keys = self._feature_keys()
        return [abs(f_new.get(k, 0) - f_old.get(k, 0)) for k in keys]

    def to_vector(self, features):
        """Return feature dict as an ordered list (for model input)."""
        return [features.get(k, 0) for k in self._feature_keys()]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _empty_features(self):
        return {k: 0 for k in self._feature_keys()}

    @staticmethod
    def _feature_keys():
        """Canonical ordering of the 19 paper features."""
        return [
            "word_count",
            "attack_ids_count",
            "tags_count",
            "export_count",
            "targeted_countries_count",
            "industries_count",
            "subscriber_count",
            "malware_families_count",
            "indicator_type_counts.5",
            "noun_density",
            "days_since_modified",
            "verb_density",
            "indicators_count",
            "indicator_type_counts.2",
            "indicator_type_counts.10",
            "indicator_type_counts.1",
            "indicator_count",
            "days_since_created",
            "references_count",
        ]

    @staticmethod
    def _days_since(timestamp_str):
        """Convert ISO timestamp string to days-since-epoch integer."""
        if not timestamp_str:
            return 0
        try:
            dt = datetime.fromisoformat(
                timestamp_str.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            return max(0, (now - dt).days)
        except Exception:
            return 0
