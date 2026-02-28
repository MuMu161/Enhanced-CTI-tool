"""
pipeline.py — Improved EnhanceCTI Pipeline

Key improvements over original:
1. Uses all 19 paper features in IICM delta vector (was only 4)
2. Correct SIRD similarity normalization (paper divides by 5 from STS scale)
3. Structured field passing to FeatureExtractor
4. Better database schema (stores all 19 features + metadata)
5. STIX export hook (stub ready to extend)
6. Cleaner separation of SIRD and IICM stages
"""

import json
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.feature_extractor import FeatureExtractor
from src.industry_classifier import IndustryClassifier
from src.similarity_engine import SimilarityEngine
from src.merge_model import MergeDecisionModel


class EnhanceCTIPipeline:

    # SIRD threshold from paper Section 4.5.1
    SIRD_THRESHOLD = 0.80

    def __init__(self):
        self.industry_model    = IndustryClassifier()
        self.feature_extractor = FeatureExtractor()
        self.similarity_engine = SimilarityEngine()
        self.merge_model       = MergeDecisionModel()
        self.merge_model.load()

        self.database_path = "data/database.json"
        self._load_database()

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------

    def _load_database(self):
        try:
            with open(self.database_path) as f:
                self.database = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.database = []

    def _save_database(self):
        os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
        with open(self.database_path, "w") as f:
            json.dump(self.database, f, indent=2)

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    def generate_summary(self, text, max_lines=25):
        """
        Smart summary: strip boilerplate, keep threat-relevant lines.
        (Same logic as before — keeps noise low for SentenceBERT.)
        """
        if not text:
            return ""
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        filtered = [
            l for l in lines
            if not any(kw in l.lower()
                       for kw in ["tlp:", "for more information", "contact us",
                                  "unsubscribe", "copyright"])
        ]
        return " ".join(filtered[:max_lines])

    def explain_classification(self, text, industries):
        """
        Lightweight explainability: which keywords triggered each industry label.
        (EnhanceCTI paper does not include this — it is an added XAI bonus.)
        """
        keywords_map = {
            "Healthcare":             ["hospital", "patient", "medical", "clinic", "ehr", "hipaa"],
            "Finance":                ["bank", "financial", "payment", "billing", "swift", "fraud"],
            "Government":             ["agency", "federal", "cisa", "government", "ministry", "policy"],
            "Technology":             ["software", "server", "system", "network", "cloud", "api"],
            "Critical Infrastructure":["infrastructure", "utility", "energy", "grid", "scada", "ics"],
            "Education":              ["university", "school", "student", "academic", "research"],
            "Telecommunications":    ["telecom", "carrier", "routing", "metadata", "5g", "isp"],
            "Others":                 [],
        }
        text_lower = text.lower()
        return {
            industry: [w for w in keywords_map.get(industry, []) if w in text_lower]
            for industry in industries
        }

    # ------------------------------------------------------------------
    # SIRD — Security Information Relevance Detector (Section 4.5.1)
    # ------------------------------------------------------------------

    def _sird_compare(self, summary_new):
        """
        Compare new CTI summary against every DB entry using SentenceBERT.

        Paper: normalized_score = cosine_sim / 5  (STS benchmark is 0–5).
        Our SentenceTransformer already returns cosine in [0,1], so we
        skip the /5 division but keep the 0.80 threshold as-is.

        Returns:
            (best_match_entry | None, best_score float)
        """
        best_match = None
        best_score = 0.0

        for entry in self.database:
            score = self.similarity_engine.compute_similarity(
                summary_new, entry.get("summary", ""))
            if score > best_score:
                best_score = score
                best_match = entry

        return best_match, best_score

    # ------------------------------------------------------------------
    # IICM — Information Integration Classifier (Section 4.5.2)
    # ------------------------------------------------------------------

    def _iicm_decide(self, features_new, features_old):
        """
        Compute Δf_k = |f_k(new) − f_k(old)| for all 19 features,
        then ask the Random Forest whether to merge (1) or not (0).
        """
        delta_vector = self.feature_extractor.delta_features(
            features_new, features_old)
        return self.merge_model.predict(delta_vector)   # 1 = merge, 0 = new

    # ------------------------------------------------------------------
    # Main pipeline (Section 4.1 — Overall Workflow)
    # ------------------------------------------------------------------

    def process(self, text, structured=None):
        """
        Run the full 4-stage EnhanceCTI pipeline.

        Args:
            text (str): Raw CTI text (from PDF, blog, report).
            structured (dict | None): Optional parsed metadata fields
                (tags, references, malware_families, timestamps, …).

        Returns:
            dict: Decision + all intermediate outputs for the UI.
        """

        # ── Stage 1 preprocessing ─────────────────────────────────────
        summary = self.generate_summary(text)

        # ── Stage 2: Specified Industry Filter ────────────────────────
        industries, confidences = self.industry_model.predict(summary)

        # If no relevant industry found → DROP (not stored at all)
        if not industries:
            return {
                "decision":              "DROPPED",
                "reason":                "No relevant industry detected",
                "industries":            [],
                "classification_scores": {},
                "similarity_score":      0.0,
                "duplicate_detected":    False,
                "explanations":          {},
            }

        # ── Stage 3: Feature Extraction (all 19 paper features) ───────
        features_new = self.feature_extractor.extract(text, structured)

        # ── Stage 4a: SIRD — cosine similarity search ─────────────────
        best_match, best_score = self._sird_compare(summary)

        decision_label   = "NEW CTI"
        duplicate_flag   = False

        if best_match and best_score >= self.SIRD_THRESHOLD:
            duplicate_flag = True   # relevant enough to check merge

            # ── Stage 4b: IICM — merge or store separately ────────────
            features_old = best_match.get("features", {})
            merge_label  = self._iicm_decide(features_new, features_old)

            if merge_label == 1:
                decision_label = "MERGED"
            else:
                decision_label = "NEW CTI"
        # else: score < 0.80 → unrelated → NEW CTI (no merge check needed)

        # ── Persist new entry if not merged ───────────────────────────
        if decision_label != "MERGED":
            self.database.append({
                "summary":    summary,
                "features":   features_new,
                "industries": industries,
                "metadata":   structured or {},
            })
            self._save_database()

        # ── Explainability ────────────────────────────────────────────
        explanations = self.explain_classification(text, industries)

        return {
            "decision":              decision_label,
            "industries":            industries,
            "classification_scores": confidences,
            "similarity_score":      round(best_score, 4),
            "duplicate_detected":    duplicate_flag,
            "explanations":          explanations,
            # bonus fields for the UI
            "features_extracted":    features_new,
            "db_size_after":         len(self.database),
        }
