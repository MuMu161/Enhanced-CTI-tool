import json
from src.feature_extractor import FeatureExtractor
from src.industry_classifier import IndustryClassifier
from src.similarity_engine import SimilarityEngine
from src.merge_model import MergeDecisionModel
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))


class EnhanceCTIPipeline:

    def __init__(self):

        self.industry_model = IndustryClassifier()
        self.feature_extractor = FeatureExtractor()
        self.similarity_engine = SimilarityEngine()

        self.merge_model = MergeDecisionModel()
        self.merge_model.load()

        self.database_path = "data/database.json"

        try:
            with open(self.database_path) as f:
                self.database = json.load(f)
        except:
            self.database = []

    # ---------------------------------------------------
    # 🧠 Smart Summary Extraction (NEW)
    # ---------------------------------------------------
    def generate_summary(self, text, max_lines=25):

        if not text:
            return ""

        lines = [line.strip() for line in text.split("\n") if line.strip()]

        filtered_lines = []
        for line in lines:
            lower = line.lower()
            if "tlp:" in lower:
                continue
            if "for more information" in lower:
                continue
            if "contact" in lower:
                continue
            filtered_lines.append(line)

        return " ".join(filtered_lines[:max_lines])

    # ---------------------------------------------------
    # 🔎 Explainable Classification
    # ---------------------------------------------------
    def explain_classification(self, text, industries):

        explanations = {}

        keywords_map = {
            "Healthcare": ["hospital", "patient", "medical", "clinic"],
            "Finance": ["bank", "financial", "payment", "billing"],
            "Government": ["agency", "federal", "cisa", "policy"],
            "Technology": ["software", "server", "system", "network"],
            "Critical Infrastructure": ["infrastructure", "utility", "energy", "grid"],
        }

        text_lower = text.lower()

        for industry in industries:
            matched = []
            for word in keywords_map.get(industry, []):
                if word in text_lower:
                    matched.append(word)

            explanations[industry] = matched

        return explanations

    # ---------------------------------------------------
    # 🚀 Main Processing (UPDATED WITH SIRD + IICM)
    # ---------------------------------------------------
    def process(self, text):

        # 🔥 Use summary instead of full text
        summary = self.generate_summary(text)

        # 1️⃣ Industry Classification (DistilBERT on summary)
        industries, confidences = self.industry_model.predict(summary)

        # 2️⃣ Extract structured features (full text for richer features)
        features_new = self.feature_extractor.extract(text)

        # 3️⃣ SentenceBERT similarity comparison (SIRD stage)
        best_match = None
        best_score = 0.0

        for old_entry in self.database:

            similarity = self.similarity_engine.compute_similarity(
                summary, old_entry["summary"]
            )

            if similarity > best_score:
                best_score = similarity
                best_match = old_entry

        # 🔥 SIRD Relevance Threshold
        RELEVANCE_THRESHOLD = 0.80

        duplicate_flag = False
        decision_label = "NEW CTI"

        # Only proceed if relevant
        if best_match and best_score >= RELEVANCE_THRESHOLD:

            duplicate_flag = True  # relevant enough to check merge

            features_old = best_match["features"]

            feature_vector = [
                best_score,
                abs(features_new["word_count"] - features_old["word_count"]),
                abs(features_new["cve_count"] - features_old["cve_count"]),
                abs(features_new["noun_density"] - features_old["noun_density"]),
            ]

            # IICM Decision
            decision = self.merge_model.predict(feature_vector)

            if decision == 1:
                decision_label = "MERGED"
            else:
                decision_label = "NEW CTI"

        else:
            decision_label = "NEW CTI"

        # 4️⃣ Store new entry if not merged
        if decision_label != "MERGED":

            self.database.append({
                "summary": summary,
                "features": features_new
            })

            with open(self.database_path, "w") as f:
                json.dump(self.database, f)

        # 5️⃣ Explainable AI
        explanations = self.explain_classification(text, industries)

        return {
            "decision": decision_label,
            "industries": industries,
            "classification_scores": confidences,
            "similarity_score": round(best_score, 3),
            "duplicate_detected": duplicate_flag,
            "explanations": explanations
        }
