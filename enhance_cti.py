import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# Confidence-aware sector override (POLICY LAYER)
# =====================================================

SECTOR_KEYWORDS = {
    "Healthcare": [
        "healthcare", "hospital", "hospitals",
        "public health", "hph", "medical",
        "clinic", "clinical", "patient",
        "care provider", "health system"
    ],
    "Finance": [
        "finance", "financial", "bank", "banking",
        "credit card", "payment", "transaction",
        "fintech", "atm", "loan"
    ],
    "Government": [
        "government", "federal", "public sector",
        "agency", "ministry", "state-sponsored",
        "defense", "military", "law enforcement"
    ],
    "Energy": [
        "energy", "power grid", "electric",
        "oil", "gas", "pipeline", "utility",
        "generation", "substation"
    ],
    "Telecommunications": [
        "telecom", "telecommunications", "isp",
        "mobile network", "cellular",
        "5g", "network provider"
    ],
    "Education": [
        "education", "university", "college",
        "school", "campus", "academic institution"
    ],
    "Technology": [
        "software", "technology", "it company",
        "cloud service", "saas", "platform",
        "service provider", "vendor"
    ]
}

def sector_override(text, predicted_label, confidence, threshold=0.35):
    """
    Apply sector override ONLY when:
    - Model confidence is low
    - CTI text explicitly mentions a sector
    """
    if confidence >= threshold:
        return predicted_label

    text_lower = text.lower()

    for sector, keywords in SECTOR_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return sector

    return predicted_label


# =====================================================
# Industry Classifier (DistilBERT CORE)
# =====================================================
class IndustryClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = DistilBertTokenizerFast.from_pretrained("model/distilbert_cti")
        self.model = DistilBertForSequenceClassification.from_pretrained(
            "model/distilbert_cti"
        ).to(self.device)

        self.label_map = torch.load("model/label_map.pt")
        self.model.eval()

        # Temperature scaling for confidence calibration
        self.temperature = 0.7

    def predict(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits / self.temperature, dim=1)[0]
        idx = torch.argmax(probs).item()

        return self.label_map[idx], float(probs[idx])


# =====================================================
# Sentence-BERT Similarity Engine
# =====================================================
class SimilarityEngine:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.corpus = []

    def analyze(self, text):
        if not self.corpus:
            self.corpus.append(text)
            return "New CTI", 0.0

        embeddings = self.model.encode(self.corpus + [text])
        scores = cosine_similarity([embeddings[-1]], embeddings[:-1])[0]
        max_score = max(scores)

        self.corpus.append(text)

        if max_score > 0.85:
            return "Duplicate CTI", max_score
        elif max_score > 0.50:
            return "Related CTI", max_score
        else:
            return "New CTI", max_score


# =====================================================
# Load CTI Reports
# =====================================================
def load_cti_reports(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    return [
        block.strip()
        for block in content.split("=== CTI REPORT ===")
        if block.strip()
    ]


# =====================================================
# Main Execution
# =====================================================
if __name__ == "__main__":

    reports = load_cti_reports("data/cti_overviews.txt")

    classifier = IndustryClassifier()
    similarity_engine = SimilarityEngine()

    print("\n====== EnhanceCTI Output ======\n")

    for i, report in enumerate(reports, start=1):
        industry, confidence = classifier.predict(report)

        # 🔹 Policy override applied here
        industry = sector_override(report, industry, confidence)

        relation, score = similarity_engine.analyze(report)

        print(f"[CTI REPORT {i}]")
        print("Industry     :", industry)
        print("Confidence   :", round(confidence, 3))
        print("Relation     :", relation)
        if score > 0:
            print("Similarity   :", round(score, 3))
        print("-" * 50)
        
