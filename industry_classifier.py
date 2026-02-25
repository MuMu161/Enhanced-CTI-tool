import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import os


class IndustryClassifier:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_path = os.path.join("model", "distilbert_cti")
        label_path = os.path.join("model", "label_map.pt")

        # If model files are missing, disable classifier safely
        if not os.path.exists(model_path) or not os.path.exists(label_path):
            print("⚠ Industry model files not found. Classifier disabled.")
            self.model = None
            self.tokenizer = None
            self.label_map = {}
            return

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_path
        ).to(self.device)

        self.label_map = torch.load(label_path, map_location=self.device)
        self.model.eval()

    def predict(self, text, threshold=0.4):

        if self.model is None:
            return [], {}

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.sigmoid(outputs.logits)[0]

        industries = []
        confidences = {}

        for i, p in enumerate(probs):
            if p.item() > threshold:
                industries.append(self.label_map[i])
                confidences[self.label_map[i]] = float(p.item())

        return industries, confidences
