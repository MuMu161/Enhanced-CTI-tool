import os
import joblib
from sklearn.ensemble import RandomForestClassifier


class MergeDecisionModel:

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=200)

    def train(self, X, y):
        self.model.fit(X, y)
        os.makedirs("model", exist_ok=True)
        joblib.dump(self.model, "model/merge_model.pkl")

    def load(self):
        model_path = "model/merge_model.pkl"
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            print("⚠ merge_model.pkl not found. Model not loaded.")

    def predict(self, features):
        if not hasattr(self.model, "estimators_"):
            return 0  # Not trained → treat as NEW

        return self.model.predict([features])[0]
