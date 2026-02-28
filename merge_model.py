"""
merge_model.py — IICM component (Section 4.5.2)

Paper: Random Forest trained on 750 manually annotated pairs.
       Input: Δf = [|f_k(x_i) − f_k(x_j)|] for all 19 features.
       Output: 1 = merge, 0 = store separately.
       F1 = 0.97, AUC = 1.00 on 6:4 train/val split.

Improvement over original:
- train() now accepts feature-difference vectors properly
- Added evaluate() method for checking model quality
- Added feature importance reporting
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, roc_auc_score


class MergeDecisionModel:

    # Paper: n_estimators chosen for robustness on small dataset (750 pairs)
    N_ESTIMATORS = 200

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=self.N_ESTIMATORS,
            random_state=42,            # reproducibility
            class_weight="balanced",    # handles label imbalance
        )
        self._trained = False

    # ------------------------------------------------------------------
    # Training (Section 4.5.2)
    # ------------------------------------------------------------------

    def train(self, X, y):
        """
        Train on delta-feature vectors.

        Args:
            X: list of 19-element delta vectors (|f_k(x_i) − f_k(x_j)|)
            y: labels — 1 = should merge, 0 = store separately
        """
        self.model.fit(X, y)
        self._trained = True
        os.makedirs("model", exist_ok=True)
        joblib.dump(self.model, "model/merge_model.pkl")
        print(f"[MergeModel] Trained on {len(X)} samples. Saved.")

    def evaluate(self, X_val, y_val):
        """
        Evaluate on validation set. Paper reports F1=0.97, AUC=1.00.
        """
        if not self._is_ready():
            print("[MergeModel] Not trained — cannot evaluate.")
            return

        y_pred = self.model.predict(X_val)
        y_prob = self.model.predict_proba(X_val)[:, 1]

        print(classification_report(y_val, y_pred,
              target_names=["New CTI", "Merge"]))
        print(f"AUC: {roc_auc_score(y_val, y_prob):.4f}")

        return {
            "f1":  f1_score(y_val, y_pred, average="weighted"),
            "auc": roc_auc_score(y_val, y_prob),
        }

    def feature_importances(self, feature_names=None):
        """Return feature importance scores (Gini impurity reduction)."""
        if not self._is_ready():
            return {}
        importances = self.model.feature_importances_
        if feature_names:
            return dict(zip(feature_names, importances))
        return dict(enumerate(importances))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, delta_features):
        """
        Predict merge decision for one pair.

        Args:
            delta_features (list[float]): 19-element delta vector.

        Returns:
            int: 1 = merge, 0 = store as new CTI.
        """
        if not self._is_ready():
            print("[MergeModel] Not trained — defaulting to NEW CTI.")
            return 0

        return int(self.model.predict([delta_features])[0])

    def predict_proba(self, delta_features):
        """Return merge probability (useful for UI confidence display)."""
        if not self._is_ready():
            return 0.0
        return float(self.model.predict_proba([delta_features])[0][1])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self):
        path = "model/merge_model.pkl"
        if os.path.exists(path):
            self.model    = joblib.load(path)
            self._trained = True
            print("[MergeModel] Loaded from", path)
        else:
            print("[MergeModel] ⚠ merge_model.pkl not found. "
                  "Run training first (see train_merge_model.py).")

    def _is_ready(self):
        return self._trained and hasattr(self.model, "estimators_")
