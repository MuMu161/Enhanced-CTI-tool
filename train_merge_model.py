"""
train_merge_model.py — CRITICAL MISSING PIECE

The paper manually annotated 750 CTI pairs and trained the IICM
Random Forest on the delta features. This script shows you how to:

1. Load your CTI database
2. Generate pairs (similar + dissimilar)
3. Compute delta features
4. Train and evaluate the Random Forest

HOW TO RUN:
    python train_merge_model.py

WHAT YOU NEED:
    - data/cti_train.csv  (your annotated CTI entries)
    - At least 50+ entries in data/database.json  (to form pairs)
    - OR: Manually labelled pairs in data/annotated_pairs.json
"""

import json
import random
import numpy as np
from sklearn.model_selection import train_test_split

import sys, os
sys.path.append(os.path.dirname(__file__))

from src.feature_extractor import FeatureExtractor
from src.similarity_engine  import SimilarityEngine
from src.merge_model        import MergeDecisionModel


# ── Config ────────────────────────────────────────────────────────────
N_PAIRS        = 750    # paper used 750 annotated pairs
MERGE_THRESHOLD = 0.85  # cosine above this → candidate for merge label=1
VAL_SPLIT       = 0.4   # paper: 6:4 train/val


def load_cti_entries(path="data/database.json"):
    """Load existing database entries."""
    with open(path) as f:
        entries = json.load(f)
    print(f"Loaded {len(entries)} CTI entries.")
    return entries


def generate_pairs_auto(entries, n_pairs=750, seed=42):
    """
    Automatically generate similar/dissimilar pairs using cosine similarity.
    
    In the paper, pairs were MANUALLY annotated by security analysts.
    This function gives you a starting point, but you should review and
    correct the labels for best results (especially edge cases).
    
    Label 1 = should merge (very similar CTI entries)
    Label 0 = store separately (different incidents)
    """
    sim_engine = SimilarityEngine()
    feat_ext   = FeatureExtractor()
    random.seed(seed)

    X, y = [], []
    entries_with_summary = [e for e in entries if e.get("summary")]

    if len(entries_with_summary) < 10:
        raise ValueError("Need at least 10 entries to generate pairs. "
                         "Add more CTI to database.json first.")

    summaries = [e["summary"] for e in entries_with_summary]
    features  = [e.get("features") or feat_ext.extract(e.get("summary",""))
                 for e in entries_with_summary]

    n = len(entries_with_summary)
    attempts = 0

    while len(X) < n_pairs and attempts < n_pairs * 10:
        attempts += 1
        i, j = random.sample(range(n), 2)

        score = sim_engine.compute_similarity(summaries[i], summaries[j])
        delta = feat_ext.delta_features(features[i], features[j])

        # Auto-label: high similarity → merge candidate
        # NOTE: You should manually review and correct these labels!
        label = 1 if score >= MERGE_THRESHOLD else 0

        X.append(delta)
        y.append(label)

    print(f"Generated {len(X)} pairs "
          f"({sum(y)} merge / {len(y)-sum(y)} new CTI)")
    return X, y


def load_manual_annotations(path="data/annotated_pairs.json"):
    """
    Load manually annotated pairs if you have them.
    
    Format of annotated_pairs.json:
    [
        {
            "cti_a": "summary text of first CTI",
            "cti_b": "summary text of second CTI",
            "label": 1   // 1=merge, 0=new
        },
        ...
    ]
    """
    feat_ext   = FeatureExtractor()
    sim_engine = SimilarityEngine()

    with open(path) as f:
        pairs = json.load(f)

    X, y = [], []
    for pair in pairs:
        fa = feat_ext.extract(pair["cti_a"])
        fb = feat_ext.extract(pair["cti_b"])
        delta = feat_ext.delta_features(fa, fb)
        X.append(delta)
        y.append(int(pair["label"]))

    print(f"Loaded {len(X)} manually annotated pairs.")
    return X, y


def main():
    print("=" * 60)
    print("EnhanceCTI — IICM Training Script")
    print("=" * 60)

    # --- Load data ---
    manual_path = "data/annotated_pairs.json"

    if os.path.exists(manual_path):
        print(f"\n✓ Found manual annotations at {manual_path}")
        X, y = load_manual_annotations(manual_path)
    else:
        print(f"\n⚠ No manual annotations found at {manual_path}")
        print("  Falling back to auto-generated pairs from database.json")
        print("  (Recommended: manually annotate pairs for best F1!)")
        entries = load_cti_entries()
        X, y = generate_pairs_auto(entries, n_pairs=N_PAIRS)

    # --- Train/val split (paper: 6:4) ---
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=VAL_SPLIT,
        random_state=42,
        stratify=y if len(set(y)) > 1 else None
    )
    print(f"\nTrain: {len(X_train)} | Val: {len(X_val)}")

    # --- Train ---
    model = MergeDecisionModel()
    model.train(X_train, y_train)

    # --- Evaluate ---
    print("\n── Validation Results ──")
    results = model.evaluate(X_val, y_val)

    if results:
        print(f"\nPaper target: F1=0.97, AUC=1.00")
        print(f"Your result:  F1={results['f1']:.4f}, AUC={results['auc']:.4f}")

    print("\n✅ merge_model.pkl saved to model/")
    print("   Restart your Flask app to use the trained model.")


if __name__ == "__main__":
    main()
