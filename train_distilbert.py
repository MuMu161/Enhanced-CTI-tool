import os
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset

# =====================================================
# Resolve Project Paths (CRITICAL FIX)
# =====================================================

# Absolute path to project root (EnhanceCTI/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
OUTPUT_MODEL_DIR = os.path.join(MODEL_DIR, "distilbert_cti")

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

# =====================================================
# Load Training Dataset
# =====================================================

data_path = os.path.join(DATA_DIR, "cti_train.csv")

df = pd.read_csv(data_path)

label_encoder = LabelEncoder()
df["label_id"] = label_encoder.fit_transform(df["label"])

# Save label map for inference
label_map = dict(enumerate(label_encoder.classes_))
torch.save(label_map, os.path.join(MODEL_DIR, "label_map.pt"))

# Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df[["text", "label_id"]])

# =====================================================
# Tokenization
# =====================================================

tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label_id", "labels")
dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

# =====================================================
# Load Model
# =====================================================

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label_encoder.classes_)
)

# =====================================================
# Training Configuration
# =====================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_MODEL_DIR,
    per_device_train_batch_size=4,
    num_train_epochs=4,
    learning_rate=2e-5,
    logging_steps=5,
    save_steps=500,
    save_total_limit=1
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# =====================================================
# Train Model
# =====================================================

trainer.train()

# =====================================================
# Save Model & Tokenizer
# =====================================================

model.save_pretrained(OUTPUT_MODEL_DIR)
tokenizer.save_pretrained(OUTPUT_MODEL_DIR)

print("\n✅ DistilBERT fine-tuning completed successfully.")
print(f"📁 Model saved at: {OUTPUT_MODEL_DIR}")
print(f"📁 Label map saved at: {os.path.join(MODEL_DIR, 'label_map.pt')}")

