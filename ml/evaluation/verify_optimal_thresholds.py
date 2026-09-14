import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_dir = Path("models/bert_multilabel")
test_csv = Path("ml/preprocessing/test_data.csv")

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.eval()

test_df = pd.read_csv(test_csv)
labels = ["DV", "SH", "ST", "CA", "WH", "OV"]

with open(model_dir / "optimal_thresholds.json") as f:
    thresholds = json.load(f)

all_logits = []
all_labels = []

for _, r in test_df.iterrows():
    text = str(r["narrative_text"])
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
        all_logits.append(out.logits.squeeze(0).cpu().numpy())
    all_labels.append([float(r[col]) for col in labels])

all_logits = np.vstack(all_logits)
all_labels = np.vstack(all_labels).astype(int)

probs = 1.0 / (1.0 + np.exp(-all_logits))
preds = np.zeros_like(probs, dtype=int)

for idx, col in enumerate(labels):
    t = float(thresholds.get(col, 0.5))
    preds[:, idx] = (probs[:, idx] >= t).astype(int)

exact_match_acc = float((preds == all_labels).all(axis=1).mean())
elementwise_acc = float((preds == all_labels).mean())

tp = float(np.sum((all_labels == 1) & (preds == 1)))
fp = float(np.sum((all_labels == 0) & (preds == 1)))
fn = float(np.sum((all_labels == 1) & (preds == 0)))

micro_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
micro_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

macro_precs, macro_recs, macro_f1s = [], [], []
for idx, col in enumerate(labels):
    tp_k = float(np.sum((all_labels[:, idx] == 1) & (preds[:, idx] == 1)))
    fp_k = float(np.sum((all_labels[:, idx] == 0) & (preds[:, idx] == 1)))
    fn_k = float(np.sum((all_labels[:, idx] == 1) & (preds[:, idx] == 0)))

    p_k = tp_k / (tp_k + fp_k) if (tp_k + fp_k) > 0 else 0.0
    r_k = tp_k / (tp_k + fn_k) if (tp_k + fn_k) > 0 else 0.0
    f_k = (2 * p_k * r_k) / (p_k + r_k) if (p_k + r_k) > 0 else 0.0

    macro_precs.append(p_k)
    macro_recs.append(r_k)
    macro_f1s.append(f_k)

print("==================================================")
print("INDEPENDENT RE-EVALUATION AT OPTIMAL THRESHOLDS:")
print("==================================================")
print(f"Exact Match Accuracy  : {exact_match_acc:.4f} ({exact_match_acc*100:.2f}%)")
print(f"Elementwise Accuracy  : {elementwise_acc:.4f} ({elementwise_acc*100:.2f}%)")
print(f"Micro Precision       : {micro_p:.4f}")
print(f"Micro Recall          : {micro_r:.4f}")
print(f"Micro F1 Score        : {micro_f1:.4f}")
print(f"Macro Precision       : {np.mean(macro_precs):.4f}")
print(f"Macro Recall          : {np.mean(macro_recs):.4f}")
print(f"Macro F1 Score        : {np.mean(macro_f1s):.4f}")
print("==================================================")
