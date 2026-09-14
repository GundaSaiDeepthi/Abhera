import json
import logging
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("EvaluateBERT")


class EvaluationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, label_columns: list[str], max_length: int = 256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label_columns = label_columns
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["narrative_text"])
        labels = [float(row[col]) for col in self.label_columns]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(labels, dtype=torch.float)
        return item


def evaluate(threshold: float = 0.5):
    project_root = Path(__file__).resolve().parent.parent.parent
    prep_dir = project_root / "ml" / "preprocessing"
    model_dir = project_root / "models" / "bert_multilabel"

    test_csv = prep_dir / "test_data.csv"
    mapping_json = prep_dir / "label_mapping.json"

    if not model_dir.exists():
        logger.error(f"Trained model directory not found at: {model_dir}")
        sys.exit(1)

    if not test_csv.exists():
        logger.error(f"Held-out test dataset not found at: {test_csv}")
        sys.exit(1)

    logger.info(f"Loading trained BERT model and tokenizer from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    with open(mapping_json, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)

    label_columns = mapping_data["labels"]
    num_labels = len(label_columns)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluation Compute Device: {device}")
    model.to(device)
    model.eval()

    test_df = pd.read_csv(test_csv)
    logger.info(f"Loaded held-out test split containing {len(test_df)} samples.")

    test_dataset = EvaluationDataset(test_df, tokenizer, label_columns, max_length=256)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_logits = np.vstack(all_logits)
    all_labels = np.vstack(all_labels).astype(int)

    all_probs = 1.0 / (1.0 + np.exp(-all_logits))
    preds = (all_probs >= threshold).astype(int)

    exact_match_acc = float((preds == all_labels).all(axis=1).mean())
    elementwise_acc = float((preds == all_labels).mean())

    # Overall Micro Metrics
    tp = float(np.sum((all_labels == 1) & (preds == 1)))
    fp = float(np.sum((all_labels == 0) & (preds == 1)))
    fn = float(np.sum((all_labels == 1) & (preds == 0)))

    micro_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    micro_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    micro_f1 = (2 * micro_prec * micro_rec) / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

    # Per-Label & Macro Metrics
    per_label_metrics = []
    macro_precs, macro_recs, macro_f1s = [], [], []

    for idx, label in enumerate(label_columns):
        y_true = all_labels[:, idx]
        y_pred = preds[:, idx]

        tp_k = float(np.sum((y_true == 1) & (y_pred == 1)))
        fp_k = float(np.sum((y_true == 0) & (y_pred == 1)))
        fn_k = float(np.sum((y_true == 1) & (y_pred == 0)))

        p_k = tp_k / (tp_k + fp_k) if (tp_k + fp_k) > 0 else 0.0
        r_k = tp_k / (tp_k + fn_k) if (tp_k + fn_k) > 0 else 0.0
        f_k = (2 * p_k * r_k) / (p_k + r_k) if (p_k + r_k) > 0 else 0.0
        support = int(y_true.sum())

        macro_precs.append(p_k)
        macro_recs.append(r_k)
        macro_f1s.append(f_k)

        per_label_metrics.append({
            "label": label,
            "precision": float(p_k),
            "recall": float(r_k),
            "f1": float(f_k),
            "support": support,
        })

    per_label_df = pd.DataFrame(per_label_metrics)

    summary_results = {
        "evaluation_split": "test",
        "total_test_samples": len(test_df),
        "threshold": threshold,
        "exact_match_accuracy": float(exact_match_acc),
        "elementwise_accuracy": float(elementwise_acc),
        "micro_precision": float(micro_prec),
        "micro_recall": float(micro_rec),
        "micro_f1": float(micro_f1),
        "macro_precision": float(np.mean(macro_precs)),
        "macro_recall": float(np.mean(macro_recs)),
        "macro_f1": float(np.mean(macro_f1s)),
        "per_label_metrics": per_label_metrics,
    }

    results_json_path = model_dir / "evaluation_results.json"
    per_label_csv_path = model_dir / "evaluation_per_label.csv"

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)

    per_label_df.to_csv(per_label_csv_path, index=False)

    logger.info("==================================================")
    logger.info("BERT MULTI-LABEL MODEL TEST EVALUATION REPORT")
    logger.info("==================================================")
    logger.info(f"  Test Samples Evaluated: {len(test_df)}")
    logger.info(f"  Classification Threshold: {threshold}")
    logger.info(f"  Exact Match Subset Accuracy: {exact_match_acc:.4f}")
    logger.info(f"  Elementwise Binary Accuracy: {elementwise_acc:.4f}")
    logger.info(f"  Micro Precision: {micro_prec:.4f} | Micro Recall: {micro_rec:.4f} | Micro F1: {micro_f1:.4f}")
    logger.info(f"  Macro Precision: {np.mean(macro_precs):.4f} | Macro Recall: {np.mean(macro_recs):.4f} | Macro F1: {np.mean(macro_f1s):.4f}")
    logger.info("--------------------------------------------------")
    logger.info("PER-LABEL METRICS BREAKDOWN:")
    for m in per_label_metrics:
        logger.info(
            f"  Label '{m['label']}': Precision={m['precision']:.4f}, Recall={m['recall']:.4f}, F1={m['f1']:.4f} (Support: {m['support']})"
        )
    logger.info("==================================================")
    logger.info(f"Evaluation metrics saved to: {results_json_path}")
    logger.info(f"Per-label breakdown CSV saved to: {per_label_csv_path}")

    return summary_results


if __name__ == "__main__":
    evaluate()
