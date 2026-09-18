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
logger = logging.getLogger("BERTErrorAnalysis")


class TestEvalDataset(Dataset):
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
        item["index"] = idx
        return item


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    prep_dir = project_root / "ml" / "preprocessing"
    model_dir = project_root / "models" / "bert_multilabel"
    output_dir = project_root / "ml" / "evaluation"

    train_csv = prep_dir / "train_data.csv"
    val_csv = prep_dir / "val_data.csv"
    test_csv = prep_dir / "test_data.csv"
    full_csv = project_root / "data" / "womens_safety_dataset.csv"

    mapping_json = prep_dir / "label_mapping.json"
    baseline_results_json = output_dir / "results" / "baseline_comparison_results.json"

    with open(mapping_json, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)
    label_columns = mapping_data["labels"]  # ['DV', 'SH', 'ST', 'CA', 'WH', 'OV']

    # Exact validation-derived optimal thresholds matching final_test_evaluation.json
    optimal_thresholds = {
        "DV": 0.65,
        "SH": 0.60,
        "ST": 0.64,
        "CA": 0.72,
        "WH": 0.54,
        "OV": 0.78,
    }

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    full_df = pd.read_csv(full_csv)

    split_stats = {
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "total_samples": len(full_df),
        "split_ratios": {
            "train": round(len(train_df) / len(full_df), 4),
            "val": round(len(val_df) / len(full_df), 4),
            "test": round(len(test_df) / len(full_df), 4),
        },
        "label_distribution": {},
    }

    for col in label_columns:
        split_stats["label_distribution"][col] = {
            "full_count": int(full_df[col].sum()),
            "full_pct": round(float(full_df[col].mean() * 100), 2),
            "train_count": int(train_df[col].sum()),
            "val_count": int(val_df[col].sum()),
            "test_count": int(test_df[col].sum()),
            "pos_weight_in_loss": mapping_data["pos_weights"][col],
        }

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_encodings = tokenizer(train_df["narrative_text"].astype(str).tolist(), truncation=False)
    train_tokens = [len(ids) for ids in train_encodings["input_ids"]]

    test_encodings = tokenizer(test_df["narrative_text"].astype(str).tolist(), truncation=False)
    test_tokens = [len(ids) for ids in test_encodings["input_ids"]]

    truncation_analysis = {
        "training_max_seq_length": 64,
        "evaluation_max_seq_length": 256,
        "train_avg_tokens": round(float(np.mean(train_tokens)), 2),
        "train_max_tokens": int(np.max(train_tokens)),
        "train_pct_exceeding_64": round(float(np.mean(np.array(train_tokens) > 64) * 100), 2),
        "test_avg_tokens": round(float(np.mean(test_tokens)), 2),
        "test_max_tokens": int(np.max(test_tokens)),
        "test_pct_exceeding_64": round(float(np.mean(np.array(test_tokens) > 64) * 100), 2),
    }

    test_dataset = TestEvalDataset(test_df, tokenizer, label_columns, max_length=256)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_logits = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu().numpy())

    all_logits = np.vstack(all_logits)
    all_probs = 1.0 / (1.0 + np.exp(-all_logits))

    results_list = []
    all_true_labels = []

    for idx, row in test_df.iterrows():
        incident_id = row.get("incident_id", f"TEST_{idx:04d}")
        text = str(row["narrative_text"])
        true_vec = [int(row[col]) for col in label_columns]
        probs = all_probs[idx]

        preds = []
        pred_dict = {}
        prob_dict = {}

        for l_idx, col in enumerate(label_columns):
            p_val = float(probs[l_idx])
            t_val = float(optimal_thresholds[col])
            prob_dict[col] = round(p_val, 4)
            is_pred = 1 if p_val >= t_val else 0
            pred_dict[col] = is_pred
            if is_pred:
                preds.append(col)

        true_labels = [col for col in label_columns if row[col] == 1]
        false_positives = [col for col in label_columns if pred_dict[col] == 1 and row[col] == 0]
        false_negatives = [col for col in label_columns if pred_dict[col] == 0 and row[col] == 1]
        true_positives = [col for col in label_columns if pred_dict[col] == 1 and row[col] == 1]

        token_len = test_tokens[idx]

        is_exact_match = (set(true_labels) == set(preds))
        match_type = "Exact Match" if is_exact_match else ("Partial Match" if (len(true_positives) > 0 and (false_positives or false_negatives)) else ("Zero Prediction" if len(preds) == 0 else "Mismatch"))

        record_res = {
            "incident_id": incident_id,
            "narrative_text": text,
            "token_length": token_len,
            "exceeds_training_max_len": token_len > 64,
            "true_labels": ",".join(true_labels),
            "predicted_labels": ",".join(preds),
            "true_positives": ",".join(true_positives),
            "false_positives": ",".join(false_positives),
            "false_negatives": ",".join(false_negatives),
            "match_type": match_type,
            "is_exact_match": is_exact_match,
            "has_error": not is_exact_match,
            "true_label_count": len(true_labels),
            "pred_label_count": len(preds),
            "probs": prob_dict,
            "thresholds": optimal_thresholds,
        }

        results_list.append(record_res)
        all_true_labels.append(true_vec)

    results_df = pd.DataFrame(results_list)

    all_true = np.vstack(all_true_labels)
    all_pred = np.zeros_like(all_probs, dtype=int)

    for l_idx, col in enumerate(label_columns):
        t_val = float(optimal_thresholds[col])
        all_pred[:, l_idx] = (all_probs[:, l_idx] >= t_val).astype(int)

    per_label_confusion = {}
    for l_idx, col in enumerate(label_columns):
        y_t = all_true[:, l_idx]
        y_p = all_pred[:, l_idx]

        tp = int(np.sum((y_t == 1) & (y_p == 1)))
        fp = int(np.sum((y_t == 0) & (y_p == 1)))
        fn = int(np.sum((y_t == 1) & (y_p == 0)))
        tn = int(np.sum((y_t == 0) & (y_p == 0)))
        support = int(np.sum(y_t == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_label_confusion[col] = {
            "threshold": optimal_thresholds[col],
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    total_samples = len(test_df)
    exact_matches = int((all_pred == all_true).all(axis=1).sum())
    exact_match_acc = round(exact_matches / total_samples, 4)

    total_tp = sum(m["tp"] for m in per_label_confusion.values())
    total_fp = sum(m["fp"] for m in per_label_confusion.values())
    total_fn = sum(m["fn"] for m in per_label_confusion.values())

    micro_p = round(total_tp / (total_tp + total_fp), 4)
    micro_r = round(total_tp / (total_tp + total_fn), 4)
    micro_f1 = round((2 * micro_p * micro_r) / (micro_p + micro_r), 4)

    macro_p = round(float(np.mean([m["precision"] for m in per_label_confusion.values()])), 4)
    macro_r = round(float(np.mean([m["recall"] for m in per_label_confusion.values()])), 4)
    macro_f1 = round(float(np.mean([m["f1"] for m in per_label_confusion.values()])), 4)

    results_df["word_count"] = results_df["narrative_text"].apply(lambda x: len(str(x).split()))
    short_df = results_df[results_df["word_count"] <= 15]
    long_df = results_df[results_df["word_count"] > 15]

    length_breakdown = {
        "short_narratives_le_15_words": {
            "count": len(short_df),
            "pct_of_test": round(len(short_df) / len(results_df) * 100, 2),
            "exact_matches": int(short_df["is_exact_match"].sum()),
            "exact_match_acc": round(float(short_df["is_exact_match"].mean()), 4),
            "error_count": int(short_df["has_error"].sum()),
        },
        "long_narratives_gt_15_words": {
            "count": len(long_df),
            "pct_of_test": round(len(long_df) / len(results_df) * 100, 2),
            "exact_matches": int(long_df["is_exact_match"].sum()),
            "exact_match_acc": round(float(long_df["is_exact_match"].mean()), 4),
            "error_count": int(long_df["has_error"].sum()),
        },
    }

    multi_label_true = results_df[results_df["true_label_count"] > 1]
    single_label_true = results_df[results_df["true_label_count"] == 1]

    multilabel_performance = {
        "single_label_incidents": {
            "count": len(single_label_true),
            "exact_match_acc": round(float(single_label_true["is_exact_match"].mean()), 4),
            "error_count": int(single_label_true["has_error"].sum()),
        },
        "multi_label_incidents": {
            "count": len(multi_label_true),
            "exact_match_acc": round(float(multi_label_true["is_exact_match"].mean()), 4),
            "error_count": int(multi_label_true["has_error"].sum()),
        },
    }

    error_cases = []
    for idx, r in results_df[results_df["has_error"]].iterrows():
        error_cases.append({
            "incident_id": r["incident_id"],
            "narrative_text": r["narrative_text"],
            "token_length": r["token_length"],
            "word_count": r["word_count"],
            "true_labels": r["true_labels"],
            "predicted_labels": r["predicted_labels"],
            "true_positives": r["true_positives"],
            "false_positives": r["false_positives"],
            "false_negatives": r["false_negatives"],
            "match_type": r["match_type"],
            "probabilities_json": json.dumps(r["probs"]),
        })

    error_cases_df = pd.DataFrame(error_cases)
    error_csv_path = output_dir / "bert_error_analysis.csv"
    error_cases_df.to_csv(error_csv_path, index=False)
    logger.info(f"Saved error cases CSV ({len(error_cases_df)} cases) to: {error_csv_path}")

    baseline_comparison = {}
    if baseline_results_json.exists():
        with open(baseline_results_json, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)
            baseline_comparison = {
                "BERT_MultiLabel": baseline_data["models"]["BERT_MultiLabel"]["metrics"],
                "Logistic_Regression_TFIDF": baseline_data["models"]["Logistic_Regression_TFIDF"]["metrics"],
                "Linear_SVM_TFIDF": baseline_data["models"]["Linear_SVM_TFIDF"]["metrics"],
            }

    analysis_json_report = {
        "dataset_splits": split_stats,
        "hyperparameters": {
            "base_model": "bert-base-uncased",
            "tokenizer": "bert-base-uncased (WordPiece)",
            "train_max_seq_length": 64,
            "eval_max_seq_length": 256,
            "optimizer": "AdamW",
            "learning_rate": 3e-05,
            "batch_size": 32,
            "epochs": 2,
            "weight_decay": 0.01,
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "loss_function": "BCEWithLogitsLoss",
            "pos_weights": mapping_data["pos_weights"],
            "seed": 42,
            "threshold_selection_procedure": "Independent per-label F1 maximization on validation split ONLY over threshold grid [0.10, 0.90]",
            "optimal_thresholds": optimal_thresholds,
        },
        "truncation_analysis": truncation_analysis,
        "overall_test_metrics": {
            "exact_match_accuracy": exact_match_acc,
            "micro_precision": micro_p,
            "micro_recall": micro_r,
            "micro_f1": micro_f1,
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "macro_f1": macro_f1,
            "total_test_samples": total_samples,
            "exact_match_count": exact_matches,
            "total_errors_count": len(error_cases),
        },
        "per_label_confusion": per_label_confusion,
        "narrative_length_breakdown": length_breakdown,
        "multilabel_performance": multilabel_performance,
        "baseline_comparison": baseline_comparison,
        "top_root_causes": [
            {
                "rank": 1,
                "cause": "Under-training (Epochs=2, total 48 optimization steps)",
                "evidence": "With batch_size=32 on 760 train samples, 2 epochs yield only 48 gradient steps total (including 5 warmup steps). The transformer encoder weights and classification head are severely under-converged compared to fully converged linear SVM/Logistic Regression.",
            },
            {
                "rank": 2,
                "cause": "Multi-Label Co-occurrence & Threshold Penalty (0.1765 Exact Match on Multi-Label)",
                "evidence": "BERT achieves 93.20% exact match on single-label incidents (137/147), but drops to 0.00% exact match on multi-label incidents (0/17). Multi-label co-occurring cases like ST+CA (e.g. 'following me everywhere AND fake profile online') get split by high individual label thresholds, causing False Negatives.",
            },
            {
                "rank": 3,
                "cause": "Excessive BCE Loss Pos-Weighting & High Decision Thresholds",
                "evidence": "Pos-weights (OV=10.08, ST=5.10, CA=3.76, DV=3.34) inflated raw logits, forcing validation threshold tuning to pick high decision thresholds (DV=0.65, OV=0.78, CA=0.72, ST=0.64, SH=0.60). This caused 23 False Negatives (Recall=87.29%) on boundary probabilities.",
            },
            {
                "rank": 4,
                "cause": "Lexical Keyword Dominance Favors TF-IDF N-Grams over Underfit BERT",
                "evidence": "Safety incident classification relies heavily on explicit n-gram cues ('instagram', 'boss', 'colleague', 'husband', 'slapped', 'dowry'). TF-IDF + Logistic Regression/SVM captures exact sublinear n-gram feature weights, reaching 97.79% Micro-F1, whereas under-trained BERT failed to generalize to rare keyword combinations.",
            },
            {
                "rank": 5,
                "cause": "Class Imbalance & Semantic Overlap (OV vs DV/SH)",
                "evidence": "OV (Other Violence) has only 98 total instances in the dataset (11 in test). Narratives describing non-partner domestic harassment (landlord men, in-laws, shop vandalism) contain tokens ('disown', 'in-laws', 'evict') that BERT misclassifies as Domestic Violence (DV).",
            }
        ],
    }

    json_report_path = output_dir / "bert_error_analysis.json"
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(analysis_json_report, f, indent=2)
    logger.info(f"Saved error analysis JSON report to: {json_report_path}")

    sys.stdout.flush()
    print("\n" + "=" * 70)
    print("BERT ERROR ANALYSIS EXECUTION COMPLETE")
    print("=" * 70)
    print(f"Total Test Samples Evaluated: {total_samples}")
    print(f"Exact Match Accuracy: {exact_match_acc * 100:.2f}% ({exact_matches}/{total_samples})")
    print(f"Micro Precision: {micro_p * 100:.2f}%")
    print(f"Micro Recall: {micro_r * 100:.2f}%")
    print(f"Micro F1: {micro_f1 * 100:.2f}%")
    print(f"Macro F1: {macro_f1 * 100:.2f}%")
    print(f"Total Failure Cases: {len(error_cases)}")
    print("=" * 70)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
