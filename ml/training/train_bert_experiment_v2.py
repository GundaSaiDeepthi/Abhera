import json
import logging
from pathlib import Path
import random
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("TrainBERT_ExpV2")


def set_seed(seed: int = 42):
    """Sets reproducible random seeds across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WomensSafetyDataset(Dataset):
    """PyTorch Dataset for multi-label women's safety incident text classification."""

    def __init__(self, df: pd.DataFrame, tokenizer, label_columns: list[str], max_length: int = 128):
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


def compute_metrics_at_thresholds(logits, labels, thresholds):
    """Computes exact match, micro, macro, and per-label metrics given per-label thresholds."""
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = np.zeros_like(probs, dtype=int)

    if isinstance(thresholds, (float, int)):
        threshold_map = [float(thresholds)] * probs.shape[1]
    elif isinstance(thresholds, list):
        threshold_map = thresholds
    elif isinstance(thresholds, dict):
        threshold_map = [float(thresholds[col]) for col in ["DV", "SH", "ST", "CA", "WH", "OV"]]
    else:
        threshold_map = [0.5] * probs.shape[1]

    for k in range(probs.shape[1]):
        preds[:, k] = (probs[:, k] >= threshold_map[k]).astype(int)

    labels = labels.astype(int)

    exact_accuracy = float((preds == labels).all(axis=1).mean())
    elementwise_accuracy = float((preds == labels).mean())

    # Overall Micro Metrics
    tp = float(np.sum((labels == 1) & (preds == 1)))
    fp = float(np.sum((labels == 0) & (preds == 1)))
    fn = float(np.sum((labels == 1) & (preds == 0)))

    micro_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    micro_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    micro_f1 = (2 * micro_prec * micro_rec) / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

    # Per-Label & Macro Metrics
    num_labels = labels.shape[1]
    macro_precs, macro_recs, macro_f1s = [], [], []
    per_label_details = {}

    label_names = ["DV", "SH", "ST", "CA", "WH", "OV"]

    for k in range(num_labels):
        l_k = labels[:, k]
        p_k = preds[:, k]

        tp_k = int(np.sum((l_k == 1) & (p_k == 1)))
        fp_k = int(np.sum((l_k == 0) & (p_k == 1)))
        fn_k = int(np.sum((l_k == 1) & (p_k == 0)))
        tn_k = int(np.sum((l_k == 0) & (p_k == 0)))

        prec_k = tp_k / (tp_k + fp_k) if (tp_k + fp_k) > 0 else 0.0
        rec_k = tp_k / (tp_k + fn_k) if (tp_k + fn_k) > 0 else 0.0
        f1_k = (2 * prec_k * rec_k) / (prec_k + rec_k) if (prec_k + rec_k) > 0 else 0.0

        macro_precs.append(prec_k)
        macro_recs.append(rec_k)
        macro_f1s.append(f1_k)

        col_name = label_names[k] if k < len(label_names) else f"label_{k}"
        per_label_details[col_name] = {
            "threshold": threshold_map[k],
            "precision": float(prec_k),
            "recall": float(rec_k),
            "f1": float(f1_k),
            "support": int(l_k.sum()),
            "tp": tp_k,
            "fp": fp_k,
            "fn": fn_k,
            "tn": tn_k,
        }

    return {
        "exact_match_accuracy": exact_accuracy,
        "elementwise_accuracy": elementwise_accuracy,
        "micro_precision": float(micro_prec),
        "micro_recall": float(micro_rec),
        "micro_f1": float(micro_f1),
        "macro_precision": float(np.mean(macro_precs)),
        "macro_recall": float(np.mean(macro_recs)),
        "macro_f1": float(np.mean(macro_f1s)),
        "per_label": per_label_details,
    }


def optimize_thresholds_on_val(logits, labels, label_columns):
    """
    Performs independent threshold optimization on the Validation Set ONLY.
    Searches thresholds t in [0.10, 0.90] (step 0.01) for each label to maximize per-label F1.
    """
    probs = 1.0 / (1.0 + np.exp(-logits))
    best_thresholds = {}

    for idx, col in enumerate(label_columns):
        y_true = labels[:, idx]
        y_prob = probs[:, idx]

        best_t = 0.50
        best_f1 = -1.0

        for t in np.arange(0.10, 0.91, 0.01):
            t_val = round(float(t), 2)
            y_pred = (y_prob >= t_val).astype(int)

            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            if f1 > best_f1:
                best_f1 = f1
                best_t = t_val

        best_thresholds[col] = best_t
        logger.info(f"  Optimized Val Threshold for '{col}': {best_t:.2f} (Val F1: {best_f1:.4f})")

    return best_thresholds


def main():
    set_seed(42)

    project_root = Path(__file__).resolve().parent.parent.parent
    prep_dir = project_root / "ml" / "preprocessing"
    eval_dir = project_root / "ml" / "evaluation"
    output_model_dir = project_root / "models" / "bert_multilabel_experiment_v2"
    output_model_dir.mkdir(parents=True, exist_ok=True)

    train_csv = prep_dir / "train_data.csv"
    val_csv = prep_dir / "val_data.csv"
    test_csv = prep_dir / "test_data.csv"
    mapping_json = prep_dir / "label_mapping.json"
    reference_model_dir = project_root / "models" / "bert_multilabel"

    with open(mapping_json, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)

    label_columns = mapping_data["labels"]  # ['DV', 'SH', 'ST', 'CA', 'WH', 'OV']
    num_labels = len(label_columns)
    pos_weights_dict = mapping_data.get("pos_weights", {})

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    logger.info(f"Loaded datasets - Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    base_model_name = "bert-base-uncased"
    max_length = 128
    batch_size = 32
    epochs = 6
    learning_rate = 3e-5
    weight_decay = 0.01

    logger.info("==================================================")
    logger.info("BERT EXPERIMENT V2: CONTROLLED TRAINING EXPERIMENT")
    logger.info("==================================================")
    logger.info(f"  Base Model: {base_model_name}")
    logger.info(f"  Max Sequence Length: {max_length}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch Size: {batch_size}")
    logger.info(f"  Learning Rate: {learning_rate}")
    logger.info(f"  Weight Decay: {weight_decay}")
    logger.info(f"  Seed: 42")
    logger.info(f"  Loss: BCEWithLogitsLoss with Pos Weights: {pos_weights_dict}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    config = AutoConfig.from_pretrained(
        base_model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        id2label={idx: label for idx, label in enumerate(label_columns)},
        label2id={label: idx for idx, label in enumerate(label_columns)},
    )

    model = AutoModelForSequenceClassification.from_pretrained(base_model_name, config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Device: {device}")
    model.to(device)

    pos_weight_tensor = torch.tensor(
        [pos_weights_dict.get(col, 1.0) for col in label_columns],
        dtype=torch.float,
    ).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    train_dataset = WomensSafetyDataset(train_df, tokenizer, label_columns, max_length)
    val_dataset = WomensSafetyDataset(val_df, tokenizer, label_columns, max_length)
    test_dataset = WomensSafetyDataset(test_df, tokenizer, label_columns, max_length=256)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_val_micro_f1 = -1.0
    best_epoch = 1
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation evaluation at default 0.50 threshold for epoch selection
        model.eval()
        total_val_loss = 0.0
        val_logits_list = []
        val_labels_list = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)

                total_val_loss += loss.item()
                val_logits_list.append(outputs.logits.cpu().numpy())
                val_labels_list.append(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_logits_arr = np.vstack(val_logits_list)
        val_labels_arr = np.vstack(val_labels_list)

        val_metrics_at_50 = compute_metrics_at_thresholds(val_logits_arr, val_labels_arr, 0.50)
        val_micro_f1 = val_metrics_at_50["micro_f1"]
        val_macro_f1 = val_metrics_at_50["macro_f1"]
        val_exact_acc = val_metrics_at_50["exact_match_accuracy"]

        logger.info(
            f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Val Micro F1 (t=0.5): {val_micro_f1:.4f} | Val Macro F1: {val_macro_f1:.4f} | Val Exact Acc: {val_exact_acc:.4f}"
        )
        sys.stdout.flush()

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_exact_match_acc": round(val_exact_acc, 4),
            "val_micro_precision": round(val_metrics_at_50["micro_precision"], 4),
            "val_micro_recall": round(val_metrics_at_50["micro_recall"], 4),
            "val_micro_f1": round(val_micro_f1, 4),
            "val_macro_precision": round(val_metrics_at_50["macro_precision"], 4),
            "val_macro_recall": round(val_metrics_at_50["macro_recall"], 4),
            "val_macro_f1": round(val_macro_f1, 4),
            "is_best_epoch": False,
        }

        if val_micro_f1 > best_val_micro_f1:
            best_val_micro_f1 = val_micro_f1
            best_epoch = epoch
            logger.info(f"--> [NEW BEST CHECKPOINT] Saving Epoch {epoch} to {output_model_dir} (Val Micro F1: {best_val_micro_f1:.4f})")
            model.save_pretrained(output_model_dir)
            tokenizer.save_pretrained(output_model_dir)

        history.append(epoch_record)

    # Mark best epoch in history
    for record in history:
        if record["epoch"] == best_epoch:
            record["is_best_epoch"] = True

    history_df = pd.DataFrame(history)
    history_csv_path = eval_dir / "bert_experiment_v2_history.csv"
    history_df.to_csv(history_csv_path, index=False)
    logger.info(f"Saved epoch training history to: {history_csv_path}")

    # =========================================================================
    # STEP F: THRESHOLD OPTIMIZATION ON VALIDATION SET ONLY USING BEST CHECKPOINT
    # =========================================================================
    logger.info("==================================================")
    logger.info(f"OPTIMIZING DECISION THRESHOLDS ON VALIDATION SET ONLY (BEST CHECKPOINT: EPOCH {best_epoch})")
    logger.info("==================================================")

    best_model = AutoModelForSequenceClassification.from_pretrained(output_model_dir).to(device)
    best_model.eval()

    val_logits_list = []
    val_labels_list = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = best_model(input_ids=input_ids, attention_mask=attention_mask)
            val_logits_list.append(outputs.logits.cpu().numpy())
            val_labels_list.append(labels.cpu().numpy())

    val_logits_arr = np.vstack(val_logits_list)
    val_labels_arr = np.vstack(val_labels_list).astype(int)

    optimized_thresholds = optimize_thresholds_on_val(val_logits_arr, val_labels_arr, label_columns)

    # Save optimal thresholds into experimental model directory
    thresholds_json_path = output_model_dir / "optimal_thresholds.json"
    with open(thresholds_json_path, "w", encoding="utf-8") as f:
        json.dump(optimized_thresholds, f, indent=2)
    logger.info(f"Saved optimized experimental thresholds to: {thresholds_json_path}")

    # Save training config
    training_config = {
        "base_model_name": base_model_name,
        "max_length": max_length,
        "batch_size": batch_size,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_micro_f1_at_50": best_val_micro_f1,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "optimizer": "AdamW",
        "loss_function": "BCEWithLogitsLoss",
        "pos_weights": pos_weights_dict,
        "seed": 42,
        "device": str(device),
    }
    with open(output_model_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(training_config, f, indent=2)

    with open(output_model_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping_data, f, indent=2)

    # =========================================================================
    # STEP G: FINAL HELD-OUT TEST SET EVALUATION (RUN ONCE)
    # =========================================================================
    logger.info("==================================================")
    logger.info("RUNNING FINAL EVALUATION ON UNTOUCHED HELD-OUT TEST SET (ONCE)")
    logger.info("==================================================")

    test_logits_list = []
    test_labels_list = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = best_model(input_ids=input_ids, attention_mask=attention_mask)
            test_logits_list.append(outputs.logits.cpu().numpy())
            test_labels_list.append(labels.cpu().numpy())

    test_logits_arr = np.vstack(test_logits_list)
    test_labels_arr = np.vstack(test_labels_list).astype(int)

    test_results = compute_metrics_at_thresholds(test_logits_arr, test_labels_arr, optimized_thresholds)

    # Narrative length breakdown & Multi-label exact match analysis on test set
    test_probs = 1.0 / (1.0 + np.exp(-test_logits_arr))
    test_preds = np.zeros_like(test_probs, dtype=int)
    for k, col in enumerate(label_columns):
        test_preds[:, k] = (test_probs[:, k] >= optimized_thresholds[col]).astype(int)

    exact_match_vector = (test_preds == test_labels_arr).all(axis=1)

    test_df["word_count"] = test_df["narrative_text"].apply(lambda x: len(str(x).split()))
    test_df["true_label_count"] = test_df[label_columns].sum(axis=1)

    single_label_mask = (test_df["true_label_count"] == 1).values
    multi_label_mask = (test_df["true_label_count"] > 1).values

    single_label_exact_acc = float(exact_match_vector[single_label_mask].mean()) if single_label_mask.sum() > 0 else 0.0
    multi_label_exact_acc = float(exact_match_vector[multi_label_mask].mean()) if multi_label_mask.sum() > 0 else 0.0

    exp_v2_results_dict = {
        "experiment_name": "BERT_MultiLabel_Experiment_v2",
        "description": "Controlled BERT training experiment: epochs=6, max_length=128, batch_size=32, lr=3e-5, weight_decay=0.01",
        "best_epoch_selected": best_epoch,
        "optimal_thresholds_val_derived": optimized_thresholds,
        "test_metrics": {
            "exact_match_accuracy": round(test_results["exact_match_accuracy"], 4),
            "elementwise_accuracy": round(test_results["elementwise_accuracy"], 4),
            "micro_precision": round(test_results["micro_precision"], 4),
            "micro_recall": round(test_results["micro_recall"], 4),
            "micro_f1": round(test_results["micro_f1"], 4),
            "macro_precision": round(test_results["macro_precision"], 4),
            "macro_recall": round(test_results["macro_recall"], 4),
            "macro_f1": round(test_results["macro_f1"], 4),
            "single_label_exact_match_acc": round(single_label_exact_acc, 4),
            "multi_label_exact_match_acc": round(multi_label_exact_acc, 4),
        },
        "per_label_metrics": test_results["per_label"],
    }

    results_json_path = eval_dir / "bert_experiment_v2_results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(exp_v2_results_dict, f, indent=2)
    logger.info(f"Saved experimental test results to: {results_json_path}")

    # Load existing baseline comparison for direct side-by-side report
    ref_results_file = reference_model_dir / "final_test_evaluation.json"
    ref_metrics = {}
    if ref_results_file.exists():
        with open(ref_results_file, "r", encoding="utf-8") as f:
            ref_metrics = json.load(f)["metrics"]

    comparison_dict = {
        "metrics_comparison": {
            "Existing_BERT_v1": {
                "epochs": 2,
                "max_length": 64,
                "exact_match_accuracy": round(ref_metrics.get("exact_match_accuracy", 0.8354), 4),
                "micro_precision": round(ref_metrics.get("micro_precision", 0.9576), 4),
                "micro_recall": round(ref_metrics.get("micro_recall", 0.8729), 4),
                "micro_f1": round(ref_metrics.get("micro_f1", 0.9133), 4),
                "macro_f1": round(ref_metrics.get("macro_f1", 0.9037), 4),
            },
            "Experimental_BERT_v2": {
                "epochs": 6,
                "best_epoch": best_epoch,
                "max_length": 128,
                "exact_match_accuracy": round(test_results["exact_match_accuracy"], 4),
                "micro_precision": round(test_results["micro_precision"], 4),
                "micro_recall": round(test_results["micro_recall"], 4),
                "micro_f1": round(test_results["micro_f1"], 4),
                "macro_f1": round(test_results["macro_f1"], 4),
            },
            "delta_experimental_minus_existing": {
                "exact_match_accuracy_diff": round(test_results["exact_match_accuracy"] - ref_metrics.get("exact_match_accuracy", 0.8354), 4),
                "micro_precision_diff": round(test_results["micro_precision"] - ref_metrics.get("micro_precision", 0.9576), 4),
                "micro_recall_diff": round(test_results["micro_recall"] - ref_metrics.get("micro_recall", 0.8729), 4),
                "micro_f1_diff": round(test_results["micro_f1"] - ref_metrics.get("micro_f1", 0.9133), 4),
                "macro_f1_diff": round(test_results["macro_f1"] - ref_metrics.get("macro_f1", 0.9037), 4),
            },
        }
    }

    comp_json_path = eval_dir / "bert_experiment_v2_comparison.json"
    with open(comp_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_dict, f, indent=2)
    logger.info(f"Saved comparison JSON to: {comp_json_path}")

    # Printable summary report
    print("\n" + "=" * 70)
    print("EXPERIMENTAL BERT V2 RESULTS SUMMARY")
    print("=" * 70)
    print(f"Best Epoch Selected (Validation Micro-F1): Epoch {best_epoch} of {epochs}")
    print(f"Optimized Thresholds (Val Derived): {optimized_thresholds}")
    print("--------------------------------------------------")
    print(f"Exact Match Accuracy : {test_results['exact_match_accuracy']*100:.2f}% (Ref: 83.54%)")
    print(f"Micro Precision      : {test_results['micro_precision']*100:.2f}% (Ref: 95.76%)")
    print(f"Micro Recall         : {test_results['micro_recall']*100:.2f}% (Ref: 87.29%)")
    print(f"Micro F1 Score       : {test_results['micro_f1']*100:.2f}% (Ref: 91.33%)")
    print(f"Macro F1 Score       : {test_results['macro_f1']*100:.2f}% (Ref: 90.37%)")
    print(f"Single-Label Exact Match : {single_label_exact_acc*100:.2f}%")
    print(f"Multi-Label Exact Match  : {multi_label_exact_acc*100:.2f}%")
    print("==================================================")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
