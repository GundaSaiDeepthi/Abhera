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
logger = logging.getLogger("TrainBERT")


def set_seed(seed: int = 42):
    """Sets reproducible seeds across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WomensSafetyDataset(Dataset):
    """PyTorch Dataset for multi-label women's safety incident text classification."""

    def __init__(self, df: pd.DataFrame, tokenizer, label_columns: list[str], max_length: int = 64):
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


def compute_metrics(logits, labels, threshold: float = 0.5):
    """Computes Micro/Macro F1, Precision, Recall, and Exact Accuracy using NumPy."""
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)
    labels = labels.astype(int)

    exact_accuracy = float((preds == labels).all(axis=1).mean())

    # Micro Metrics
    tp = float(np.sum((labels == 1) & (preds == 1)))
    fp = float(np.sum((labels == 0) & (preds == 1)))
    fn = float(np.sum((labels == 1) & (preds == 0)))

    micro_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    micro_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    micro_f1 = (2 * micro_prec * micro_rec) / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

    # Macro Metrics
    num_labels = labels.shape[1]
    macro_f1s, macro_precs, macro_recs = [], [], []

    for k in range(num_labels):
        l_k = labels[:, k]
        p_k = preds[:, k]
        tp_k = float(np.sum((l_k == 1) & (p_k == 1)))
        fp_k = float(np.sum((l_k == 0) & (p_k == 1)))
        fn_k = float(np.sum((l_k == 1) & (p_k == 0)))

        prec_k = tp_k / (tp_k + fp_k) if (tp_k + fp_k) > 0 else 0.0
        rec_k = tp_k / (tp_k + fn_k) if (tp_k + fn_k) > 0 else 0.0
        f1_k = (2 * prec_k * rec_k) / (prec_k + rec_k) if (prec_k + rec_k) > 0 else 0.0

        macro_precs.append(prec_k)
        macro_recs.append(rec_k)
        macro_f1s.append(f1_k)

    return {
        "exact_accuracy": exact_accuracy,
        "micro_f1": float(micro_f1),
        "macro_f1": float(np.mean(macro_f1s)),
        "micro_precision": float(micro_prec),
        "micro_recall": float(micro_rec),
        "macro_precision": float(np.mean(macro_precs)),
        "macro_recall": float(np.mean(macro_recs)),
    }


def train():
    set_seed(42)

    project_root = Path(__file__).resolve().parent.parent.parent
    prep_dir = project_root / "ml" / "preprocessing"
    output_model_dir = project_root / "models" / "bert_multilabel"
    output_model_dir.mkdir(parents=True, exist_ok=True)

    train_csv = prep_dir / "train_data.csv"
    val_csv = prep_dir / "val_data.csv"
    mapping_json = prep_dir / "label_mapping.json"

    if not (train_csv.exists() and val_csv.exists() and mapping_json.exists()):
        logger.error(
            "Preprocessing artifacts missing! Run python"
            " ml/preprocessing/prepare_classification.py first."
        )
        sys.exit(1)

    with open(mapping_json, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)

    label_columns = mapping_data["labels"]
    num_labels = len(label_columns)
    pos_weights_dict = mapping_data.get("pos_weights", {})

    logger.info(f"Loaded {num_labels} labels: {label_columns}")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    logger.info(f"Train dataset size: {len(train_df)} | Val dataset size: {len(val_df)}")

    base_model_name = "bert-base-uncased"
    max_length = 64
    batch_size = 32
    epochs = 2
    learning_rate = 3e-5

    logger.info(f"Initializing Tokenizer and Pretrained Model: {base_model_name}")
    sys.stdout.flush()
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    config = AutoConfig.from_pretrained(
        base_model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        id2label={idx: label for idx, label in enumerate(label_columns)},
        label2id={label: idx for idx, label in enumerate(label_columns)},
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        config=config,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training Compute Device: {device}")
    model.to(device)

    pos_weight_tensor = torch.tensor(
        [pos_weights_dict.get(col, 1.0) for col in label_columns],
        dtype=torch.float,
    ).to(device)
    logger.info(f"Configured BCEWithLogitsLoss Pos-Weights for Class Imbalance: {pos_weights_dict}")
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    train_dataset = WomensSafetyDataset(train_df, tokenizer, label_columns, max_length)
    val_dataset = WomensSafetyDataset(val_df, tokenizer, label_columns, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    best_val_f1 = 0.0
    history = []

    logger.info("==================================================")
    logger.info("STARTING BERT MULTI-LABEL FINE-TUNING")
    logger.info("==================================================")
    sys.stdout.flush()

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

        model.eval()
        total_val_loss = 0.0
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = loss_fn(logits, labels)

                total_val_loss += loss.item()
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        all_logits = np.vstack(all_logits)
        all_labels = np.vstack(all_labels)

        metrics = compute_metrics(all_logits, all_labels)
        val_micro_f1 = metrics["micro_f1"]

        logger.info(
            f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Val Micro F1: {val_micro_f1:.4f} | Val Exact Acc: {metrics['exact_accuracy']:.4f}"
        )
        sys.stdout.flush()

        epoch_stats = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            **metrics,
        }
        history.append(epoch_stats)

        if val_micro_f1 >= best_val_f1 or epoch == 1:
            best_val_f1 = val_micro_f1
            logger.info(f"--> Saving Best Model Checkpoint to: {output_model_dir} (Val Micro F1: {best_val_f1:.4f})")
            sys.stdout.flush()
            model.save_pretrained(output_model_dir)
            tokenizer.save_pretrained(output_model_dir)

            with open(output_model_dir / "label_mapping.json", "w", encoding="utf-8") as f:
                json.dump(mapping_data, f, indent=2)

            training_config = {
                "base_model_name": base_model_name,
                "max_length": max_length,
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "optimizer": "AdamW",
                "loss_function": "BCEWithLogitsLoss",
                "pos_weights": pos_weights_dict,
                "seed": 42,
                "device": str(device),
            }
            with open(output_model_dir / "training_config.json", "w", encoding="utf-8") as f:
                json.dump(training_config, f, indent=2)

    with open(output_model_dir / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"best_val_micro_f1": best_val_f1, "history": history}, f, indent=2)

    logger.info("==================================================")
    logger.info("BERT MULTI-LABEL FINE-TUNING COMPLETED")
    logger.info(f"Best Validation Micro F1: {best_val_f1:.4f}")
    logger.info(f"Model & Tokenizer artifacts saved in: {output_model_dir}")
    logger.info("==================================================")
    sys.stdout.flush()


if __name__ == "__main__":
    train()
