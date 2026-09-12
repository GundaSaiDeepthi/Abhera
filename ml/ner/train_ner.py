import json
import logging
import random
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("TrainNER")

BASE_MODEL_NAME = "bert-base-uncased"
SEED = 42
MAX_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
EPOCHS = 12


def set_seed(seed: int = 42):
    """Ensures full reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_project_root() -> Path:
    current_file = Path(__file__).resolve()
    candidates = [
        current_file.parent.parent.parent,
        current_file.parent.parent,
        Path.cwd(),
        Path.cwd().parent,
    ]
    for candidate in candidates:
        if (candidate / "data").exists() and (candidate / "ml").exists():
            return candidate
    return current_file.parent.parent.parent


def extract_spans_from_bio(tags: List[str]) -> List[Tuple[int, int, str]]:
    """
    Extracts entity spans (start_token_idx, end_token_idx, label) from a list of BIO tags.
    """
    spans = []
    current_label = None
    start_idx = -1

    for idx, tag in enumerate(tags):
        if tag == "O" or tag == "-100":
            if current_label is not None:
                spans.append((start_idx, idx - 1, current_label))
                current_label = None
                start_idx = -1
        elif tag.startswith("B-"):
            if current_label is not None:
                spans.append((start_idx, idx - 1, current_label))
            current_label = tag[2:]
            start_idx = idx
        elif tag.startswith("I-"):
            ent_type = tag[2:]
            if current_label == ent_type and current_label is not None:
                pass
            else:
                if current_label is not None:
                    spans.append((start_idx, idx - 1, current_label))
                current_label = ent_type
                start_idx = idx

    if current_label is not None:
        spans.append((start_idx, len(tags) - 1, current_label))

    return spans


class NERDataset(Dataset):
    """PyTorch Dataset for Token Classification NER."""

    def __init__(self, records: List[Dict[str, Any]], tokenizer, label2id: Dict[str, int], max_length: int = 256):
        self.records = records
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        tokens = rec["tokens"]
        bio_tags = rec["bio_tags"]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        labels = []
        for i, (tok, tag) in enumerate(zip(tokens, bio_tags)):
            if tok in ["[CLS]", "[SEP]", "[PAD]"] or i == 0 or i == len(tokens) - 1:
                labels.append(-100)
            else:
                labels.append(self.label2id.get(tag, self.label2id["O"]))

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

        attention_mask = [1] * len(input_ids)

        return {
            "id": rec["id"],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def ner_collate_fn(batch, pad_token_id=0):
    max_len = max(len(item["input_ids"]) for item in batch)

    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    for item in batch:
        seq_len = len(item["input_ids"])
        pad_len = max_len - seq_len

        input_ids = item["input_ids"] + [pad_token_id] * pad_len
        attention_mask = item["attention_mask"] + [0] * pad_len
        labels = item["labels"] + [-100] * pad_len

        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
        batch_labels.append(labels)

    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
    }


def evaluate_ner(model, dataloader, device, id2label: Dict[int, str], all_entity_types: List[str]):
    model.eval()
    total_loss = 0.0
    total_batches = 0

    all_gt_spans_total = []
    all_pred_spans_total = []

    label_tp = {lbl: 0 for lbl in all_entity_types}
    label_fp = {lbl: 0 for lbl in all_entity_types}
    label_fn = {lbl: 0 for lbl in all_entity_types}

    token_correct = 0
    token_total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            total_batches += 1

            preds = torch.argmax(logits, dim=-1)

            for doc_preds, doc_labels in zip(preds, labels):
                doc_pred_tags = []
                doc_gt_tags = []
                for p, l in zip(doc_preds, doc_labels):
                    lbl_val = l.item()
                    if lbl_val != -100:
                        token_total += 1
                        pred_val = p.item()
                        if pred_val == lbl_val:
                            token_correct += 1
                        doc_gt_tags.append(id2label[lbl_val])
                        doc_pred_tags.append(id2label[pred_val])

                gt_spans = extract_spans_from_bio(doc_gt_tags)
                pred_spans = extract_spans_from_bio(doc_pred_tags)

                gt_set = set(gt_spans)
                pred_set = set(pred_spans)

                # Overall spans tracking
                all_gt_spans_total.extend(gt_spans)
                all_pred_spans_total.extend(pred_spans)

                # Per-label counts
                for lbl in all_entity_types:
                    gt_lbl_spans = set(s for s in gt_spans if s[2] == lbl)
                    pred_lbl_spans = set(s for s in pred_spans if s[2] == lbl)

                    tp = len(gt_lbl_spans.intersection(pred_lbl_spans))
                    fp = len(pred_lbl_spans - gt_lbl_spans)
                    fn = len(gt_lbl_spans - pred_lbl_spans)

                    label_tp[lbl] += tp
                    label_fp[lbl] += fp
                    label_fn[lbl] += fn

    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    token_acc = token_correct / token_total if token_total > 0 else 0.0

    # Global entity-level metrics
    total_tp = sum(label_tp.values())
    total_fp = sum(label_fp.values())
    total_fn = sum(label_fn.values())

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Per-label metrics breakdown
    per_label_metrics = {}
    for lbl in all_entity_types:
        tp = label_tp[lbl]
        fp = label_fp[lbl]
        fn = label_fn[lbl]
        p_lbl = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r_lbl = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_lbl = (2 * p_lbl * r_lbl) / (p_lbl + r_lbl) if (p_lbl + r_lbl) > 0 else 0.0
        per_label_metrics[lbl] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": float(p_lbl),
            "recall": float(r_lbl),
            "f1": float(f1_lbl),
        }

    return {
        "val_loss": float(avg_loss),
        "token_accuracy": float(token_acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "per_label": per_label_metrics,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


def train_ner_model():
    set_seed(SEED)
    project_root = get_project_root()

    ner_dir = project_root / "ml" / "ner"
    training_dir = ner_dir / "training"
    models_dir = project_root / "models" / "ner"
    models_dir.mkdir(parents=True, exist_ok=True)

    train_path = training_dir / "train.jsonl"
    val_path = training_dir / "validation.jsonl"
    test_path = training_dir / "test.jsonl"
    label_map_path = training_dir / "label_mapping.json"
    report_path = training_dir / "ner_training_report.txt"

    if not train_path.exists() or not val_path.exists() or not label_map_path.exists():
        logger.error("Required dataset files missing. Please run prepare_training_data.py first.")
        sys.exit(1)

    # 1. Load label mapping
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_mapping_data = json.load(f)

    label2id = {k: int(v) for k, v in label_mapping_data["label2id"].items()}
    id2label = {int(k): v for k, v in label_mapping_data["id2label"].items()}
    num_labels = len(id2label)
    all_entity_types = ["PERP_REL", "LOCATION", "TIME_FREQ", "PLATFORM", "EVIDENCE", "LAW_SEC"]

    logger.info(f"Loaded label mapping with {num_labels} labels: {id2label}")

    # 2. Load splits
    def load_jsonl(path: Path) -> List[Dict[str, Any]]:
        recs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    recs.append(json.loads(line))
        return recs

    train_recs = load_jsonl(train_path)
    val_recs = load_jsonl(val_path)
    test_recs = load_jsonl(test_path)

    logger.info(f"Loaded splits - Train: {len(train_recs)} | Val: {len(val_recs)} | Test: {len(test_recs)}")

    # 3. Load Tokenizer & Model
    logger.info(f"Initializing AutoTokenizer and AutoModelForTokenClassification from '{BASE_MODEL_NAME}'")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    config = AutoConfig.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        finetuning_task="ner",
    )

    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL_NAME,
        config=config,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")
    model.to(device)

    # 4. Create Datasets & DataLoaders
    train_dataset = NERDataset(train_recs, tokenizer, label2id, max_length=MAX_LENGTH)
    val_dataset = NERDataset(val_recs, tokenizer, label2id, max_length=MAX_LENGTH)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: ner_collate_fn(b, pad_token_id=tokenizer.pad_token_id),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: ner_collate_fn(b, pad_token_id=tokenizer.pad_token_id),
    )

    # 5. Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_training_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_training_steps * 0.1),
        num_training_steps=total_training_steps,
    )

    logger.info(f"Starting training for {EPOCHS} epochs ({total_training_steps} total steps)...")

    epoch_history = []
    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0.0

        # Evaluate on validation set
        val_metrics = evaluate_ner(model, val_loader, device, id2label, all_entity_types)
        
        epoch_info = {
            "epoch": epoch,
            "train_loss": float(avg_train_loss),
            "val_loss": val_metrics["val_loss"],
            "token_accuracy": val_metrics["token_accuracy"],
            "precision": val_metrics["precision"],
            "recall": val_metrics["recall"],
            "f1": val_metrics["f1"],
            "per_label": val_metrics["per_label"],
        }
        epoch_history.append(epoch_info)

        logger.info(
            f"Epoch {epoch:02d}/{EPOCHS:02d} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | Val F1: {val_metrics['f1']:.4f} "
            f"(P: {val_metrics['precision']:.4f}, R: {val_metrics['recall']:.4f})"
        )

        # Track best model checkpoint based on Validation Entity F1 (or Val Loss if F1 == 0)
        is_best = False
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_val_loss = val_metrics["val_loss"]
            best_epoch = epoch
            is_best = True
        elif val_metrics["f1"] == best_val_f1 and val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_epoch = epoch
            is_best = True

        if is_best:
            logger.info(f"--> New best model checkpoint achieved at epoch {epoch}! Saving to {models_dir}")
            model.save_pretrained(models_dir)
            tokenizer.save_pretrained(models_dir)

            # Save label mapping & config inside models/ner
            with open(models_dir / "label_mapping.json", "w", encoding="utf-8") as f:
                json.dump(label_mapping_data, f, indent=2)

    # Save training metrics and config JSON files
    training_config = {
        "base_model": BASE_MODEL_NAME,
        "seed": SEED,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS,
        "num_train_examples": len(train_recs),
        "num_val_examples": len(val_recs),
        "num_test_examples": len(test_recs),
        "num_labels": num_labels,
    }

    with open(models_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(training_config, f, indent=2)

    metrics_summary = {
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "best_val_loss": best_val_loss,
        "epoch_history": epoch_history,
    }

    with open(models_dir / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    logger.info(f"Saved model, tokenizer, config, and metrics to {models_dir}")

    # 6. Generate Comprehensive Training Report
    best_epoch_data = epoch_history[best_epoch - 1]

    report_lines = [
        "=" * 80,
        "BERT-BASED NER TOKEN CLASSIFICATION TRAINING REPORT",
        "=" * 80,
        "1. MODEL & DATASET CONFIGURATION:",
        f"   - Base Model: {BASE_MODEL_NAME}",
        f"   - Model Output Directory: {models_dir}",
        f"   - Training Examples: {len(train_recs)} records",
        f"   - Validation Examples: {len(val_recs)} records",
        f"   - Test Examples: {len(test_recs)} records (HELD OUT - UNTOUCHED)",
        f"   - Total BIO Labels: {num_labels}",
        f"   - Random Seed: {SEED}",
        "",
        "2. HYPERPARAMETERS:",
        f"   - Learning Rate: {LEARNING_RATE}",
        f"   - Batch Size: {BATCH_SIZE}",
        f"   - Epochs Trained: {EPOCHS}",
        f"   - Weight Decay: {WEIGHT_DECAY}",
        f"   - Max Sequence Length: {MAX_LENGTH}",
        "",
        "3. EPOCH-BY-EPOCH METRICS:",
        "   Epoch | Train Loss | Val Loss   | Val Token Acc | Val Entity Prec | Val Entity Rec | Val Entity F1",
        "   " + "-" * 90,
    ]

    for ep in epoch_history:
        report_lines.append(
            f"   {ep['epoch']:>5} | {ep['train_loss']:>10.4f} | {ep['val_loss']:>10.4f} | "
            f"{ep['token_accuracy']:>13.4f} | {ep['precision']:>15.4f} | {ep['recall']:>14.4f} | {ep['f1']:>13.4f}"
        )

    report_lines.extend([
        "",
        "4. BEST CHECKPOINT SUMMARY:",
        f"   - Best Epoch: {best_epoch}",
        f"   - Best Validation Loss: {best_epoch_data['val_loss']:.4f}",
        f"   - Best Validation Precision: {best_epoch_data['precision']:.4f}",
        f"   - Best Validation Recall: {best_epoch_data['recall']:.4f}",
        f"   - Best Validation Entity F1: {best_epoch_data['f1']:.4f}",
        "",
        "5. BEST CHECKPOINT PER-ENTITY BREAKDOWN:",
    ])

    for lbl, m in best_epoch_data["per_label"].items():
        note = " [ZERO-SHOT / NO EXAMPLES]" if lbl in ["EVIDENCE", "LAW_SEC"] else ""
        report_lines.append(
            f"   - {lbl:<12}: Precision: {m['precision']:.4f} | Recall: {m['recall']:.4f} | "
            f"F1: {m['f1']:.4f} (TP: {m['tp']}, FP: {m['fp']}, FN: {m['fn']}){note}"
        )

    report_lines.extend([
        "",
        "6. MANDATORY DATASET & MODEL LIMITATIONS NOTICE:",
        "   EVIDENCE and LAW_SEC were retained in the label mapping but have no supervised training examples",
        "   in the current provisional dataset. No synthetic examples were created.",
        "",
        "7. ARTIFACTS MAP:",
        f"   - Model Checkpoint: {models_dir}",
        f"   - Training Config: {models_dir / 'training_config.json'}",
        f"   - Metrics Summary: {models_dir / 'training_metrics.json'}",
        f"   - Training Report: {report_path}",
        "=" * 80,
    ])

    report_text = "\n".join(report_lines)
    print(report_text)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info(f"Saved NER training report to: {report_path}")

    return {
        "status": "SUCCESS",
        "model_dir": str(models_dir),
        "epochs_completed": EPOCHS,
        "best_epoch": best_epoch,
        "best_train_loss": best_epoch_data["train_loss"],
        "best_val_loss": best_epoch_data["val_loss"],
        "best_precision": best_epoch_data["precision"],
        "best_recall": best_epoch_data["recall"],
        "best_f1": best_epoch_data["f1"],
        "report_path": str(report_path),
    }


if __name__ == "__main__":
    train_ner_model()
