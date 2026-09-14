import json
import logging
from pathlib import Path
import sys
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("EvaluateLegalRetrieval")


def get_project_root() -> Path:
    current_file = Path(__file__).resolve()
    candidates = [
        current_file.parent.parent.parent,
        current_file.parent.parent,
        Path.cwd(),
    ]
    for candidate in candidates:
        if (candidate / "data").exists() and (candidate / "ml").exists():
            return candidate
    return current_file.parent.parent.parent


def evaluate_retrieval():
    project_root = get_project_root()
    sys.path.insert(0, str(project_root / "backend"))
    sys.path.insert(0, str(project_root / "ml" / "inference"))

    from predict_bert import BERTIncidentPredictor
    from app.services.legal_mapping import normalize_to_applicable_label

    laws_csv = project_root / "data" / "laws.csv"
    test_csv = project_root / "ml" / "preprocessing" / "test_data.csv"
    output_dir = project_root / "ml" / "evaluation" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_out_path = output_dir / "legal_retrieval_metrics.json"
    txt_out_path = output_dir / "legal_retrieval_report.txt"

    if not laws_csv.exists():
        logger.error(f"Laws CSV missing at {laws_csv}")
        sys.exit(1)
    if not test_csv.exists():
        logger.error(f"Test CSV missing at {test_csv}")
        sys.exit(1)

    laws_df = pd.read_csv(laws_csv)
    test_df = pd.read_csv(test_csv)

    logger.info(f"Loaded {len(laws_df)} legal provisions from {laws_csv}")
    logger.info(f"Loaded {len(test_df)} test queries from {test_csv}")

    # Build DB law list & mapping
    db_laws = []
    for idx, row in laws_df.iterrows():
        db_laws.append({
            "law_id": idx + 1,
            "act_name": str(row["act_name"]).strip(),
            "section_number": str(row["section_number"]).strip(),
            "applicable_label": str(row["applicable_label"]).strip(),
            "section_text": str(row["section_text"]).strip(),
        })

    # Group laws by category label
    laws_by_label = {}
    for law in db_laws:
        lbl = law["applicable_label"]
        if lbl not in laws_by_label:
            laws_by_label[lbl] = []
        laws_by_label[lbl].append(law)

    predictor = BERTIncidentPredictor()

    # Evaluation loop
    k_values = [1, 3, 5, 10]

    end_to_end_metrics = {f"hit_rate_at_{k}": [] for k in k_values}
    end_to_end_metrics.update({f"precision_at_{k}": [] for k in k_values})
    end_to_end_metrics.update({f"recall_at_{k}": [] for k in k_values})
    end_to_end_metrics["mrr"] = []

    oracle_metrics = {f"hit_rate_at_{k}": [] for k in k_values}
    oracle_metrics.update({f"precision_at_{k}": [] for k in k_values})
    oracle_metrics.update({f"recall_at_{k}": [] for k in k_values})
    oracle_metrics["mrr"] = []

    per_query_results = []
    unmatched_queries = 0

    for idx, row in test_df.iterrows():
        query_id = str(row.get("id", f"Q_{idx}"))
        text = str(row["narrative_text"])

        # Ground truth active labels
        gt_labels_raw = eval(str(row["active_labels"])) if isinstance(row["active_labels"], str) else row["active_labels"]
        gt_labels = [normalize_to_applicable_label(l) for l in gt_labels_raw if normalize_to_applicable_label(l)]

        # Relevant ground truth laws
        gt_relevant_laws = [l for l in db_laws if l["applicable_label"] in gt_labels]
        gt_relevant_ids = set(l["law_id"] for l in gt_relevant_laws)

        # 1. End-to-End BERT Prediction
        pred_res = predictor.predict(text)
        pred_labels_raw = pred_res.get("predicted_labels", [])
        pred_codes = []
        for item in pred_labels_raw:
            if isinstance(item, dict):
                pred_codes.append(item.get("code") or item.get("name"))
            elif isinstance(item, str):
                pred_codes.append(item)
        pred_labels = [normalize_to_applicable_label(l) for l in pred_codes if normalize_to_applicable_label(l)]

        # Retrieve laws for predicted labels (ordered as in laws.csv)
        retrieved_laws = [l for l in db_laws if l["applicable_label"] in pred_labels]
        retrieved_ids = [l["law_id"] for l in retrieved_laws]

        # 2. Compute End-to-End Metrics for this query
        if not gt_relevant_ids:
            unmatched_queries += 1
            continue

        # MRR calculation
        first_rel_rank = 0
        for r_idx, l_id in enumerate(retrieved_ids, 1):
            if l_id in gt_relevant_ids:
                first_rel_rank = r_idx
                break
        e2e_mrr = (1.0 / first_rel_rank) if first_rel_rank > 0 else 0.0
        end_to_end_metrics["mrr"].append(e2e_mrr)

        for k in k_values:
            top_k_retrieved = retrieved_ids[:k]
            top_k_set = set(top_k_retrieved)

            hits = len(top_k_set.intersection(gt_relevant_ids))

            hit_rate = 1.0 if hits > 0 else 0.0
            prec = (hits / len(top_k_retrieved)) if len(top_k_retrieved) > 0 else 0.0
            rec = (hits / len(gt_relevant_ids)) if len(gt_relevant_ids) > 0 else 0.0

            end_to_end_metrics[f"hit_rate_at_{k}"].append(hit_rate)
            end_to_end_metrics[f"precision_at_{k}"].append(prec)
            end_to_end_metrics[f"recall_at_{k}"].append(rec)

        # 3. Oracle Metrics (assuming 100% accurate classification)
        oracle_retrieved_laws = [l for l in db_laws if l["applicable_label"] in gt_labels]
        oracle_retrieved_ids = [l["law_id"] for l in oracle_retrieved_laws]

        first_rel_rank_oracle = 0
        for r_idx, l_id in enumerate(oracle_retrieved_ids, 1):
            if l_id in gt_relevant_ids:
                first_rel_rank_oracle = r_idx
                break
        oracle_mrr = (1.0 / first_rel_rank_oracle) if first_rel_rank_oracle > 0 else 0.0
        oracle_metrics["mrr"].append(oracle_mrr)

        for k in k_values:
            top_k_oracle = oracle_retrieved_ids[:k]
            top_k_oracle_set = set(top_k_oracle)

            hits_oracle = len(top_k_oracle_set.intersection(gt_relevant_ids))

            hit_rate_o = 1.0 if hits_oracle > 0 else 0.0
            prec_o = (hits_oracle / len(top_k_oracle)) if len(top_k_oracle) > 0 else 0.0
            rec_o = (hits_oracle / len(gt_relevant_ids)) if len(gt_relevant_ids) > 0 else 0.0

            oracle_metrics[f"hit_rate_at_{k}"].append(hit_rate_o)
            oracle_metrics[f"precision_at_{k}"].append(prec_o)
            oracle_metrics[f"recall_at_{k}"].append(rec_o)

        per_query_results.append({
            "query_id": query_id,
            "text": text,
            "gt_labels": gt_labels,
            "pred_labels": pred_labels,
            "num_gt_laws": len(gt_relevant_ids),
            "num_retrieved_laws": len(retrieved_ids),
            "mrr": e2e_mrr,
            "hit_at_1": 1.0 if len(set(retrieved_ids[:1]).intersection(gt_relevant_ids)) > 0 else 0.0,
            "hit_at_5": 1.0 if len(set(retrieved_ids[:5]).intersection(gt_relevant_ids)) > 0 else 0.0,
        })

    num_eval_queries = len(end_to_end_metrics["mrr"])

    # Aggregate metric results
    def agg(arr):
        return float(np.mean(arr)) if arr else 0.0

    summary_results = {
        "dataset_name": "test_data.csv",
        "total_test_records": len(test_df),
        "evaluated_queries": num_eval_queries,
        "unmapped_queries": unmatched_queries,
        "retrieval_methodology": "Category-Based Relational Database Query (Rule-Based Filter on data/laws.csv)",
        "ground_truth_definition": "Laws in data/laws.csv matching ground-truth incident category labels",
        "end_to_end_system_metrics": {
            "mrr": agg(end_to_end_metrics["mrr"]),
            "hit_rate_at_1": agg(end_to_end_metrics["hit_rate_at_1"]),
            "hit_rate_at_3": agg(end_to_end_metrics["hit_rate_at_3"]),
            "hit_rate_at_5": agg(end_to_end_metrics["hit_rate_at_5"]),
            "hit_rate_at_10": agg(end_to_end_metrics["hit_rate_at_10"]),
            "precision_at_1": agg(end_to_end_metrics["precision_at_1"]),
            "precision_at_3": agg(end_to_end_metrics["precision_at_3"]),
            "precision_at_5": agg(end_to_end_metrics["precision_at_5"]),
            "precision_at_10": agg(end_to_end_metrics["precision_at_10"]),
            "recall_at_1": agg(end_to_end_metrics["recall_at_1"]),
            "recall_at_3": agg(end_to_end_metrics["recall_at_3"]),
            "recall_at_5": agg(end_to_end_metrics["recall_at_5"]),
            "recall_at_10": agg(end_to_end_metrics["recall_at_10"]),
        },
        "oracle_category_retrieval_metrics": {
            "mrr": agg(oracle_metrics["mrr"]),
            "hit_rate_at_1": agg(oracle_metrics["hit_rate_at_1"]),
            "hit_rate_at_3": agg(oracle_metrics["hit_rate_at_3"]),
            "hit_rate_at_5": agg(oracle_metrics["hit_rate_at_5"]),
            "hit_rate_at_10": agg(oracle_metrics["hit_rate_at_10"]),
            "precision_at_1": agg(oracle_metrics["precision_at_1"]),
            "precision_at_3": agg(oracle_metrics["precision_at_3"]),
            "precision_at_5": agg(oracle_metrics["precision_at_5"]),
            "precision_at_10": agg(oracle_metrics["precision_at_10"]),
            "recall_at_1": agg(oracle_metrics["recall_at_1"]),
            "recall_at_3": agg(oracle_metrics["recall_at_3"]),
            "recall_at_5": agg(oracle_metrics["recall_at_5"]),
            "recall_at_10": agg(oracle_metrics["recall_at_10"]),
        },
    }

    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)

    # Format Human-Readable Report
    e2e = summary_results["end_to_end_system_metrics"]
    orc = summary_results["oracle_category_retrieval_metrics"]

    report_text = f"""================================================================================
LEGAL RETRIEVAL EVALUATION REPORT
================================================================================
1. EVALUATION METADATA:
   - Project Root: {project_root}
   - Legal Knowledge Base: {laws_csv} (33 law records)
   - Evaluation Split: {test_csv} (164 held-out incident queries)
   - Evaluated Queries: {num_eval_queries}
   - Retrieval Methodology: Category-Based Relational Database Query (Rule-Based Relational Filter)

2. RETRIEVAL ARCHITECTURE DEFINITION:
   - Method: Predicted BERT Incident Labels -> SQL Filter on data/laws.csv
   - Ranking Criterion: Primary sorting by canonical law ID in laws.csv
   - Fallback Strategy: Exact "Information not available" message if no categories match

3. END-TO-END SYSTEM RETRIEVAL PERFORMANCE (BERT + Legal Mapping):
   - Mean Reciprocal Rank (MRR) : {e2e['mrr']:.4f}
   ----------------------------------------------------------------------
   Metric         | Top-1   | Top-3   | Top-5   | Top-10
   ----------------------------------------------------------------------
   Hit Rate@K     | {e2e['hit_rate_at_1']:.4f}  | {e2e['hit_rate_at_3']:.4f}  | {e2e['hit_rate_at_5']:.4f}  | {e2e['hit_rate_at_10']:.4f}
   Precision@K    | {e2e['precision_at_1']:.4f}  | {e2e['precision_at_3']:.4f}  | {e2e['precision_at_5']:.4f}  | {e2e['precision_at_10']:.4f}
   Recall@K       | {e2e['recall_at_1']:.4f}  | {e2e['recall_at_3']:.4f}  | {e2e['recall_at_5']:.4f}  | {e2e['recall_at_10']:.4f}

4. ORACLE CATEGORY RETRIEVAL PERFORMANCE (Ground-Truth Labels -> Legal Mapping):
   - Mean Reciprocal Rank (MRR) : {orc['mrr']:.4f}
   ----------------------------------------------------------------------
   Metric         | Top-1   | Top-3   | Top-5   | Top-10
   ----------------------------------------------------------------------
   Hit Rate@K     | {orc['hit_rate_at_1']:.4f}  | {orc['hit_rate_at_3']:.4f}  | {orc['hit_rate_at_5']:.4f}  | {orc['hit_rate_at_10']:.4f}
   Precision@K    | {orc['precision_at_1']:.4f}  | {orc['precision_at_3']:.4f}  | {orc['precision_at_5']:.4f}  | {orc['precision_at_10']:.4f}
   Recall@K       | {orc['recall_at_1']:.4f}  | {orc['recall_at_3']:.4f}  | {orc['recall_at_5']:.4f}  | {orc['recall_at_10']:.4f}

5. ERROR & LIMITATION ANALYSIS:
   - System relies on BERT multi-label classification accuracy to route to legal provisions.
   - For multi-provision categories (e.g. SH has 8 laws), simple category filtering returns all matching laws without dense relevance ranking within the category.
   - Dense semantic vector search (e.g. BM25 or embedding similarity) could be used in future research to rank provisions within the matched category.

6. ARTIFACT LOCATION MAP:
   - JSON Results: {json_out_path}
   - Text Report: {txt_out_path}
================================================================================
"""

    with open(txt_out_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    logger.info(f"Legal retrieval evaluation complete. Artifacts saved to {output_dir}")


if __name__ == "__main__":
    evaluate_retrieval()
