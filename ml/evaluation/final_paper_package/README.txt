ABHERA PAPER-READY EVALUATION PACKAGE
=====================================
This directory contains the complete, paper-ready evaluation artifacts, metrics tables,
error analyses, comparison datasets, research text sections, figures, and reproducibility notes for ABHERA.

FOLDER GUIDE:
  - 01_bert/            : BERT multi-label classification metrics, thresholds, and confusion matrices.
  - 02_ner/             : NER entity extraction overall & per-class metrics.
  - 03_legal_retrieval/ : Category-conditioned legal retrieval metrics (MRR, Hit@K, Prec@K, Rec@K).
  - 04_grounding_safety/: Legal grounding success rates, anti-hallucination pass rates & negative test results.
  - 05_end_to_end/      : End-to-end pipeline success rates & fully grounded pipeline formula definition.
  - 06_error_analysis/  : Detailed breakdown of the 9 classification error cases & error pattern analysis.
  - 07_comparisons/     : Old uncalibrated vs. New validation-calibrated threshold performance comparison.
  - 08_paper_text/      : Paper-ready Results, Discussion, and Limitations draft sections.
  - 09_figures/         : High-resolution (300 DPI) publication charts & confusion matrix heatmaps.
  - 10_reproducibility/ : Reproducibility notes, environment specifications, and dataset split integrity.

KEY BENCHMARK RESULTS:
  - BERT Micro F1 Score     : 91.33%
  - BERT Exact Match Acc     : 83.54%
  - NER Micro F1 Score       : 99.24%
  - Legal Retrieval MRR      : 0.9400
  - Legal Grounding Pass     : 100.00%
  - Anti-Hallucination Pass  : 100.00%
  - Fully Grounded Pipeline  : 94.51%
