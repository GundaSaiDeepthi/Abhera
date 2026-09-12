# Machine Learning Pipeline Architecture

## Architecture Overview
The machine learning component comprises two main sub-systems:

### 1. BERT Multi-Label Classifier (`ml/bert/`)
- **Task**: Multi-Label Incident Classification on labels `DV` (Domestic Violence), `SH` (Sexual Harassment), `ST` (Stalking), `CA` (Cyber Abuse), `WH` (Workplace Harassment), and `OV` (Other Violence).
- **Base Architecture**: `bert-base-uncased` fine-tuned with binary cross-entropy loss (`BCEWithLogitsLoss`) for multi-label prediction.
- **Dataset**: `data/womens_safety_dataset.csv`.

### 2. NER Entity Extraction (`ml/ner/`)
- **Task**: Extraction of key incident entities (Perpetrator Info, Location, Time/Frequency, Digital Platform, Physical Harm / Threat details).
- **Dataset**: Formatted annotation datasets derived from training narratives.

## Model Output Path
All trained model checkpoints, tokenizers, and label encoders will be saved to `models/bert/` and `models/ner/`.
