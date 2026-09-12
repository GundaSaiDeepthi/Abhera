# Model Artifacts Directory

This directory stores trained machine learning models, tokenizers, and configuration files.

## Subdirectory Structure
- `models/bert/`: Fine-tuned BERT model weights (`pytorch_model.bin` / `model.safetensors`), `config.json`, `tokenizer.json`, and `label_encoder.pkl`.
- `models/ner/`: Fine-tuned NER entity extraction weights and configs.

> Note: Large model binaries are ignored by git via `.gitignore`.
