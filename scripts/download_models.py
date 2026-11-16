"""
ABHERA — Secure Production Model Artifact Downloader & Checksum Verifier

This utility downloads model artifacts from a private Hugging Face repository
into the expected local directory structure, strictly verifying file sizes
and SHA-256 checksums before allowing application boot.

Security & Environment Requirements:
- HF_TOKEN: Hugging Face read-only user access token (configured in Render environment).
- HF_MODEL_REPO: Hugging Face private repository ID (e.g., 'username/abhera-models').
- Never hardcodes or logs tokens.
"""

import hashlib
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ModelArtifactManager")

# Authoritative Manifest of Exactly 11 Production Model Artifacts
MODEL_MANIFEST = {
    # 1. BERT Multi-Label Experiment v2 Artifacts
    "bert_multilabel_experiment_v2/config.json": {
        "target_path": PROJECT_ROOT / "models" / "bert_multilabel_experiment_v2" / "config.json",
        "expected_size": 1069,
        "expected_sha256": "2d791de317f20a833be68f842713fbece23c67babcdd9657462ecbe03d27327f",
    },
    "bert_multilabel_experiment_v2/label_mapping.json": {
        "target_path": PROJECT_ROOT / "models" / "bert_multilabel_experiment_v2" / "label_mapping.json",
        "expected_size": 721,
        "expected_sha256": "3d475d368bcf151f42bd18b04733387846ba4cf5dc9dd8e87bc0352edba35693",
    },
    "bert_multilabel_experiment_v2/optimal_thresholds.json": {
        "target_path": PROJECT_ROOT / "models" / "bert_multilabel_experiment_v2" / "optimal_thresholds.json",
        "expected_size": 92,
        "expected_sha256": "2d544c8d1129134bdd123f0fded8af6a955c84336f1eaa5b25c9e4d46a47acf8",
    },
    "bert_multilabel_experiment_v2/tokenizer.json": {
        "target_path": PROJECT_ROOT / "models" / "bert_multilabel_experiment_v2" / "tokenizer.json",
        "expected_size": 711661,
        "expected_sha256": "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0",
    },
    "bert_multilabel_experiment_v2/tokenizer_config.json": {
        "target_path": PROJECT_ROOT / "models" / "bert_multilabel_experiment_v2" / "tokenizer_config.json",
        "expected_size": 366,
        "expected_sha256": "71cd48543bec670f50e0c9c3556df16b1f58ea720ead6a1b4675f08870c60b17",
    },
    "bert_multilabel_experiment_v2/model.safetensors": {
        "target_path": PROJECT_ROOT / "models" / "bert_multilabel_experiment_v2" / "model.safetensors",
        "expected_size": 437970928,
        "expected_sha256": "56298549b697a6763da0c4c67e32bcea03acf8a9da74b030e2ebb717b86c9029",
    },

    # 2. Named Entity Recognition (NER) Artifacts
    "ner/config.json": {
        "target_path": PROJECT_ROOT / "models" / "ner" / "config.json",
        "expected_size": 1423,
        "expected_sha256": "d50bb8e5f65e579d1b97b9019aa08e27faf9d660204b59e9ad98686b46b97b2e",
    },
    "ner/label_mapping.json": {
        "target_path": PROJECT_ROOT / "models" / "ner" / "label_mapping.json",
        "expected_size": 992,
        "expected_sha256": "d079653d4edcfe3585cd05da70ac6de89f5b9a9d246fc76eae027f79413ac2fc",
    },
    "ner/tokenizer.json": {
        "target_path": PROJECT_ROOT / "models" / "ner" / "tokenizer.json",
        "expected_size": 711396,
        "expected_sha256": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    },
    "ner/tokenizer_config.json": {
        "target_path": PROJECT_ROOT / "models" / "ner" / "tokenizer_config.json",
        "expected_size": 366,
        "expected_sha256": "71cd48543bec670f50e0c9c3556df16b1f58ea720ead6a1b4675f08870c60b17",
    },
    "ner/model.safetensors": {
        "target_path": PROJECT_ROOT / "models" / "ner" / "model.safetensors",
        "expected_size": 435629900,
        "expected_sha256": "45568b7c4b87392de5662dd4c638c3f7dbb7848aea94a27f39dc4cd4c04a4529",
    },
}


def verify_file_checksum(file_path: Path, expected_size: int, expected_sha256: str) -> bool:
    """Verifies that an existing file matches the exact byte size and SHA-256 digest."""
    if not file_path.exists():
        return False

    actual_size = file_path.stat().st_size
    if actual_size != expected_size:
        logger.warning(f"Size mismatch for {file_path.name}: expected {expected_size}, got {actual_size}.")
        return False

    hasher = hashlib.sha256()
    with open(file_path, "rb") as fp:
        while chunk := fp.read(8192 * 1024):
            hasher.update(chunk)

    actual_sha256 = hasher.hexdigest()
    if actual_sha256.lower() != expected_sha256.lower():
        logger.error(f"SHA-256 MISMATCH for {file_path.name}!\n  Expected: {expected_sha256}\n  Actual:   {actual_sha256}")
        return False

    return True


def ensure_models_available():
    logger.info("=================================================================")
    logger.info("ABHERA — PRODUCTION MODEL ARTIFACT VERIFICATION & INGESTION")
    logger.info("=================================================================")

    # Step 1: Check if all 11 model artifacts already exist locally and pass checksum
    all_present_and_valid = True
    for rel_path, spec in MODEL_MANIFEST.items():
        target_path = spec["target_path"]
        if not verify_file_checksum(target_path, spec["expected_size"], spec["expected_sha256"]):
            all_present_and_valid = False
            logger.info(f"Artifact {rel_path} requires download/refresh.")
            break

    if all_present_and_valid:
        logger.info(f"All {len(MODEL_MANIFEST)} required model artifacts are verified byte-for-byte on disk. Ready for inference!")
        return

    # Step 2: Download missing models from Private Hugging Face repository
    hf_token = os.getenv("HF_TOKEN")
    hf_repo = os.getenv("HF_MODEL_REPO")

    if not hf_token or not hf_repo:
        logger.error(
            "CRITICAL ERROR: Model artifacts are missing or invalid, and required Hugging Face credentials "
            "are not set! Please configure HF_TOKEN and HF_MODEL_REPO in Render's Environment Variables."
        )
        sys.exit(1)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error("huggingface_hub package is not installed. Add huggingface_hub to requirements.")
        sys.exit(1)

    for rel_path, spec in MODEL_MANIFEST.items():
        target_path = spec["target_path"]
        if verify_file_checksum(target_path, spec["expected_size"], spec["expected_sha256"]):
            logger.info(f"Artifact {rel_path} already valid. Skipping download.")
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        filename = rel_path
        logger.info(f"Downloading {filename} from private repository '{hf_repo}'...")

        try:
            downloaded_temp_path = hf_hub_download(
                repo_id=hf_repo,
                filename=filename,
                token=hf_token,
                local_dir=str(PROJECT_ROOT / "models"),
            )
            logger.info(f"Successfully downloaded: {downloaded_temp_path}")
        except Exception as e:
            logger.error(f"Failed to download {filename} from Hugging Face: {e}")
            sys.exit(1)

        # Re-verify downloaded file
        if not verify_file_checksum(target_path, spec["expected_size"], spec["expected_sha256"]):
            logger.error(f"Downloaded artifact {target_path} failed integrity verification! Removing file.")
            if target_path.exists():
                target_path.unlink()
            sys.exit(1)

    logger.info("=================================================================")
    logger.info(f"STATUS: ALL {len(MODEL_MANIFEST)} PRODUCTION MODEL ARTIFACTS VERIFIED AND READY")
    logger.info("=================================================================")


if __name__ == "__main__":
    ensure_models_available()
