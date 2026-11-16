"""
ABHERA — Secure Production Model Artifact Uploader to Private Hugging Face Hub

SAFETY & OPERATIONAL RULES:
1. STRICT PRIVATE VISIBILITY: Checks with Hugging Face API that the target repository
   is marked PRIVATE. Aborts immediately if the repository is PUBLIC.
2. SELECTIVE INGESTION: Only uploads the 11 required production inference artifacts
   (no training metrics, datasets, or evaluation reports).
3. SECURE CREDENTIALS: Takes HF_WRITE_TOKEN from interactive hidden input or environment.
   Never logs, echoes, or writes tokens to files or Git history.
4. CHECKSUM VERIFICATION: Verifies every file's SHA-256 hash locally before uploading.

Usage:
  python scripts/upload_models_to_hf.py --repo-id "<your-username>/abhera-models" --dry-run
  python scripts/upload_models_to_hf.py --repo-id "<your-username>/abhera-models"
"""

import argparse
import getpass
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
logger = logging.getLogger("HFModelUploader")

# Authoritative Manifest of Exactly 11 Required Production Files
UPLOAD_MANIFEST = [
    {
        "local_path": PROJECT_ROOT / "models" / "bert_multilabel_experiment_v2" / "config.json",
        "path_in_repo": "bert_multilabel_experiment_v2/config.json",
        "expected_size": 1069,
        "expected_sha256": "2d791de317f20a833be68f842713fbece23c67babcdd9657462ecbe03d27327f",
    },
    {
        "local_path": PROJECT_ROOT / "models" / "bert_multilabel_experiment_v2" / "label_mapping.json",
        "path_in_repo": "bert_multilabel_experiment_v2/label_mapping.json",
        "expected_size": 721,
        "expected_sha256": "3d475d368bcf151f42bd18b04733387846ba4cf5dc9dd8e87bc0352edba35693",
    },
    {
        "local_path": PROJECT_ROOT / "models" / "bert_multilabel_experiment_v2" / "optimal_thresholds.json",
        "path_in_repo": "bert_multilabel_experiment_v2/optimal_thresholds.json",
        "expected_size": 92,
        "expected_sha256": "2d544c8d1129134bdd123f0fded8af6a955c84336f1eaa5b25c9e4d46a47acf8",
    },
    {
        "local_path": PROJECT_ROOT / "models" / "bert_multilabel_experiment_v2" / "tokenizer.json",
        "path_in_repo": "bert_multilabel_experiment_v2/tokenizer.json",
        "expected_size": 711661,
        "expected_sha256": "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0",
    },
    {
        "local_path": PROJECT_ROOT / "models" / "bert_multilabel_experiment_v2" / "tokenizer_config.json",
        "path_in_repo": "bert_multilabel_experiment_v2/tokenizer_config.json",
        "expected_size": 366,
        "expected_sha256": "71cd48543bec670f50e0c9c3556df16b1f58ea720ead6a1b4675f08870c60b17",
    },
    {
        "local_path": PROJECT_ROOT / "models" / "bert_multilabel_experiment_v2" / "model.safetensors",
        "path_in_repo": "bert_multilabel_experiment_v2/model.safetensors",
        "expected_size": 437970928,
        "expected_sha256": "56298549b697a6763da0c4c67e32bcea03acf8a9da74b030e2ebb717b86c9029",
    },
    {
        "local_path": PROJECT_ROOT / "models" / "ner" / "config.json",
        "path_in_repo": "ner/config.json",
        "expected_size": 1423,
        "expected_sha256": "d50bb8e5f65e579d1b97b9019aa08e27faf9d660204b59e9ad98686b46b97b2e",
    },
    {
        "local_path": PROJECT_ROOT / "models" / "ner" / "label_mapping.json",
        "path_in_repo": "ner/label_mapping.json",
        "expected_size": 992,
        "expected_sha256": "d079653d4edcfe3585cd05da70ac6de89f5b9a9d246fc76eae027f79413ac2fc",
    },
    {
        "local_path": PROJECT_ROOT / "models" / "ner" / "tokenizer.json",
        "path_in_repo": "ner/tokenizer.json",
        "expected_size": 711396,
        "expected_sha256": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    },
    {
        "local_path": PROJECT_ROOT / "models" / "ner" / "tokenizer_config.json",
        "path_in_repo": "ner/tokenizer_config.json",
        "expected_size": 366,
        "expected_sha256": "71cd48543bec670f50e0c9c3556df16b1f58ea720ead6a1b4675f08870c60b17",
    },
    {
        "local_path": PROJECT_ROOT / "models" / "ner" / "model.safetensors",
        "path_in_repo": "ner/model.safetensors",
        "expected_size": 435629900,
        "expected_sha256": "45568b7c4b87392de5662dd4c638c3f7dbb7848aea94a27f39dc4cd4c04a4529",
    },
]


def verify_file(item: dict) -> bool:
    path = item["local_path"]
    if not path.exists():
        logger.error(f"Missing file: {path}")
        return False
    size = path.stat().st_size
    if size != item["expected_size"]:
        logger.error(f"Size mismatch for {path.name}: expected {item['expected_size']}, got {size}")
        return False
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        while chunk := fp.read(8192 * 1024):
            h.update(chunk)
    digest = h.hexdigest()
    if digest.lower() != item["expected_sha256"].lower():
        logger.error(f"SHA-256 mismatch for {path.name}!\n  Expected: {item['expected_sha256']}\n  Actual:   {digest}")
        return False
    return True


def run_upload(repo_id: str, dry_run: bool = False):
    logger.info("=================================================================")
    logger.info("ABHERA — SECURE HUGGING FACE MODEL UPLOADER")
    logger.info("=================================================================")
    logger.info(f"Target Repository: {repo_id}")
    logger.info(f"Dry Run Mode: {dry_run}")

    # 1. Verify all 11 files locally before doing anything
    logger.info("Verifying all 11 local model artifacts and SHA-256 digests...")
    total_size = 0
    for item in UPLOAD_MANIFEST:
        if not verify_file(item):
            logger.error("Pre-upload local validation failed! Aborting.")
            sys.exit(1)
        total_size += item["expected_size"]

    logger.info(f"All 11 local files verified successfully! Total upload size: {total_size / (1024*1024):.2f} MB.")

    if dry_run:
        logger.info("[DRY RUN] All files are valid and ready. No upload was performed.")
        return

    # 2. Acquire Write Token securely
    token = os.getenv("HF_TOKEN") or os.getenv("HF_WRITE_TOKEN")
    if not token:
        if sys.stdin.isatty():
            token = getpass.getpass("Enter your Hugging Face Write Token: ").strip()
        else:
            logger.error("HF_TOKEN or HF_WRITE_TOKEN environment variable required in non-interactive mode.")
            sys.exit(1)

    if not token:
        logger.error("Token cannot be empty. Aborting.")
        sys.exit(1)

    from huggingface_hub import HfApi

    api = HfApi()

    # 3. Verify Repository Visibility is strictly PRIVATE
    logger.info(f"Verifying repository '{repo_id}' visibility and permissions...")
    try:
        repo_info = api.repo_info(repo_id=repo_id, repo_type="model", token=token)
        if not repo_info.private:
            logger.error(
                "CRITICAL SECURITY VIOLATION: The repository '%s' is PUBLIC! "
                "ABHERA model weights must never be uploaded to a public repository. "
                "Please change the repository visibility to PRIVATE on Hugging Face before proceeding.",
                repo_id,
            )
            sys.exit(1)
        logger.info("Repository visibility check PASSED: Repository is PRIVATE.")
    except Exception as e:
        logger.error(f"Failed to access repository '{repo_id}' on Hugging Face: {e}")
        logger.error("Ensure the repository exists and your token has WRITE permissions.")
        sys.exit(1)

    # 4. Upload each verified file
    logger.info("Beginning secure artifact upload...")
    for idx, item in enumerate(UPLOAD_MANIFEST, 1):
        rel_repo_path = item["path_in_repo"]
        local_file = item["local_path"]
        logger.info(f"[{idx}/{len(UPLOAD_MANIFEST)}] Uploading {rel_repo_path} ({local_file.stat().st_size / (1024*1024):.2f} MB)...")
        try:
            api.upload_file(
                path_or_fileobj=str(local_file),
                path_in_repo=rel_repo_path,
                repo_id=repo_id,
                repo_type="model",
                token=token,
            )
            logger.info(f"[{idx}/{len(UPLOAD_MANIFEST)}] Successfully uploaded {rel_repo_path}.")
        except Exception as e:
            logger.error(f"Failed to upload {rel_repo_path}: {e}")
            sys.exit(1)

    logger.info("=================================================================")
    logger.info("STATUS: ALL 11 MODEL ARTIFACTS SUCCESSFULLY UPLOADED TO PRIVATE REPO")
    logger.info("=================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload ABHERA models to private Hugging Face repository")
    parser.add_argument("--repo-id", type=str, required=True, help="Target repository, e.g. 'username/abhera-models'")
    parser.add_argument("--dry-run", action="store_true", help="Validate local files without uploading")
    args = parser.parse_args()

    run_upload(repo_id=args.repo_id, dry_run=args.dry_run)
