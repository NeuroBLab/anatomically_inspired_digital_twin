import logging
import zipfile
import tempfile
from pathlib import Path
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

REPO_ID = "NeuroBLab/anatomically_inspired_digital_twin_models"
FILENAME = "selected_models.zip"
TARGET_DIR = "data/models"

def download_and_extract_from_hub(
    repo_id: str,
    filename: str,                 # path inside repo (e.g. "models/microns.zip")
    target_dir: str | Path,        # where to extract
    *,
    repo_type: str = "model",
    revision: str = "main",
    token: str | None = None,
    verbose: bool = True,
) -> Path:
    """
    Download a ZIP file from a Hugging Face Hub repo, extract it to `target_dir`,
    and delete the downloaded zip to save space.

    Returns
    -------
    Path : the extraction directory
    """
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        zip_path = tmpdir / Path(filename).name

        # Download zip from HF Hub directly into the temporary directory
        logger.info(f"Downloading {filename} from {repo_id} …")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=revision,
            token=token,
            local_dir=tmpdir,
            local_dir_use_symlinks=False,
        )

        logger.info(f"Extracting {zip_path} → {target_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)

        # Delete zip after extraction (since it’s inside a TemporaryDirectory, this happens automatically)
        logger.info("Extraction complete. ZIP file removed.")

    return target_dir

if __name__ == "__main__":
    download_and_extract_from_hub(
        REPO_ID,
        FILENAME,
        TARGET_DIR,
        repo_type="model",
        verbose=True,
    )