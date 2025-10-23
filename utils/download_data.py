import logging
import zipfile
import tempfile
from pathlib import Path
from huggingface_hub import hf_hub_download

from utils import white_noise

logger = logging.getLogger(__name__)

REPO_ID = "NeuroBLab/anatomically_inspired_digital_twin_data"
SESSIONS = [
    "4_7",
    "5_6",
    "5_7",
    "6_2",
    "6_4",
    "6_6",
    "6_7",
    "7_3",
    "7_4",
    "7_5",
    "8_5",
    "9_3",
    "9_4",
    "9_6",
]

def download_and_extract_from_hub(
    repo_id: str,
    filename: str,                 # e.g. "data/microns30/9_6.zip"
    target_dir: str | Path,
    *,
    repo_type: str = "dataset",    # "model" / "dataset" / "space"
    revision: str = "main",
    token: str | None = None,
    verbose: bool = True,
) -> Path:
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        logger.info("Downloading %s from %s@%s …", filename, repo_id, revision)
        zip_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type=repo_type,
                revision=revision,
                token=token,
                local_dir=tmpdir,
            )
        )
        assert zip_path.exists(), f"Download failed or path unexpected: {zip_path}"

        logger.info("Extracting %s → %s", zip_path, target_dir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)

    logger.info("Done. Extracted to: %s", target_dir.resolve())
    return target_dir

if __name__ == "__main__":
    for session in SESSIONS:
        download_and_extract_from_hub(
            REPO_ID,
            f"data/microns30/{session}.zip",
            target_dir=".",
            repo_type="dataset",
            verbose=True,
        )
    
    download_and_extract_from_hub(
        REPO_ID,
        "data/object_rotations.zip",
        target_dir=".",
        repo_type="dataset",
        verbose=True,
    )
    
    download_and_extract_from_hub(
        REPO_ID,
        "data/objects_scaled.zip",
        target_dir=".",
        repo_type="dataset",
        verbose=True,
    )

    download_and_extract_from_hub(
        REPO_ID,
        "data/microns30/microns30_metadata.zip",
        target_dir=".",
        repo_type="dataset",
        verbose=True,
    )
    
    white_noise.main()