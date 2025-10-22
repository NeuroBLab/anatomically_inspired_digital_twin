import tarfile, zstandard as zstd
from pathlib import Path
import hashlib, json

ARCHIVES_DIR = Path("data/hf_cache/downloaded_archives")  # where HF saved shards
RESTORE_DIR  = Path("data/restored")  # where you want the folder tree

def sha256_file(p: Path, bufsize=1024*1024):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b: break
            h.update(b)
    return h.hexdigest()

def extract_all():
    manifest = json.loads((ARCHIVES_DIR / "manifest.json").read_text())
    RESTORE_DIR.mkdir(parents=True, exist_ok=True)
    for shard in manifest["shards"]:
        path = ARCHIVES_DIR / shard["name"]
        # optional integrity check
        assert sha256_file(path) == shard["sha256"], f"Checksum mismatch: {path}"
        dctx = zstd.ZstdDecompressor()
        with open(path, "rb") as raw, dctx.stream_reader(raw) as zfh, tarfile.open(fileobj=zfh, mode="r|") as tar:
            tar.extractall(path=RESTORE_DIR)  # preserves relative paths inside
    print("Extraction complete.")

if __name__ == "__main__":
    extract_all()
