"""
Download TableFormer model for local use with Docling
"""
from pathlib import Path
import sys

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("ERROR: huggingface_hub not installed")
    print("Install with: pip install huggingface_hub")
    sys.exit(1)

# Set up paths
models_dir = Path(__file__).parent / "models" / "ds4sd--docling-models"

print(f"Downloading TableFormer model to: {models_dir}")
print("This may take several minutes...")
print()

try:
    snapshot_download(
        repo_id="ds4sd/docling-models",
        revision="v2.3.0",
        local_dir=str(models_dir),
        local_dir_use_symlinks=False,
    )
    print()
    print("✓ Download complete!")
    print(f"Model saved to: {models_dir}")
except Exception as e:
    print(f"✗ Error downloading model: {e}")
    sys.exit(1)
