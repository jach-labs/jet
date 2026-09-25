"""Files shared by the release scripts (merge_release, validate_release, publish_release)."""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = ROOT / 'releases' / 'jet-v6'
# Prompt and answer code ships from src/ so the release never carries a stale copy.
SHARED_CODE = ['format.py', 'inference.py']


def sync_code(release: Path) -> None:
    for name in SHARED_CODE:
        shutil.copy2(ROOT / 'src' / name, release / name)
