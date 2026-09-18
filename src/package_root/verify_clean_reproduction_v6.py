from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from verify_clean_reproduction_v6 import main  # noqa: E402


if __name__ == "__main__":
    main()
