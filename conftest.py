"""Make the repository root importable so `tests/` can import `scripts.*`.

Without this, `pytest` collects tests/test_osf_evidence.py only when invoked as
`python -m pytest` (which happens to put the working directory on sys.path).
Pinning it here makes the suite invocation independent and therefore reproducible.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
