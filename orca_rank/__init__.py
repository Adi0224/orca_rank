import sys
from pathlib import Path

__version__ = "0.1.0"

# Vendored ORCA OTDD package (import otdd/pytorch/distance ...)
_OTDD_PARENT = Path(__file__).resolve().parent.parent / "third_party" / "orca_otdd"
if _OTDD_PARENT.is_dir() and str(_OTDD_PARENT) not in sys.path:
    sys.path.insert(0, str(_OTDD_PARENT))
