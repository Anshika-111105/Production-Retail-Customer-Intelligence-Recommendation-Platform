"""Retail Customer Intelligence & Recommendation Engine Package."""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path automatically
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

__version__ = "1.0.0"
