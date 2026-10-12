import sys
from pathlib import Path
import runpy

# Ensure repository root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Run dashboard module directly
_DASHBOARD_SCRIPT = _PROJECT_ROOT / "src" / "dashboard" / "app.py"
runpy.run_path(str(_DASHBOARD_SCRIPT), run_name="__main__")
