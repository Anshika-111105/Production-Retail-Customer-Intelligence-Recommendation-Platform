"""Configuration loader and validator."""

import sys
from pathlib import Path

# Ensure project root is in sys.path for direct execution or IDE imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load and parse a YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class AppConfig:
    """Consolidated application configuration container."""

    def __init__(
        self,
        base_config_path: str = "configs/config.yaml",
        segmentation_config_path: str = "configs/segmentation.yaml",
        recommendation_config_path: str = "configs/recommendation.yaml",
    ):
        self.base = load_yaml_config(base_config_path)
        self.segmentation = load_yaml_config(segmentation_config_path)
        self.recommendation = load_yaml_config(recommendation_config_path)

    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """Fetch config value with fallback."""
        data = getattr(self, section, {}) if hasattr(self, section) else self.base.get(section, {})
        if key is None:
            return data
        return data.get(key, default)
