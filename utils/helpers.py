import yaml
from pathlib import Path


def load_yaml(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent
