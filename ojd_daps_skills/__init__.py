"""ojd_daps_skills."""

import warnings
from pathlib import Path
from typing import Optional

import yaml
from spacy.tokens import Doc

warnings.filterwarnings("ignore")


def get_yaml_config(file_path: Path) -> Optional[dict]:
    """Fetch yaml config and return as dict if it exists."""
    if file_path.exists():
        with open(file_path, "rt") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)


PROJECT_DIR = Path(__file__).resolve().parents[1]

info_out = str(PROJECT_DIR / "info.log")
error_out = str(PROJECT_DIR / "errors.log")

_base_config_path = Path(__file__).parent.resolve() / "config/base.yaml"
config = get_yaml_config(_base_config_path)

bucket_name = "open-jobs-lake"

PUBLIC_DATA_FOLDER_PATH = PROJECT_DIR / "ojd_daps_skills_data"
PUBLIC_MODEL_FOLDER_PATH = PROJECT_DIR / "ojd_daps_skills_models"


def setup_spacy_extensions():
    if not Doc.has_extension("skill_spans"):
        Doc.set_extension("skill_spans", default=[], force=True)
    if not Doc.has_extension("mapped_skills"):
        Doc.set_extension("mapped_skills", default=[], force=True)
