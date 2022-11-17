"""ojd_daps_skills."""
import logging
import logging.config
from pathlib import Path
from typing import Optional
import re
import sentence_transformers
import boto3

import yaml
import warnings

warnings.filterwarnings("ignore")


def get_yaml_config(file_path: Path) -> Optional[dict]:
    """Fetch yaml config and return as dict if it exists."""
    if file_path.exists():
        with open(file_path, "rt") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)


PROJECT_DIR = Path(__file__).resolve().parents[1]

info_out = str(PROJECT_DIR / "info.log")
error_out = str(PROJECT_DIR / "errors.log")


class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    bold_yellow = "\x1b[33;20;1;1m"
    bold_red = "\x1b[31;1m"
    bold_blue = "\x1b[94;1;1m"
    reset = "\x1b[0m"
    format = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: bold_blue + format + reset,
        logging.WARNING: bold_yellow + format + reset,
        logging.ERROR: bold_red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logger = logging.getLogger(
    "SkillsExtractor"
)  # NOTE: change logger name once we decide what library will be called

prefix_re = re.compile(fr'^(?:{ "|".join(["sentence_transformers", "boto"]) })')
for name in logging.root.manager.loggerDict:
    if re.match(prefix_re, name):
        logging.getLogger(name).setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setFormatter(CustomFormatter())
if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(ch)
logger.propagate = False

_base_config_path = Path(__file__).parent.resolve() / "config/base.yaml"
config = get_yaml_config(_base_config_path)

bucket_name = "open-jobs-lake"

PUBLIC_DATA_FOLDER_NAME = "ojd_daps_skills_data"
