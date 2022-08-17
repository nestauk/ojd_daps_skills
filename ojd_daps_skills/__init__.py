# %%
"""ojd_daps_skills."""
import logging
import logging.config
from pathlib import Path
from typing import Optional

# %%
import yaml
# %%
def get_yaml_config(file_path: Path) -> Optional[dict]:
    """Fetch yaml config and return as dict if it exists."""
    if file_path.exists():
        with open(file_path, "rt") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)


# %%
# Define project base directory
PROJECT_DIR = Path(__file__).resolve().parents[1]

# %%
# Define log output locations
info_out = str(PROJECT_DIR / "info.log")
error_out = str(PROJECT_DIR / "errors.log")

# %%
# Read log config file
_log_config_path = Path(__file__).parent.resolve() / "config/logging.yaml"
_logging_config = get_yaml_config(_log_config_path)
if _logging_config:
    logging.config.dictConfig(_logging_config)

# %%
# Define module logger
class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    bold_red = "\x1b[31;1m"
    blue = '\033[94m'
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: bold_red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
if (logger.hasHandlers()):
    logger.handlers.clear()
    
logger.addHandler(ch)
# %%
# base/global config
_base_config_path = Path(__file__).parent.resolve() / "config/base.yaml"
config = get_yaml_config(_base_config_path)

# The S3 bucket we are using
bucket_name = "open-jobs-lake"
