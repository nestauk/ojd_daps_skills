import os
from zipfile import ZipFile

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
from wasabi import msg

from ojd_daps_skills import PROJECT_DIR, PUBLIC_DATA_FOLDER_PATH


def download_data():
    """Download public data. Expected to run once on first use."""
    s3 = boto3.client(
        "s3", region_name="eu-west-2", config=Config(signature_version=UNSIGNED)
    )

    bucket_name = "nesta-open-data"
    key = "escoe_extension/ojd_daps_skills_data_refactor.zip"

    try:
        s3.download_file(
            bucket_name, key, f"{str(PUBLIC_DATA_FOLDER_PATH)}_refactor.zip"
        )

        with ZipFile(f"{PUBLIC_DATA_FOLDER_PATH}_refactor.zip", "r") as zip_ref:
            zip_ref.extractall(PROJECT_DIR)

        os.remove(f"{PUBLIC_DATA_FOLDER_PATH}_refactor.zip")
        msg.info(f"Data folder downloaded from {PUBLIC_DATA_FOLDER_PATH}")

    except ClientError as ce:
        msg.warn(f"Error: {ce}")
    except FileNotFoundError as fnfe:
        msg.warn(f"Error: {fnfe}")
