from ojd_daps_skills import PUBLIC_DATA_FOLDER_NAME, PROJECT_DIR

import os
import boto3
from botocore.exceptions import ClientError
from botocore import UNSIGNED
from botocore.config import Config
from zipfile import ZipFile

def download():
    """Download public data. Expected to run once on first use."""
    s3 = boto3.client(
        "s3", region_name="eu-west-1", config=Config(signature_version=UNSIGNED)
    )

    bucket_name = "open-jobs-indicators"
    key = f"escoe_extension/{PUBLIC_DATA_FOLDER_NAME}.zip"

    public_data_dir = os.path.join(PROJECT_DIR, PUBLIC_DATA_FOLDER_NAME)

    try:
        s3.download_file(bucket_name, key, f"{public_data_dir}.zip")

        with ZipFile(f"{public_data_dir}.zip", "r") as zip_ref:
            zip_ref.extractall(PROJECT_DIR)

        os.remove(f"{public_data_dir}.zip")

    except ClientError as ce:
        print(f"Error: {ce}")
    except FileNotFoundError as fnfe:
        print(f"Error: {fnfe}")


if __name__ == "__main__":
    download()
