import streamlit as st
import boto3
import zipfile

import os

session = boto3.Session(
    aws_access_key_id=st.secrets["aws_access_key_id"],
    aws_secret_access_key=st.secrets["aws_secret_access_key"],
)

s3 = session.resource("s3")

BUCKET_NAME = "open-jobs-lake"
FILE_NAME = "escoe_extension/inputs/data/analysis/AvertaDemo-Regular.otf"

PATH = os.path.dirname(__file__)


def download_file_from_s3(
    local_path: str, bucket_name: str = BUCKET_NAME, file_name: str = FILE_NAME
):
    """Download a file from S3 to a local path.

    Args:
        local_path (str): Path to save file to.
        bucket_name (str, optional): Bucket name in s3. Defaults to BUCKET_NAME.
        file_name (str, optional): File name in s3. Defaults to FILE_NAME.
    """
    bucket = s3.Bucket(bucket_name)
    bucket.download_file(file_name, local_path)


def download():
    """Downloads all relevant files for the library."""

    public_data_dir = os.path.join(PATH, 'ojd_daps_skills_data')

    os.system(
            f"aws --no-sign-request --region=eu-west-1 s3 cp s3://open-jobs-indicators/escoe_extension/ojd_daps_skills_data.zip {PATH}/ojd_daps_skills_data.zip"
        )

    os.system(f"unzip {public_data_dir}.zip -d {PATH}")
    os.system(f"rm {public_data_dir}.zip")

