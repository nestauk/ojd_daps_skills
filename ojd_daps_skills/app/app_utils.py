import streamlit as st
import boto3

session = boto3.Session(
    aws_access_key_id=st.secrets["aws_access_key_id"],
    aws_secret_access_key=st.secrets["aws_secret_access_key"],
)

s3 = session.resource("s3")

BUCKET_NAME = "open-jobs-lake"
FILE_NAME = "escoe_extension/inputs/data/analysis/AvertaDemo-Regular.otf"


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
