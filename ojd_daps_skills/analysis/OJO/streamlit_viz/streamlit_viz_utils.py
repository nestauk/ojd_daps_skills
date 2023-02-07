import altair as alt
import pandas as pd

from fnmatch import fnmatch
import json
import pickle
import gzip

import boto3
import os
import streamlit as st

ChartType = alt.vegalite.v4.api.Chart

FILE_NAME = "escoe_extension/inputs/data/analysis/AvertaDemo-Regular.otf"
PATH = os.path.dirname(__file__)

bucket_name = "open-jobs-lake"

NESTA_COLOURS = [
    "#0000FF",
    "#FDB633",
    "#18A48C",
    "#9A1BBE",
    "#EB003B",
    "#FF6E47",
    "#646363",
    "#0F294A",
    "#97D9E3",
    "#A59BEE",
    "#F6A4B7",
    "#D2C9C0",
    # "#FFFFFF",
    "#000000",
]

session = boto3.Session(
    aws_access_key_id=st.secrets["aws_access_key_id"],
    aws_secret_access_key=st.secrets["aws_secret_access_key"],
)

s3 = session.resource("s3")

s3_folder = "escoe_extension/outputs/data"


def download_file_from_s3(
    local_path: str, bucket_name: str = bucket_name, file_name: str = FILE_NAME
):
    """Download a file from S3 to a local path.
    Args:
        local_path (str): Path to save file to.
        bucket_name (str, optional): Bucket name in s3. Defaults to BUCKET_NAME.
        file_name (str, optional): File name in s3. Defaults to FILE_NAME.
    """
    bucket = s3.Bucket(bucket_name)
    bucket.download_file(file_name, local_path)


def load_s3_data(s3, bucket_name, file_name):
    """
    Load data from S3 location.

    s3: S3 boto3 resource
    bucket_name: The S3 bucket name
    file_name: S3 key to load
    """
    obj = s3.Object(bucket_name, file_name)
    if fnmatch(file_name, "*.jsonl.gz"):
        with gzip.GzipFile(fileobj=obj.get()["Body"]) as file:
            return [json.loads(line) for line in file]
    elif fnmatch(file_name, "*.jsonl"):
        file = obj.get()["Body"].read().decode()
        return [json.loads(line) for line in file]
    elif fnmatch(file_name, "*.json.gz"):
        with gzip.GzipFile(fileobj=obj.get()["Body"]) as file:
            return json.load(file)
    elif fnmatch(file_name, "*.json"):
        file = obj.get()["Body"].read().decode()
        return json.loads(file)
    elif fnmatch(file_name, "*.csv"):
        return pd.read_csv("s3://" + bucket_name + "/" + file_name)

        
    elif fnmatch(file_name, "*.pkl") or fnmatch(file_name, "*.pickle"):
        file = obj.get()["Body"].read().decode()
        return pickle.loads(file)
    else:
        print(
            'Function not supported for file type other than "*.csv", "*.jsonl.gz", "*.jsonl", or "*.json"'
        )

def nestafont(font: str = "Averta Demo"):
    """Define Nesta fonts"""
    return {
        "config": {
            "title": {"font": font, "anchor": "start"},
            "axis": {"labelFont": font, "titleFont": font},
            "header": {"labelFont": font, "titleFont": font},
            "legend": {"labelFont": font, "titleFont": font},
            "range": {
                "category": NESTA_COLOURS,
                "ordinal": {
                    "scheme": NESTA_COLOURS
                },  # this will interpolate the colors
            },
        }
    }


alt.themes.register("nestafont", nestafont)
alt.themes.enable("nestafont")


def configure_plots(
    fig,
    font: str = "Averta Demo",
    chart_title: str = "",
    chart_subtitle: str = "",
    fontsize_title: int = 16,
    fontsize_subtitle: int = 13,
    fontsize_normal: int = 13,
):
    """Adds titles, subtitles; configures font sizes; adjusts gridlines"""
    return (
        fig.properties(
            title={
                "anchor": "start",
                "text": chart_title,
                "fontSize": fontsize_title,
                "subtitle": chart_subtitle,
                "subtitleFont": font,
                "subtitleFontSize": fontsize_subtitle,
            },
        )
        .configure_axis(
            gridDash=[1, 7],
            gridColor="grey",
            labelFontSize=fontsize_normal,
            titleFontSize=fontsize_normal,
        )
        .configure_legend(
            titleFontSize=fontsize_title,
            labelFontSize=fontsize_normal,
        )
        .configure_view(strokeWidth=0)
    )

