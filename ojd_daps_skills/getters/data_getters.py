from fnmatch import fnmatch
import json
import pickle
import gzip
import os

import pandas as pd
from pandas import DataFrame
import boto3


def load_data(file_name: str, local=True) -> DataFrame:
    """Loads data from path.
    Args:
            file_name (str): Local path to data.
    Returns:
            file (pd.DataFrame): Loaded Data in pd.DataFrame
    """
    if local:
        if fnmatch(file_name, "*.csv"):
            return pd.read_csv(file_name)
        else:
            print(f'{file_name} has wrong file extension! Only supports "*.csv"')


def load_json_dict(file_name: str) -> dict:

    """Loads a dict stored in a json file from path.
    Args:
            file_name (str): Local path to json.
    Returns:
            file (dict): Loaded dict
    """
    if fnmatch(file_name, "*.json"):
        with open(file_name, "r") as file:
            return json.load(file)
    else:
        print(f'{file_name} has wrong file extension! Only supports "*.json"')


def save_json_dict(dictionary: dict, file_name: str):

    """Saves a dict to a json file.

    Args:
            dictionary (dict): The dictionary to be saved
            file_name (str): Local path to json.
    """
    if fnmatch(file_name, "*.json"):
        with open(file_name, "w") as file:
            json.dump(dictionary, file)
    else:
        print(f'{file_name} has wrong file extension! Only supports "*.json"')


def load_txt_lines(file_name: str) -> list:
    txt_list = []
    with open(file_name) as file:
        for line in file:
            txt_list.append(line.rstrip())
    return txt_list


def get_s3_resource():
    s3 = boto3.resource("s3")
    return s3


def save_to_s3(s3, bucket_name, output_var, output_file_dir):

    obj = s3.Object(bucket_name, output_file_dir)

    if fnmatch(output_file_dir, "*.pkl") or fnmatch(output_file_dir, "*.pickle"):
        byte_obj = pickle.dumps(output_var)
    elif fnmatch(output_file_dir, "*.gz"):
        byte_obj = gzip.compress(json.dumps(output_var).encode())
    elif fnmatch(output_file_dir, "*.txt"):
        byte_obj = output_var
    else:
        byte_obj = json.dumps(output_var)
    obj.put(Body=byte_obj)

    print(f"Saved to s3://{bucket_name} + {output_file_dir} ...")


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
        return pd.read_csv(os.path.join("s3://" + bucket_name, file_name))
    elif fnmatch(file_name, "*.pkl") or fnmatch(file_name, "*.pickle"):
        file = obj.get()["Body"].read().decode()
        return pickle.loads(file)
    else:
        print(
            'Function not supported for file type other than "*.jsonl.gz", "*.jsonl", or "*.json"'
        )


def get_s3_data_paths(s3, bucket_name, root, file_types=["*.jsonl"]):
    """
    Get all paths to particular file types in a S3 root location

    s3: S3 boto3 resource
    bucket_name: The S3 bucket name
    root: The root folder to look for files in
    file_types: List of file types to look for, or one
    """
    if isinstance(file_types, str):
        file_types = [file_types]

    bucket = s3.Bucket(bucket_name)

    s3_keys = []
    for files in bucket.objects.filter(Prefix=root):
        key = files.key
        if any([fnmatch(key, pattern) for pattern in file_types]):
            s3_keys.append(key)

    return s3_keys
