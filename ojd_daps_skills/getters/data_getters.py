"""Script to load data locally and from s3.
"""
#########################################
from fnmatch import fnmatch
import pandas as pd
import glob
import os
import json

#########################################


def load_local_data(data_path: str):
    """Loads local data as pd.DataFrame.
	Args:
		data_path (str): Local data path.

	"""
    if fnmatch(data_path, "*.csv"):
        return pd.read_csv(data_path)
    else:
        print("only supports '*.csv' file extension.")


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
