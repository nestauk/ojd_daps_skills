"""
Currently the code in this script needs to be run in the ojd_daps repo folder using it's conda environment.
It is kept in this repo for completeness.

It will output to S3 a random sample of the job adverts for each month,
each month's data will be saved in a separate file.

This needs to be run whilst being on Nesta's VPN.

There is a runtime function since for some of the latest months we get the message and would like to skip over it:

WARNING:ojd_daps.dqa.shared_cache: No cache found for get_db_job_ads with args () and kwargs
{'chunksize': 10000, 'return_description': False, 'return_features': True, 'deduplicate': True,
'min_dupe_weight': 0.95, 'max_dupe_weight': 1, 'split_dupes_by_location': True,
'from_date': '11-04-2022', 'to_date': '23-05-2022'}.
Evaluating this now, but it may take some time.

This causes problems in the data files for the most recent months

"""

import json
import random
import os
import threading
import time

from tqdm import tqdm
import boto3

from ojd_daps.dqa.data_getters import (
    get_cached_job_ads,
    fetch_descriptions,
    get_valid_cache_dates,
)

BUCKET_NAME = "open-jobs-lake"
S3_FOLDER = "escoe_extension/inputs/data/skill_ner/data_sample/"
sample_size_per_date = 100

date_blocks = get_valid_cache_dates()


def save_to_s3(s3, bucket_name, output_var, output_file_dir):
    obj = s3.Object(bucket_name, output_file_dir)
    byte_obj = json.dumps(output_var, default=str)
    obj.put(Body=byte_obj)
    print(f"Saved to s3://{bucket_name} + {output_file_dir} ...")


class RunWithTimeout(object):
    def __init__(self, get_cached_job_ads, start_date, end_date):
        self.get_cached_job_ads = get_cached_job_ads
        self.start_date = start_date
        self.end_date = end_date
        self.answer = None

    def worker(self):
        self.answer = self.get_cached_job_ads(self.start_date, self.end_date)

    def run(self, timeout):
        thread = threading.Thread(target=self.worker)
        thread.start()
        thread.join(timeout)
        return self.answer


s3 = boto3.resource("s3")
for start_date, end_date in tqdm(date_blocks):
    # job_ads = get_cached_job_ads(start_date, end_date)
    n = RunWithTimeout(get_cached_job_ads, start_date, end_date)
    job_ads = n.run(200)
    if not job_ads:
        print(f"Skipping cache {start_date} to {end_date} due to time out")
        continue
    random.seed(42)
    sampled_job_ads = random.sample(job_ads, min(len(job_ads), sample_size_per_date))
    sampled_job_ads_dict = {}
    for j in sampled_job_ads:
        sampled_job_ads_dict[j.get("id")] = j
    # Query in chunks
    print(f"Getting job descriptions for {len(sampled_job_ads_dict)} random jobs")
    sample_job_ids = list(sampled_job_ads_dict.keys())
    chunk_n = 50
    for chunk in tqdm(
        [
            sample_job_ids[i : i + chunk_n]
            for i in range(0, len(sample_job_ids), chunk_n)
        ]
    ):
        descriptions = fetch_descriptions(chunk)
        for k, v in descriptions.items():
            sampled_job_ads_dict[k]["description"] = v
        save_to_s3(
            s3,
            BUCKET_NAME,
            sampled_job_ads_dict,
            os.path.join(S3_FOLDER, f"sampled_job_ads_{start_date}_{end_date}.json"),
        )
