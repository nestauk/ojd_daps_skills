# Analysis

## Deduplication of job adverts

To get the ids of deduplicated job adverts run the `ojd_daps_skills/analysis/deduplicate_job_ids.py` script with correct arguments.

For the sample of job adverts this is:

```
python ojd_daps_skills/analysis/deduplicate_job_ids.py --s3_folder 'escoe_extension/outputs/data/model_application_data' --raw_job_adverts_file_name 'raw_job_adverts_sample.csv' --itl_file_name 'job_ad_to_itl_sample.csv' --duplicates_file_name 'job_ad_duplicates_sample.csv' --output_file_name 'deduplicated_job_ids_sample.csv' --num_units 3 --unit_type 'days'
```

This will save a .csv file of the job ids with the date chunk it is assigned to. The job adverts will be deduplicated by chunking up the job advert dates according to the num_units and unit_type arguments (e.g. 3 days), then if within a chunk of job adverts the raw job advert text and raw location are the same, then only the first will be kept.

For the full dataset and a 14 day time chunk for job stocks (this takes about 10 hours to run):

```
python ojd_daps_skills/analysis/deduplicate_job_ids.py --s3_folder 'escoe_extension/outputs/data/model_application_data' --raw_job_adverts_file_name 'raw_job_adverts_no_desc.csv' --itl_file_name 'job_ad_to_itl.csv' --duplicates_file_name 'job_ad_duplicates.csv' --output_file_name 'deduplicated_job_ids.csv' --num_units 14 --unit_type 'days'

```

Running this file will reduce 4918511 job adverts to the deduplicated dataset of 4009973 job adverts.
