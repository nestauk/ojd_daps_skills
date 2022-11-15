# OJO analysis

The analysis of skills data was done by applying the skills extraction class to a large sample of job adverts in Nesta's Open Jobs Observatory (OJO). Skills were extracted from these adverts and the data was saved out. The data from OJO is not available in the public S3 bucket, and thus this analysis is only for internal Nesta purposes.

## Deduplication of job adverts

To get the ids of deduplicated job adverts run the `ojd_daps_skills/analysis/deduplicate_job_ids.py` script with correct arguments.

For the full dataset and a 6 week time chunk for job stocks (this takes about 4 hours to run):

```
python ojd_daps_skills/analysis/deduplicate_job_ids.py --s3_folder 'escoe_extension/outputs/data/model_application_data' --raw_job_adverts_file_name 'raw_job_adverts_no_desc.csv' --itl_file_name 'job_ad_to_itl_v2.csv' --duplicates_file_name 'job_ad_duplicates.csv' --output_file_name 'deduplicated_job_ids_6_weeks_v2.csv' --num_units 6 --unit_type 'weeks'

```

Running this file will reduce 4918511 job adverts to the deduplicated dataset of 3693313 job adverts. This will save a .csv file of the job ids with the date chunk it is assigned to. The job adverts will be deduplicated by chunking up the job advert dates according to the num_units and unit_type arguments (e.g. 3 days), then if within a chunk of job adverts the raw job advert text and raw location are the same, then only the first will be kept.

## Creating datasets for the analysis

There are two files needed for all analysis pieces:

1. Deduplicated and merged job advert metadata (no skills)
2. A sample of deduplicated skills data merged with the metadata - the skills data is big, so we can only deal with using a sample of it.

Since the skills data file is so large, we first need to download it from S3 to somewhere locally (this will be defined in the "local_skills_file_name" argument).

Running:

```
python ojd_daps_skills/analysis/process_analysis_data.py --s3_folder "escoe_extension/outputs/data/model_application_data" --local_skills_file_name "ojd_daps_skills/analysis/OJO/job_ad_to_skills_v2.json" --dedupe_ids_file_name "deduplicated_job_ids_6_weeks_v2.csv" --itl_file_name "job_ad_to_itl_v2.csv" --occupations_file_name "raw_job_adverts_additional_fields.csv" --sample_skills_output "dedupe_analysis_skills_sample.json" --metadata_output "dedupe_analysis_metadata.csv"

```

will read in the skills data, the deduplicated ids saved out by running `deduplicate_job_ids.py`, and the relevant metadata files. It will output two files:

1. `dedupe_analysis_metadata.csv` the metadata for all job adverts
2. `dedupe_analysis_skills_sample.json` (+ a `.csv` version of this file) the skills + metadata for 100,000 job adverts
   where metadata is the date, location, occupation and summarised skill information (e.g. how many skills were in the job advert).

## Analysis

All the analysis is done in Jupyter notebooks.

### OJO data overview

Note: To plot the Altair figures as pngs in the notebook you will need to run them in jupyterlab.
