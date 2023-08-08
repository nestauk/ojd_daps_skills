# Using Prodigy to tag entities

Activate your own prodigy environment.

For example:

```
conda create --name liz_prodigy pip python=3.8
conda activate liz_prodigy
pip install prodigy -f https://[YOUR_LICENSE_KEY]@download.prodi.gy
```

## Data

Merge 5000 random job adverts into a format readable for Prodigy by running

```
python ojd_daps_skills/pipeline/skill_ner/prodigy/process_data.py

```

in the `ojd-daps-skills` conda environment.

This will create `s3://open-jobs-lake/escoe_extension/outputs/labelled_job_adverts/prodigy/processed_sample_20230801.jsonl`.

## Tagging skills

This is all to be done in your own Prodigy environment, and the commands in this section should be runnable independently from this repo (so could be run in a new and empty EC2 instance).

First download the data locally to the file location you are running prodigy from

```
aws s3 cp s3://open-jobs-lake/escoe_extension/outputs/labelled_job_adverts/prodigy/processed_sample_20230801.jsonl prodigy_data/processed_sample_20230801.jsonl

```

Create two empty folders for the outputs.

```
mkdir ./prodigy_data/models/
mkdir ./prodigy_data/labelled_data/

```

Copy the original model (trained on 375 job adverts) to this location:

```
aws s3 cp --recursive s3://open-jobs-lake/escoe_extension/outputs/models/ner_model/20220825/ ./prodigy_data/models/20220825_model/

```

Then open up the tagging task by running.

```
prodigy ner.correct_skills dataset-skills ./prodigy_data/models/20220825_model/ prodigy_data/processed_sample_20230801.jsonl --label SKILL,MULTISKILL,EXPERIENCE,BENEFIT -F skill_recipe.py --update
```

Your task is to manually annotate all the SKILLs, MULTISKILL,EXPERIENCE,BENEFIT in the sentences you are provided with. These are job adverts cut up into lengths of 5 sentences (separated by full stop).

You must provide the session url argument (with your name) when labelling the tasks if this is hosted on EC2, e.g. `http://18.XXX:8080/?session=liz`. This makes it so no two labellers will end up annotating the same task. Without it each time someone tried to label the stream of tasks will be exactly the same as another labeller.

Output the annotations

```
prodigy db-out dataset-skills > ./prodigy_data/labelled_data/dataset_skills_080823.jsonl
aws s3 cp ./prodigy_data/labelled_data/dataset_skills_080823.jsonl s3://open-jobs-lake/escoe_extension/outputs/labelled_job_adverts/prodigy/labelled_dataset_skills_080823.jsonl
```
