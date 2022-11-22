## Evaluation methods

This directory contains the approaches taken to evaluate the current skills extraction algorithm.

### 1. Aggregated ESCO-OJO occupation evaluation

This approach compares a list of ESCO skills with a list of extracted OJO skills per occupation that is in both ESCO and OJO. In this instance, we are using a sample of 100,000 job adverts with extracted skills.

To ensure that there are a reasonable amount of job adverts per ESCO occupation, we only examine occupations that have at least 100 job adverts associated to them. We also only compare top skills in the 75% quartile or greater.

The output is a .json where we report on the skills that are mentioned in at least X percent of OJO job adverts AND ESCO skills per occuption, skills mentioned in OJO but not ESCO, skills mentioned in ESCO but not OJO and % of top OJO skills mentioned in essential ESCO skills.

To run the script:

`python aggregate_ojo_esco_evaluation.py`

### 2. Lightcast skills

To extract Lightcast skills from a random sample of 50 OJO job adverts, you need to [first create an account with Lightcast]("https://skills.lightcast.io/extraction"). They will send you API credentials that you will need to run the script.

With your emailed credentials, to run the script:
`python ojd_daps_skills/pipeline/evaluation/lightcast_evaluation.py --client-id CLIENT_ID --client-secret CLIENT_SECRET`

This will output a saved .json with job ids, job description, extracted OJO Lightcast skills and Lightcast skills. BEWARE: you can only call the API 50 times A MONTH! So running the script will take you out for the whole month.
