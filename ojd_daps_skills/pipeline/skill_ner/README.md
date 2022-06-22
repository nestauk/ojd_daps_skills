# Skill NER

## Label data

### Creating a sample of the OJO data

We first create a sample of job adverts to label.

First, connected to the Nesta VPN and in your activated conda environment, export your SQL credentials location to an environmental variable:

```
export SQL_DB_CREDS="$HOME/path/to/mysqldb_team_ojo_may22.config"
```

Then run the script with:

```
python ojd_daps_skills/pipeline/skill_ner/create_data_sample.py
```

The random sample of job adverts created from this will be stored in S3 in the `open-jobs-lake` bucket in the `/escoe_extension/inputs/data/skill_ner/data_sample/` folder.

### Processing the job adverts

After creating this random sample we process it into a form suitable for labelling in label-studio. To make the process less overwhelming, we decided that the labelling task would be performed sentence by sentence rather than labelling an entire document at a time. Thus we had to save out all the sentences of our sample of job adverts.

Back to this `ojd_daps_skills` repo and conda environment, this processing is done by first downloading a Spacy model needed for sentence separation:

```
python -m spacy download en_core_web_sm
```

then running:

```
python ojd_daps_skills/pipeline/skill_ner/create_label_data.py
```

An output file which can be inputted into label-studio (sentences from the job advert sample) is stored in `s3://open-jobs-lake/escoe_extension/inputs/data/skill_ner/20220622_sample_labelling_text_data.txt`, and a sister file which maps the sentences to the job ID of the advert they were in is stored in `s3://open-jobs-lake/escoe_extension/inputs/data/skill_ner/20220622_sample_labelling_metadata.json`. This latter file is important for any analysis of the sample data since we can link back to the job advert's metadata, so could analysis which years/job occupations the sample came from.
