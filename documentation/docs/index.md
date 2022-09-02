# Welcome to Nesta's Skills Extractor Library

Welcome to the documentation of Nesta's skills extractor library version 1.0.0.

This page contains information on how to install and use Nesta's skills extraction library. The skills library allows you to extract skills from job advertisments and map then onto a skills taxonomy of your choice.

If you'd like to learn more about the models used in the library, please refer to the [model card page](https://nestauk.github.io/ojd_daps_skills/modelcard/).

## Installation

`pip install ojd_daps_skills`

## Pre-defined configurations

There are currently three configurations available for running the skills extraction algorithm.

1. `extract_skills_toy` - Configuration for a toy taxonomy example, useful for testing.
2. `extract_skills_esco` - Configuration for extracting skills and matching them to the [ESCO](https://esco.ec.europa.eu/en) skills taxonomy.
3. `extract_skills_lightcast` - Configuration for extracting skills and matching them to the [lightcast](https://skills.lightcast.io/) skills taxonomy.

These configurations contain information about parameter values, trained models and data locations.

## Using Nesta's Skills Extractor library

You can extract skills from job adverts and then map them onto a taxonomy of your choice. In this instance, we map onto a toy taxonomy.

```
from ojd_daps_skills.pipeline.extract_skills.extract_skills import ExtractSkills #import the module

es = ExtractSkills(config_name="extract_skills_toy", s3=True) #instantiate with toy taxonomy configuartion file

es.load() #load necessary models

job_adverts = [
    "The job involves communication and maths skills",
    "The job involves excel and presenting skills. You need good excel skills",
] #toy job advert examples

predicted_skills = es.get_skills(job_adverts) #extract skills from list of job adverts

job_skills_matched = es.map_skills(predicted_skills) #match extracted skills to toy taxonomy
```

The outputs are as follows:

```
predicted_skills
>>> [{'EXPERIENCE': [], 'SKILL': ['maths skills'], 'MULTISKILL': []}, {'EXPERIENCE': [], 'SKILL': ['presenting skills', 'excel skills'], 'MULTISKILL': []}]

job_skills_matched
>>> [{'SKILL': [('maths skills', ('communicate with others', 'S1.1'))]}, {'SKILL': [('presenting skills', ('communicate with others', 'S1.1')), ('excel skills', ('computational skills', 'K2.1'))]}]
```

Alternatively, you can extract and map skills onto a taxonomy in one step:

```
from ojd_daps_skills.pipeline.extract_skills.extract_skills import ExtractSkills #import the module

es = ExtractSkills(config_name="extract_skills_toy", s3=True) #instantiate with toy taxonomy configuartion file

es.load() #load necessary models

job_adverts = [
    "The job involves communication and maths skills",
    "The job involves excel and presenting skills. You need good excel skills",
] #toy job advert examples

job_skills_matched = es.extract_skills(predicted_skills) #match and extract skills to toy taxonomy

```

The outputs are as follows:

```
job_skills_matched
>>> [{'SKILL': [('maths skills', ('communicate with others', 'S1.1'))]}, {'SKILL': [('presenting skills', ('communicate with others', 'S1.1')), ('excel skills', ('computational skills', 'K2.1'))]}]
```

If you don't have access to the Nesta S3 bucket for this repo then you will need to set s3=False, and make sure to have relevant files downloaded locally.