# ojd_daps_skills

ojd_daps_skills is a package designed to extract skills from job adverts, and match them to an existing taxonomy if needed.

This works by:

1. Finding skills in job adverts using a Named Entity Recognition (NER) model.
2. Matching these skills to an existing skills taxonomy using semantic similarity.

Much more about these steps can be found in [this report](outputs/reports/skills_extraction.md).

This package is split into the three pipeline steps:

1. [skill_ner](ojd_daps_skills/ojd_daps_skills/pipeline/skill_ner)
2. [skill_ner_mapping](ojd_daps_skills/ojd_daps_skills/pipeline/skill_ner_mapping)
3. [evaluation](ojd_daps_skills/ojd_daps_skills/pipeline/evaluation)

## Installation

```
pip install ojd_daps_skills
```

## Pre-defined configurations

There are two configurations available for running the skills extraction algorithm.

1. "extract_skills_toy" - Configuration for a toy taxonomy example, useful for testing.
2. "extract_skills_esco" - Configuration for extracting skills and matching them to the [ESCO](https://esco.ec.europa.eu/en) skills taxonomy.

These configurations contain all the information about parameter values, and trained model and data locations.

## Basic usage

```
from ojd_daps_skills.pipeline.extract_skills.extract_skills import ExtractSkills

es = ExtractSkills(config_name="extract_skills_toy", s3=True)

es.load()

job_adverts = [
    "The job involves communication and maths skills",
    "The job involves excel and presenting skills. You need good excel skills",
]

predicted_skills = es.get_skills(job_adverts)
job_skills_matched = es.map_skills(predicted_skills)

predicted_skills
>>> [{'EXPERIENCE': [], 'SKILL': ['maths skills'], 'MULTISKILL': []}, {'EXPERIENCE': [], 'SKILL': ['presenting skills', 'excel skills'], 'MULTISKILL': []}]
job_skills_matched
>>> [{'SKILL': [('maths skills', ('communicate with others', 'S1.1'))]}, {'SKILL': [('presenting skills', ('communicate with others', 'S1.1')), ('excel skills', ('computational skills', 'K2.1'))]}]

```

If you don't have access to the Nesta S3 bucket for this repo then you will need to set s3=False, and make sure to have relevant files downloaded locally.

TOD0:

How to download the relevant files locally:

- ner_model_path: "outputs/models/ner_model/20220729/"
- hier_name_mapper_file_name: "escoe_extension/outputs/data/skill_ner_mapping/esco_hier_mapper.json"
- taxonomy_path : "escoe_extension/outputs/data/skill_ner_mapping/esco_data_formatted.csv"
- (optional) taxonomy_embedding_file_name : "escoe_extension/outputs/data/skill_ner_mapping/esco_embeddings.json"
- (optional) prev_skill_matches_file_name : "escoe_extension/outputs/data/skill_ner_mapping/ojo_esco_lookup.json"

## Testing

Some functions have tests, these can be checked by running

```
pytest
```

## Setup

- Meet the data science cookiecutter [requirements](http://nestauk.github.io/ds-cookiecutter/quickstart), in brief:
  - Install: `direnv` and `conda`
- Create a blank cookiecutter conda log file:
  - `mkdir .cookiecutter/state`
  - `touch .cookiecutter/state/conda-create.log`
- Run `make install` to configure the development environment:
  - Setup the conda environment
  - Configure `pre-commit`

## Contributor guidelines

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

---

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>
