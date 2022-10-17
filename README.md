# ojd_daps_skills

ojd_daps_skills is a package designed to extract skills from job adverts, and match them to an existing taxonomy if needed.

This works by:

1. Finding skills in job adverts using a Named Entity Recognition (NER) model.
2. Matching these skills to an existing skills taxonomy using semantic similarity.

Much more about these steps can be found in [this report](outputs/reports/skills_extraction.md).

This package is split into the three pipeline steps:

- [skill_ner](https://github.com/nestauk/ojd_daps_skills/tree/dev/ojd_daps_skills/pipeline/skill_ner)
- [skill_ner_mapping](https://github.com/nestauk/ojd_daps_skills/tree/dev/ojd_daps_skills/pipeline/skill_ner_mapping)
- [evaluation](https://github.com/nestauk/ojd_daps_skills/tree/dev/ojd_daps_skills/pipeline/evaluation)

![](outputs/reports/figures/overview.png)
![](outputs/reports/figures/overview_example.png)

## Installation

```
pip install ojd_daps_skills
```

## Pre-defined configurations

There are three configurations available for running the skills extraction algorithm.

1. [extract_skills_toy](ojd_daps_skills/config/extract_skills_toy.yaml) - Configuration for a toy taxonomy example, useful for testing.
2. [extract_skills_esco](ojd_daps_skills/config/extract_skills_esco.yaml) - Configuration for extracting skills and matching them to the [ESCO](https://esco.ec.europa.eu/en) skills taxonomy.
3. [extract_skills_lightcast](ojd_daps_skills/config/extract_skills_lightcast.yaml) - Configuration for extracting skills and matching them to the [Lightcast](https://skills.emsidata.com/) skills taxonomy.

These configurations contain all the information about parameter values, and trained model and data locations.

## Basic usage

```
from ojd_daps_skills.pipeline.extract_skills.extract_skills import ExtractSkills

es = ExtractSkills(config_name="extract_skills_toy", s3=True)

es.load()

job_adverts = [
        "You will need to have good communication and excellent mathematics skills. You will have experience in the IT sector.",
        "You will need to have good excel and presenting skills. You need good excel software skills",
    ]

predicted_skills = es.get_skills(job_adverts)
job_skills_matched = es.map_skills(predicted_skills)

predicted_skills
>>> [{'EXPERIENCE': ['experience in the IT sector'], 'SKILL': ['communication', 'excellent mathematics skills'], 'MULTISKILL': []}, {'EXPERIENCE': [], 'SKILL': ['excel', 'presenting', 'excel software skills'], 'MULTISKILL': []}]
job_skills_matched
>>> [{'SKILL': [
  ('communication', ('communication', '15d76317-c71a-4fa2-aadc-2ecc34e627b7')),
  ('excellent mathematics skills', ('practice mathematics', 'db77825e-0f3e-47d0-abdb-356794484272'))],
  'EXPERIENCE': [
  'experience in the IT sector']},
 {'SKILL': [
  ('excel software skills', ('use spreadsheets software', '1973c966-f236-40c9-b2d4-5d71a89019be')),
  ('excel', ('use spreadsheets', 'db77825e-0f3e-47d0-abdb-356794484272')),
  ('presenting', ('presenting exhibition', 'c45848bc-33c6-45fa-b791-bc5b06c21b87'))]}]
```

If you don't have access to the Nesta S3 bucket for this repo then you will need to set s3=False, and make sure to have relevant files downloaded locally.

TOD0:

How to download the relevant files locally:

- ner_model_path: "outputs/models/ner_model/20220825/"
- hier_name_mapper_file_name: "escoe_extension/outputs/data/skill_ner_mapping/esco_hier_mapper.json"
- taxonomy_path : "escoe_extension/outputs/data/skill_ner_mapping/esco_data_formatted.csv"
- (optional) taxonomy_embedding_file_name : "escoe_extension/outputs/data/skill_ner_mapping/esco_embeddings.json"
- (optional) prev_skill_matches_file_name : "escoe_extension/outputs/data/skill_ner_mapping/ojo_esco_lookup.json"

## User-defined configurations

For guidance about matching skills to a different taxonomy see [here](ojd_daps_skills/pipeline/extract_skills/README.md).

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
