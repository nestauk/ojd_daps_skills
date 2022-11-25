# Custom Usage

`extract_skills.py` combines the prediction of skills using code from [skill_ner](https://github.com/nestauk/ojd_daps_skills/tree/dev/ojd_daps_skills/pipeline/skill_ner) with the mapping of skills to a taxonomy using code from [skill_ner_mapping](https://github.com/nestauk/ojd_daps_skills/tree/dev/ojd_daps_skills/pipeline/skill_ner_mapping).

This page explains more about the custom usage of this class including creating a custom config file and mapping to another taxonomy. To do this you will need to clone the repo. Please refer to the main documentation page for the [development setup instructions](https://nestauk.github.io/ojd_daps_skills/build/html/about.html#development-a-name-development-a) for this package and the core usage.

## Configuration files <a name="config_files"></a>

Core to the Extract Skills package, and in particular the taxonomy mapping functionality, is config files. These are included in the instantiation of the class, as so:

```
es = ExtractSkills(config_name="extract_skills_toy")
```

### Predefined configurations <a name="predefined_config"></a>

There are currently three configurations available for running the skills extraction algorithm. These configurations contain information about parameter values, trained models and directory locations of stored data.

1. `extract_skills_toy` - Configuration for a toy taxonomy example, useful for testing.
2. `extract_skills_esco` - Configuration for extracting skills and matching them to the ESCO skills taxonomy. This configuration is correct to v1.1.1 of ESCO.
3. `extract_skills_lightcast` - Configuration for extracting skills and matching them to the Lightcast skills taxonomy. This configuration is correct to the version of Lightcast as of 22/11/22.

If you are mapping to the ESCO skills taxonomy using `extract_skills_esco.yaml`, we reviewed the top 100 skills and ultimately hard coded 43 of the most common skills which were not well matched from a random sample of 100,000 job adverts in the [Open Jobs Observatory](https://www.nesta.org.uk/data-visualisation-and-interactive/open-jobs-observatory/) project with the most appropriate skills from the taxonomy.

### Configuration definitions <a name="config_defs"></a>

Every predefined configuration includes the following parameters:

| Parameter                                        | Description                                                                                                                                                                                                                                                                                                                                  |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ner_model_path`: str                            | The relative path to the NER model folder used to predict skill spans in job adverts.                                                                                                                                                                                                                                                        |
| `taxonomy_name`: str                             | The name of the taxonomy to map onto.                                                                                                                                                                                                                                                                                                        |
| `taxonomy_path`: str                             | The relative path to the formatted taxonomy. Formatted taxonomy must be in `.csv` format.                                                                                                                                                                                                                                                    |
| `clean_job_ads`: bool, default=True              | Whether to perform light text cleaning on job adverts or not. Text cleaning includes detecting and splitting camelcase in job adverts, replacing various characters and converting bullet points to full stops. Defaults to True.                                                                                                            |
| `min_multiskill_length`: int                     | The minimum character length a predicted multi-skill sentence must be to apply splitting rules to.                                                                                                                                                                                                                                           |
| (optional) `taxonomy_embedding_file_name`: str   | The relative path to a taxonomy embedding file if it exists. If left unset the embeddings will be generated when the code is run.                                                                                                                                                                                                            |
| (optional) `prev_skill_matches_file_name`: str   | The relative path to a previous skill matches file if it exists.                                                                                                                                                                                                                                                                             |
| (optional) `hard_labelled_skills_file_name`: str | The relative path to a hard labelled skills file if it exists.                                                                                                                                                                                                                                                                               |
| (optional) `hier_name_mapper_file_name`: str     | The relative path to a hierarchy name mapper file if it exists.                                                                                                                                                                                                                                                                              |
| `num_hier_levels`: int                           | The number of levels in the skills taxonomy hierarchy. This can be set to 0 if the taxonomy has no levels.                                                                                                                                                                                                                                   |
| `skill_type_dict`: dict                          | A dictionary that defines skill types and hierarchy types. <br /><br /> `{ "skill_types": [A list of the values of the 'type' column which code skills], "hier_types": [A list of the values of the 'type' column which code skill groups, these need to be in order from least to most granular]}`                                          |
| `match_thresholds_dict`: dict                    | A dictionary that defines thresholds at each level of the skills taxonomy hierarchy. For example,<br /> <br /> `{"skill_match_thresh": 0.7, "top_tax_skills": {1: 0.5, 2: 0.5, 3: 0.5},“max_share”: {1: 0, 2: 0.2, 3: 0.2}}`<br /> <br /> See **Model Card: Skills to Taxonomy Mapping** for the details of what these thresholds represent. |
| `skill_name_col`: str                            | The name of the skill/hierarchy level description text column in formatted taxonomy `.csv`.                                                                                                                                                                                                                                                  |
| `skill_id_col`: str                              | Name of skill id column in formatted taxonomy `.csv`. Each row should contain a unique ID for the skill/hierarchy.                                                                                                                                                                                                                           |
| (optional) `skill_hier_info_col`: str            | Name of hierarchy info column in formatted taxonomy `.csv`. The hierarchy info column contains which hierarchy levels a skill is in (from least to most granular). If not a skill, then NA.                                                                                                                                                  |
| `skill_type_col`: str                            | Name of what column name the skill/hier description is from (category, subcategory) in formatted taxonomy `.csv`.                                                                                                                                                                                                                            |

## Mapping to your own taxonomy <a name="mapping"></a>

Although we currently support three configurations for running the skills extraction algorithm, you are also able to map extracted skills onto a taxonomy of your choice by defining your own configuration file. In order to map skills onto your own taxonomy you must:

1. Format your taxonomy
2. Define your own configuration file

#### Format your taxonomy <a name="format_tax"></a>

You must also format your taxonomy in such a way that looks like the following:

| skill_type_col | skill_name_col                              | skill_id_col | (optional) skill_hier_info_col                                   |
| -------------- | ------------------------------------------- | ------------ | ---------------------------------------------------------------- |
| skill          | use spreadsheets software                   | abcd         | `[["S", "S5", "S5.6", "S5.6.1"], ["S", "S5", "S5.5", "S5.5.2"]]` |
| skill          | use communication techniques                | cdef         | `[["S", "S1", "S1.0", "S1.0.0"]]`                                |
| skill_group_3  | communication, collaboration and creativity | S1.0.0       | NaN                                                              |
| skill_group_3  | mathematics                                 | S1.2.1       | NaN                                                              |
| skill_group_2  | presenting information                      | S1.4         | NaN                                                              |

You will see the `skill_type_col` column contains skills and skill groups. This is because we try to match to individual skills, but if this isn't possible we then try to match to a skill group in the taxonomy (if given).

For rows which correspond to individual skills (rather than skill groups) the `skill_hier_info_col` column values show all the parts of the taxonomy where this skill is situated. It is helpful to link these codes to names, so you may also want to create a taxonomy name mapper file for this data, e.g. `{"S1.2.1": "mathematics"}`. For rows which correspond to skill groups (rather than individual skills) the `skill_hier_info_col` column will be blank since the hierarchy information is contained in the `skill_id_col` column. The contents of `skill_hier_info_col` need to be a list of lists, or a list of strings, but not a combination of both.

The number of levels in the taxonomy will correspond to the length of the lists in the `skill_hier_info_col` column.

Although we don’t provide guidance on re-formatting your taxonomy, we have re-formatted the ESCO taxonomy to this format in [this script](https://github.com/nestauk/ojd_daps_skills/blob/dev/ojd_daps_skills/pipeline/skill_ner_mapping/esco_formatting.py) and we have re-formatted the Lightcast taxonomy to this format in [this script](https://github.com/nestauk/ojd_daps_skills/blob/dev/ojd_daps_skills/pipeline/skill_ner_mapping/lightcast_formatting.py).

#### Define your own configuration file <a name="custom_config"></a>

Create your own configuration `yaml` file in the format `extract_skills_taxonomy_name.yaml`. This config should contain all the parameters as described in [Predefined configuration definitions](https://nestauk.github.io/ojd_daps_skills/build/html/custom_usage.html#configuration-definitions-a-name-config-defs-a). The file should be saved to `your_current_path/ojd_daps_skills/config/`.

We provide a template config file [here](https://github.com/nestauk/ojd_daps_skills/blob/dev/ojd_daps_skills/config/extract_skills_template.yaml).

It is important that the list given in `skill_type_dict['hier_types']` is in the order from the least to most granular parts of the taxonomy. For example, in the ESCO taxonomy we match against the second and third skill group levels, so this is set to `["level_2", "level_3"]` i.e. level 3 is more granular than level 2, where `level 2 skill groups > level 3 skill groups > individual skill`.

Now you can use your custom taxonomy as:

```
from ojd_daps_skills.pipeline.extract_skills.extract_skills import ExtractSkills #import the module

es = ExtractSkills(config_name="my_custom_config_name", local=True)

es.load()

```
