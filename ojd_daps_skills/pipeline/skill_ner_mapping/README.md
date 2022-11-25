# Taxonomy mapper

This folder contains the `SkillMapper` class in `skill_ner_mapper.py` needed to extract skills. This class is used to find the closest matches from a skill span to a skill or skill group from a chosen taxonomy. It does this using BERT embeddings and cosine similarities. It will try to match to a skill, and if it isn't possible to get a close match to a skill, it will try to match to a skill group with descreasing granularity.

It also contains a few one-off scripts to create data for the mapping process.

## Taxonomy data formatting

The data from different taxonomies needs to be formatted for use in the `SkillMapper` class.

This is done for ESCO and Lightcast in `esco_formatting.py` and `lightcast_formatting.py` respectively. Running these scripts save out the two formatted taxonomies `escoe_extension/outputs/data/skill_ner_mapping/esco_data_formatted.csv` and `escoe_extension/outputs/data/skill_ner_mapping/lightcast_data_formatted.csv` which are used when extracting and matching skills.

## Pre-calculating taxonomy embeddings

Running:

```
python ojd_daps_skills/pipeline/skill_ner_mapping/taxonomy_matcher_embeddings.py --config_name CONFIG_NAME --embed_fn EMBEDDING_FILE_NAME
```

will create the taxonomy embeddings for a given taxonomy in a config file. This script just needs to be run once and is useful to do as a one off for speeding up the matching skills algorithm. It will save the file `escoe_extension/outputs/data/skill_ner_mapping/EMBEDDING_FILE_NAME.json`.
