# Taxonomy mapper

This folder contains the `SkillsMapper` class in `skill_ner_mapper.py` needed to extract skills. This class is used to find the closest matches from a skill span to a skill or skill group from a chosen taxonomy. It does this using BERT embeddings and cosine similarities. It will try to match to a skill, and if it isn't possible to get a close match to a skill, it will try to match to a skill group with descreasing granularity.

It also contains a few one-off scripts to create data for the mapping process.

## Taxonomy data formatting

The data from different taxonomies needs to be formatted for use in the `SkillsMapper` class.

This is done for ESCO and Lightcast in `esco_formatting.py` and `lightcast_formatting.py` respectively. Running these scripts save out the two formatted taxonomies `escoe_extension/outputs/data/skill_ner_mapping/esco_data_formatted.csv` and `escoe_extension/outputs/data/skill_ner_mapping/lightcast_data_formatted.csv` which are used when extracting and matching skills.

## Pre-calculating taxonomy embeddings

Running:

```
python ojd_daps_skills/pipeline/skill_ner_mapping/taxonomy_matcher_embeddings.py --config_name CONFIG_NAME --embed_fn EMBEDDING_FILE_NAME
```

will create the taxonomy embeddings for a given taxonomy in a config file. This script just needs to be run once and is useful to do as a one off for speeding up the matching skills algorithm. It will save the file `escoe_extension/outputs/data/skill_ner_mapping/EMBEDDING_FILE_NAME.json`.

"""
The taxonomy being mapped to in the script needs to be in a specific format.
There should be the 3 columns skill_name_col, skill_id_col, skill_type_col
with an optional 4th column (skill_hier_info_col).

### Example 1:

At the most basic level your taxonomy input could be:
"name" | "id" | "type"
---|---|---
"driving a car" | 123 | "skill"
"give presentations" | 333 | "skill"
"communicating well" | 456 | "skill"
...
with skill_type_dict = {'skill_types': ['skill']}.
Your output match for the OJO skill "communicate" might look like this:
{
'ojo_ner_skills': "communicate",
'top_5_tax_skills': [("communicating well", 456, 0.978), ("give presentations", 333, 0.762), ..]
}

- the closest skill to this ojo skill is "communicating well" which is code 456 and had a cosine distance of 0.978

### Example 2:

A more complicated example would have hierarchy levels given too
"name" | "id" | "type" | "hierarchy_levels"
---|---|---|---
"driving a car" | 123 | "skill" | ['A2.1']
"give presentations" | 333 | "skill" | ['A1.2']
"communicating well" | 456 | "skill"| ['A1.3']
...
with skill_type_dict = {'skill_types': ['skill']}.
This might give the result:
{
'ojo_ner_skills': "communicate",
'top_5_tax_skills': [("communicating well", 456, 0.978), ("give presentations", 333, 0.762), ..],
'high_tax_skills': {'num_over_thresh': 2, 'most_common_level_0: ('A1', 1) , 'most_common_level_1': ('A1.3', 0.5)},
}

- 100% of the skills where the similarity is greater than the threshold are in the 'A1' skill level 0 group
- 50% of the skills where the similarity is greater than the threshold are in the 'A1.3' skill level 1 group

### Example 3:

And an even more complicated example would have skill level names given too (making use
of the 'type' column to differentiate them).
"name" | "id" | "type" | "hierarchy*levels"
---|---|---|---
"driving a car" | 123 | "skill" | ['A2.1']
"give presentations" | 333 | "skill" | ['A1.2']
"communicating well" | 456 | "skill"| ['A1.3']
"communication" | 'A1' | "level 1"| None
"driving" | 'A2' | "level 0"| None
"communicate verbally" | 'A1.3' | "level 1"| None
...
with skill_type_dict = {'skill_types': ['skill'], 'hier_types': ["level A", "level B"]} and num_hier_levels=2
This might give the result:
{
'ojo_ner_skills': "communicate",
'top_5_tax_skills': [("communicating well", 456, 0.978), ("give presentations", 333, 0.762), ..],
'high_tax_skills': {'num_over_thresh': 2, 'most_common_level_0: ('A1', 1) , 'most_common_level_1': ('A1.3', 0.5)},
"top*'level 0'_tax_level": ('communication', 'A1', 0.998),
"top_'level 1'\_tax_level": ('communicate verbally', 'A1.3', 0.98),
}

- the skill level 0 group 'communication' (code 'A1') is the closest to thie ojo skill with distance 0.998
- the skill level 1 group 'communicate verbally' (code 'A1.3') is the closest to thie ojo skill with distance 0.98
  """
