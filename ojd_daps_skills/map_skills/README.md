# Taxonomy mapper

## How SkillMapper chooses a match

`SkillsMapper` maps an **extracted skill phrase**, not a whole job advert or job
title. `get_embeddings` embeds the unique phrases and loads or computes taxonomy
embeddings. `map_skills` calculates cosine similarities and prepares candidates;
`match_skills` applies the configured thresholds and returns the first eligible
candidate. This is a description of the current algorithm, not a guarantee that
every semantic match is correct.

For the ESCO configuration in `ojd_daps_skills/configs/extract_skills_esco.yaml`:

1. Compare the phrase with taxonomy rows of type `preferredLabel` or
   `altLabels`. Retain the ten most similar rows. If the highest similarity is
   at least `skill_match_thresh` (**0.7**), the exact taxonomy skill wins.
2. For hierarchy voting, take the hierarchy codes of those ten skill rows.
   Each row contributes `round(similarity * 10)` copies of its codes. At each
   level, `get_most_common_code` returns the most frequent code and its share of
   all votes. Thus `max_share` tests a **weighted vote share**, not a cosine
   similarity. A level absent from `max_share` is not eligible by this route.
3. Independently, compare the phrase with taxonomy hierarchy **descriptions**
   for each configured `hier_types` entry (`level_2` and `level_3` in ESCO).
   The best description is eligible when its cosine similarity meets that
   level's `top_tax_skills` threshold. This is a different score from the vote
   share in step 2.
4. If the exact skill failed, visit hierarchy levels in descending index order
   (`3`, `2`, `1`, `0` for ESCO). At each level, try the majority-code candidate
   first, then the best hierarchy-description candidate. The **first eligible**
   candidate becomes the returned match; candidates are not ranked against
   one another by their numeric values. If none qualifies, there is no match.
   Finally, ESCO's hard-coded lookup can override a computed match.

The current ESCO thresholds are:

| Candidate | ESCO setting | Meaning |
| --- | --- | --- |
| Exact skill | `skill_match_thresh: 0.7` | Minimum cosine similarity to the closest skill row. |
| Majority hierarchy code | `max_share: {1: 0, 2: 0.2, 3: 0.2}` | Minimum fraction of weighted votes at hierarchy indices 1, 2, 3. Index 0 is omitted. |
| Closest hierarchy description | `top_tax_skills: {1: 0.5, 2: 0.5, 3: 0.5}` | Minimum cosine similarity at an available hierarchy-description level. In ESCO, descriptions are configured only for indices 2 and 3. |

For example, if an extracted phrase's best exact skill scores **0.68**, it
fails the 0.7 exact threshold. Suppose its level-3 codes receive 6 of 20
weighted votes for the most common code. The share **0.30** meets ESCO's
level-3 `max_share` of 0.2, so that code is returned before a level-3
description candidate is considered. If that share were 0.15 but the best
level-3 description scored 0.55, the description would qualify instead.
These numbers illustrate the decision rules; they are not measured model
performance.

The result's `match_type` tells you which route won: `skill`,
`most_common_level_N`, or `top_level_N_tax_level`. `match_score` is the
cosine similarity for an exact or description match, but a **vote share** for
`most_common_level_N`; these values should not be compared as though they
were the same measure. `match_id` is the selected taxonomy code. A hard-coded
ESCO result may take precedence over these computed fields. Thresholds and
available levels differ in the Lightcast and toy YAML configurations.

This folder contains `SkillsMapper` in `skill_mapper.py`. The extractor finds
skill spans first; the mapper then compares those spans with a chosen taxonomy.

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
