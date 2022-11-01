# Extract Skills

`extract_skills.py` combines the prediction of skills using code from [skill_ner](https://github.com/nestauk/ojd_daps_skills/tree/dev/ojd_daps_skills/pipeline/skill_ner) with the mapping of skills to a taxonomy using code from [skill_ner_mapping](https://github.com/nestauk/ojd_daps_skills/tree/dev/ojd_daps_skills/pipeline/skill_ner_mapping).

If you are mapping to the ESCO skills taxonomy, we reviewed the top 100 skills and ultimately hard coded 43 of the most common skills which were not well matched from a random sample of 100,000 job adverts in OJO with the most appropriate skills from the taxonomy.

## Public access to relevant models and embeddings

The ExtractSkills class relies on the aws cli tool to download relevant models and embeddings required to extract and map skill spans from a given job advert(s). If you do not have the aws cli tool downloaded, you can simply download all the files `open-jobs-indicators/escoe_extension/outputs` using [aws's front end via our public access bucket.](https://s3.console.aws.amazon.com/s3/buckets/open-jobs-indicators?region=eu-west-1&prefix=escoe_extension/&showversions=false) These files will need to be downloaded to a folder called `escoe_extension` in your project directory.

## Mapping a list of skills

If you would like to map an existing list of skills to a taxonomy rather than extract skills from free job advert text, you can do so by:

`
es = ExtractSkills(config_name="extract_skills_esco", s3=True)
es.load()

skill_list = ['communication and writing', 'numerical skills', 'collaboration']

#re-format list of skills for mapping
skill_list_reformatted = es.format_skills(skill_list)
#then map reformatted list of skills to taxonomy - in this example, esco
mapped_skills = es.map_skills(skill_list_reformatted)
`

## Matching skills to a new taxonomy

If you'd like to match skills extracted from job adverts to a different taxonomy than our pre-defined ones you can do so by creating a new config.yaml file and loading it in `ExtractSkills(config_name="new_config_name")`. You should also process the taxonomy in the following format and save it as a '.json' file. This should have the following format.

| type          | description                                 | id     | hierarchy_levels                               |
| ------------- | ------------------------------------------- | ------ | ---------------------------------------------- |
| skill         | use spreadsheets software                   | abcd   | [[S, S5, S5.6, S5.6.1], [S, S5, S5.5, S5.5.2]] |
| skill         | use communication techniques                | cdef   | [[S, S1, S1.0, S1.0.0]]                        |
| skill_group_3 | communication, collaboration and creativity | S1.0.0 | NaN                                            |
| skill_group_3 | mathematics                                 | S1.2.1 | NaN                                            |
| skill_group_2 | presenting information                      | S1.4   | NaN                                            |

You will see the `type` column contains skills and skill groups. This is because we try to match to individual skills, but if this isn't possible we then try to match to a skill group in the taxonomy (if given).

For rows which correspond to individual skills (rather than skill groups) the `hierarchy_levels` column values show all the parts of the taxonomy where this skill is situated. It is helpful to link these codes to names, so you may also want to create a taxonomy name mapper file for this data, e.g. `{"S1.2.1": "mathematics"}`. For rows which correspond to skill groups (rather than individual skills) the `hierarchy_levels` column will be blank since the hierarchy information is contained in the `id`. The contents of `hierarchy_levels` need to be a list of lists, or a list of strings, but not a combination of both.

The number of levels in the taxonomy will correspond to the length of the lists in the `hierarchy_levels` column.

The config should contain the following:

```
taxonomy_name: [Name]
taxonomy_path: [Path to your taxonomy dataset]
hier_name_mapper_file_name: [(optional) Path to your taxonomy name mapper]
num_hier_levels: [Number of skill group levels your taxonomy has]
skill_type_dict:
  {
    "skill_types": [A list of the values of the 'type' column which code skills],
    "hier_types": [A list of the values of the 'type' column which code skill groups, these need to be in order from least to most granular],
  }
ner_model_path: "outputs/models/ner_model/20220825/"
clean_job_ads: True
min_multiskill_length: 75
match_thresholds_dict:
  {
    "skill_match_thresh": 0.7,
    "top_tax_skills": { 1: 0.5, 2: 0.5, 3: 0.5 },
    "max_share": { 1: 0, 2: 0.2, 3: 0.2 },
  }
skill_name_col: "description"
skill_id_col: "id"
skill_hier_info_col: "hierarchy_levels"
skill_type_col: "type"
```

Note:

- It is important that the list given in `skill_type_dict['hier_types']` is in the order from the least to most granular parts of the taxonomy. For example, in the ESCO taxonomy we match against the second and third skill group levels, so this is set to `["level_2", "level_3"]` i.e. level 3 is more granular than level 2, where `level 2 skill groups > level 3 skill groups > individual skill`.
