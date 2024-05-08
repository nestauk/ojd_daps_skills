To install as a package:

```
pipx install poetry
poetry shell
poetry install
```

To extract skills from a job advert:

```
from ojd_daps_skills.extract_skills.extract_skills import SkillsExtractor

sm = SkillsExtractor(taxonomy_name="toy")

✘ nestauk/en_skillner NER model not loaded. Downloading model...
Collecting en-skillner==any
  Downloading https://huggingface.co/nestauk/en_skillner/resolve/main/en_skillner-any-py3-none-any.whl (587.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 587.7/587.7 MB 5.1 MB/s eta 0:00:0000:0100:01
Installing collected packages: en-skillner
Successfully installed en-skillner-3.7.1
✘ Multi-skill classifier not loaded. Downloading model...
Fetching 4 files: 100%|██████████| 4/4 [00:00<00:00, 26843.55it/s]
✘ Neccessary data files are not downloaded. Downloading ~0.5GB of
neccessary data files to
/Users/india.kerlenesta/Projects/nesta/ojd_daps/ojd_daps_extension/ojd_daps_skills/ojd_daps_skills_data.
ℹ Data folder downloaded from
/Users/india.kerlenesta/Projects/nesta/ojd_daps/ojd_daps_extension/ojd_daps_skills/ojd_daps_skills_data

job_ad = "You should be skilled in Python, Java and R."
job_ad_with_skills = sm(job_ad)

ℹ Getting embeddings for 3 texts ...
ℹ Took 0.018199920654296875 seconds
```

To access the extracted and mapped skills:

```
job_ad_with_skills_doc = job_ad_with_skills[0]

#print raw ents (i.e. multiskills are not split, also include 'BENEFIT' and 'EXPERIENCE' spans)
job_ad_with_skills_doc.ents
>> (Python, Java, R.)

#print SKILL spans (where SKILL spans are predicted as multiskills, split them)

job_ad_with_skills._.skill_spans
>> [Python, Java, R.]

#print mapped skills to the "toy" taxonomy
job_ad_with_skills._.mapped_skills
>> [{'ojo_skill': 'Python',
  'ojo_skill_id': 2232581233191055,
  'match_skill': 'working with computers',
  'match_score': 0.75,
  'match_type': 'most_common_level_1',
  'match_id': 'S5'},
 {'ojo_skill': 'Java',
  'ojo_skill_id': 2833100423969322,
  'match_skill': 'working with computers',
  'match_score': 0.6666666666666666,
  'match_type': 'most_common_level_1',
  'match_id': 'S5'},
 {'ojo_skill': 'R.',
  'ojo_skill_id': 8622187230313821,
  'match_skill': 'working with computers',
  'match_score': 0.6666666666666666,
  'match_type': 'most_common_level_1',
  'match_id': 'S5'}]
```

To run tests:

```
pytest tests/
```
