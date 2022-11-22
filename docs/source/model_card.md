# Model Cards

- [Model Card: Full pipeline](#skills_ex_card)
- [Pipeline High level summary](#pipeline_summary)
- [Pipeline Intended Use](#intended_use)
- [Pipeline Out of Scope Uses](#out_of_scope)
- [Pipeline Metrics](#metrics)
  - [Comparison 1 - Degree of overlap between Lightcast’s extracted skills and our Lightcast skills](#comp_1)
  - [Comparison 2 - Top skills per occupation comparison to ESCO essential skills per occupation](#comp_2)
  - [Evaluation - Manual judgement of false positive rate](#eval_man)
- [Model Card: Extract Skills](#extract_skills)
- [Summary](#extract_skills_summary)
- [NER Metrics](#ner_metrics)
- [Multiskill Metrics](#multiskill_metrics)
- [Caveats and Recommendations](#caveats)
- [Model Card: Skills to Taxonomy Mapping](#mapping_card)
- [Summary](#mapping_summary)
- [Model Factors](#mapping_factors)
- [Metrics and Evaluation](#mapping_metrics)
- [Caveats and Recommendations](#mapping_caveats)

This page contains information for different parts of the skills extraction and mapping pipeline. We detail a high level summary of the pipeline and the pipeline's overall intended use. We then detail different parts of the pipeline.

Developed by data scientists in Nesta’s Data Analytics Practice, (last updated on 15-11-2022).

## Model Card: Full pipeline <a name="skills_ex_card"></a>

### Pipeline High level summary <a name="pipeline_summary"></a>

![](../../outputs/reports/figures/overview.png)

High level, the overall pipeline includes:

- Named Entity Recognition (NER) model to extract skill, multi skill or experience entities in job adverts;
- Support Vector Machine (SVM) model to predict whether the skill entity is a skill or multiskill; if multiskill, apply rules to split multiskills into individual skill entities;
- Embed all entities (skill and multi skill entities) and taxonomy skills using huggingface’s [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) pre-trained model;
- Map extracted skills (skill and multi skill) onto taxonomy skills using cosine similarity of embeddings.

For further information or feedback please contact Liz Gallagher, India Kerle or Cath Sleeman.

### Pipeline Intended Use <a name="intended_use"></a>

- Extract skills from online job adverts and match extracted skills to a user’s skill taxonomy of choice, such as the European Commission’s European Skills, Competences, and Occupations (ESCO) or Lightcast’s Open Skills.
- Intended users include researchers in labour statistics or related government bodies.

### Pipeline Out of Scope Uses <a name="out_of_scope"></a>

Out of scope is extracting and matching skills from job adverts in non-English languages; extracting and matching skills from texts other than job adverts; drawing conclusions on new, unidentified skills.

Skills extracted should not be used to determine skill demand without expert steer and input nor should be used for any discriminatory hiring practices.

### Pipeline Metrics <a name="metrics"></a>

#### Comparison 1 - Degree of overlap between Lightcast’s extracted skills and our Lightcast skills <a name="comp_1"></a>

- We compare extracted Lightcast skills from Lightcasts’ Open Skills algorithm and our current approach from 50 job adverts, with a minimum cosine similarity threshold between an extracted skill and taxonomy skill set to 0 to guarantee we only match at the skill level
- We extract skills from 50 job adverts given the limits of use of Lightcast’s Open Skills algorithm
- We extract an average of 10.22 skills per job advert while Lightcast’s Open Skills algorithm extracts an average of 6.42 skills per job advert
- There no overlap for 40% of job adverts between the two approaches
- Of the job adverts where there is overlap, on average, 39.3% of extracted Lightcast skills are present in our current approach. The median percentage is 33.3%.
- Of the job adverts where there is overlap, on average, 25.09% of our skills are present in Lightcast skills. The median percentage is 21.43%
- Qualitatively, there are a number of limitations to the degree of overlap approach for comparison:
- The two skill lists may contain very similar skills i.e. Financial Accounting vs. Finance but will be considered different as a result
- For exact comparison, we set the cosine similarity threshold to 0 to guarantee extracted skill-level skills but would otherwise not do so. This allows for inappropriate skill matches i.e. ‘Eye Examination’ for a supply chain role
- Lightcast’s algorithm may not be a single source of truth and it also extracts inappropriate skill matches i.e. ‘Flooring’ for a care assistant role

#### Comparison 2 - Top skills per occupation comparison to ESCO essential skills per occupation <a name="comp_2"></a>

We compare ESCO’s essential skills per occupation with the top ESCO-mapped skills per occupation. We identify top skills per occupation by:

- Identifying occupations for which we have at least 100 job adverts;
- Identify skills extracted at ONLY the skill level;
- Identify a top skill threshold by calculating the 50 percentile % of jobs that require a given skill for a given occupation + 0.25 standard deviation
- Identify the % of top ESCO-mapped skills in ESCO’s essential skills per occupation

At a high level, we find that:

- 58 occupations with 100 or more job adverts were found in both ESCO and a sample of deduplicated 100,000 job adverts
- the average # of adverts per occupation is 345.54
- We extract essential ESCO skills, transversal skills and additional skills
- On average, 19 percent of essential ESCO skills were also in the top skills extracted per occupation
- The occupation with the maximum % of top extracted skills that were also essential ESCO skills is 54.5, for the occupation ‘project manager’
- There is 1 occupation for which there was no overlap between the list of top extracted skills and ESCO essential skills

Similarly to the Lightcast approach, ESCO’s essential skill list per occupation may be very similar to the extracted skills, but not identical, leading to artificially lower overlap.

#### Evaluation - Manual judgement of false positive rate <a name="eval_man"></a>

We looked at the ESCO-mapped skills extracted from a random sample of 64 job adverts, and manually judged how many skills shouldn’t have been extracted from the job advert i.e. the false positives. Our results showed on average 27% of the skills extracted from a job advert are false positives.

We also performed this analysis when looking at the skills extracted from 22 job adverts using Lightcast’s Skills Extractor API. We found on average 12% of the skills extracted from a job advert are false positives. We find on average Lightcast extract more skills (8 skills per job advert) than our algorithm does (5 skills per job advert).

## Model Card: Extract Skills <a name="extract_skills"></a>

![](../../outputs/reports/figures/predict_flow.png)

### Summary <a name="extract_skills_summary"></a>

- Train a NER spaCy component to extract skills, multiskills and experience from job adverts.
- Predict whether or not a skill is multi-skill or not using scikit learn's SVM model. Features are length of entity; if 'and' in entity; if ',' in entity.
- Split multiskills above 75 characters based on rules: split on and, duplicate verbs, split skill mentions. If multi skills are less than 75 characters but not split into single skills, keep in and treat them as single skills. If the length is more than 75 characters, still match to taxonomy. - The current predefined configurations ensures that every extracted skill will be matched to a taxonomy. However, if a skill is matched to the highest skill group, we label it as ‘unmatched’. Under this definition, we identify approximately 2% of skills as ‘unmatched’.

### NER Metrics <a name="ner_metrics"></a>

- For the NER model, 375 job adverts were labelled for skills, multiskills and experience.
- As of 15th November 2022, **5641** entities in 375 job adverts from OJO were labelled;
- **354** are multiskill, **4696** are skill, and **608** were experience entities. 20% of the labelled entities were held out as a test set to evaluate the models.
- A partial metric in the python library nerevaluate ([read more here](https://pypi.org/project/nervaluate/)) was used to calculate F1, precision and recall for the NER and SVM classifier on the held-out test set. As of 15th November 2022, the results are as follows:

| Entity     | F1    | Precision | Recall |
| ---------- | ----- | --------- | ------ |
| Skill      | 0.586 | 0.679     | 0.515  |
| Experience | 0.506 | 0.648     | 0.416  |
| All        | 0.563 | 0.643     | 0.500  |

- More details of the evaluation performance across both the NER model and the SVM model can be found in `outputs/models/ner_model/20220825/train_details.json`

### Multiskill Metrics <a name="multiskill_metrics"></a>

- The same training data and held out test set used for the NER model was used to evaluate the SVM model. On a held out test set, the SVM model achieved 91% accuracy.
- When evaluating the multi skill splitter algorithm, 253 multiskill spans were labelled as ‘good’, ‘ok’ or ‘bad’ splits. Of the 253 multiskill spans, 80 were split. Of the splits, 66% were ‘good’, 9% were ‘ok’ and 25% were ‘bad’.
- More details of the evaluation performance across both the NER model and the SVM model can be found in `outputs/models/ner_model/20220825/train_details.json`

### Caveats and Recommendations <a name="caveats"></a>

- As we take a rules based approach to splitting multiskills, many multiskills do not get split. If a multiskill is unable to be split, we still match to a taxonomy of choice. Future work should add more rules to split multiskills.
- We deduplicate at the extracted skill level but not at the taxonomy level. This means that if ‘excel skills’ is extracted twice from a job advert, it will be deduplicated and one will be matched to a taxonomy. However, if it mentions the strings "excel skills" and "Excel skill", both occurrences will be matched to a taxonomy and therefore the output will have two excel skills. If deduplicating is important, you will need to deduplicate at the taxonomy level.

## Model Card: Skills to Taxonomy Mapping <a name="mapping_card"></a>

![](../../outputs/reports/figures/match_flow.png)

![](../../outputs/reports/figures/overview_example.png)

### Summary <a name="mapping_summary"></a>

- Match to a taxonomy based on different similarity thresholds.
- First try to match at the most granular level of a taxonomy based on cosine similarity between embedded, extracted skill and taxonomy skills. Extracted and taxonomy skills are embedded using huggingface’s sentence-transformers/all-MiniLM-L6-v2.
- If there is no close granular skill above 0.7 cosine similarity (can be changed in configuration file), we then assign the skill to different levels of the taxonomy in one of two approaches.
- The first approach defines the taxonomy levels based on the number of skills where the similarity score is above a threshold. The second approach calculates the cosine distance between the extracted skill and the embedded taxonomy level description and chooses the closest taxonomy level.
- if matching to ESCO, the top 100 skills from a sample of 100,000 job adverts are hard coded.

### Model Factors <a name="mapping_factors"></a>

The main factors in this matching approach are: 1) the different thresholds at different levels of a taxonomy and 2) the different matching approaches.

### Metrics and Evaluation <a name="mapping_metrics"></a>

The thresholds at different levels of the taxonomy with different approaches are determined by labelling 100 unique randomly sampled skills and calculating the accuracy:

| Skill score threshold window | top\_‘Level 2 preferred term’\_tax_level_accuracy | top\_‘Level 3 preferred term’\_tax_level_accuracy | most_common_1_level_accuracy | most_common_2_level_accuracy | most_common_3_level_accuracy |
| ---------------------------- | ------------------------------------------------- | ------------------------------------------------- | ---------------------------- | ---------------------------- | ---------------------------- |
| 0.416 - 0.5                  | 0.063                                             | 0.000                                             | 0.750                        | 0.438                        | 0.438                        | 0.000 |
| 0.5 - 0.584                  | 0.234                                             | 0.188                                             | 0.578                        | 0.688                        | 0.688                        | 0.6406 |
| 0.584 - 0.667                | 0.558                                             | 0.225                                             | 0.876                        | 0.900                        | 0.876                        | 0.876 |
| 0.667 - 0.75                 | 0.713                                             | 0.324                                             | 0.936                        | 0.889                        | 0.824                        | 0.704 |
| 0.75 - 0.833                 | 0.893                                             | 0.661                                             | 0.982                        | 0.893                        | 0.893                        | 0.893 |
| 0.833 - 0.917                | 0.931                                             | 0.517                                             | 1.000                        | 0.966                        | 0.966                        | 0.897 |
| 0.917 - 1.000                | 1.000                                             | 0.667                                             | 1.000                        | 1.000                        | 1.000                        | 1.000 |

The configuration file contains the relevant thresholds per taxonomy. These thresholds will need to be manually tuned based on different taxonomies.

### Caveats and Recommendations <a name="mapping_caveats"></a>

This step does less well when:

- the extracted skill is a metaphor: i.e. 'understand the bigger picture' gets matched to 'take pictures'
- the extracted skill is an acronym: i.e. 'drafting ORSAs' gets matched to 'fine arts'
- the extracted skill is not a skill (poor NER model performance): i.e. 'assist with the' gets matched to providing general assistance to people

Future work should look to train embeddings with job-specific texts, disambiguate acronyms and improve NER model performance.
