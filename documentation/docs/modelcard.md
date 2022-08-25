# Skills Extractor Model Card

This page contains information for different parts of the skills extraction and mapping pipeline. We detail a high level summary of the pipeline and the pipeline's overall intended use. We then detail different parts of the pipeline.

Developed by data scientists in Nesta’s Data Analytics Practice, (last updated on 18-08-2022).

### Pipeline High level summary

<p><img alt="full pipeline" src="/img/full_pipeline.jpeg" /></p>

Overall pipeline includes:

- NER model to extract skill or experience entities in job adverts;
- SVM model to predict whether the skill entity is a skill or multiskill; if multiskill, apply rules to split multiskills into individual skill entities;
- embed all skill, split multiskills, multiskill sentences and taxonomy skills using [huggingface’s "sentence-transformers/all-MiniLM-L6-v2"](https://www.google.com/url?q=https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2&sa=D&source=docs&ust=1660824880413352&usg=AOvVaw2dpSwPbxZibtqIcyi7sIIT)
- map skills (skill, split multiskill and multiskills) onto taxonomy skills at different levels

For further information or feedback please contact (Liz Gallagher)[mailto:elizabeth.gallagher@nesta.org.uk], (India Kerle)[india.kerle@nesta.org.uk] or(Cath Sleeman)[mailto:cath.sleeman@nesta.org.uk].

### Pipeline Intended Use

- Extract skills from online job adverts and match extracted skills to a user’s skill taxonomy of choice, such as the [European Commission’s European Skills, Competences, and Occupations (ESCO)](https://esco.ec.europa.eu/en) or [Lightcast’s Open Skills](https://www.economicmodeling.com/2022/03/08/open-skills-taxonomy/#:~:text=Skills%20Taxonomy%2FOpen%20methodology%3A&text=To%20help%20everyone%20speak%20the,resumes%E2%80%94updated%20every%20two%20weeks.).
- Intended users include researchers in labour statistics or related government bodies.

### Pipeline Out of Scope Uses

Out of scope is extracting and matching skills from job adverts in non-English languages; extracting and matching skills from texts other than job adverts; drawing conclusions on new, unidentified skills.

Skills extracted should not be used to determine skill demand without expert steer and input nor should be used for any discriminatory hiring practices.

### Component Model Card: Extracting Skills

<p><img alt="predicting entities" src="/img/predicting_entities.jpeg" /></p>

**_Summary_**

- Train a [NER spaCy component](https://spacy.io/api/entityrecognizer) to extract skills, multiskills and experience from job adverts.
- Predict whether or not a sentence is multiskill or not using [scikit learn's SVM model](https://scikit-learn.org/stable/modules/svm.html). Features include: length of entity; if 'and' in entity; if ',' in entity.
- Split multiskills based on rules: split on and, duplicate verbs, split skill mentions

**_Metrics and Evaluation_**

- For the NER model, **X** job adverts werre labelled for skills, multiskills and experience.
- As of 29th July 2022, **X** entities in **X** job adverts from OJO were labelled; **X** are multiskill, **X** are skill, and **393** are experience entities. 20% of the labelled entities were held out as a test set to evaluate the models.

- A partial metric in the python library nerevaluate (read more here) was used to calculate F1, precision and recall for the NER and SVM classifier on the held-out test set. As of **MOST RECENT DATE**, the results are as follows:

| Entity     | F1    | Precision | Recall |
| ---------- | ----- | --------- | ------ |
| Skill      | 0.543 | 0.715     | 0.437  |
| Experience | 0.385 | 0.467     | 0.328  |
| All        | 0.524 | 0.577     | 0.512  |

- The same training data and held out test set was used to evaluate the SVM model. On a held out test set, the SVM model achieved 87% accuracy.

- More details of the evaluation performance across both the NER model and the SVM model can be found within the [evaluation directory of the repo](https://github.com/nestauk/ojd_daps_skills/blob/dev/ojd_daps_skills/pipeline/evaluation/20220729_ner_svm_model_evaluation.json).

**_Caveats and Reccomendations_**

- As we take a rules based approach to splitting multiskills, many multiskills do not get split. If a multiskill is unable to be split, we still match to a taxonomy of choice. Future work should add more rules to split multiskills.

### Component Model Card: Skills to Taxonomy Mapping

<p><img alt="match to taxonomy" src="/img/matching_to_taxonomy.jpeg" /></p>

**_Summary_**

- Match to a taxonomy based on differrent similarity thresholds.
- First try to match at the most graular level of a taxonomy.
- If there is no close granular skill above 0.7, we then assign the skill to different levels of the taxonomy in different approaches.
- The first approach defines the taxonomy levels based on the number of skills where the similarity score is above a threshold. The second approach calculates the cosine distance between the ojo skill and the embedded taxonomy level description and chooses the closest taxonomy level.

**_Model Factors_**

The main factors in this matching approach are: 1) the different thresholds at different levels of a taxonomy and 2) the different matching approaches.

**_Metrics and Evaluation_**

The thresholds at different levels of the taxonomy with differernt approaches are determined by labelling 100 unique randomly sampled ojo skills:

|                                              | **skill_level_accuracy** | **top\_'Level 2 preferred term'\_tax_level_accuracy** | **top\_'Level 3 preferred term'\_tax_level_accuracy** | **most_common_1_level_accuracy** | **most_common_2_level_accuracy** | **most_common_3_level_accuracy** |
| -------------------------------------------- | ------------------------ | ----------------------------------------------------- | ----------------------------------------------------- | -------------------------------- | -------------------------------- | -------------------------------- |
| **skill_score_threshold_window_0.416_0.5**   | 0.0625                   | 0.0                                                   | 0.75                                                  | 0.4375                           | 0.4375                           | 0.0                              |
| **skill_score_threshold_window_0.5_0.584**   | 0.234375                 | 0.1875                                                | 0.578125                                              | 0.6875                           | 0.6875                           | 0.640625                         |
| **skill_score_threshold_window_0.584_0.667** | 0.5581395348837210       | 0.2248062015503880                                    | 0.875968992248062                                     | 0.8992248062015500               | 0.875968992248062                | 0.875968992248062                |
| **skill_score_threshold_window_0.667_0.75**  | 0.7129629629629630       | 0.32407407407407400                                   | 0.9351851851851850                                    | 0.8888888888888890               | 0.8240740740740740               | 0.7037037037037040               |
| **skill_score_threshold_window_0.75_0.833**  | 0.8928571428571430       | 0.6607142857142860                                    | 0.9821428571428570                                    | 0.8928571428571430               | 0.8928571428571430               | 0.8928571428571430               |
| **skill_score_threshold_window_0.833_0.917** | 0.9310344827586210       | 0.5172413793103450                                    | 1.0                                                   | 0.9655172413793100               | 0.9655172413793100               | 0.896551724137931                |
| **skill_score_threshold_window_0.917_1.0**   | 1.0                      | 0.6666666666666670                                    | 1.0                                                   | 1.0                              | 1.0                              | 1.0                              |

A configuration file will contain the relevant thresholds per taxonomy. These thresholds will need to be manually changed based on different taxonomies.

**_Caveats and Reccomendations_**

This step does less well when:

- the extracted skill is a metaphor: i.e. 'understand the bigger picture' gets matched to 'take pictures'
- the extracted skill is an acronym: i.e. 'drafting ORSAs' gets matched to 'fine arts'
- the extracted skill is not a skill (poor NER model performance): i.e. 'assist with the' gets matched to providing general assistance to people

Future work should look to train embeddings with job-specific texts, disambiguate acronymns and improve NER model performance.
