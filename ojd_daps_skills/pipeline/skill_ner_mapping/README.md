# SkillMapper

This directory contains the class to map extracted skill spans to a list of taxonomy skills. It also contains scripts to define a proximity threshold, based on labelled skill matches.

## skill ner mapper

This class maps skill spans to skill names from an inputted taxonomy based on cosine similarity. The outputs of this class are saved to s3.

`python skill_ner_mapper.py`

## get skill mapper threshold sample

This script generates a sample of n size of skill matches per threshold window. It samples the sample to s3.

To run script, you can define the minimum threshold, the maximum threshold, the threshold length and the sample size.

`python get_skill_mapper_threshold_sample.py --min 0.3 --max 1 --threshold_len 10 --sample_size 20`

## get skill mapper threshold

This script loads labelled skill matches and prints accuracy per threshold window. It also prints the percentage of skill spans that we would be able to label based on thresholding.

`python get_skill_mapper_threshold.py --thresh threshold`

### Thresholding (20220704 skill_predictions)

In terms of thresholding, a sample of skill matchs were labelled as either being good or not based off of different threshold windows. The accuracy and percentage of skill matches that would be labelled based on thresholds in a sample of 5,000 job adverts is reported below:

| threshold window | percentage labelled (upper band) | accuracy |
| ---------------- | -------------------------------- | -------- |
| 0.71 - 0.72      | 44%                              | 0.75     |
| 0.72 - 0.73      | 41%                              | 0.7      |
| 0.73 - 0.75      | 35%                              | 0.83     |
| 0.75 - 0.76      | 33%                              | 0.96     |
| 0.76 - 0.77      | 31%                              | 0.83     |
| 0.77 - 0.79      | 26%                              | 0.85     |
| 0.79 - 0.8       | 24%                              | 0.86     |

### Evaluation (20220714 skill_predictions)

The SkillMapper class now also assigns a skill to different levels of the taxonomy in different approaches. The first approach defines the taxonomy levels based on the number of skills where the similarity score is above a threshold. The second approach calculates the cosine distance between the ojo skill and the embedded taxonomy level description and chooses the closest taxonomy level.

A dataset of 100 unique randomly sampled ojo skills in the updated SkillMapper output (`"escoe_extension/outputs/data/skill_ner/skill_mappings/20220714_skills_to_esco.json"`) was labelled (0 - bad match, 1- good match) at both the skill- and taxonomy- level.

Based on the below skill-level threshold windows, accuracy at the skill- and taxonomy- level (for both approaches) are reported.

|                                              | **skill_level_accuracy** | **top\_'Level 2 preferred term'\_tax_level_accuracy** | **top\_'Level 3 preferred term'\_tax_level_accuracy** | **most_common_1_level_accuracy** | **most_common_2_level_accuracy** | **most_common_3_level_accuracy** |
| -------------------------------------------- | ------------------------ | ----------------------------------------------------- | ----------------------------------------------------- | -------------------------------- | -------------------------------- | -------------------------------- |
| **skill_score_threshold_window_0.416_0.5**   | 0.0625                   | 0.0                                                   | 0.75                                                  | 0.4375                           | 0.4375                           | 0.0                              |
| **skill_score_threshold_window_0.5_0.584**   | 0.234375                 | 0.1875                                                | 0.578125                                              | 0.6875                           | 0.6875                           | 0.640625                         |
| **skill_score_threshold_window_0.584_0.667** | 0.5581395348837210       | 0.2248062015503880                                    | 0.875968992248062                                     | 0.8992248062015500               | 0.875968992248062                | 0.875968992248062                |
| **skill_score_threshold_window_0.667_0.75**  | 0.7129629629629630       | 0.32407407407407400                                   | 0.9351851851851850                                    | 0.8888888888888890               | 0.8240740740740740               | 0.7037037037037040               |
| **skill_score_threshold_window_0.75_0.833**  | 0.8928571428571430       | 0.6607142857142860                                    | 0.9821428571428570                                    | 0.8928571428571430               | 0.8928571428571430               | 0.8928571428571430               |
| **skill_score_threshold_window_0.833_0.917** | 0.9310344827586210       | 0.5172413793103450                                    | 1.0                                                   | 0.9655172413793100               | 0.9655172413793100               | 0.896551724137931                |
| **skill_score_threshold_window_0.917_1.0**   | 1.0                      | 0.6666666666666670                                    | 1.0                                                   | 1.0                              | 1.0                              | 1.0                              |
