# SkillMapper

This directory contains the class to map extracted OJO skill spans to a list of taxonomy skills. It also contains scripts to define a proximity threshold, based on labelled skill matches.

## skill ner mapper

This class maps skill spans to (currently) ESCO skill names based on cosine similarity. It reports on the top 5 skills and skill scores associated to each OJO skill span. The outputs of this class are saved to s3.

`python skill_ner_mapper.py`

## get skill mapper threshold sample

This script generates a sample of n size of skill matches per threshold window. It samples the sample to s3.

To run script, you can define the minimum threshold, the maximum threshold, the threshold length and the sample size.

`python get_skill_mapper_threshold_sample.py --min 0.3 --max 1 --threshold_len 10 --sample_size 20`

## get skill mapper threshold

This script loads labelled skill matches and prints accuracy per threshold window. It also prints the percentage of skill spans that we would be able to label based on thresholding.

`python get_skill_mapper_threshold.py --thresh threshold`

### Thresholding

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
