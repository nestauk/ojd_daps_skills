# SkillMapper

This directory contains the class to map extracted OJO skill spans to a list of taxonomy skills. It also contains scripts to define a proximity threshold, based on labelled skill matches.

## skill_ner_mapper

This class maps skill spans to (currently) ESCO skill names based on cosine similarity. It reports on the top 5 skills and skill scores associated to each OJO skill span. The outputs of this class are saved to s3.

To run the script, python skill_ner_mapper.py

## skill_ner_mapper_utils

This utils file has functions to preprocess a given skill span.

## get_skill_mapper_threshold_sample

This script generates a sample of n size of skill matches per threshold window. It samples the sample to s3.

To run script, you can define the minimum threshold, the maximum threshold, the threshold length and the sample size.

python get_skill_mapper_threshold_sample.py --min 0.3 --max 1 --threshold_len 10 --sample_size 20

## get_skill_mapper_threshold

This script loads labelled skill matches and prints true positive and false negatives per threshold window. It also prints the percentage of skill spans that we would be able to label based on thresholding.

python get_skill_mapper_threshold.py
