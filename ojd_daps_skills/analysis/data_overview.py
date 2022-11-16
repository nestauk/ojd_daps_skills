# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Data Overview
#
# Basic analysis of the NER training data and the skill matches found in 5000 job adverts.
#
# ### [1. Training data analysis](#training_data)
# - How many did we label?
# - What were the camelcases found?
#
# ### [2. NER model results](#model_results)
# - How did the model perform on the test set?
#
# ### [3. Match data for 5000 job adverts](#match_data)
# - What did the model predict?
#
# ### [4. Quality of skill matches](#match_quality)
# - When we manually labelled predictions, what were the results like?
#
# ### [5. Make predictions](#use_model)
# - Load the model to make skill predictions

# %%
import pandas as pd
import numpy as np
from collections import Counter
import re

# %%
from ojd_daps_skills.getters.data_getters import (
    load_s3_json,
    get_s3_resource,
    load_file,
)

from ojd_daps_skills import bucket_name, PROJECT_DIR
from ojd_daps_skills.pipeline.skill_ner.ner_spacy_utils import (
    compiled_missing_space_pattern,
)
from ojd_daps_skills.pipeline.extract_skills.extract_skills import ExtractSkills

# %%
s3 = get_s3_resource()

# %%
labelled_date_filename = (
    "escoe_extension/outputs/labelled_job_adverts/combined_labels_20220824.json"
)
train_details_file = (
    "escoe_extension/outputs/models/ner_model/20220825/train_details.json"
)
sample_matches_file_name = "escoe_extension/outputs/data/extract_skills/20220901_ojo_sample_skills_extracted.json"
manually_matches_tagged_file = (
    "escoe_extension/outputs/quality_analysis/26_aug_22_matches_shuffle_manual.csv"
)

# %% [markdown]
# ### 1. Training data analysis<a id='training_data'></a>

# %%
job_advert_labels = load_s3_json(s3, bucket_name, labelled_date_filename)

# %%
all_raw_entities = []
all_labels = set()
for job_id, labels in job_advert_labels.items():
    text = labels["text"]
    ent_list = labels["labels"]
    for ent in ent_list:
        all_raw_entities.append(ent["value"]["text"])
        for label_tag in ent["value"]["labels"]:
            all_labels.add(label_tag)

# %%
all_exp = []
all_skills = []
all_multiskills = []
weird_ents = []
all_raw_entities = []
for job_id, labels in job_advert_labels.items():
    text = labels["text"]
    job_skills = []
    job_multiskills = []
    job_exp = []
    for ent in labels["labels"]:
        ent_val = ent["value"]
        all_raw_entities.append(ent_val["text"])
        if len(ent_val["labels"]) != 1:
            weird_ents.append((job_id, labels))
        if ent_val["labels"][0] == "SKILL":
            job_skills.append(ent_val["text"])
        elif ent_val["labels"][0] == "MULTISKILL":
            job_multiskills.append(ent_val["text"])
        elif ent_val["labels"][0] == "EXPERIENCE":
            job_exp.append(ent_val["text"])
        else:
            weird_ents.append((job_id, labels))
    all_exp.append(job_exp)
    all_skills.append(job_skills)
    all_multiskills.append(job_multiskills)


# %%

flatten_all_skills = [v for m in all_skills for v in m]
flatten_all_multiskills = [v for m in all_multiskills for v in m]
flatten_all_exp = [v for m in all_exp for v in m]
num_ents = len(flatten_all_skills + flatten_all_multiskills + flatten_all_exp)

print(f"On a sample of {len(job_advert_labels)} job adverts manually labelled...\n")

print(f"{num_ents} entities were labelled in total")
print(
    f"There were {len(flatten_all_skills)} ({round(len(flatten_all_skills)*100/num_ents,2)}%) skill entities ({len(set(flatten_all_skills))} unique values)"
)
print(
    f"There were {len(flatten_all_multiskills)} ({round(len(flatten_all_multiskills)*100/num_ents,2)}%) multiskill entities ({len(set(flatten_all_multiskills))} unique values)"
)
print(
    f"There were {len(flatten_all_exp)} ({round(len(flatten_all_exp)*100/num_ents,2)}%) experience entities ({len(set(flatten_all_exp))} unique values)"
)

print(
    f"Each job advert has an average of {np.mean([len(m) for m in all_skills])} skill entities"
)
print(
    f"Each job advert has an average of {np.mean([len(m) for m in all_multiskills])} multiskill entities"
)
print(
    f"Each job advert has an average of {np.mean([len(m) for m in all_exp])} experience entities"
)

# %%
camel_cases = []
for text in all_raw_entities:
    words = text.split()
    for word in words:
        if len(re.findall(compiled_missing_space_pattern, word)) != 0:
            camel_cases.append(word)

# %%
print("Camel cases found in the training data:")
Counter(camel_cases).most_common()

# %% [markdown]
# ### 2. NER model results<a id='model_results'></a>

# %%

train_details = load_s3_json(s3, bucket_name, train_details_file)

# %%
print(f"The model was trained on {train_details['train_data_length']} job adverts")
print(f"The model was evaluated on {train_details['eval_data_length']} job adverts")

print(
    f"The multiskill classifier scored a mean accuracy of {round(train_details['ms_classifier_train_evaluation'], 3)} on the training data"
)
print(
    f"The multiskill classifier scored a mean accuracy of {round(train_details['ms_classifier_test_evaluation'], 3)} on the evaluation data"
)

# %%
result_df = []
for res_type, result in train_details["results_summary"].items():
    result_df.append(
        {
            "entity type": res_type,
            "f1": round(result["f1"], 2),
            "precision": round(result["precision"], 2),
            "recall": round(result["recall"], 2),
        }
    )
pd.DataFrame(result_df)

# %% [markdown]
# ### 3. Match data for 5000 job adverts<a id='match_data'></a>

# %%
match_results = load_file(sample_matches_file_name, s3=True)

# %%
num_skills = []
num_multiskills = []
num_experiences = []
for job_ad_results in match_results.values():
    num_skills.append(len(job_ad_results.get("SKILL", [])))
    num_multiskills.append(len(job_ad_results.get("MULTISKILL", [])))
    num_experiences.append(len(job_ad_results.get("EXPERIENCE", [])))

# %%
sum(num_multiskills)

# %%
print(f"On a sample of {len(match_results)} job adverts labelled using the model...\n")

num_pred_ents = sum(num_skills) + sum(num_experiences)

print(f"There were {num_pred_ents} entities predicted in total")
print(
    f"There were {sum(num_skills)} ({round(sum(num_skills)*100/num_pred_ents,2)}%) skill entities"
)
print(
    f"There were {sum(num_experiences)} ({round(sum(num_experiences)*100/num_pred_ents,2)}%) experience entities"
)

print(f"There is an average of {round(np.mean(num_skills), 2)} skills per job advert")
print(
    f"There is an average of {round(np.mean(num_experiences), 2)} experiences per job advert"
)
print(
    f"{len([i for i in num_skills if i==0])*100/len(match_results)}% job adverts have no skills"
)
print(
    f"{len([i for i in num_experiences if i==0])*100/len(match_results)}% job adverts have no experiences"
)

# %%
ojo_skills = []
ojo_exp = []
esco_skills = []
esco_codes = []
for r in match_results.values():
    if r.get("SKILL"):
        for skills in r["SKILL"]:
            ojo_skills.append(skills[0])
            esco_codes.append(skills[1][1])
            esco_skills.append(skills[1][0])
    if r.get("EXPERIENCE"):
        for exps in r["EXPERIENCE"]:
            ojo_exp.append(exps)

# %%
print("Most common skill entities:")
print([i for i, v in Counter(ojo_skills).most_common(10)])

# %%
print(
    f"There were {len(ojo_skills)} skill entities, {len(set(ojo_skills))} unique entities"
)
print(
    f"There were {len(ojo_exp)} experience entities, {len(set(ojo_exp))} unique entities"
)


# %%
def match_type(x):
    if len(x) > 10:
        return "skill"
    else:
        if x[0] == "S":
            if len(x) > 5:
                return "skill group level 3"
            elif len(x) > 3:
                return "skill group level 2"
            else:
                return "skill group level 1"
        elif x[0] == "K":
            if len(x) >= 5:
                return "knowledge group level 3"
            elif len(x) > 3:
                return "knowledge group level 2"
            else:
                return "knowledge group level 1"
        elif x[0] == "A":
            if len(x) > 5:
                return "attitudes group level 3"
            elif len(x) > 3:
                return "attitudes group level 2"
            else:
                return "attitudes group level 1"
        elif x[0] == "T":
            if len(x) > 5:
                return "transversal skills and competences group level 3"
            elif len(x) > 3:
                return "transversal skills and competences group level 2"
            else:
                return "transversal skills and competences group level 1"
        elif x[0] == "L":
            return "language skills and knowledge"
        else:
            return "other"


# %%
match_results_pd = pd.DataFrame(
    {"ojo_skills": ojo_skills, "esco_skills": esco_skills, "esco_codes": esco_codes}
)
match_results_grouped = (
    match_results_pd.groupby(
        ["ojo_skills", "esco_skills", "esco_codes"], as_index=False
    )
    .size()
    .reset_index(drop=True)
)
match_results_grouped = match_results_grouped.sort_values(
    "size", ascending=False
).reset_index(drop=True)
match_results_grouped["match_type"] = match_results_grouped["esco_codes"].apply(
    lambda x: match_type(x)
)

# %%
not_match_type_list = [
    "skill group level 1",
    "knowledge group level 1",
    "transversal skills and competences group level 1",
    "attitudes group level 1",
]
match_results_grouped["matched?"] = match_results_grouped["match_type"].apply(
    lambda x: False if x in not_match_type_list else True
)

# %%
print(
    f"{round(sum(match_results_grouped['matched?'])*100/len(match_results_grouped),2)}% of the {len(match_results_grouped)} unique skill entities could be matched to ESCO"
)

# %%
match_results_grouped["match_type"].value_counts()

# %%
print("Most common ESCO skills matched:")
print(match_results_grouped["esco_skills"][0:10].tolist())

# %% [markdown]
# ### 4. Quality of skill matches<a id='match_quality'></a>
#
# Use our manually tagged ojo_skill<->esco skill quality checks.
#
# Use `match_results_grouped` from previous section.

# %%
# Merge with the tagged dataset if the ojo name, esco name and esco codes are all exactly the same
manual_tagged_pairs = load_file(manually_matches_tagged_file)
manual_tagged = pd.merge(
    match_results_grouped,
    manual_tagged_pairs[
        [
            "ojo_skills",
            "esco_skills",
            "esco_codes",
            "good skill? (0-bad, 1-ok, 2-good)",
            "good match? (0-bad, 1-ok, 2-good)",
            "comments",
        ]
    ],
    how="left",
    on=["ojo_skills", "esco_skills", "esco_codes"],
)

manual_tagged = manual_tagged[
    pd.notnull(manual_tagged["good skill? (0-bad, 1-ok, 2-good)"])
]
manual_tagged.rename(
    columns={
        "good skill? (0-bad, 1-ok, 2-good)": "skill_quality",
        "good match? (0-bad, 1-ok, 2-good)": "match_quality",
    },
    inplace=True,
)
manual_tagged.replace("?", "-1", inplace=True)
manual_tagged["skill_quality"] = pd.to_numeric(manual_tagged["skill_quality"])
manual_tagged["match_quality"] = pd.to_numeric(manual_tagged["match_quality"])
len(manual_tagged)

# %%
not_match_type_list = [
    "skill group level 1",
    "knowledge group level 1",
    "transversal skills and competences group level 1",
    "attitudes group level 1",
]
manual_tagged["matched"] = manual_tagged["match_type"].apply(
    lambda x: True if x not in not_match_type_list else False
)

# %%
manual_tagged["matched"].value_counts()

# %%
manual_tagged["skill_quality"].value_counts()

# %%
manual_tagged["skill_quality"].value_counts() / len(manual_tagged)

# %%
manual_tagged[manual_tagged["skill_quality"] == 1]["ojo_skills"][0:10].tolist()

# %%
manual_tagged["match_quality"].value_counts()

# %%
manual_tagged["match_quality"].value_counts() / len(manual_tagged)

# %%
manual_tagged[["skill_quality", "match_quality"]].value_counts()

# %%
print(f"|ojo_skills|esco_skills|esco_codes|")
print("|---|---|---|")
for i, row in manual_tagged[manual_tagged["match_quality"] == 0].iterrows():
    print(f"|{row['ojo_skills']}|{row['esco_skills']}|{row['esco_codes']}|")

# %% [markdown]
# ### 5. Make predictions <a id='use_model'></a>

# %%
from ojd_daps_skills.pipeline.extract_skills.extract_skills import ExtractSkills

es = ExtractSkills(config_name="extract_skills_esco", local=False, verbose=False)
es.load()

job_adverts = [
    (
        "You will need to have good communication and excellent mathematics skills. "
        "You will have experience in the IT sector."
    ),
    (
        "You will need to have good excel and presenting skills. "
        "You need good excel software skills"
    ),
]

predicted_skills = es.get_skills(job_adverts)
job_skills_matched = es.map_skills(predicted_skills)

print(job_skills_matched)

# %%
for text in job_adverts:
    es.job_ner.display_prediction(text)

# %%
