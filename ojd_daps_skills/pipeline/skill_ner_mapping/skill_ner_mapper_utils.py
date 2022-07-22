"""Utils to map extracted skills from NER model to
taxonomy skills."""

import re
import nltk
from nltk.corpus import stopwords
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
import numpy as np
import json

lem = nltk.WordNetLemmatizer()

nltk.download("omw-1.4")
nltk.download("stopwords")
stopwords = set(stopwords.words("english"))

# load punctuation replacement rules
punctuation_replacement_rules = {
    # old patterns: replacement pattern
    "[\u2022,\u2023,\u25E6,\u2043,\u2219]": " ",  # Convert bullet points to space
    r"[-/:\\]": " ",  # Convert colon, hyphens and forward and backward slashes to spaces
    r"[^a-zA-Z0-9,.; #(++)]": "",  # Preserve spaces, commas, full stops, semicollons
}

compiled_punct_patterns = [re.compile(p) for p in punctuation_replacement_rules.keys()]
punct_replacement = list(punctuation_replacement_rules.values())
compiled_missing_space_pattern = re.compile("([a-z])([A-Z])([a-z])")


def preprocess_skill(skill):
    """Preprocess skill to remove bullet points, convert colon, hyphens
    and slashes to spaces, lowercase, remove trailing whitespace and
    lemmatise skill.

    Inputs:
        skill (str): skill to be preprocessed.

    Outputs:
        skill (str): preprocessed skill.
    """

    # job_stopwords = list("skill", "skills")
    # get rid of bullet points
    for j, pattern in enumerate(compiled_punct_patterns):
        skill = pattern.sub(punct_replacement[j], skill)

    # # get first half of skill where skills contain camel cases (assumes skill was first half)
    # skill = compiled_missing_space_pattern.sub(r"\1. \2\3", skill).split(".")[0]

    # # remove stopwords
    # skill = " ".join(
    #     filter(
    #         lambda token: token not in stopwords | set(job_stopwords), skill.split(" ")
    #     )
    # )

    # # lemmatise tokens in skill
    # skill = " ".join([lem.lemmatize(token) for token in skill.split(" ")])

    # # lowercase and remove trailing with spaces
    # skill = skill.lower().strip()
    return skill


def get_top_skill_score_df(ojo_to_taxonomy: dict, taxonomy: str) -> pd.DataFrame:
    """Convert to DataFrame and get top skill and top score
    per ojo to taxonomy skill match.

    Inputs:
        ojo_to_taxonomy (dict): Saved data from ojo skill span to taxonomy skill.
        taxonomy (str): Name of taxonomy

    Outputs:
        ojo_to_taxonomy (pd.DataFrame): DataFrame where each row has a ojo skill,
        taxonomy skill and a closeness score.

    """
    ojo_to_taxonomy = pd.DataFrame(ojo_to_taxonomy).T
    for col in (taxonomy + "_taxonomy_skills", taxonomy + "_taxonomy_scores"):
        col_name = "top_" + col.split("_")[-1]
        ojo_to_taxonomy[col_name] = ojo_to_taxonomy[col].apply(
            lambda x: [i[0] for i in x]
        )

    ojo_to_taxonomy = ojo_to_taxonomy.apply(pd.Series.explode)[
        ["ojo_ner_skills", "top_skills", "top_scores"]
    ]

    return ojo_to_taxonomy


def get_top_comparisons(ojo_embs, taxonomy_embs, match_sim_thresh=0.5):
    """
    Get the cosine similarities between two embedding matrices and
    output the top index and score

    Need to convert score to float for saving to JSON
    """

    emb_sims = cosine_similarity(ojo_embs, taxonomy_embs)

    top_sim_indxs = [list(np.argsort(sim)[::-1][:10]) for sim in emb_sims]
    top_sim_scores = [[float(s) for s in np.sort(sim)[::-1][:10]] for sim in emb_sims]

    return top_sim_indxs, top_sim_scores


def get_most_common_code(split_possible_codes, lev_n):
    """
    split_possible_codes = [['S4', 'S4.8', 'S4.8.1'],['S1', 'S1.8', 'S1.8.1'],['S1', 'S1.8', 'S1.8.1'], ['S1', 'S1.12', 'S1.12.3']]
    lev_n = 0
    will output ('S1', 0.75) [i.e. 'S1' is 75% of the level 0 codes]
    """
    lev_codes = [w[lev_n] for w in split_possible_codes if w[lev_n]]
    if lev_codes:
        lev_code, lev_num = Counter(lev_codes).most_common(1)[0]
        lev_prop = (
            0 if len(split_possible_codes) == 0 else lev_num / len(split_possible_codes)
        )
        return lev_code, lev_prop
    else:
        return None, None


def get_top_match(score_0, score_1, threshold_0, threshold_1):
    # Returns the index of which one is bigger
    # To deal with times where there is no score
    if not score_0:
        score_0 = 0
    if not score_1:
        score_1 = 0

    if score_0 < threshold_0:
        if score_1 < threshold_1:
            return None
        else:
            return 1
    else:
        if score_1 < threshold_1:
            return 0
        else:
            return np.argmax([score_0, score_1])
