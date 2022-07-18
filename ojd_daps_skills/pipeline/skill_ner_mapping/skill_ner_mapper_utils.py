"""Utils to map extracted skills from NER model to
taxonomy skills."""

import re
import nltk
from nltk.corpus import stopwords
import pandas as pd

lem = nltk.WordNetLemmatizer()

nltk.download("omw-1.4")
nltk.download("stopwords")
stopwords = set(stopwords.words("english"))

# load punctuation replacement rules
punctuation_replacement_rules = {
    # old patterns: replacement pattern
    "[\u2022,\u2023,\u25E6,\u2043,\u2219]": "",  # Convert bullet points to empty string
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

    job_stopwords = list("skill", "skills")
    # get rid of bullet points
    for j, pattern in enumerate(compiled_punct_patterns):
        skill = pattern.sub(punct_replacement[j], skill)

    # get first half of skill where skills contain camel cases (assumes skill was first half)
    skill = compiled_missing_space_pattern.sub(r"\1. \2\3", skill).split(".")[0]

    # remove stopwords
    skill = " ".join(
        filter(
            lambda token: token not in stopwords | set(job_stopwords), skill.split(" ")
        )
    )

    # lemmatise tokens in skill
    skill = " ".join([lem.lemmatize(token) for token in skill.split(" ")])

    # lowercase and remove trailing with spaces
    return skill.lower().strip()


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
