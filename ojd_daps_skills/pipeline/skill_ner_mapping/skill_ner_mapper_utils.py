"""Utils to map extracted skills from NER model to 
taxonomy skills."""

import re
import nltk
from nltk.corpus import stopwords

lem = nltk.WordNetLemmatizer()

nltk.download("omw-1.4")
nltk.download("stopwords")
stopwords = set(stopwords.words("english"))


def preprocess_skill(skill):
    """Preprocess skill to remove bullet points, convert colon, hyphens
    and slashes to spaces, lowercase, remove trailing whitespace and
    lemmatise skill.
    
    Inputs:
        skill (str): skill to be preprocessed.
    
    Outputs:
        skill (str): preprocessed skill. 
    """
    punctuation_replacement_rules = {
        # old patterns: replacement pattern
        "[\u2022,\u2023,\u25E6,\u2043,\u2219]": "",  # Convert bullet points to empty string
        r"[-/:\\]": " ",  # Convert colon, hyphens and forward and backward slashes to spaces
        r"[^a-zA-Z0-9,.; #(++)]": "",  # Preserve spaces, commas, full stops, semicollons
    }

    compiled_punct_patterns = [
        re.compile(p) for p in punctuation_replacement_rules.keys()
    ]
    punct_replacement = list(punctuation_replacement_rules.values())
    compiled_missing_space_pattern = re.compile("([a-z])([A-Z])([a-z])")

    # get rid of bullet points
    for j, pattern in enumerate(compiled_punct_patterns):
        skill = pattern.sub(punct_replacement[j], skill)

    # get first half of skill where skills contain camel cases (assumes skill was first half)
    skill = compiled_missing_space_pattern.sub(r"\1. \2\3", skill).split(".")[0]

    # remove stopwords
    skill = " ".join(filter(lambda token: token not in stopwords, skill.split(" ")))

    # lemmatise tokens in skill
    skill = " ".join([lem.lemmatize(token) for token in skill.split(" ")])

    # lowercase and remove trailing with spaces
    return skill.lower().strip()
