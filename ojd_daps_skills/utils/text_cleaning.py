"""
Various text cleaning functions for times you aren't also trying to
clean entity spans too (these functions are in ner_spacy_utils).

"""
import re
from toolz import pipe
from hashlib import md5

from ojd_daps_skills.pipeline.skill_ner.ner_spacy_utils import detect_camelcase

# load punctuation replacement rules
punctuation_replacement_rules = {
    # old patterns: replacement pattern
    "[\u2022\u2023\u25E6\u2043\u2219*]": ".",  # Convert bullet points to fullstops
    r"[/:\\]": " ",  # Convert colon and forward and backward slashes to spaces
}

compiled_punct_patterns = {
    re.compile(p): v for p, v in punctuation_replacement_rules.items()
}


def replacements(text):
    """
    Ampersands and bullet points need some tweaking to be most useful in the pipeline.

    Some job adverts have different markers for a bullet pointed list. When this happens
    we want them to be in a fullstop separated format.

    e.g. ";• managing the grants database;• preparing financial and interna"
    ":•\xa0NMC registration paid every year•\xa0Free train"

    """
    text = text.replace("&", "and").replace("\xa0", " ")

    for pattern, rep in compiled_punct_patterns.items():
        text = pattern.sub(rep, text)

    return text


def clean_text(text):

    return pipe(text, detect_camelcase, replacements)


def short_hash(text):
    """Generate a unique short hash for this string - from ojd_daps"""
    hx_code = md5(text.encode()).hexdigest()
    int_code = int(hx_code, 16)
    short_code = str(int_code)[:16]
    return int(short_code)
