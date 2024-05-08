"""
Text cleaning utilities for the skills extraction pipeline.
"""

import re
from hashlib import md5

from toolz import pipe

compiled_missing_space_pattern = re.compile("([a-z])([A-Z])")
exception_camelcases = [
    "JavaScript",
    "WordPress",
    "PowerPoint",
    "CloudFormation",
    "CommVault",
    "InDesign",
    "GitHub",
    "GitLab",
    "DevOps",
    "QuickBooks",
    "TypeScript",
    "XenDesktop",
    "DevSecOps",
    "CircleCi",
    "LeDeR",
    "CeMap",
    "MavenAutomation",
    "SaaS",
    "iOS",
    "MySQL",
    "MongoDB",
    "NoSQL",
    "GraphQL",
    "VoIP",
    "PhD",
    "HyperV",
    "PaaS",
    "ArgoCD",
    "WinCC",
    "AutoCAD",
]


def detect_camelcase(text: str) -> str:
    """Split camelcase words into separate sentences.

        i.e. "skillsBe" --> "skills. Be"

        Some camelcases are allowed though - these are found and replaced. e.g. JavaScript

        Reference: https://stackoverflow.com/questions/1097901/regular-expression-split-string-by-capital-letter-but-ignore-tla

    Args:
        text (str): Text to be cleaned.

    Returns:
        str: Split text with spaces based on camelcase.
    """

    text = compiled_missing_space_pattern.sub(r"\1. \2", str(text))
    for exception in exception_camelcases:
        exception_cleaned = compiled_missing_space_pattern.sub(r"\1. \2", exception)
        if exception_cleaned in text:
            text = text.replace(exception_cleaned, exception)

    return text


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


def clean_text(text: str) -> str:
    """Clean text by replacing punctuation and camelcase.

    Args:
        text (str): Text to be cleaned.

    Returns:
        str: Cleaned text.
    """

    return pipe(text, detect_camelcase, replacements)


def short_hash(text: str) -> int:
    """Create a short hash from a string.

    Args:
        text (str): Text to be hashed.

    Returns:
        int: Short hash of the text.
    """

    hx_code = md5(text.encode()).hexdigest()
    int_code = int(hx_code, 16)
    short_code = str(int_code)[:16]
    return int(short_code)
