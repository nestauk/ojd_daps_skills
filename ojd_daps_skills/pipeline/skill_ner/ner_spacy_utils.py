"""
There are two main functions in this script
1. fix_entity_annotations: to clean badly annotated entities, e.g. when trailing whitespace
is included, or the entity starts in the middle of a "word" due to bad parsing, e.g. "fixMe"
2. fix_all_formatting: to clean all the text - removing any occurences of camelcase, but not
neccessarily to do with the entity spans.

These are combined in clean_entities_text.

Warning: It is important to not apply any old text cleaning function where labelled data is concerned -
changing the text means the span label information should also be changed.
e.g. "This is   a SKILL" has SKILL entity at characters [13, 17]
but "This is a SKILL" has SKILL entity at characters [11, 15]

"""
import re
import difflib
from toolz import pipe

# Pattern for fixing a missing space between enumerations, for split_sentences()
compiled_missing_space_pattern = re.compile("([a-z])([A-Z])")
# Characters outside these rules will be padded, for pad_punctuation()
compiled_nonalphabet_nonnumeric_pattern = re.compile(r"([^a-zA-Z0-9] )")

# The list of camel cases which should be kept in
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

# Any trailing chars that match these are removed
trim_chars = [" ", ".", ",", ";", ":", "\xa0"]


def edit_ents(text, orig_ents):
    """
    A function to fix the text and entity spans,
    will remove trailing whitespace/punctuation
    from the text and spans
    """

    editted = False
    # Don't include trailing whitespace from entity spans
    trimmed_ents = []
    for b, e, l in orig_ents:
        if text[b] in trim_chars:
            new_b = b + 1
            editted = True
        else:
            new_b = b

        if text[e - 1] in trim_chars:
            new_e = e - 1
            editted = True
        else:
            new_e = e
        trimmed_ents.append((new_b, new_e, l))
    return trimmed_ents, editted


def fix_entity_annotations(text, ents):
    """
    Clean the text and entity spans for cases
    where the entity ends but the next character is not a space
    e.g. "this is OK you need to fixMe please and hereToo please"
    ents = [(8, 10, "LABEL"), (15, 26, "LABEL"), (36,44,"LABEL")]

    Also:
    - if start or end of entity is a space then trim it
    """
    ent_additions = [0] * len(ents)
    insert_index_space = []
    for i, (b, e, l) in enumerate(ents):

        # If the char before the start of this span is not a space,
        # Then update from this ent onwards
        if text[b - 1] != " ":
            ent_additions[i:] = [ea + 1 for ea in ent_additions[i:]]
            insert_index_space.append(b)

        # If the next char after this span is not a space,
        # then update the start and endings of all entities after this
        if (e) < len(text):
            if text[e] != " ":
                ent_additions[(i + 1) :] = [ea + 1 for ea in ent_additions[(i + 1) :]]
                insert_index_space.append(e)

    # Fix entity spans
    new_ents = []
    for (b, e, l), add_n in zip(ents, ent_additions):
        new_ents.append((b + add_n, e + add_n, l))

    # Add spaces in the correct places
    b = 0
    new_texts = []
    for e in insert_index_space:
        new_texts.append(text[b:e])
        b = e
    new_texts.append(text[b:])
    new_text = " ".join(new_texts)

    editted = True
    trimmed_ents = new_ents
    while editted:
        trimmed_ents, editted = edit_ents(new_text, trimmed_ents)

    return new_text, trimmed_ents


def pad_punctuation(text):
    """Pad punctuation marks with spaces (to facilitate lemmatisation)"""
    text = compiled_nonalphabet_nonnumeric_pattern.sub(r" \1 ", text)
    return text


def detect_camelcase(text):
    """
    Splits a word written in camel-case into separate sentences. This fixes a case
    when the last word of a sentence in not seperated from the capitalised word of
    the next sentence. This tends to occur with enumerations.

    For example, the string "skillsBe" will be converted to "skills. Be"

    Some camelcases are allowed though - these are found and replaced. e.g. JavaScript

    Note that the present solution doesn't catch all such cases (e.g. "UKSkills")

    Reference: https://stackoverflow.com/questions/1097901/regular-expression-split-string-by-capital-letter-but-ignore-tla
    """
    text = compiled_missing_space_pattern.sub(r"\1. \2", str(text))
    for exception in exception_camelcases:
        exception_cleaned = compiled_missing_space_pattern.sub(r"\1. \2", exception)
        if exception_cleaned in text:
            text = text.replace(exception_cleaned, exception)

    return text


def clean_text_pipeline(text):
    """
    Pipeline for preprocessing online job vacancy and skills-related text.
    This should ONLY insert characters (eg spaces, fullstops) - not delete or replace any.
    This is because when it comes to cross referencing the cleaned text with entity spans
    our algorithm depends on only insertion.

    Args:
            text (str): Text to be processed via the pipeline
    """
    return pipe(
        text,
        detect_camelcase,
        pad_punctuation,  # messes up entity spans
    )


def get_old2new_chars_dict(orig_text, new_text):
    """
    This is a function to map the orig_text character indices to the new_text indices
    e.g.
    orig_text = "abcd"
    new_text = "ab cd"
    old2new_chars_dict = {0:0, 1:1, 2:3, 3:4}
    """
    seq_matcher = difflib.SequenceMatcher(None, orig_text, new_text)
    old2new_chars_dict = {}
    for tag, i1, i2, j1, j2 in seq_matcher.get_opcodes():
        if tag == "equal":
            step_up = j1 - i1
            for i in range(i1, i2):
                old2new_chars_dict[i] = i + step_up
        elif tag == "insert":
            old2new_chars_dict[i1] = j1
        elif tag == "replace":
            # This shouldnt really be happening since our cleaning is only
            # inserting, but sometimes it does categorise as "replace" in cases where it
            # thinks adding whitespace to either side is a replacement
            # e.g. "abcd" -> " abcd "
            step_up = j1 - i1
            for i in range(i1, i2):
                old2new_chars_dict[i] = i + 1 + step_up

    return old2new_chars_dict


def fix_all_formatting(text, ents):
    new_text = clean_text_pipeline(text)
    old2new_chars_dict = get_old2new_chars_dict(text, new_text)

    new_ents = []
    num_index_problems = 0
    for b, e, t in ents:
        new_b = old2new_chars_dict.get(b)
        new_e = old2new_chars_dict.get(e)
        if new_b and new_e:
            new_ents.append((new_b, new_e, t))
        else:
            num_index_problems += 1

    if num_index_problems != 0:
        print(
            f"Problems with {num_index_problems} entity spans - these will be left out of any training or testing"
        )

    return new_text, new_ents


def clean_entities_text(text, ents):
    text, ents = fix_all_formatting(text, ents)
    text, ents = fix_entity_annotations(
        text, ents
    )  # apply after to deal with the padding
    return text, ents
