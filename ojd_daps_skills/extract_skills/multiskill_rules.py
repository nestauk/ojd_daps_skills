"""
Phrase splitting rules for multi-skill phrases. 
"""

import re
from typing import List

from spacy.tokens import Doc


def _split_duplicate_object(parsed_sent: Doc) -> List[str]:
    """Split phrases with duplicate objects (2 verbs + 1 object).

    i.e. 'using and providing clinical supervision'
        --> ['using clinical supervision', 'providing clinical supervision']


    Args:
        parsed_sent (Doc): Spacy parsed sentence.

    Returns:
        List[str]: List of split skills.
    """

    for token in parsed_sent:
        if token.pos_ == "VERB" and token.dep_ == "ROOT":
            has_AND = False
            has_second_verb = False
            has_dobj = False

            for child in token.children:
                if child.pos_ == "CCONJ" and child.lemma_ == "and":
                    has_AND = True

                if child.pos_ == "VERB" and child.dep_ == "conj":
                    has_second_verb = True
                    second_verb = child
                    first_verb = token

                    has_dobj = "dobj" in [o.dep_ for o in second_verb.subtree]

                    if has_dobj:
                        has_dobj = True
                        dobj = " ".join(
                            [
                                c.text
                                for c in second_verb.subtree
                                if c.text != second_verb.text
                            ]
                        )

            if has_AND and has_second_verb and has_dobj:
                first_skill = "{} {}".format(first_verb, dobj)
                second_skill = "{} {}".format(second_verb, dobj)

                return [first_skill, second_skill]

    return None


def _split_on_and(text: str) -> List[str]:
    """Split text on the word 'and' and commas,
        but deal with oxford commas

    Args:
        text (str): Text to split.

    Returns:
        List[str]: List of split text.
    """

    # Get rid of any double spacing
    text = re.sub("\s\s+", " ", text)
    split_on = " and "
    # Normalize combinations of 'and' with commas or semicolons.
    text = text.replace(";", ",")
    replacements = [
        ", and ,",
        ", and,",
        ",and ,",
        ", and ",
        " and ,",
        ",and,",
        " and,",
        ",and ",
    ]
    for replacement in replacements:
        text = text.replace(replacement, split_on)

    # Split on commas and 'and'
    text = text.replace(",", split_on).split(" and ")
    return [t.strip() for t in text]


def _split_duplicate_verb(parsed_phrase: Doc) -> List[str]:
    """Split phrases with duplicate verbs (1 verb + 2 objects).

    i.e. 'using smartphones and apps'
        --> ['using smartphones', 'using apps']


    Args:
        parsed_phrase (Doc): Spacy parsed sentence.

    Returns:
        List[str]: List of split skills.
    """
    for token in parsed_phrase:
        if token.pos_ == "VERB" and token.dep_ == "ROOT":
            has_AND = False
            has_dobj = False
            has_sec_obj = False

            for child in token.children:
                if child.dep_ == "dobj":
                    has_dobj = True

                    objects = " ".join(
                        [c.text for c in token.subtree if c.text != token.text]
                    )

                    split_objects = _split_on_and(objects)

                    object_list = []
                    for split_skill in split_objects:
                        object_list.append(split_skill)

                    for subchild in child.children:
                        if subchild.pos_ == "CCONJ" and subchild.lemma_ == "and":
                            has_AND = True

                        if subchild.dep_ == "conj":
                            has_sec_obj = True

                    if has_AND and has_dobj and has_sec_obj:
                        skill_lists = [
                            "{} {}".format(token.text, split_skill)
                            for split_skill in object_list
                        ]

                        return skill_lists

    return None


def _split_skill_mentions(parsed_phrase: Doc) -> List[str]:
    """Split skill mentions.

    i.e. 'written and oral communication skills'
        --> ['written skills', 'oral communication skills']


    Args:
        parsed_phrase (Doc): Spacy parsed sentence.

    Returns:
        List[str]: List of split skills.
    """
    for token in parsed_phrase:
        if (
            token.pos_ == "NOUN"
            and token.lemma_ == "skill"
            and token.idx == parsed_phrase[-1].idx
        ):
            has_AND = False

            root = [token for token in parsed_phrase if token.dep_ == "ROOT"]
            if root:
                root = root[0]

                for child in root.subtree:
                    if child.pos_ == "CCONJ" and child.lemma_ == "and":
                        has_AND = True

                if has_AND:
                    skill_def = " ".join(
                        [c.text for c in root.subtree if c.text != token.text]
                    )

                    split_skills = _split_on_and(skill_def)

                    skill_lists = []
                    for split_skill in split_skills:
                        skill_lists.append("{} {}".format(split_skill, token.text))

                    return skill_lists
    return None
