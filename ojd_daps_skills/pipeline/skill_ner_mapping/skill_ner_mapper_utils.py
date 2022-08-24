"""Utils to map extracted skills from NER model to
taxonomy skills."""

import re
from collections import Counter
import json

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
    if any([isinstance(el, str) for el in split_possible_codes]):
        split_possible_codes = [split_possible_codes]

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
