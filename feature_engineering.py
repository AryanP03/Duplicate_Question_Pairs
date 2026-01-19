import numpy as np
from fuzzywuzzy import fuzz
from collections import Counter

STOP_WORDS = set([
    'the','is','in','it','to','and','a','of','for','on','with','as','by'
])

def basic_features(q1, q2):
    q1_words = q1.split()
    q2_words = q2.split()

    common_words = set(q1_words) & set(q2_words)

    return [
        len(q1),                         # q1_len
        len(q2),                         # q2_len
        len(q1_words),                   # q1_num_words
        len(q2_words),                   # q2_num_words
        len(common_words),               # word_common
        len(q1_words) + len(q2_words),   # word_total
        len(common_words) / (len(q1_words) + len(q2_words) + 1)  # word_share
    ]

def token_features(q1, q2):
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    q1_words = set([w for w in q1_tokens if w not in STOP_WORDS])
    q2_words = set([w for w in q2_tokens if w not in STOP_WORDS])

    q1_stops = set([w for w in q1_tokens if w in STOP_WORDS])
    q2_stops = set([w for w in q2_tokens if w in STOP_WORDS])

    common_words = q1_words & q2_words
    common_stops = q1_stops & q2_stops
    common_tokens = set(q1_tokens) & set(q2_tokens)

    return [
        len(common_words) / (min(len(q1_words), len(q2_words)) + 1),
        len(common_words) / (max(len(q1_words), len(q2_words)) + 1),
        len(common_stops) / (min(len(q1_stops), len(q2_stops)) + 1),
        len(common_stops) / (max(len(q1_stops), len(q2_stops)) + 1),
        len(common_tokens) / (min(len(q1_tokens), len(q2_tokens)) + 1),
        len(common_tokens) / (max(len(q1_tokens), len(q2_tokens)) + 1),
        int(q1_tokens[-1] == q2_tokens[-1]),
        int(q1_tokens[0] == q2_tokens[0])
    ]

def length_features(q1, q2):
    return [
        (len(q1) + len(q2)) / 2,
        abs(len(q1) - len(q2)),
        fuzz.partial_ratio(q1, q2) / 100
    ]

def fuzzy_features(q1, q2):
    return [
        fuzz.QRatio(q1, q2),
        fuzz.partial_ratio(q1, q2),
        fuzz.token_sort_ratio(q1, q2),
        fuzz.token_set_ratio(q1, q2)
    ]
