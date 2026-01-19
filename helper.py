import numpy as np
from fuzzywuzzy import fuzz

# -----------------------------------------------
# 1. Feature Engineering Functions
# -----------------------------------------------

def basic_features(q1, q2):
    # Length of questions
    q1_len = len(q1)
    q2_len = len(q2)
    
    # Number of words
    q1_words = q1.split()
    q2_words = q2.split()
    q1_num_words = len(q1_words)
    q2_num_words = len(q2_words)
    
    # Common words
    w1 = set(map(lambda word: word.lower().strip(), q1_words))
    w2 = set(map(lambda word: word.lower().strip(), q2_words))
    word_common = len(w1 & w2)
    
    # Total words
    word_total = len(w1) + len(w2)
    
    # Word Share
    word_share = round(word_common / word_total, 2) if word_total > 0 else 0
    
    return [q1_len, q2_len, q1_num_words, q2_num_words, word_common, word_total, word_share]

def token_features(q1, q2):
    SAFE_DIV = 0.0001
    
    token_features = [0.0] * 8
    
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features
        
    # Stopwords (you can use NLTK or a hardcoded list like typical in these projects)
    STOP_WORDS = set(['the', 'is', 'in', 'on', 'at', 'to', 'a', 'an', 'of']) # Simplified list
    
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])
    
    common_word_count = len(q1_words.intersection(q2_words))
    common_stop_count = len(q1_stops.intersection(q2_stops))
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
    
    # cwc_min
    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    # cwc_max
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    # csc_min
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    # csc_max
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    # ctc_min
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    # ctc_max
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    # last_word_eq
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    # first_word_eq
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    
    return token_features

def length_features(q1, q2):
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    mean_len = (len(q1_tokens) + len(q2_tokens)) / 2
    abs_len_diff = abs(len(q1_tokens) - len(q2_tokens))
    
    # Longest substring ratio (Simplified logic for demo)
    # Typically uses difflib.SequenceMatcher in production code
    import difflib
    strs = [q1, q2]
    match = difflib.SequenceMatcher(None, q1, q2).find_longest_match(0, len(q1), 0, len(q2))
    longest_substr_ratio = match.size / min(len(q1), len(q2)) if min(len(q1), len(q2)) > 0 else 0
    
    return [mean_len, abs_len_diff, longest_substr_ratio]

def fuzzy_features(q1, q2):
    fuzz_ratio = fuzz.QRatio(q1, q2)
    fuzz_partial_ratio = fuzz.partial_ratio(q1, q2)
    token_sort_ratio = fuzz.token_sort_ratio(q1, q2)
    token_set_ratio = fuzz.token_set_ratio(q1, q2)
    
    return [fuzz_ratio, fuzz_partial_ratio, token_sort_ratio, token_set_ratio]

# -----------------------------------------------
# 2. Query Point Creator (The main pipeline function)
# -----------------------------------------------

def query_point_creator(q1, q2, cv):
    """
    Takes two raw strings and the loaded CountVectorizer.
    Returns the final feature array (1, 6022).
    """
    
    # 1. Generate Manual Features (22 features)
    features = []
    features.extend(basic_features(q1, q2))
    features.extend(token_features(q1, q2))
    features.extend(length_features(q1, q2))
    features.extend(fuzzy_features(q1, q2))
    
    # Reshape to shape (1, 22)
    features_array = np.array(features).reshape(1, -1)
    
    # 2. Generate Bag of Words Features (3000 + 3000)
    q1_bow = cv.transform([q1]).toarray()
    q2_bow = cv.transform([q2]).toarray()
    
    # 3. Concatenate all
    # Order must match training: q1_bow + q2_bow + manual_features
    return np.hstack((q1_bow, q2_bow, features_array))