import pandas as pd
import numpy as np
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from preprocess import preprocess_text
from feature_engineering import (
    basic_features,
    token_features,
    length_features,
    fuzzy_features
)

# Load data
df = pd.read_csv("train.csv")

# Drop missing rows
df.dropna(inplace=True)

# Sample (because full dataset is huge)
df = df.sample(30000, random_state=42)

# Preprocess text
df['q1_clean'] = df['question1'].apply(preprocess_text)
df['q2_clean'] = df['question2'].apply(preprocess_text)

# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
feature_list = []

for q1, q2 in zip(df['q1_clean'], df['q2_clean']):
    features = []
    features.extend(basic_features(q1, q2))
    features.extend(token_features(q1, q2))
    features.extend(length_features(q1, q2))
    features.extend(fuzzy_features(q1, q2))
    feature_list.append(features)

X_features = np.array(feature_list)

# ----------------------------
# BAG OF WORDS
# ----------------------------
bow = CountVectorizer(max_features=3000)

q1_bow = bow.fit_transform(df['q1_clean'])
q2_bow = bow.transform(df['q2_clean'])

X_bow = np.hstack((q1_bow.toarray(), q2_bow.toarray()))

# Combine all features
X = np.hstack((X_bow, X_features))
y = df['is_duplicate'].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# MODEL
# ----------------------------
rf = RandomForestClassifier(
    n_estimators=60,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train, y_train)

print("Train Accuracy:", rf.score(X_train, y_train))
print("Test Accuracy:", rf.score(X_test, y_test))

# ----------------------------
# SAVE MODEL
# ----------------------------
pickle.dump(rf, open("model.pkl", "wb"))
pickle.dump(bow, open("bow.pkl", "wb"))

print("Model and Vectorizer saved successfully!")
