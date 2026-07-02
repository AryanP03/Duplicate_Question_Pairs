# ❓ Quora Duplicate Question Pairs

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Deployed-success)

An NLP-based Machine Learning web application that identifies whether two questions have the same intent or meaning. This project is based on the famous **Quora Question Pairs** Kaggle competition.

## 🔗 Live Demo
Check out the live application deployed on Render:  
👉 **[Click Here to View App](https://quora-question-checker.onrender.com)** *(Note: Due to this is hosted on a free tier, it may take 30-50 seconds to wake up if it hasn't been used recently.)*

---

## 📖 Overview
Quora, a popular Q&A platform, often struggles with duplicate questions (e.g., *"How do I learn Python?"* vs. *"What is the best way to start learning Python?"*). These duplicates dilute the quality of answers. 

This project uses **Natural Language Processing (NLP)** and **Machine Learning** to predict whether two questions are duplicates (1) or not (0).

## 🛠️ Tech Stack
* **Frontend:** [Streamlit](https://streamlit.io/) (Web Interface)
* **Language:** Python
* **Libraries:** Scikit-learn, Pandas, NumPy, NLTK, FuzzyWuzzy
* **Model:** Random Forest Classifier
* **Deployment:** Render

---

## 🧠 Approach & Methodology

### 1. Data Preprocessing
* Removed HTML tags, special characters, and punctuation.
* Performed stemming and stop-word removal.
* Replaced common contractions (e.g., "won't" -> "will not").

### 2. Feature Engineering
To improve accuracy, we extracted **22 advanced features** from the text:
* **Basic Features:** Length of questions, number of words, common words ratio.
* **Token Features:** `cwc_min`, `cwc_max`, `csc_min`, `csc_max` (ratios of common words/stop-words).
* **Length Features:** Absolute length difference, mean length.
* **Fuzzy Features:** Uses `fuzzywuzzy` library (Fuzz Ratio, Partial Ratio, Token Sort Ratio, Token Set Ratio).

### 3. Text Vectorization
* Used **Bag of Words (BoW)** with `CountVectorizer` (max features = 3000) to convert text into numerical vectors.

### 4. Model Training
* Algorithm: **Random Forest Classifier**
* Total Features: ~6022 (3000 BoW Q1 + 3000 BoW Q2 + 22 Manual Features).
* Metric: Accuracy.

---

## 📂 Project Structure
```bash
├── app.py               # Streamlit frontend application
├── helper.py            # Feature extraction & helper functions
├── preprocess.py        # Text cleaning logic
├── train_model.py       # Script to train the model
├── requirements.txt     # List of dependencies
├── model.pkl            # Trained Random Forest Model
├── bow.pkl              # CountVectorizer Object
└── README.md            # Project Documentation
