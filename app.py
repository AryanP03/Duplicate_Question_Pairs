import streamlit as st
import pickle
import numpy as np
from preprocess import preprocess_text
import helper # Importing the file we created above

# Load the saved models
model = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('bow.pkl', 'rb'))

st.title("Duplicate Question Pairs Classifier")

# Create input fields
q1_input = st.text_input("Enter Question 1")
q2_input = st.text_input("Enter Question 2")

if st.button("Check Similarity"):
    if q1_input and q2_input:
        # 1. Preprocess the text (using your preprocess.py logic)
        q1_clean = preprocess_text(q1_input)
        q2_clean = preprocess_text(q2_input)
        
        # 2. Generate the full feature vector (6022 features)
        query_point = helper.query_point_creator(q1_clean, q2_clean, cv)
        
        # 3. Predict
        prediction = model.predict(query_point)[0]
        
        # 4. Display Result
        if prediction == 1:
            st.header("Result: Duplicate Questions")
        else:
            st.header("Result: Not Duplicate")
    else:
        st.warning("Please enter both questions.")