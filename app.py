import streamlit as st
import pickle
import numpy as np
from preprocess import preprocess_text
import helper  # Importing the file we created above

# ----------------------------------------------------------------
# 1. PAGE CONFIGURATION & CSS STYLING
# ----------------------------------------------------------------
st.set_page_config(
    page_title="Duplicate Question Pairs",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS for "Exciting" UI
st.markdown("""
<style>
    /* Main Background adjustments */
    .stApp {
        background-color: #f0f2f6;
    }
    
    /* Title Styling - Gradient Text */
    .title-text {
        font-family: 'Helvetica', sans-serif;
        font-weight: 700;
        font-size: 3em;
        background: -webkit-linear-gradient(45deg, #FF4B4B, #FF914D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
    }
    
    /* Subtitle/Description Styling */
    .subtitle-text {
        text-align: center;
        color: #555;
        font-size: 1.1em;
        margin-bottom: 30px;
    }

    /* Input Card Styling */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 1px solid #ddd;
        padding: 10px;
    }

    /* Button Styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #FF4B4B 0%, #FF416C 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 18px;
        border-radius: 12px;
        transition-duration: 0.4s;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
        color: white;
    }

    /* Footer Styling */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: #333;
        text-align: center;
        padding: 10px;
        font-family: 'Courier New', Courier, monospace;
        font-weight: bold;
        border-top: 1px solid #eaeaea;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------
# 2. LOAD MODELS
# ----------------------------------------------------------------
# Using st.cache_resource so we don't reload models on every interaction (faster)
@st.cache_resource
def load_models():
    model = pickle.load(open('model.pkl', 'rb'))
    cv = pickle.load(open('bow.pkl', 'rb'))
    return model, cv

model, cv = load_models()

# ----------------------------------------------------------------
# 3. UI LAYOUT
# ----------------------------------------------------------------

# Title Section
st.markdown('<p class="title-text">Duplicate Question Pairs</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Powered by Machine Learning & NLP</p>', unsafe_allow_html=True)

# Input Section (Using a container for better grouping)
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        q1_input = st.text_input("First Question", placeholder="e.g. How do I learn Python?")
    with col2:
        q2_input = st.text_input("Second Question", placeholder="e.g. What is the best way to learn Python?")

# Space
st.write("")
st.write("")

# ----------------------------------------------------------------
# 4. PREDICTION LOGIC
# ----------------------------------------------------------------
if st.button("Check Similarity 🔍"):
    if q1_input and q2_input:
        
        # Spinner adds a professional feel
        with st.spinner("Analyzing semantics..."):
            try:
                # 1. Preprocess
                q1_clean = preprocess_text(q1_input)
                q2_clean = preprocess_text(q2_input)
                
                # 2. Feature Engineering
                query_point = helper.query_point_creator(q1_clean, q2_clean, cv)
                
                # 3. Predict
                prediction = model.predict(query_point)[0]
                
                # 4. Display Result with Colors
                st.write("---")
                if prediction == 1:
                    st.success("Result: **Duplicate Questions** ✅")
                    st.caption("These questions likely have the same intent.")
                    st.balloons()
                else:
                    st.error("Result: **Not Duplicate** ❌")
                    st.caption("These questions seem to have different meanings.")
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("⚠️ Please fill in both question fields.")

# ----------------------------------------------------------------
# 5. FOOTER
# ----------------------------------------------------------------
st.markdown('<div class="footer">Made by Aryan Phanse</div>', unsafe_allow_html=True)