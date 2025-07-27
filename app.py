import streamlit as st
import joblib
import numpy as np

# Load the model, vectorizer, and label encoder
model = joblib.load(‪"drug_classifier.pkl")
vectorizer = joblib.load("‪tfidf_vectorizer.pkl")
label_encoder = joblib.load("‪label_encoder.pkl")

# App title and intro
st.set_page_config(page_title="Drug Type Predictor", layout="centered")
st.title("💊 Personalized Medicine Recommendation")
st.markdown("This app predicts the **type of drug** (Oral, Injectable, Topical, etc.) based on your medical symptoms and description.")

# User input
user_input = st.text_area("📝 Enter medical reason & description", height=150)

# Predict button
if st.button("🔍 Predict Drug Type"):
    if user_input.strip() == "":
        st.warning("Please enter some text to predict.")
    else:
        # Transform input text
        input_vec = vectorizer.transform([user_input]).toarray()
        
        # Make prediction
        pred = model.predict(input_vec)
        pred_label = label_encoder.inverse_transform(pred)[0]
        
        # Show result
        st.success(f"✅ **Predicted Drug Type:** {pred_label}")
        
        # Show probability scores
        probs = model.predict_proba(input_vec)[0]
        st.subheader("🔬 Prediction Probabilities:")
        for i, prob in enumerate(probs):
            st.write(f"- {label_encoder.classes_[i]}: **{prob*100:.2f}%**")
