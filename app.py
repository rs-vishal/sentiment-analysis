import streamlit as st
import joblib
from sklearn.feature_extraction.text import CountVectorizer


logistic_model = joblib.load('./Model/Logistic_Regression_model.pkl')
naive_bayes_model = joblib.load('./Model/Naive_Bayes_model.pkl')

vectorizer = joblib.load('./Model/vectorizer.pkl')  

def predict_sentiment(sentence, model):
    sentence_vectorized = vectorizer.transform([sentence])
    prediction = model.predict(sentence_vectorized)
    return prediction[0]

st.title("Sentiment Analysis")
st.write("Enter a sentence below to get the sentiment prediction:")
sentence = st.text_area("Input Sentence", "")

if st.button("Predict"):
    if sentence:
        logistic_prediction = predict_sentiment(sentence, logistic_model)
        naive_bayes_prediction = predict_sentiment(sentence, naive_bayes_model)

        st.write(f"Logistic Regression Prediction: {'Positive' if logistic_prediction == 1 else 'Negative'}")
        st.write(f"Naive Bayes Prediction: {'Positive' if naive_bayes_prediction == 1 else 'Negative'}")
  
    else:
        st.write("Please enter a sentence.")