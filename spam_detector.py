import pandas as pd
import streamlit as st
import joblib
from scipy.sparse import hstack
import numpy as np

def main():
  html_temp = """<h1>Spam Email Predictor</h1>"""

  model = joblib.load("knn_model.joblib")
  vectorizer = joblib.load('vectorizer.joblib')

  st.markdown(html_temp, unsafe_allow_html=True)
  st.markdown("This app will help you to predict whether the email is spam or not")

  p1 = st.text_input(label="Enter your email subject", value="")
  p2 = st.number_input("Enter email length ", 0, 300, step=1)
  p3 = st.number_input("Enter number of capital words ", 0, 60, step=1)

  s1 = st.selectbox("Select the Has Attachment", ('Yes', 'No'))
  if s1 == 'Yes':
    p4 = 1
  else:
    p4 = 0

  p5 = st.slider("Enter number of special chars", 0, 30)
  p6 = st.slider("Enter number of links", 0, 15)

  if st.button("Predict"):
    subject_vector = vectorizer.transform([p1])

    numeric_features = np.array([[p2, p6, p5, p3, p4]])

    final_input = hstack([subject_vector, numeric_features])

    pred = model.predict(final_input)

    if pred[0] == 1:
      st.error("Spam Email")
    else:
      st.success("Not Spam")

if __name__ == '__main__':
  main()