import streamlit as st
import numpy as np
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
import pandas as pd

df = pd.read_csv("~/Documents/ML_VLU/lab2/Data/Education.csv")
text, label = df['Text'], df['Label']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(text, label, test_size=0.2, random_state=42)

# Convert data into numerical features
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english')
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

X_train_vect = X_train_vect.toarray()
X_test_vect = X_test_vect.toarray()

Bernoulli, Multinomial = BernoulliNB(), MultinomialNB()
Bernoulli.fit(X_train_vect, y_train)
Multinomial.fit(X_train_vect, y_train)

title = st.text_input("Tôi sẽ phân tích tâm trạng của bạn ", "0")

user = vectorizer.transform(np.array([title]))
ans = Bernoulli.predict(user)

st.write("Tâm trạng của bạn là", "TOT" if ans == "positive" else ("KO TOT" if title == "" else ""))