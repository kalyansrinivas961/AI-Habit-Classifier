import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

data = {
    'entry': [
        "Woke up early, exercised, completed assignments",
        "Spent the whole day watching Netflix and eating junk food",
        "Organized my workspace and planned tomorrow",
        "Did nothing useful today, just browsed social media",
        "Went to class and took notes",
        "Argued with a friend, felt bad all evening",
        "Worked on my project and learned new things",
        "Slept all day and skipped meals",
        "Helped mom with chores and read a book",
        "Played games all night and missed my deadlines"
    ],
    'label': [
        'Productive',
        'Unproductive',
        'Productive',
        'Unproductive',
        'Neutral',
        'Unproductive',
        'Productive',
        'Unproductive',
        'Productive',
        'Unproductive'
    ]
}

df = pd.DataFrame(data)
X_train, X_test, y_train, y_test = train_test_split(df['entry'], df['label'], test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

pipeline.fit(X_train, y_train)

st.set_page_config(page_title="Habit Classifier", layout="centered")
st.title("🧠 AI Habit Classifier")
st.write("Enter a short journal of your day to find out if it was Productive, Unproductive, or Neutral.")

user_input = st.text_area("✍️ Write your journal entry below:", height=150)

if st.button("Classify Entry"):
    if user_input.strip():
        prediction = pipeline.predict([user_input])[0]
        st.success(f"🏷️ Your day was classified as: **{prediction}**")
    else:
        st.warning("Please enter some text to classify.")
