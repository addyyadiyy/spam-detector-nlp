import streamlit as st
import pandas as pd
import pickle

# -----------------------------

# PAGE CONFIG

# -----------------------------

st.set_page_config(
page_title="Email Spam Detector",
page_icon="📧",
layout="wide"
)

# -----------------------------

# LOAD DATA

# -----------------------------

df = pd.read_csv("data/spam.csv")

# -----------------------------

# LOAD MODEL

# -----------------------------

model = pickle.load(
open("models/best_model.pkl", "rb")
)

vectorizer = pickle.load(
open("models/vectorizer.pkl", "rb")
)

# -----------------------------

# SIDEBAR

# -----------------------------

page = st.sidebar.selectbox(
"Navigation",
[
"Home",
"Text Analyzer",
"Data Explorer",
"Visualizations",
"Model Info"
]
)

# ==================================================

# HOME PAGE

# ==================================================

if page == "Home":

st.title("📧 Email Spam Detector")

st.header("Project Description")

st.write("""
This project uses Natural Language Processing (NLP)
and Machine Learning techniques to classify SMS
messages as Spam or Ham (Not Spam).
""")

st.header("Problem Statement")

st.write("""
Spam messages often contain advertisements,
phishing attempts, and malicious content.
This system helps users automatically identify
unwanted messages.
""")

st.header("How to Use")

st.write("""
1. Navigate to Text Analyzer.
2. Enter an SMS message.
3. Click Analyze.
4. View prediction and confidence score.
""")

st.header("Team Members")

st.write("""
- Nur Adina binti Zahrulkamar
- Laila Khadija binti Mohd Nazri
- Kesshi Akshainie Thevi A/P Saravanan
- Nurin Afiqah Binti Rosman
""")


# ==================================================

# TEXT ANALYZER

# ==================================================

elif page == "Text Analyzer":


st.title("📧 Email Spam Detector")

st.write(
    "Enter SMS message below to determine whether it is Spam or Ham."
)

message = st.text_area(
    "SMS Message"
)

if st.button("Analyze"):

    transformed_message = vectorizer.transform(
        [message]
    )

    prediction = model.predict(
        transformed_message
    )

    probability = model.predict_proba(
        transformed_message
    )

    confidence = max(
        probability[0]
    ) * 100

    if prediction[0] == 1:

        st.error(
            f"🚨 SPAM\n\nConfidence: {confidence:.2f}%"
        )

    else:

        st.success(
            f"✅ HAM\n\nConfidence: {confidence:.2f}%"
        )

    st.subheader(
        "Words Influencing Prediction"
    )

    words = message.split()

    st.write(
        ", ".join(words[:10])
    )


# ==================================================

# DATA EXPLORER

# ==================================================

elif page == "Data Explorer":


st.title("📊 Data Explorer")

st.subheader("Dataset Preview")

st.dataframe(
    df.head()
)

st.subheader("Dataset Statistics")

st.write(
    f"Total Messages: {len(df)}"
)

st.write(
    df["label"].value_counts()
)

st.subheader("Label Distribution")

st.bar_chart(
    df["label"].value_counts()
)


# ==================================================

# VISUALIZATIONS

# ==================================================

elif page == "Visualizations":


st.title("📈 Visualizations")

st.image(
    "visualizations/Visualization 1 - Spam vs Ham Distribution.png"
)

st.image(
    "visualizations/Visualization 2 - Spam Word Cloud.png"
)

st.image(
    "visualizations/Visualization 3 - Ham Word Cloud.png"
)

st.image(
    "visualizations/Visualization 4 - Top 20 Words.png"
)

st.image(
    "visualizations/Visualization 5 - Model Accuracy Comparison.png"
)

st.image(
    "visualizations/Visualization 6 - Best Model Confusion Matrix.png"
)

st.image(
    "visualizations/Visualization 7 - Performance Metrics Comparison.png"
)


# ==================================================

# MODEL INFO

# ==================================================

elif page == "Model Info":


st.title("🤖 Model Information")

st.subheader("Models Used")

st.write("""
• Bag of Words + Naive Bayes

• Bag of Words + KNN

• TF-IDF + Naive Bayes

• TF-IDF + KNN
""")

st.subheader(
    "Performance Comparison"
)

results = pd.DataFrame(
    {
        "Model": [
            "BoW + Naive Bayes",
            "BoW + KNN",
            "TF-IDF + Naive Bayes",
            "TF-IDF + KNN"
        ],
        "Accuracy": [
            0.9812,
            0.9444,
            0.9740,
            0.9444
        ],
        "Precision": [
            0.9156,
            1.0000,
            0.9918,
            1.0000
        ],
        "Recall": [
            0.9463,
            0.5839,
            0.8121,
            0.5839
        ],
        "F1 Score": [
            0.9307,
            0.7373,
            0.8930,
            0.7373
        ]
    }
)

st.dataframe(results)

st.subheader("Best Model")

st.success(
    "BoW + Naive Bayes achieved the highest accuracy (98.12%)."
)

st.subheader(
    "Training Details"
)

st.write("""
Dataset:
SMS Spam Collection Dataset

Feature Extraction:
Bag of Words (BoW)
and TF-IDF

Classification Models:
Naive Bayes and KNN

Train-Test Split:
80% Training
20% Testing
""")

