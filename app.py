import streamlit as st
import pickle

# Load model
model = pickle.load(
    open("spam_model.pkl", "rb")
)

# Load vectorizer
vectorizer = pickle.load(
    open("vectorizer.pkl", "rb")
)

st.title("📧 Email Spam Detector")

st.write(
    "Enter an email or SMS message below."
)

message = st.text_area(
    "Message"
)

if st.button("Analyze"):

    transformed_message = vectorizer.transform(
        [message]
    )

    prediction = model.predict(
        transformed_message
    )

    if prediction[0] == 1:
        st.error("🚨 SPAM")
    else:
        st.success("✅ NOT SPAM")