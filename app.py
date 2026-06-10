import streamlit as st
import pickle

# Load best model
model = pickle.load(
    open("models/best_model.pkl", "rb")
)

# Load vectorizer
vectorizer = pickle.load(
    open("models/vectorizer.pkl", "rb")
)

st.title("📧 Email Spam Detector")

st.write(
    "Enter SMS message below to check whether it is Spam or Ham."
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