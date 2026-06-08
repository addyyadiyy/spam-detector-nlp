import streamlit as st
import pickle

# Load model
model = pickle.load(
    open("spam_model_nb.pkl", "rb")
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
            f"✅ NOT SPAM\n\nConfidence: {confidence:.2f}%"
        )