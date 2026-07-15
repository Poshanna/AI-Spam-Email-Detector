import streamlit as st
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Spam Email Detector",
    page_icon="📧",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📌 About")

st.sidebar.info(
"""
This project detects whether an Email/SMS is Spam or Not Spam.

Machine Learning Model:
- TF-IDF Vectorizer
- Multinomial Naive Bayes

Developed using Python & Streamlit.
"""
)

st.sidebar.success("Model Accuracy : 96.68%")

# -----------------------------
# Title
# -----------------------------
st.title("📧 AI Spam Email Detector")

st.write(
"""
Paste any Email or SMS below and click **Detect Spam**.
"""
)

# -----------------------------
# Text Area
# -----------------------------
message = st.text_area(
    "Enter Message",
    height=180
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Detect Spam"):

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:

        transformed = vectorizer.transform([message])

        prediction = model.predict(transformed)[0]

        confidence = model.predict_proba(transformed).max() * 100

        st.divider()

        st.subheader("Prediction")

        if prediction == 1:

            st.error("🚨 SPAM MESSAGE")

        else:

            st.success("✅ NOT SPAM")

        st.write(f"Confidence : **{confidence:.2f}%**")

        st.progress(float(confidence / 100))

# -----------------------------
# Examples
# -----------------------------
st.divider()

st.subheader("Example Messages")

col1, col2 = st.columns(2)

with col1:

    st.error("Spam Example")

    st.code(
"""Congratulations!

You have won ₹50,000.

Click the link below to claim your prize."""
    )

with col2:

    st.success("Ham Example")

    st.code(
"""Hi,

Can we meet after class today?

Thanks."""
    )

# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption("Developed using Python | Scikit-Learn | Streamlit")