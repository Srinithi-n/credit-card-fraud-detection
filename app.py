import streamlit as st
from model import predict_transaction
from llm_explainer import generate_explanation

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection")

st.markdown("""
This application demonstrates a complete fraud detection pipeline using:

- Machine Learning (Logistic Regression)
- SMOTE (Handling Imbalanced Data)
- Autoencoder (Deep Learning)
- BERT (NLP)
- LLM Fraud Explanation
""")

st.divider()

amount = st.number_input(
    "Transaction Amount (₹)",
    min_value=0.0,
    value=100.0
)

description = st.text_input(
    "Transaction Description",
    placeholder="Example: Online Electronics Purchase"
)

if st.button("Predict Transaction"):

    prediction, confidence = predict_transaction(
        amount,
        description
    )

    if prediction == "Fraud":
        st.error("🚨 Fraudulent Transaction Detected")
    else:
        st.success("✅ Normal Transaction")

    st.metric(
        label="Confidence",
        value=f"{confidence}%"
    )

    st.divider()

    st.subheader("🤖 LLM Fraud Explanation")

    explanation = generate_explanation(
        amount,
        description,
        prediction
    )

    st.info(explanation)