import streamlit as st
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

from preprocess import preprocess_data


def show_shap():

    st.header("📊 SHAP Feature Importance")

    try:
        model = joblib.load("models/churn_model.pkl")
    except Exception as e:
        st.error(f"Could not load model: {e}")
        return

    try:
        df = pd.read_csv("data/Telco-Customer-Churn.csv")
    except Exception as e:
        st.error(f"Could not load dataset: {e}")
        return

    # Use the SAME preprocessing as training
    X, y = preprocess_data(df)

    st.success("Dataset loaded successfully!")

    if st.checkbox("Show Processed Data"):
        st.dataframe(X.head())

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        st.subheader("Feature Importance")

        fig, ax = plt.subplots(figsize=(10, 6))

        shap.summary_plot(
            shap_values,
            X,
            show=False
        )

        st.pyplot(fig)

    except Exception as e:
        st.error(f"SHAP Error: {e}")