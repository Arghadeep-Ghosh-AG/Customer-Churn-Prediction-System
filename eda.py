import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def show_eda():
    st.header("📊 Exploratory Data Analysis")

    try:
        df = pd.read_csv("data/Telco-Customer-Churn.csv")
    except Exception as e:
        st.error(f"Could not load dataset: {e}")
        return

    # Dataset Preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Shape
    st.subheader("Dataset Shape")
    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")

    # Missing Values
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Data Types
    st.subheader("Data Types")
    st.write(df.dtypes)

    # Summary Statistics
    st.subheader("Summary Statistics")
    st.dataframe(df.describe(include='all'))

    # Churn Distribution
    if "Churn" in df.columns:
        st.subheader("Churn Distribution")
        churn_counts = df["Churn"].value_counts()
        st.bar_chart(churn_counts)

    # Contract Distribution
    if "Contract" in df.columns:
        st.subheader("Contract Distribution")
        st.bar_chart(df["Contract"].value_counts())

    # Internet Service Distribution
    if "InternetService" in df.columns:
        st.subheader("Internet Service")
        st.bar_chart(df["InternetService"].value_counts())

    # Monthly Charges Histogram
    if "MonthlyCharges" in df.columns:
        st.subheader("Monthly Charges Distribution")
        fig, ax = plt.subplots()
        ax.hist(df["MonthlyCharges"], bins=20)
        ax.set_xlabel("Monthly Charges")
        ax.set_ylabel("Customers")
        st.pyplot(fig)

    # Tenure Histogram
    if "tenure" in df.columns:
        st.subheader("Tenure Distribution")
        fig, ax = plt.subplots()
        ax.hist(df["tenure"], bins=20)
        ax.set_xlabel("Tenure (Months)")
        ax.set_ylabel("Customers")
        st.pyplot(fig)