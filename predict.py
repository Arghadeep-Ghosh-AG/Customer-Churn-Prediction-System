import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("models/churn_model.pkl")
encoders = joblib.load("models/encoders.pkl")


def predict_churn(customer_data):

    # Convert dictionary to DataFrame
    df = pd.DataFrame([customer_data])

    # Convert TotalCharges
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df.dropna(inplace=True)

    # Feature Engineering
    df["AvgMonthlyCharge"] = (
        df["TotalCharges"] /
        (df["tenure"] + 1)
    )

    # Outlier Treatment
    for col in ["TotalCharges", "MonthlyCharges"]:

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df[col] = df[col].clip(
            lower=lower_bound,
            upper=upper_bound
        )

    # Drop customerID
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    # Encode categorical columns using saved encoders
    categorical_cols = df.select_dtypes(
        include="object"
    ).columns

    for col in categorical_cols:

        if col in encoders:

            encoder = encoders[col]

            value = df[col].iloc[0]

            if value not in encoder.classes_:
                raise ValueError(
                    f"Unknown category '{value}' in column '{col}'"
                )

            df[col] = encoder.transform(df[col])

    # Prediction
    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0][1]

    return prediction, probability


if __name__ == "__main__":

    sample_customer = {
        "customerID": "7590-VHVEG",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
    }

    prediction, probability = predict_churn(
        sample_customer
    )

    print(
        "Customer will Churn"
        if prediction == 1
        else "Customer will Stay"
    )

    print(
        f"Churn Probability: {probability:.2%}"
    )

