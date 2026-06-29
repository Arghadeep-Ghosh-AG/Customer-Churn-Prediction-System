import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder


def preprocess_data(df, save_encoders=False):

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Remove missing values
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

    # Drop Customer ID
    if "customerID" in df.columns:
        df.drop(
            "customerID",
            axis=1,
            inplace=True
        )

    # Encode Categorical Columns
    categorical_cols = df.select_dtypes(
        include="object"
    ).columns

    encoders = {}

    for col in categorical_cols:

        encoder = LabelEncoder()

        df[col] = encoder.fit_transform(df[col])

        encoders[col] = encoder

    # Save encoders ONLY while training
    if save_encoders:
        joblib.dump(
            encoders,
            "models/encoders.pkl"
        )

    # Separate Features and Target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    return X, y