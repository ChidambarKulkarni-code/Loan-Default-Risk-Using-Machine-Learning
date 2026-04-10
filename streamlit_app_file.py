import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Loan Default Prediction App",
    page_icon="💳",
    layout="centered"
)

DATA_FILE = "Loan_default.csv"

@st.cache_resource
def train_model():
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        st.error("❌ 'Loan_default.csv' not found. Place it in the same folder as this script.")
        st.stop()

    if "LoanID" in df.columns:
        df = df.drop(columns=["LoanID"])

    # Normalize binary columns: 0/1 → "Yes"/"No"
    binary_cols = ["HasMortgage", "HasDependents", "HasCoSigner"]
    for col in binary_cols:
        if col in df.columns and df[col].dtype != object:
            df[col] = df[col].map({1: "Yes", 0: "No"})

    X = df.drop(columns=["Default"])
    y = df["Default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [col for col in X_train.columns if col not in categorical_cols]

    col_ranges = {}
    for col in numeric_cols:
        col_ranges[col] = (float(X[col].min()), float(X[col].max()))

    float_cols = X_train[numeric_cols].select_dtypes(include=["float"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols)
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)
    val_accuracy = model.score(X_test, y_test)

    return model, col_ranges, float_cols, val_accuracy


model, col_ranges, float_cols, val_accuracy = train_model()

st.title("💳 Loan Default Prediction App")
st.sidebar.metric("Model Validation Accuracy", f"{val_accuracy:.2%}")


def get_input(label, col):
    lo, hi = col_ranges.get(col, (0.0, 100.0))
    # Default = midpoint, always clamped within [lo, hi]
    default = max(lo, min(hi, (lo + hi) / 2))
    if col in float_cols:
        return st.number_input(label, min_value=float(lo), max_value=float(hi), value=float(default), step=0.01)
    else:
        return st.number_input(label, min_value=int(lo), max_value=int(hi), value=int(default))


with st.form("prediction_form"):
    age              = get_input("Age", "Age")
    income           = get_input("Income", "Income")
    loan_amount
