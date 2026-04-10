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
        st.error("❌ 'Loan_default.csv' not found. Please place it in the same folder as this script.")
        st.stop()

    if "LoanID" in df.columns:
        df = df.drop(columns=["LoanID"])

    # FIX #4: Normalize binary columns to "Yes"/"No" strings if they are 0/1
    binary_cols = ["HasMortgage", "HasDependents", "HasCoSigner"]
    for col in binary_cols:
        if col in df.columns and df[col].dtype != object:
            df[col] = df[col].map({1: "Yes", 0: "No"})

    X = df.drop(columns=["Default"])
    y = df["Default"]

    # FIX #5: Train/test split for validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [col for col in X_train.columns if col not in categorical_cols]

    # Store column ranges for dynamic input bounds
    col_ranges = {col: (int(X[col].min()), int(X[col].max())) for col in numeric_cols}
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

    # Validation accuracy shown in sidebar
    val_accuracy = model.score(X_test, y_test)

    return model, col_ranges, float_cols, val_accuracy


model, col_ranges, float_cols, val_accuracy = train_model()

st.title("💳 Loan Default Prediction App")
st.sidebar.metric("Model Validation Accuracy", f"{val_accuracy:.2%}")

def get_input(label, col, default=None):
    """Helper to auto-set min/max from dataset ranges."""
    lo, hi = col_ranges.get(col, (0, 100))
    default = default if default is not None else lo
    if col in float_cols:
        return st.number_input(label, float(lo), float(hi), float(default), step=0.01)
    else:
        return st.number_input(label, int(lo), int(hi), int(default))


with st.form("prediction_form"):
    age             = get_input("Age", "Age", 30)
    income          = get_input("Income", "Income", 50000)
    loan_amount     = get_input("Loan Amount", "LoanAmount", 10000)
    credit_score    = get_input("Credit Score", "CreditScore", 650)
    months_employed = get_input("Months Employed", "MonthsEmployed", 24)
    num_credit_lines= get_input("Number of Credit Lines", "NumCreditLines", 2)
    interest_rate   = get_input("Interest Rate (%)", "InterestRate", 12.5)
    loan_term       = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
    dti_ratio       = get_input("DTI Ratio", "DTIRatio", 0.25)
    education       = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
    employment_type = st.selectbox("Employment Type", ["Unemployed", "Self-employed", "Part-time", "Full-time"])
    marital_status  = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    has_mortgage    = st.selectbox("Has Mortgage", ["Yes", "No"])
    has_dependents  = st.selectbox("Has Dependents", ["Yes", "No"])
    loan_purpose    = st.selectbox("Loan Purpose", ["Auto", "Business", "Education", "Home", "Other"])
    has_cosigner    = st.selectbox("Has Co-Signer", ["Yes", "No"])

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = pd.DataFrame([{
        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "MonthsEmployed": months_employed,
        "NumCreditLines": num_credit_lines,
        "InterestRate": interest_rate,
        "LoanTerm": loan_term,
        "DTIRatio": dti_ratio,
        "Education": education,
        "EmploymentType": employment_type,
        "MaritalStatus": marital_status,
        "HasMortgage": has_mortgage,
        "HasDependents": has_dependents,
        "LoanPurpose": loan_purpose,
        "HasCoSigner": has_cosigner
    }])

    prediction  = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.metric("Default Probability", f"{probability:.2%}")

    if prediction == 1:
        st.error("⚠️ High Risk — Likely to Default")
    else:
        st.success("✅ Low Risk — Unlikely to Default")
