import os
import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(
    page_title="Loan Default Prediction App",
    page_icon="💳",
    layout="centered"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "Loan_default.csv")


@st.cache_resource
def train_model():
    df = pd.read_csv(DATA_FILE)

    if "LoanID" in df.columns:
        df = df.drop(columns=["LoanID"])

    if "Default" not in df.columns:
        raise ValueError("Target column 'Default' not found in Loan_default.csv")

    X = df.drop(columns=["Default"])
    y = df["Default"]

    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X, y)
    return model, y


try:
    model, y = train_model()
except Exception as e:
    st.error(f"Could not train model from Loan_default.csv: {e}")
    st.stop()

st.title("💳 Loan Default Prediction App")
st.write("Enter borrower details and predict the probability of loan default.")

with st.form("prediction_form"):
    st.subheader("Borrower Details")

    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    income = st.number_input("Income", min_value=0.0, max_value=10000000.0, value=50000.0, step=1000.0)
    loan_amount = st.number_input("Loan Amount", min_value=0.0, max_value=5000000.0, value=10000.0, step=1000.0)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650, step=1)
    months_employed = st.number_input("Months Employed", min_value=0, max_value=600, value=24, step=1)
    num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, max_value=50, value=3, step=1)
    interest_rate = st.number_input("Interest Rate", min_value=0.0, max_value=100.0, value=12.5, step=0.1)
    loan_term = st.selectbox("Loan Term", [12, 24, 36, 48, 60], index=1)
    dti_ratio = st.number_input("DTI Ratio", min_value=0.0, max_value=1.0, value=0.25, step=0.01)

    education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
    employment_type = st.selectbox("Employment Type", ["Unemployed", "Self-employed", "Part-time", "Full-time"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    has_mortgage = st.selectbox("Has Mortgage", ["Yes", "No"])
    has_dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    loan_purpose = st.selectbox("Loan Purpose", ["Auto", "Business", "Education", "Home", "Other"])
    has_cosigner = st.selectbox("Has Co-Signer", ["Yes", "No"])

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

    try:
        prediction = model.predict(input_data)[0]

        proba = model.predict_proba(input_data)[0]
        class_labels = model.named_steps["classifier"].classes_

        if 1 in class_labels:
            default_index = list(class_labels).index(1)
        else:
            default_index = 1

        probability = proba[default_index]

        st.subheader("Prediction Result")
        st.metric("Default Probability", f"{probability:.2%}")

        if probability >= 0.60:
            st.error("🔴 High Risk of Default")
        elif probability >= 0.30:
            st.warning("🟠 Moderate Risk of Default")
        else:
            st.success("🟢 Low Risk of Default")

        st.subheader("Model Decision")
        st.write("Predicted Class:", "Default" if prediction == 1 else "No Default")

        st.subheader("Input Summary")
        st.dataframe(input_data, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
