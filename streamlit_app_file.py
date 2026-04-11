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
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "Loan_default.csv")


def calculate_emi(principal, annual_rate, months):
    monthly_rate = annual_rate / 12 / 100
    if months == 0:
        return 0
    if monthly_rate == 0:
        return principal / months
    emi = principal * monthly_rate * (1 + monthly_rate) ** months / ((1 + monthly_rate) ** months - 1)
    return emi


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
            n_estimators=250,
            max_depth=14,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X, y)
    return model


def apply_business_rules(base_probability, age, income, loan_amount, credit_score,
                         months_employed, num_credit_lines, interest_rate,
                         loan_term, dti_ratio, employment_type, has_cosigner,
                         has_mortgage, has_dependents):
    """
    Adjust ML output using practical lending rules.
    """
    adjusted_probability = base_probability
    flags = []
    reasons = []

    # Employment consistency logic
    if employment_type == "Unemployed":
        adjusted_probability = max(adjusted_probability, 0.45)
        reasons.append("Unemployed applicants generally carry higher repayment uncertainty")

        if income > 0:
            flags.append("Applicant is unemployed but has declared income. Manual verification required.")
            adjusted_probability += 0.08

        if months_employed > 0:
            flags.append("Months employed is greater than 0 while employment type is Unemployed.")
            adjusted_probability += 0.05

        if has_cosigner == "No":
            adjusted_probability += 0.07
            reasons.append("No co-signer support for unemployed borrower")

    # Credit score logic
    if credit_score < 580:
        adjusted_probability += 0.12
        reasons.append("Very low credit score")
    elif credit_score < 650:
        adjusted_probability += 0.06
        reasons.append("Below-average credit score")

    # DTI logic
    if dti_ratio >= 0.50:
        adjusted_probability += 0.12
        reasons.append("Very high debt-to-income ratio")
    elif dti_ratio >= 0.35:
        adjusted_probability += 0.06
        reasons.append("Elevated debt-to-income ratio")

    # Income vs loan burden logic
    if income > 0:
        income_to_loan_ratio = loan_amount / income
        if income_to_loan_ratio > 8:
            adjusted_probability += 0.10
            reasons.append("Loan amount is very high relative to income")
        elif income_to_loan_ratio > 5:
            adjusted_probability += 0.05
            reasons.append("Loan amount is high relative to income")
    else:
        adjusted_probability += 0.15
        reasons.append("No declared income")

    # Employment stability logic
    if months_employed < 6 and employment_type != "Unemployed":
        adjusted_probability += 0.05
        reasons.append("Low employment stability")

    # Interest burden
    if interest_rate > 18:
        adjusted_probability += 0.07
        reasons.append("High interest burden")
    elif interest_rate > 14:
        adjusted_probability += 0.03
        reasons.append("Above-average interest rate")

    # Support factors
    if has_cosigner == "Yes":
        adjusted_probability -= 0.04
        reasons.append("Co-signer support reduces lender risk")

    if has_mortgage == "Yes":
        adjusted_probability += 0.02

    if has_dependents == "Yes" and income < 40000:
        adjusted_probability += 0.04
        reasons.append("Dependents with limited income may affect repayment ability")

    adjusted_probability = max(0.0, min(adjusted_probability, 0.99))
    return adjusted_probability, flags, reasons


try:
    model = train_model()
except Exception as e:
    st.error(f"Could not train model from Loan_default.csv: {e}")
    st.stop()


st.title("💳 Loan Default Risk Prediction App")
st.caption("Machine learning + business rule underwriting for loan default assessment")

st.sidebar.header("About the App")
st.sidebar.info(
    "This app predicts loan default risk using a machine learning model "
    "enhanced with lending business rules for more realistic decisions."
)

st.sidebar.subheader("What makes this useful?")
st.sidebar.write(
    "- Predicts default probability\n"
    "- Applies underwriting-style logic\n"
    "- Gives approval recommendation\n"
    "- Explains likely risk drivers\n"
    "- Calculates EMI"
)

with st.form("prediction_form"):
    st.subheader("Borrower Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        income = st.number_input("Income", min_value=0.0, max_value=10000000.0, value=50000.0, step=1000.0)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650, step=1)
        employment_type = st.selectbox("Employment Type", ["Unemployed", "Self-employed", "Part-time", "Full-time"])
        months_employed = st.number_input("Months Employed", min_value=0, max_value=600, value=24, step=1)

    with col2:
        loan_amount = st.number_input("Loan Amount", min_value=1000.0, max_value=5000000.0, value=10000.0, step=1000.0)
        interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=12.5, step=0.1)
        loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60], index=1)
        dti_ratio = st.number_input("DTI Ratio", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
        num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, max_value=50, value=3, step=1)

    with col3:
        education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        has_mortgage = st.selectbox("Has Mortgage", ["Yes", "No"])
        has_dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        loan_purpose = st.selectbox("Loan Purpose", ["Auto", "Business", "Education", "Home", "Other"])
        has_cosigner = st.selectbox("Has Co-Signer", ["Yes", "No"])

    submitted = st.form_submit_button("Predict Risk")


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

        base_probability = float(proba[default_index])

        adjusted_probability, flags, reasons = apply_business_rules(
            base_probability=base_probability,
            age=age,
            income=income,
            loan_amount=loan_amount,
            credit_score=credit_score,
            months_employed=months_employed,
            num_credit_lines=num_credit_lines,
            interest_rate=interest_rate,
            loan_term=loan_term,
            dti_ratio=dti_ratio,
            employment_type=employment_type,
            has_cosigner=has_cosigner,
            has_mortgage=has_mortgage,
            has_dependents=has_dependents
        )

        emi = calculate_emi(loan_amount, interest_rate, loan_term)

        # Decision logic
        if adjusted_probability >= 0.65:
            risk_label = "High Risk"
            decision = "Reject"
        elif adjusted_probability >= 0.40:
            risk_label = "Moderate Risk"
            decision = "Manual Review"
        else:
            risk_label = "Low Risk"
            decision = "Approve"

        # Hard rule: unemployed applicants should not be auto-approved
        if employment_type == "Unemployed" and decision == "Approve":
            decision = "Manual Review"
            risk_label = "Moderate Risk"
            adjusted_probability = max(adjusted_probability, 0.40)
            reasons.append("Unemployed applicant routed to manual review")

        st.subheader("Underwriting Result")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Base ML Risk", f"{base_probability:.2%}")
        m2.metric("Adjusted Risk", f"{adjusted_probability:.2%}")
        m3.metric("Risk Level", risk_label)
        m4.metric("Decision", decision)

        st.metric("Estimated EMI", f"₹{emi:,.2f}")

        if risk_label == "High Risk":
            st.error("🔴 High risk of default. Strong caution advised.")
        elif risk_label == "Moderate Risk":
            st.warning("🟠 Moderate risk. Manual underwriting review recommended.")
        else:
            st.success("🟢 Low risk based on current inputs.")

        if flags:
            st.subheader("Validation Flags")
            for flag in flags:
                st.warning(flag)

        if reasons:
            st.subheader("Key Risk Drivers")
            for reason in dict.fromkeys(reasons):
                st.write(f"- {reason}")

        st.subheader("Input Summary")
        st.dataframe(input_data, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
