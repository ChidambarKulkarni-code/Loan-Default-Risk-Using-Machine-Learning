import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Loan Default Prediction App", page_icon="💳", layout="centered")

@st.cache_resource
def load_model():
    model_path = "model.pkl"
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

model = load_model()

st.title("💳 Loan Default Prediction App")
st.write("Enter borrower details and predict the probability of loan default.")

if model is None:
    st.error("model.pkl not found. Add your trained model.")
    st.stop()

with st.form("prediction_form"):
    age = st.number_input("Age", 18, 100, 30)
    income = st.number_input("Income", 0, 50000)
    loan_amount = st.number_input("Loan Amount", 0, 10000)
    credit_score = st.number_input("Credit Score", 300, 900, 650)
    months_employed = st.number_input("Months Employed", 0, 24)
    num_credit_lines = st.number_input("Number of Credit Lines", 0, 3)
    interest_rate = st.number_input("Interest Rate", 0.0, 100.0, 12.5)
    loan_term = st.selectbox("Loan Term", [12, 24, 36, 48, 60])
    dti_ratio = st.number_input("DTI Ratio", 0.0, 100.0, 25.0)

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

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_data)[0][1]
        else:
            probability = None

        st.subheader("Prediction Result")

        if probability is not None:
            st.metric("Default Probability", f"{probability:.2%}")

        if prediction == 1:
            st.error("⚠️ High Risk: Likely to default")
        else:
            st.success("✅ Low Risk: Less likely to default")

        st.dataframe(input_data)

    except Exception as e:
        st.error(f"Error: {e}")
