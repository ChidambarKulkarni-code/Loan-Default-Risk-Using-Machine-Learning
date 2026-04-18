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


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def calculate_emi(principal, annual_rate, months):
    """Standard reducing-balance EMI formula."""
    if months == 0:
        return 0.0
    monthly_rate = annual_rate / 12 / 100
    if monthly_rate == 0:
        return principal / months
    emi = (principal * monthly_rate * (1 + monthly_rate) ** months
           / ((1 + monthly_rate) ** months - 1))
    return emi


def calculate_foir(emi, income):
    """
    Fixed Obligation to Income Ratio — the share of gross monthly income
    consumed by the proposed EMI.  A FOIR > 50% is a standard red flag in
    Indian retail lending.
    """
    if income <= 0:
        return 1.0          # treat zero income as 100% obligation
    monthly_income = income / 12
    return emi / monthly_income


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

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
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

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


# ─────────────────────────────────────────────────────────────────────────────
# BUSINESS RULES ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def apply_business_rules(
    base_probability,
    age, income, loan_amount, credit_score,
    months_employed, num_credit_lines, interest_rate,
    loan_term, dti_ratio, employment_type,
    has_cosigner, has_mortgage, has_dependents,
    loan_purpose, education, emi
):
    """
    Layer underwriting-style adjustments on top of the ML probability.

    Design principle: each rule targets a distinct risk dimension so that
    adjustments are additive in meaning, not redundant.  Positive
    adjustments increase default probability; negative ones reduce it.
    """
    prob = base_probability
    flags = []       # data-quality / consistency alerts (shown separately)
    reasons = []     # risk narrative shown to the underwriter

    # ── 1. Employment Type & Stability ────────────────────────────────────────
    if employment_type == "Unemployed":
        # Floor the probability — an unemployed borrower always carries
        # above-average risk regardless of what the ML model says.
        prob = max(prob, 0.50)
        reasons.append("Unemployed applicant — repayment source is uncertain")

        if income > 0:
            # Income while unemployed could be passive (rental, dividends) —
            # flag for verification but do NOT add a risk penalty here,
            # because passive income is a legitimate repayment source.
            flags.append(
                "Applicant is marked Unemployed but has declared income. "
                "Verify source (passive income, rental, investments, etc.) "
                "before processing further."
            )

        if months_employed > 0:
            # This is a data inconsistency — penalise it as a data-quality risk.
            flags.append(
                "Months Employed > 0 while Employment Type is Unemployed — "
                "data inconsistency detected. Manual review required."
            )
            prob += 0.04

        if has_cosigner == "No":
            prob += 0.07
            reasons.append("No co-signer — unemployed borrower has no secondary repayment support")

    elif employment_type == "Part-time":
        if months_employed < 12:
            prob += 0.05
            reasons.append("Part-time employment with less than 12 months of tenure")
        if income < 20000:
            prob += 0.04
            reasons.append("Low income from part-time employment limits repayment capacity")

    elif employment_type == "Self-employed":
        if months_employed < 24:
            prob += 0.05
            reasons.append(
                "Self-employed with less than 24 months of business tenure "
                "— income regularity is uncertain"
            )
        else:
            prob -= 0.02
            reasons.append("Established self-employment (≥ 24 months) — income relatively stable")

    elif employment_type == "Full-time":
        if months_employed >= 24:
            prob -= 0.03
            reasons.append("Stable full-time employment (≥ 24 months) — reduced income risk")
        elif months_employed < 6:
            prob += 0.04
            reasons.append("New full-time employee (< 6 months) — employment stability unproven")

    # ── 2. Credit Score ───────────────────────────────────────────────────────
    if credit_score < 580:
        prob += 0.14
        reasons.append("Very low credit score (< 580) — significant prior credit stress indicated")
    elif credit_score < 650:
        prob += 0.07
        reasons.append("Below-average credit score (580–649)")
    elif credit_score >= 750:
        prob -= 0.06
        reasons.append("Excellent credit score (≥ 750) — strong repayment track record")
    elif credit_score >= 700:
        prob -= 0.03
        reasons.append("Good credit score (700–749)")

    # ── 3. Debt-to-Income (DTI) Ratio ─────────────────────────────────────────
    # DTI measures existing obligations relative to income; it is distinct
    # from FOIR (which measures the proposed EMI's impact on income).
    if dti_ratio >= 0.55:
        prob += 0.14
        reasons.append("Severely high DTI (≥ 55%) — borrower is heavily over-leveraged")
    elif dti_ratio >= 0.40:
        prob += 0.08
        reasons.append("High DTI (40–54%) — existing debt burden is elevated")
    elif dti_ratio >= 0.30:
        prob += 0.04
        reasons.append("Moderate DTI (30–39%) — manageable but worth monitoring")
    elif dti_ratio < 0.20:
        prob -= 0.03
        reasons.append("Low DTI (< 20%) — borrower carries limited existing obligations")

    # ── 4. EMI Affordability — FOIR ───────────────────────────────────────────
    # FOIR is the primary repayment-capacity metric used by Indian lenders.
    # Standard threshold: FOIR > 50% is a rejection trigger; 40–50% warrants
    # caution.  This is distinct from DTI, which captures existing debt load.
    if income > 0:
        foir = calculate_foir(emi, income)
        if foir > 0.60:
            prob += 0.13
            reasons.append(
                f"EMI-to-Income ratio (FOIR) is {foir:.0%} — "
                "repayment will consume most of monthly income"
            )
        elif foir > 0.50:
            prob += 0.08
            reasons.append(
                f"FOIR is {foir:.0%} — above the standard 50% affordability threshold"
            )
        elif foir > 0.40:
            prob += 0.04
            reasons.append(
                f"FOIR is {foir:.0%} — repayment may strain monthly cash flow"
            )
        elif foir <= 0.25:
            prob -= 0.03
            reasons.append(f"FOIR is {foir:.0%} — EMI is comfortably within income capacity")
    else:
        prob += 0.15
        reasons.append("No declared income — repayment capacity cannot be assessed")

    # ── 5. Loan Amount vs Annual Income ───────────────────────────────────────
    # Compare loan principal to annual income (not monthly) for a consistent
    # leverage ratio.  A ratio > 5× annual income is typically a red flag.
    if income > 0:
        leverage = loan_amount / income      # loan ÷ annual income
        if leverage > 10:
            prob += 0.10
            reasons.append(
                f"Loan is {leverage:.1f}× annual income — extremely high leverage"
            )
        elif leverage > 5:
            prob += 0.05
            reasons.append(
                f"Loan is {leverage:.1f}× annual income — high leverage"
            )

    # ── 6. Number of Credit Lines ─────────────────────────────────────────────
    if num_credit_lines > 10:
        prob += 0.06
        reasons.append(
            "High number of open credit lines (> 10) — signs of credit dependency"
        )
    elif num_credit_lines == 0:
        prob += 0.03
        reasons.append("No credit lines — thin credit file, higher uncertainty")

    # ── 7. Interest Rate Burden ───────────────────────────────────────────────
    # High rates increase the EMI, which is already captured by FOIR.
    # Here we flag the rate itself as a signal of the borrower's risk pricing.
    if interest_rate > 20:
        prob += 0.07
        reasons.append(
            "Very high interest rate (> 20%) — suggests lender has already priced "
            "significant borrower risk; repayment pressure is elevated"
        )
    elif interest_rate > 15:
        prob += 0.03
        reasons.append("Above-average interest rate (> 15%) adds to repayment pressure")

    # ── 8. Loan Purpose ───────────────────────────────────────────────────────
    if loan_purpose == "Business":
        prob += 0.05
        reasons.append(
            "Business loans carry higher uncertainty due to venture/market risk"
        )
    elif loan_purpose in ("Auto", "Home"):
        prob -= 0.03
        reasons.append(
            f"{loan_purpose} loan is typically collateral-backed — "
            "reduces loss-given-default for the lender"
        )
    # Education / Other: neutral — no adjustment

    # ── 9. Borrower Age ───────────────────────────────────────────────────────
    if age < 25:
        prob += 0.04
        reasons.append(
            "Young borrower (< 25) — limited credit history and income stability"
        )
    elif age > 60:
        loan_end_age = age + (loan_term // 12)
        if loan_end_age > 70:
            prob += 0.04
            reasons.append(
                f"Borrower will be ~{loan_end_age} at loan maturity — "
                "income continuity beyond retirement age should be verified"
            )

    # ── 10. Dependents with Limited Income ────────────────────────────────────
    if has_dependents == "Yes" and income < 40000:
        prob += 0.05
        reasons.append(
            "Dependents with annual income < ₹40,000 — household obligations "
            "may reduce discretionary repayment capacity"
        )

    # ── 11. Existing Mortgage ─────────────────────────────────────────────────
    # A mortgage represents a large fixed monthly obligation, increasing
    # total debt burden — distinct from DTI because DTI is a self-reported
    # ratio whereas mortgage is a verified hard obligation.
    if has_mortgage == "Yes":
        prob += 0.03
        reasons.append(
            "Active mortgage adds a large fixed monthly obligation to existing debt burden"
        )

    # ── 12. Co-Signer ─────────────────────────────────────────────────────────
    # A co-signer provides a secondary repayment guarantee — this materially
    # reduces the lender's effective credit exposure.
    if has_cosigner == "Yes":
        prob -= 0.07
        reasons.append(
            "Co-signer provides secondary repayment assurance — reduces lender's effective risk"
        )

    # ── 13. Education + Employment Interaction ────────────────────────────────
    # Higher education with stable full-time employment is a mild positive
    # signal for long-term income stability and financial literacy.
    if education in ("Master's", "PhD") and employment_type == "Full-time":
        prob -= 0.02
        reasons.append(
            "Advanced education with stable employment — mild positive signal "
            "for income sustainability"
        )

    prob = max(0.0, min(prob, 0.99))
    return prob, flags, reasons


# ─────────────────────────────────────────────────────────────────────────────
# APP INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────

try:
    model = train_model()
except Exception as e:
    st.error(f"Could not train model from Loan_default.csv: {e}")
    st.stop()

st.title("💳 Loan Default Risk Prediction App")
st.caption(
    "Machine learning probability + business-rule underwriting for "
    "structured loan default assessment"
)

st.sidebar.header("About the App")
st.sidebar.info(
    "Predicts loan default risk using a Random Forest model trained on "
    "historical default data, enhanced with an underwriting rule engine "
    "that mirrors standard retail lending practices."
)
st.sidebar.subheader("What it covers")
st.sidebar.write(
    "- ML-based default probability\n"
    "- FOIR (EMI affordability) check\n"
    "- Employment & income consistency flags\n"
    "- Credit score and leverage analysis\n"
    "- Loan purpose and collateral adjustment\n"
    "- Approval / Manual Review / Reject decision\n"
    "- Key risk driver explanation\n"
    "- Estimated EMI"
)

# ─────────────────────────────────────────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────────────────────────────────────────

with st.form("prediction_form"):
    st.subheader("Borrower Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        income = st.number_input(
            "Annual Income (₹)", min_value=0.0, max_value=10_000_000.0,
            value=50_000.0, step=1_000.0
        )
        credit_score = st.number_input(
            "Credit Score", min_value=300, max_value=900, value=650, step=1
        )
        employment_type = st.selectbox(
            "Employment Type",
            ["Full-time", "Part-time", "Self-employed", "Unemployed"]
        )
        months_employed = st.number_input(
            "Months Employed", min_value=0, max_value=600, value=24, step=1
        )

    with col2:
        loan_amount = st.number_input(
            "Loan Amount (₹)", min_value=1_000.0, max_value=5_000_000.0,
            value=10_000.0, step=1_000.0
        )
        interest_rate = st.number_input(
            "Interest Rate (% p.a.)", min_value=0.0, max_value=100.0,
            value=12.5, step=0.1
        )
        loan_term = st.selectbox(
            "Loan Term (Months)", [12, 24, 36, 48, 60], index=1
        )
        dti_ratio = st.number_input(
            "DTI Ratio (existing debt ÷ income)",
            min_value=0.0, max_value=1.0, value=0.25, step=0.01,
            help="Debt-to-Income ratio based on borrower's existing obligations, excluding this loan."
        )
        num_credit_lines = st.number_input(
            "Number of Open Credit Lines", min_value=0, max_value=50, value=3, step=1
        )

    with col3:
        education = st.selectbox(
            "Education", ["High School", "Bachelor's", "Master's", "PhD"]
        )
        marital_status = st.selectbox(
            "Marital Status", ["Single", "Married", "Divorced"]
        )
        has_mortgage = st.selectbox("Has Active Mortgage", ["No", "Yes"])
        has_dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        loan_purpose = st.selectbox(
            "Loan Purpose", ["Auto", "Business", "Education", "Home", "Other"]
        )
        has_cosigner = st.selectbox("Has Co-Signer", ["No", "Yes"])

    submitted = st.form_submit_button("Predict Risk")

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION & OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

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
        # ── ML Prediction ─────────────────────────────────────────────────────
        class_labels = list(model.named_steps["classifier"].classes_)
        default_index = class_labels.index(1) if 1 in class_labels else 1

        proba = model.predict_proba(input_data)[0]
        base_probability = float(proba[default_index])

        # ML binary outcome (for display only — business rules determine final decision)
        ml_binary = "Default" if base_probability >= 0.50 else "No Default"

        # ── EMI & FOIR ────────────────────────────────────────────────────────
        emi = calculate_emi(loan_amount, interest_rate, loan_term)
        foir = calculate_foir(emi, income)

        # ── Business Rules ────────────────────────────────────────────────────
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
            has_dependents=has_dependents,
            loan_purpose=loan_purpose,
            education=education,
            emi=emi
        )

        # ── Decision Logic ────────────────────────────────────────────────────
        # Thresholds are based on the adjusted (rules-layered) probability.
        if adjusted_probability >= 0.65:
            risk_label = "High Risk"
            decision = "Reject"
        elif adjusted_probability >= 0.40:
            risk_label = "Moderate Risk"
            decision = "Manual Review"
        else:
            risk_label = "Low Risk"
            decision = "Approve"

        # Hard override: unemployed applicants must never be auto-approved —
        # a human underwriter must verify the income source before disbursement.
        if employment_type == "Unemployed" and decision == "Approve":
            decision = "Manual Review"
            risk_label = "Moderate Risk"
            adjusted_probability = max(adjusted_probability, 0.40)
            reasons.append(
                "Hard rule: unemployed applicant routed to manual review "
                "regardless of ML score — income source must be verified"
            )

        # Hard override: FOIR > 60% with no co-signer is a rejection trigger.
        if income > 0 and foir > 0.60 and has_cosigner == "No" and decision == "Approve":
            decision = "Manual Review"
            risk_label = "Moderate Risk"
            adjusted_probability = max(adjusted_probability, 0.40)
            reasons.append(
                "Hard rule: FOIR > 60% with no co-signer — auto-approval "
                "suppressed; affordability review required"
            )

        # ── Output ────────────────────────────────────────────────────────────
        st.subheader("Underwriting Result")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("ML Probability", f"{base_probability:.2%}",
                  help="Raw default probability from the Random Forest model")
        m2.metric("ML Signal", ml_binary,
                  help="Binary ML prediction before business-rule adjustment")
        m3.metric("Adjusted Risk", f"{adjusted_probability:.2%}",
                  help="Probability after applying underwriting rules")
        m4.metric("Risk Level", risk_label)
        m5.metric("Decision", decision)

        col_emi, col_foir = st.columns(2)
        col_emi.metric("Estimated Monthly EMI", f"₹{emi:,.2f}")
        col_foir.metric(
            "FOIR (EMI ÷ Monthly Income)",
            f"{foir:.1%}" if income > 0 else "N/A",
            help="Fixed Obligation to Income Ratio. Standard safe limit: ≤ 50%."
        )

        if risk_label == "High Risk":
            st.error("🔴 High risk of default. Rejection recommended.")
        elif risk_label == "Moderate Risk":
            st.warning("🟠 Moderate risk. Manual underwriting review required.")
        else:
            st.success("🟢 Low risk based on current profile. Approval recommended.")

        if flags:
            st.subheader("⚠️ Data Consistency Flags")
            st.caption(
                "These are data-quality alerts that require manual verification "
                "before a final decision is issued."
            )
            for flag in flags:
                st.warning(flag)

        if reasons:
            st.subheader("📋 Key Risk Drivers")
            st.caption(
                "Factors that influenced the final risk assessment, "
                "ordered by rule sequence."
            )
            for reason in dict.fromkeys(reasons):   # preserves order, deduplicates
                st.write(f"• {reason}")

        with st.expander("View Input Summary"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
