import streamlit as st
import numpy as np
import joblib

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="🚗 Automobile Loan Default Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .title {
            text-align: center;
            color: #2E86C1;
            font-family: 'Segoe UI', sans-serif;
            font-size: 36px;
            font-weight: bold;
        }
        .subtitle {
            text-align: center;
            color: #117A65;
            font-size: 18px;
            margin-bottom: 20px;
        }
        .stButton>button {
            background-color: #2E86C1;
            color: white;
            font-weight: 600;
            border-radius: 8px;
            padding: 10px 20px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #1B4F72;
            color: #f2f2f2;
        }
        .result-box {
            text-align: center;
            padding: 20px;
            border-radius: 10px;
            background-color: #eaf2f8;
            font-size: 20px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
model = joblib.load('models/rf_model_weighted.pkl')

# --- TITLE ---
st.markdown("<h1 class='title'>🚗 Automobile Loan Default Prediction</h1>",
            unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predict whether a client is likely to default on a loan using trained ML models.</p>", unsafe_allow_html=True)
st.markdown("---")

# --- INPUT FORM ---
st.subheader("📋 Enter Client Information")

col1, col2, col3 = st.columns(3)

with col1:
    Client_Income = st.number_input("💵 Client Income", min_value=0)
    Car_Owned = st.selectbox("🚘 Car Owned", [0, 1])
    Bike_Owned = st.selectbox("🏍️ Bike Owned", [0, 1])
    Active_Loan = st.selectbox("💳 Active Loan", [0, 1])
    House_Own = st.selectbox("🏠 House Owned", [0, 1])

with col2:
    Child_Count = st.number_input("👶 Child Count", min_value=0)
    Credit_Amount = st.number_input("💰 Credit Amount", min_value=0)
    Loan_Annuity = st.number_input("📆 Loan Annuity", min_value=0)
    Client_Family_Members = st.number_input("👪 Family Members", min_value=0)
    Age_Years = st.number_input("🎂 Age (Years)", min_value=0)

with col3:
    Employed_Years = st.number_input("💼 Employed (Years)", min_value=0)
    Workphone_Working = st.selectbox("📞 Workphone Working", [0, 1])
    Client_Gender = st.selectbox("🧍 Client Gender", ['Female', 'Male'])
    Client_Marital_Status = st.selectbox(
        "💍 Marital Status", ['D', 'M', 'S', 'W'])
    Loan_Contract_Type = st.selectbox("📜 Loan Contract Type", ['CL', 'RL'])

st.markdown("---")
st.subheader("🎓 Education & Income Details")

col4, col5 = st.columns(2)

with col4:
    Client_Education = st.selectbox("🎓 Client Education", [
        'Graduation', 'Graduation dropout', 'Junior secondary', 'Post Grad', 'Secondary'
    ])

with col5:
    Client_Income_Type = st.selectbox("💼 Client Income Type", [
        'Commercial', 'Govt Job', 'Maternity leave', 'Retired', 'Service', 'Student'
    ])

# --- ENCODING FUNCTIONS ---


def encode_client_education(education):
    education_mapping = {
        'Graduation': 0,
        'Graduation dropout': 1,
        'Junior secondary': 2,
        'Post Grad': 3,
        'Secondary': 4
    }
    return education_mapping.get(education, -1)


def encode_input():
    education_encoded = encode_client_education(Client_Education)

    income_type_encoded = [
        1 if Client_Income_Type == "Commercial" else 0,
        1 if Client_Income_Type == "Govt Job" else 0,
        1 if Client_Income_Type == "Maternity leave" else 0,
        1 if Client_Income_Type == "Retired" else 0,
        1 if Client_Income_Type == "Service" else 0,
        1 if Client_Income_Type == "Student" else 0,
    ]

    marital_status_encoded = [
        1 if Client_Marital_Status == "D" else 0,
        1 if Client_Marital_Status == "M" else 0,
        1 if Client_Marital_Status == "S" else 0,
        1 if Client_Marital_Status == "W" else 0,
    ]

    gender_encoded = [
        1 if Client_Gender == "Female" else 0,
        1 if Client_Gender == "Male" else 0,
    ]

    loan_contract_encoded = [
        1 if Loan_Contract_Type == "CL" else 0,
        1 if Loan_Contract_Type == "RL" else 0,
    ]

    input_features = [
        Client_Income, Car_Owned, Bike_Owned, Active_Loan, House_Own, Child_Count,
        Credit_Amount, Loan_Annuity, Workphone_Working, Client_Family_Members,
        Age_Years, Employed_Years, education_encoded
    ] + income_type_encoded + marital_status_encoded + gender_encoded + loan_contract_encoded

    return np.array(input_features).reshape(1, -1)


# --- PREDICTION SECTION ---
st.markdown("---")
st.subheader("🔍 Prediction Result")

if st.button("🚀 Predict Loan Status"):
    input_data = encode_input()
    prediction = model.predict(input_data)

    if Loan_Annuity > Client_Income:
        st.markdown(
            "<div class='result-box' style='background-color:#FADBD8; color:#C0392B;'>❌ The client is predicted to DEFAULT on the loan (High annuity vs income).</div>",
            unsafe_allow_html=True,
        )
    else:
        if prediction[0] == 0:
            st.markdown(
                "<div class='result-box' style='background-color:#D5F5E3; color:#117A65;'>✅ The client is predicted to NOT default on the loan.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='result-box' style='background-color:#FADBD8; color:#C0392B;'>❌ The client is predicted to DEFAULT on the loan.</div>",
                unsafe_allow_html=True,
            )

# --- FOOTER ---
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>Made with ❤️ by <b>Cluster Crew (FDM_MLB_G12)</b></p>",
            unsafe_allow_html=True)
