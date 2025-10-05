import streamlit as st
import numpy as np
import joblib
import requests
import os
from pathlib import Path

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
MODEL_PATH = Path(__file__).parent / "models" / "rf_model_weighted.pkl"


@st.cache_resource
def load_model():
    """Load model from local path or download if missing."""
    if not MODEL_PATH.exists():
        os.makedirs(MODEL_PATH.parent, exist_ok=True)
        url = "YOUR_MODEL_DOWNLOAD_LINK"  # Replace with your real URL
        st.warning("Downloading model from remote source...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
    st.success("Model loaded successfully ✅")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- TITLE ---
st.markdown("<h1 class='title'>🚗 Automobile Loan Default Prediction</h1>",
            unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predict whether a client is likely to default on a loan using a trained ML model.</p>", unsafe_allow_html=True)
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
    mapping = {
        'Graduation': 0,
        'Graduation dropout': 1,
        'Junior secondary': 2,
        'Post Grad': 3,
        'Secondary': 4
    }
    return mapping.get(education, 0)


def encode_input():
    education_encoded = encode_client_education(Client_Education)

    # One-hot encode categorical fields in consistent order
    income_type_order = ['Commercial', 'Govt Job',
                         'Maternity leave', 'Retired', 'Service', 'Student']
    income_type_encoded = [1 if Client_Income_Type ==
                           i else 0 for i in income_type_order]

    marital_status_order = ['D', 'M', 'S', 'W']
    marital_status_encoded = [
        1 if Client_Marital_Status == i else 0 for i in marital_status_order]

    gender_order = ['Female', 'Male']
    gender_encoded = [1 if Client_Gender == g else 0 for g in gender_order]

    loan_contract_order = ['CL', 'RL']
    loan_contract_encoded = [1 if Loan_Contract_Type ==
                             l else 0 for l in loan_contract_order]

    # Combine all features
    input_features = [
        Client_Income, Car_Owned, Bike_Owned, Active_Loan, House_Own,
        Child_Count, Credit_Amount, Loan_Annuity, Workphone_Working,
        Client_Family_Members, Age_Years, Employed_Years, education_encoded
    ] + income_type_encoded + marital_status_encoded + gender_encoded + loan_contract_encoded

    return np.array(input_features).reshape(1, -1)


# --- PREDICTION SECTION ---
st.markdown("---")
st.subheader("🔍 Prediction Result")

if st.button("🚀 Predict Loan Status"):
    input_data = encode_input()

    try:
        prediction = model.predict(input_data)
        pred_class = int(prediction[0])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    if Loan_Annuity >= Client_Income:
        st.markdown(
            "<div class='result-box' style='background-color:#FADBD8; color:#C0392B;'>❌ The client is predicted to DEFAULT on the loan (High annuity vs income).</div>",
            unsafe_allow_html=True,
        )
    else:
        if pred_class == 0:
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
st.markdown(
    "<p style='text-align:center; color:gray;'>Made with ❤️ by <b>Cluster Crew (FDM_MLB_G12)</b></p>",
    unsafe_allow_html=True
)
