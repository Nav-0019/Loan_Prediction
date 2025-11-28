import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- Configuration ---
st.set_page_config(page_title="Loan Risk Predictor", layout="centered", page_icon="🏠")

# --- Load Artifacts ---
@st.cache_resource
def load_artifacts():
    try:
        with open('loan_risk_artifacts.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        return artifacts
    except FileNotFoundError:
        st.error("🚨 Critical Error: Model artifacts not found. Please run the training script first.")
        st.stop()
    except Exception as e:
        st.error(f"🚨 Critical Error: Failed to load model artifacts. {e}")
        st.stop()

artifacts = load_artifacts()
model = artifacts['model']
state_map = artifacts['state_risk_map']
city_map = artifacts['city_risk_map']
TRAINING_COLUMNS = artifacts['training_columns']
PROFESSION_GROUPS = artifacts['profession_groups']
RISK_BINS = artifacts['risk_bins']
RISK_LABELS = artifacts['risk_labels']
ALL_PROFESSION_GROUPS = artifacts['all_profession_groups']

# --- Helper Functions ---
def apply_transformations(df):
    df = df.copy()
    
    # Binary encodings
    df["Married/Single"] = df["Married/Single"].map({'single':0, 'married':1}).fillna(0).astype(int)
    df["Car_Ownership"] = df["Car_Ownership"].map({'no':0, 'yes':1}).fillna(0).astype(int)

    # House OHE
    House_map = {'rented':1, 'owned':2, 'norent_noown':0}
    df['House_Ownership'] = df['House_Ownership'].map(House_map).fillna(0)
    df['House_Ownership_owned'] = (df['House_Ownership'] == 2).astype(int)
    df['House_Ownership_rented'] = (df['House_Ownership'] == 1).astype(int)

    # Profession group + OHE
    df['Profession_Group'] = df['Profession'].map(PROFESSION_GROUPS).fillna(ALL_PROFESSION_GROUPS[0])
    Prof_Encoder = pd.get_dummies(df['Profession_Group'], prefix='Profession')
    df = pd.concat([df, Prof_Encoder], axis=1)

    # State/City risk scores (Use loaded maps)
    # Handle missing keys with mean or default
    mean_state_risk = state_map.mean() if not state_map.empty else 0.125
    mean_city_risk = city_map.mean() if not city_map.empty else 0.125
    
    df['State_Risk_Score'] = df['STATE'].map(state_map).fillna(mean_state_risk)
    df['City_Risk_Score'] = df['CITY'].map(city_map).fillna(mean_city_risk)

    # Risk zones
    df['State_Risk_Zone'] = pd.cut(df['State_Risk_Score'], bins=RISK_BINS, labels=RISK_LABELS, right=True, include_lowest=True)
    df['City_Risk_Zone'] = pd.cut(df['City_Risk_Score'], bins=RISK_BINS, labels=RISK_LABELS, right=True, include_lowest=True)
    
    Encoder_State = pd.get_dummies(df['State_Risk_Zone'], prefix='Zone_Risk_State', drop_first=True)
    Encoder_City = pd.get_dummies(df['City_Risk_Zone'], prefix='Zone_Risk_City', drop_first=True)
    df = pd.concat([df, Encoder_State, Encoder_City], axis=1)

    # Drop unnecessary columns
    cols_to_drop = ['CITY', 'STATE', 'Profession', 'Profession_Group', 'Profession_Analyst_Consulting',
                    'State_Risk_Score', 'City_Risk_Score', 'State_Risk_Zone', 'City_Risk_Zone',
                    'House_Ownership', 'Id']
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            
    return df

# --- UI ---
st.title("Real-Time Loan Risk Prediction")
st.markdown("---")
st.markdown(
    """
        ### 🌟 Loan Risk Prediction Dashboard
        This application uses a **pre-trained Machine Learning model** to provide instant risk analysis.
        
        ---
        🔍 **How to Use:**
        Fill in the applicant details below and click **'Predict Loan Risk'**.
    """
)
st.markdown("---")
st.markdown("### Applicant Details")

col1, col2 = st.columns(2)
with col1:
    income = st.number_input("Annual Income", min_value=100000, value=3000000, step=100000)
    age = st.number_input("Age", min_value=18, max_value=85, value=35)
    experience_years = st.number_input("Total Work Experience (Years)", min_value=0, max_value=60, value=10)
    current_job_years = st.number_input("Years in Current Job", min_value=0, max_value=15, value=5)
    current_house_years = st.number_input("Years in Current House", min_value=0, max_value=20, value=12)

with col2:
    married_single = st.selectbox("Marital Status", ["single", "married"])
    house_ownership_input = st.selectbox("House Ownership", ["rented", "owned", "norent_noown"])
    car_ownership = st.selectbox("Car Ownership", ["no", "yes"])
    profession_input = st.selectbox("Profession", list(PROFESSION_GROUPS.keys()))
    state_input = st.selectbox("State of Residence", sorted(state_map.index.tolist()))
    city_input = st.selectbox("City of Residence", sorted(city_map.index.tolist()))

if st.button("Predict Loan Risk", type="primary"):
    try:
        # Input Data
        input_data_raw = pd.DataFrame({
            'Income': [income],
            'Age': [age],
            'Experience': [experience_years],
            'Married/Single': [married_single],
            'House_Ownership': [house_ownership_input],
            'Car_Ownership': [car_ownership],
            'Profession': [profession_input],
            'CITY': [city_input],
            'STATE': [state_input],
            'CURRENT_JOB_YRS': [current_job_years],
            'CURRENT_HOUSE_YRS': [current_house_years],
        })

        # Preprocessing
        input_data_processed = apply_transformations(input_data_raw)
        
        # Align columns with training data
        input_data_final = input_data_processed.reindex(columns=TRAINING_COLUMNS, fill_value=0)

        # Inference
        prediction_proba = model.predict_proba(input_data_final)[0]
        risk_prediction = model.predict(input_data_final)[0]

        # Result Logic
        is_high_risk = (risk_prediction == 1)
        risk_label = "HIGH RISK" if is_high_risk else "LOW RISK"
        
        # Styling
        risk_color = "#d32f2f" if is_high_risk else "#2e7d32"
        bg_color = "#ffebee" if is_high_risk else "#e8f5e9"

        st.markdown("---")
        st.markdown("### 📊 Prediction Analysis")

        st.markdown(f"""
        <div style="
            background-color: {bg_color};
            border: 2px solid {risk_color};
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            margin-top: 10px;
            margin-bottom: 20px;
        ">
            <h2 style="color: {risk_color}; margin: 0;">{risk_label}</h2>
            <p style="font-size: 18px; margin-top: 5px; color: #333;">
                Based on the provided details, this applicant is considered 
                <strong>{'likely to default' if is_high_risk else 'safe to approve'}</strong>.
            </p>
            <hr style="border-top: 1px solid {risk_color}; opacity: 0.3; margin: 15px 0;">
            <div style="display: flex; justify-content: space-around; color: #333;">
                <div>
                    <span style="font-size: 14px; color: #555;">Probability of Default</span><br>
                    <strong style="font-size: 20px; color: #d32f2f;">{prediction_proba[1]:.2%}</strong>
                </div>
                <div>
                    <span style="font-size: 14px; color: #555;">Probability of Safety</span><br>
                    <strong style="font-size: 20px; color: #2e7d32;">{prediction_proba[0]:.2%}</strong>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error("⚠️ An unexpected error occurred while processing your request.")
        st.markdown("**Please verify your inputs and try again.**")
        with st.expander("View Technical Error Details (For Developers)"):
            st.code(f"{type(e).__name__}: {e}")
