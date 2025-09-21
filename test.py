# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Set page config
st.set_page_config(
    page_title="Multiple Disease Prediction",
    page_icon="🏥",
    layout="wide"
)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Sidebar navigation
st.sidebar.title("🏥 Disease Prediction System")
disease_option = st.sidebar.selectbox(
    "Select Disease for Prediction",
    ["Indian Liver Patient", "Kidney Disease", "Parkinson's Disease"]
)

# Main content
st.title("🤖 Multiple Disease Prediction System")
st.write("Predict various diseases using machine learning models")

# ----------------------------- LIVER -----------------------------
if disease_option == "Indian Liver Patient":
    st.header("🧬 Indian Liver Patient Prediction")
    
    model_path = 'models/liver_model.pkl'
    scaler_path = 'models/liver_scaler.pkl'
    encoder_path = 'models/liver_encoder.pkl'
    
    if not all(os.path.exists(path) for path in [model_path, scaler_path, encoder_path]):
        st.warning("⚠️ Liver model not found! Run train_liver_model.py first.")
        st.stop()
    
    with open(model_path, 'rb') as f:
        liver_model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        liver_scaler = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        liver_encoder = pickle.load(f)
    
    with st.form("liver_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 0, 100, 30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            total_bilirubin = st.number_input("Total Bilirubin", 0.0, 20.0, 0.7)
            direct_bilirubin = st.number_input("Direct Bilirubin", 0.0, 10.0, 0.1)
        with col2:
            alkaline_phosphotase = st.number_input("Alkaline Phosphotase", 0, 500, 187)
            alamine_aminotransferase = st.number_input("Alamine Aminotransferase (ALT)", 0, 500, 16)
            aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase (AST)", 0, 500, 18)
            total_proteins = st.number_input("Total Proteins", 0.0, 20.0, 6.8)
            albumin = st.number_input("Albumin", 0.0, 10.0, 3.3)
            albumin_globulin_ratio = st.number_input("Albumin and Globulin Ratio", 0.0, 5.0, 0.9)
        submit_button = st.form_submit_button("Predict Liver Condition")
    
    if submit_button:
        input_data = pd.DataFrame({
            'Age':[age],
            'Gender':[1 if gender=="Male" else 0],
            'Total_Bilirubin':[total_bilirubin],
            'Direct_Bilirubin':[direct_bilirubin],
            'Alkaline_Phosphotase':[alkaline_phosphotase],
            'Alamine_Aminotransferase':[alamine_aminotransferase],
            'Aspartate_Aminotransferase':[aspartate_aminotransferase],
            'Total_Protiens':[total_proteins],
            'Albumin':[albumin],
            'Albumin_and_Globulin_Ratio':[albumin_globulin_ratio]
        })
        input_scaled = liver_scaler.transform(input_data)
        prediction = liver_model.predict(input_scaled)
        prediction_proba = liver_model.predict_proba(input_scaled)
        st.subheader("📊 Prediction Results")
        if prediction[0]==0:
            st.success("✅ No Liver Disease Detected")
        else:
            st.error("⚠️ Liver Disease Detected")
        st.metric("Confidence Level", f"{max(prediction_proba[0]):.2%}")

# ----------------------------- KIDNEY -----------------------------
elif disease_option == "Kidney Disease":
    st.header("🫀 Kidney Disease Prediction")
    
    model_path = 'models/kidney_model.pkl'
    scaler_path = 'models/kidney_scaler.pkl'
    encoder_path = 'models/kidney_encoders.pkl'
    
    if not all(os.path.exists(path) for path in [model_path, scaler_path, encoder_path]):
        st.warning("⚠️ Kidney model not found! Run train_kidney_model.py first.")
        st.stop()
    
    with open(model_path, 'rb') as f:
        kidney_model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        kidney_scaler = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        kidney_encoders = pickle.load(f)
    
    # Input form for kidney dataset (all important features)
    with st.form("kidney_form"):
        st.subheader("Patient Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 0, 100, 30)
            bp = st.number_input("Blood Pressure", 0, 200, 80)
            sg = st.number_input("Specific Gravity", 1.0, 2.0, 1.02)
            al = st.number_input("Albumin", 0, 5, 1)
            su = st.number_input("Sugar", 0, 5, 0)
            rbc = st.selectbox("Red Blood Cells", ["normal","abnormal"])
            pc = st.selectbox("Pus Cell", ["normal","abnormal"])
        with col2:
            pcc = st.selectbox("Pus Cell Clumps", ["present","notpresent"])
            ba = st.selectbox("Bacteria", ["present","notpresent"])
            bgr = st.number_input("Blood Glucose Random", 0, 500, 121)
            bu = st.number_input("Blood Urea", 0, 200, 36)
            sc = st.number_input("Serum Creatinine", 0.0, 20.0, 1.2)
            sod = st.number_input("Sodium", 0, 200, 135)
            pot = st.number_input("Potassium", 0, 20, 4.5)
        with col3:
            hemo = st.number_input("Hemoglobin", 0.0, 30.0, 15.0)
            pcv = st.number_input("Packed Cell Volume", 0, 60, 44)
            wc = st.number_input("White Blood Cell Count", 0, 20000, 8000)
            rc = st.number_input("Red Blood Cell Count", 0.0, 10.0, 5.0)
            htn = st.selectbox("Hypertension", ["yes","no"])
            dm = st.selectbox("Diabetes Mellitus", ["yes","no"])
            cad = st.selectbox("Coronary Artery Disease", ["yes","no"])
            appet = st.selectbox("Appetite", ["good","poor"])
            pe = st.selectbox("Pedal Edema", ["yes","no"])
            ane = st.selectbox("Anemia", ["yes","no"])
        submit_button = st.form_submit_button("Predict Kidney Condition")
    
    if submit_button:
        input_data = pd.DataFrame({
            'age':[age],'bp':[bp],'sg':[sg],'al':[al],'su':[su],
            'rbc':[rbc],'pc':[pc],'pcc':[pcc],'ba':[ba],'bgr':[bgr],
            'bu':[bu],'sc':[sc],'sod':[sod],'pot':[pot],'hemo':[hemo],
            'pcv':[pcv],'wc':[wc],'rc':[rc],'htn':[htn],'dm':[dm],
            'cad':[cad],'appet':[appet],'pe':[pe],'ane':[ane]
        })
        
        # Encode categorical features
        for col, le in kidney_encoders.items():
            if col in input_data.columns:
                input_data[col] = le.transform(input_data[col])
        
        input_scaled = kidney_scaler.transform(input_data)
        prediction = kidney_model.predict(input_scaled)
        prediction_proba = kidney_model.predict_proba(input_scaled)
        
        st.subheader("📊 Prediction Results")
        if prediction[0]==0:
            st.success("✅ No Kidney Disease Detected")
        else:
            st.error("⚠️ Kidney Disease Detected")
        st.metric("Confidence Level", f"{max(prediction_proba[0]):.2%}")

# ----------------------------- PARKINSON -----------------------------
elif disease_option == "Parkinson's Disease":
    st.header("🧠 Parkinson's Disease Prediction")
    st.info("🚧 Feature under development!")

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.info("App uses ML models for disease prediction (educational purposes only).")
