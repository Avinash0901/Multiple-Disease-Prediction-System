# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

if disease_option == "Indian Liver Patient":
    st.header("🧬 Indian Liver Patient Prediction")
    
    # Check if model exists
    model_path = 'models/liver_model.pkl'
    scaler_path = 'models/liver_scaler.pkl'
    encoder_path = 'models/liver_encoder.pkl'
    
    if not all(os.path.exists(path) for path in [model_path, scaler_path, encoder_path]):
        st.warning("""
        ⚠️ Model not found! Please train the model first by running:
        ```bash
        python train_liver_model.py
        ```
        """)
        st.stop()
    
    # Load the models
    try:
        with open(model_path, 'rb') as f:
            liver_model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            liver_scaler = pickle.load(f)
        with open(encoder_path, 'rb') as f:
            liver_encoder = pickle.load(f)
        
        st.success("✅ Model loaded successfully!")
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()
    
    # Create input form
    with st.form("liver_prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Patient Information")
            age = st.number_input("Age", min_value=0, max_value=100, value=30, help="Patient's age in years")
            gender = st.selectbox("Gender", ["Male", "Female"])
            
            st.subheader("Bilirubin Levels")
            total_bilirubin = st.number_input("Total Bilirubin", min_value=0.0, value=0.7, step=0.1, 
                                            help="Normal range: 0.1-1.2 mg/dL")
            direct_bilirubin = st.number_input("Direct Bilirubin", min_value=0.0, value=0.1, step=0.1,
                                             help="Normal range: 0.0-0.3 mg/dL")
        
        with col2:
            st.subheader("Liver Enzymes")
            alkaline_phosphotase = st.number_input("Alkaline Phosphotase", min_value=0, value=187,
                                                 help="Normal range: 44-147 IU/L")
            alamine_aminotransferase = st.number_input("Alamine Aminotransferase (ALT)", min_value=0, value=16,
                                                     help="Normal range: 7-56 IU/L")
            aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase (AST)", min_value=0, value=18,
                                                       help="Normal range: 10-40 IU/L")
            
            st.subheader("Protein Levels")
            total_proteins = st.number_input("Total Proteins", min_value=0.0, value=6.8, step=0.1,
                                           help="Normal range: 6.0-8.3 g/dL")
            albumin = st.number_input("Albumin", min_value=0.0, value=3.3, step=0.1,
                                    help="Normal range: 3.5-5.0 g/dL")
            albumin_globulin_ratio = st.number_input("Albumin and Globulin Ratio", min_value=0.0, value=0.9, step=0.1,
                                                   help="Normal range: 1.0-2.0")
        
        submit_button = st.form_submit_button("🔍 Predict Liver Condition")
    
    if submit_button:
        # Prepare input data
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [1 if gender == "Male" else 0],
            'Total_Bilirubin': [total_bilirubin],
            'Direct_Bilirubin': [direct_bilirubin],
            'Alkaline_Phosphotase': [alkaline_phosphotase],
            'Alamine_Aminotransferase': [alamine_aminotransferase],
            'Aspartate_Aminotransferase': [aspartate_aminotransferase],
            'Total_Protiens': [total_proteins],
            'Albumin': [albumin],
            'Albumin_and_Globulin_Ratio': [albumin_globulin_ratio]
        })
        
        # Scale the input data
        input_scaled = liver_scaler.transform(input_data)
        
        # Make prediction
        prediction = liver_model.predict(input_scaled)
        prediction_proba = liver_model.predict_proba(input_scaled)
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            if prediction[0] == 0:
                st.success("✅ **No Liver Disease Detected**")
            else:
                st.error("⚠️ **Liver Disease Detected**")
                st.warning("Please consult a healthcare professional for proper diagnosis and treatment.")
            
            confidence = max(prediction_proba[0])
            st.metric("Confidence Level", f"{confidence:.2%}")
        
        with result_col2:
            # Show probability distribution
            prob_data = {
                'Condition': ['No Disease', 'Disease'],
                'Probability': [prediction_proba[0][0], prediction_proba[0][1]]
            }
            prob_df = pd.DataFrame(prob_data)
            st.bar_chart(prob_df.set_index('Condition'))
        
        # Detailed probabilities
        st.subheader("📈 Detailed Probabilities")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Probability of No Disease", f"{prediction_proba[0][0]:.2%}")
        with col2:
            st.metric("Probability of Disease", f"{prediction_proba[0][1]:.2%}")
        
        # Model information
        st.markdown("---")
        st.subheader("ℹ️ Model Information")
        st.info("""
        **Model Type:** Random Forest Classifier  
        **Training Accuracy:** 72.65%  
        **Features Used:** Age, Gender, Bilirubin levels, Liver enzymes, Protein levels  
        """)

elif disease_option == "Kidney Disease":
    st.header("🫀 Kidney Disease Prediction")
    
    # Model paths
    model_path = 'models/kidney_model.pkl'
    scaler_path = 'models/kidney_scaler.pkl'
    encoders_path = 'models/kidney_encoders.pkl'
    
    # Check if files exist
    if not all(os.path.exists(path) for path in [model_path, scaler_path, encoders_path]):
        st.warning("""⚠️ Kidney model not found! Please train the model first by running:
        ```bash
        python train_kidney_model.py
        ```""")
        st.stop()
    
    # Load model, scaler, encoders
    try:
        with open(model_path, 'rb') as f:
            kidney_model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            kidney_scaler = pickle.load(f)
        with open(encoders_path, 'rb') as f:
            kidney_encoders = pickle.load(f)
        st.success("✅ Kidney model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading kidney model: {e}")
        st.stop()
    
    # Input form
    with st.form("kidney_prediction_form"):
        col1, col2 = st.columns(2)
        
        numeric_cols = ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']
        categorical_cols = ['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane']
        
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            bp = st.number_input("Blood Pressure (bp)", min_value=50, max_value=200, value=80)
            sg = st.number_input("Specific Gravity (sg)", min_value=1.0, max_value=1.030, value=1.02, step=0.001)
            al = st.number_input("Albumin (al)", min_value=0, max_value=5, value=1)
            su = st.number_input("Sugar (su)", min_value=0, max_value=5, value=0)
            rbc = st.selectbox("Red Blood Cells (rbc)", ['normal', 'abnormal'])
            pc = st.selectbox("Pus Cell (pc)", ['normal', 'abnormal'])
            pcc = st.selectbox("Pus Cell Clumps (pcc)", ['present', 'notpresent'])
            ba = st.selectbox("Bacteria (ba)", ['present', 'notpresent'])
            bgr = st.number_input("Blood Glucose Random (bgr)", min_value=0, max_value=500, value=120)
            bu = st.number_input("Blood Urea (bu)", min_value=0, max_value=500, value=40)
        
        with col2:
            sc = st.number_input("Serum Creatinine (sc)", min_value=0.0, max_value=20.0, value=1.2)
            sod = st.number_input("Sodium (sod)", min_value=100, max_value=200, value=135)
            pot = st.number_input("Potassium (pot)", min_value=2.0, max_value=10.0, value=4.5)
            hemo = st.number_input("Hemoglobin (hemo)", min_value=0.0, max_value=20.0, value=14.0)
            pcv = st.number_input("Packed Cell Volume (pcv)", min_value=10, max_value=60, value=40)
            wc = st.number_input("White Blood Cell Count (wc)", min_value=2000, max_value=20000, value=8000)
            rc = st.number_input("Red Blood Cell Count (rc)", min_value=2.0, max_value=8.0, value=4.5)
            htn = st.selectbox("Hypertension (htn)", ['yes', 'no'])
            dm = st.selectbox("Diabetes Mellitus (dm)", ['yes', 'no'])
            cad = st.selectbox("Coronary Artery Disease (cad)", ['yes', 'no'])
            appet = st.selectbox("Appetite (appet)", ['good', 'poor'])
            pe = st.selectbox("Pedal Edema (pe)", ['yes', 'no'])
            ane = st.selectbox("Anemia (ane)", ['yes', 'no'])
        
        submit_button = st.form_submit_button("🔍 Predict Kidney Disease")
    
    if submit_button:
        # Prepare input dataframe
        input_dict = {
            'age':[age],'bp':[bp],'sg':[sg],'al':[al],'su':[su],
            'rbc':[rbc],'pc':[pc],'pcc':[pcc],'ba':[ba],'bgr':[bgr],
            'bu':[bu],'sc':[sc],'sod':[sod],'pot':[pot],'hemo':[hemo],
            'pcv':[pcv],'wc':[wc],'rc':[rc],'htn':[htn],'dm':[dm],
            'cad':[cad],'appet':[appet],'pe':[pe],'ane':[ane]
        }
        input_df = pd.DataFrame(input_dict)
        
        # Convert numeric columns safely
        for col in numeric_cols:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
        
        # Encode categorical columns
        for col, le in kidney_encoders.items():
            input_df[col] = le.transform(input_df[col].astype(str))
        
        # Scale features
        input_scaled = kidney_scaler.transform(input_df)
        
        # Predict
        prediction = kidney_model.predict(input_scaled)
        prediction_proba = kidney_model.predict_proba(input_scaled)
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        if prediction[0] == 1:
            st.error("⚠️ **Kidney Disease Detected**")
            st.warning("Please consult a healthcare professional for proper diagnosis and treatment.")
        else:
            st.success("✅ **No Kidney Disease Detected**")
        
        confidence = max(prediction_proba[0])
        st.metric("Confidence Level", f"{confidence:.2%}")
        
        # Probability chart
        prob_df = pd.DataFrame({
            'Condition': ['No Disease', 'Disease'],
            'Probability': [prediction_proba[0][0], prediction_proba[0][1]]
        })
        st.bar_chart(prob_df.set_index('Condition'))

elif disease_option == "Parkinson's Disease":
    st.header("🧠 Parkinson's Disease Prediction")
    
    # Paths
    model_path = 'models/parkinsons_model.pkl'
    scaler_path = 'models/parkinsons_scaler.pkl'
    encoder_path = 'models/parkinsons_encoders.pkl'
    
    # Check if model exists
    if not all(os.path.exists(path) for path in [model_path, scaler_path, encoder_path]):
        st.warning("""
        ⚠️ Model not found! Please train the model first by running:
        ```bash
        python train_parkinsons_model.py
        ```
        """)
        st.stop()
    
    # Load the model, scaler, and encoders
    try:
        with open(model_path, 'rb') as f:
            parkinson_model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            parkinson_scaler = pickle.load(f)
        with open(encoder_path, 'rb') as f:
            parkinson_encoders = pickle.load(f)
        st.success("✅ Parkinson's model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()
    
    # Input form
    with st.form("parkinsons_form"):
        st.subheader("Enter Patient Voice Feature Values")
        
        col1, col2 = st.columns(2)
        
        with col1:
            mdvp_fo = st.number_input("MDVP:Fo(Hz)", min_value=0.0, value=119.0)
            mdvp_fhi = st.number_input("MDVP:Fhi(Hz)", min_value=0.0, value=150.0)
            mdvp_flo = st.number_input("MDVP:Flo(Hz)", min_value=0.0, value=115.0)
            mdvp_jitter = st.number_input("MDVP:Jitter(%)", min_value=0.0, value=0.005)
            mdvp_jitter_abs = st.number_input("MDVP:Jitter(Abs)", min_value=0.0, value=0.00004)
            mdvp_rap = st.number_input("MDVP:RAP", min_value=0.0, value=0.003)
            mdvp_ppq = st.number_input("MDVP:PPQ", min_value=0.0, value=0.004)
            jitter_ddp = st.number_input("Jitter:DDP", min_value=0.0, value=0.009)
            mdvp_shimmer = st.number_input("MDVP:Shimmer", min_value=0.0, value=0.03)
        
        with col2:
            mdvp_shimmer_db = st.number_input("MDVP:Shimmer(dB)", min_value=0.0, value=0.3)
            shimmer_apq3 = st.number_input("Shimmer:APQ3", min_value=0.0, value=0.02)
            shimmer_apq5 = st.number_input("Shimmer:APQ5", min_value=0.0, value=0.03)
            mdvp_apq = st.number_input("MDVP:APQ", min_value=0.0, value=0.025)
            shimmer_dda = st.number_input("Shimmer:DDA", min_value=0.0, value=0.045)
            nhr = st.number_input("NHR", min_value=0.0, value=0.02)
            hnr = st.number_input("HNR", min_value=0.0, value=20.0)
            rpde = st.number_input("RPDE", min_value=0.0, value=0.55)
            dfa = st.number_input("DFA", min_value=0.0, value=0.7)
            spread1 = st.number_input("spread1", min_value=-5.0, value=-4.0)
            spread2 = st.number_input("spread2", min_value=-1.0, value=-0.2)
            d2 = st.number_input("D2", min_value=1.0, value=2.0)
            ppe = st.number_input("PPE", min_value=0.0, value=0.1)
        
        submit_button = st.form_submit_button("🔍 Predict Parkinson's Disease")
    
    if submit_button:
        # Create input dataframe
        input_df = pd.DataFrame({
            'MDVP:Fo(Hz)': [mdvp_fo],
            'MDVP:Fhi(Hz)': [mdvp_fhi],
            'MDVP:Flo(Hz)': [mdvp_flo],
            'MDVP:Jitter(%)': [mdvp_jitter],
            'MDVP:Jitter(Abs)': [mdvp_jitter_abs],
            'MDVP:RAP': [mdvp_rap],
            'MDVP:PPQ': [mdvp_ppq],
            'Jitter:DDP': [jitter_ddp],
            'MDVP:Shimmer': [mdvp_shimmer],
            'MDVP:Shimmer(dB)': [mdvp_shimmer_db],
            'Shimmer:APQ3': [shimmer_apq3],
            'Shimmer:APQ5': [shimmer_apq5],
            'MDVP:APQ': [mdvp_apq],
            'Shimmer:DDA': [shimmer_dda],
            'NHR': [nhr],
            'HNR': [hnr],
            'RPDE': [rpde],
            'DFA': [dfa],
            'spread1': [spread1],
            'spread2': [spread2],
            'D2': [d2],
            'PPE': [ppe]
        })
        
        # Scale
        input_scaled = parkinson_scaler.transform(input_df)
        
        # Predict
        prediction = parkinson_model.predict(input_scaled)
        prediction_proba = parkinson_model.predict_proba(input_scaled)
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        result_col1, result_col2 = st.columns(2)
        with result_col1:
            if prediction[0] == 0:
                st.success("✅ **No Parkinson's Disease Detected**")
            else:
                st.error("⚠️ **Parkinson's Disease Detected**")
                st.warning("Please consult a healthcare professional for proper diagnosis and treatment.")
            confidence = max(prediction_proba[0])
            st.metric("Confidence Level", f"{confidence:.2%}")
        
        with result_col2:
            prob_data = {
                'Condition': ['No Disease', 'Disease'],
                'Probability': [prediction_proba[0][0], prediction_proba[0][1]]
            }
            prob_df = pd.DataFrame(prob_data)
            st.bar_chart(prob_df.set_index('Condition'))
        
        # Detailed probabilities
        st.subheader("📈 Detailed Probabilities")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Probability of No Disease", f"{prediction_proba[0][0]:.2%}")
        with col2:
            st.metric("Probability of Disease", f"{prediction_proba[0][1]:.2%}")
        


# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**About this App:**
- Uses machine learning for disease prediction
- Trained on medical datasets
- For educational purposes only
""")


# Add some styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .prediction-result {
        font-size: 1.5rem;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)
