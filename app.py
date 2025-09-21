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
                st.balloons()
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
        **Note:** This is a predictive model and should not replace professional medical advice.
        """)

elif disease_option == "Kidney Disease":
    st.header("🫀 Kidney Disease Prediction")
    st.info("""
    🚧 Kidney disease prediction feature is under development!
    
    **Coming soon features:**
    - Chronic Kidney Disease prediction
    - Kidney function analysis
    - Risk assessment
    """)
    
    # Placeholder for future implementation
    st.warning("This feature will be available in the next update.")

elif disease_option == "Parkinson's Disease":
    st.header("🧠 Parkinson's Disease Prediction")
    st.info("""
    🚧 Parkinson's disease prediction feature is under development!
    
    **Coming soon features:**
    - Parkinson's disease detection
    - Symptom analysis
    - Progression monitoring
    """)
    
    # Placeholder for future implementation
    st.warning("This feature will be available in the next update.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**About this App:**
- Uses machine learning for disease prediction
- Trained on medical datasets
- For educational purposes only
""")

st.sidebar.warning("""
**⚠️ Important Notice:**
This application provides predictions based on machine learning models 
and should not be used as a substitute for professional medical advice, 
diagnosis, or treatment. Always consult qualified healthcare providers 
for medical concerns.
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