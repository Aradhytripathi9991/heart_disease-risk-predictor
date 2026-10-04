import streamlit as st
import joblib
import numpy as np
import time

# Set page configuration for a wider layout and custom title
st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀", layout="wide")

# Load the trained model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('heart_disease_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_model()
except Exception as e:
    st.error("Error loading model or scaler. Ensure 'heart_disease_model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop()

# --- Sidebar ---
st.sidebar.title("🫀 HeartGuard AI")
st.sidebar.info(
    "Welcome to the Heart Disease Predictor! This mini-project uses a Machine Learning "
    "(Random Forest) model to predict the likelihood of heart disease based on clinical parameters."
)
st.sidebar.markdown("---")
st.sidebar.write("**Developed for College Mini-Project**")
st.sidebar.write("Dataset: UCI Heart Disease")
st.sidebar.write("Model Accuracy: ~84%")

# --- Main Page Layout ---
st.title("Cardiovascular Disease Risk Predictor")
st.write("Adjust the patient parameters below to simulate different medical profiles and see how they affect the model's prediction in real-time.")

# Create tabs for better organization
tab1, tab2 = st.tabs(["🩺 Patient Details Form", "📚 Feature Glossary"])

with tab2:
    st.markdown("""
    ### Medical Glossary
    * **Chest Pain Type (cp):** Typical angina (heart-related), Atypical angina, Non-anginal, or Asymptomatic.
    * **Resting BP (trestbps):** Blood pressure in mm Hg on admission to the hospital.
    * **Cholesterol (chol):** Serum cholestoral in mg/dl.
    * **Fasting Blood Sugar (fbs):** Is it higher than 120 mg/dl? (Indicates potential diabetes).
    * **Max Heart Rate (thalach):** Maximum heart rate achieved during exercise testing.
    * **Oldpeak:** ST depression induced by exercise relative to rest.
    * **Thalassemia (thal):** A blood disorder. Normal, Fixed defect, or Reversable defect.
    """)

with tab1:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Demographics & Vitals")
        age = st.slider("Age", min_value=20, max_value=100, value=50, help="Patient's age in years")
        
        sex_input = st.radio("Sex", options=["Male", "Female"], horizontal=True)
        sex = 1 if sex_input == "Male" else 0
        
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120, help="Normal is around 120/80")
        
        chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200, help="Desirable is under 200 mg/dl")

    with col2:
        st.subheader("Symptom Details")
        cp_input = st.selectbox("Chest Pain Type", 
                                options=["0: Typical Angina", "1: Atypical Angina", "2: Non-anginal Pain", "3: Asymptomatic"])
        cp = int(cp_input[0]) # Extract the numeric value
        
        fbs_input = st.radio("Fasting Blood Sugar > 120 mg/dl", options=["No", "Yes"], horizontal=True, help="High blood sugar can be a risk factor")
        fbs = 1 if fbs_input == "Yes" else 0
        
        restecg_input = st.selectbox("Resting ECG Results", 
                                     options=["0: Normal", "1: ST-T wave abnormality", "2: Left ventricular hypertrophy"])
        restecg = int(restecg_input[0])
        
        thalach = st.slider("Maximum Heart Rate Achieved", 70, 220, 150)

    with col3:
        st.subheader("Exercise & Angiography")
        exang_input = st.radio("Exercise Induced Angina", options=["No", "Yes"], horizontal=True, help="Does exercise cause chest pain?")
        exang = 1 if exang_input == "Yes" else 0
        
        oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1, help="Depression induced by exercise relative to rest")
        
        slope_input = st.selectbox("Slope of Peak Exercise ST Segment", 
                                   options=["0: Upsloping", "1: Flat", "2: Downsloping"])
        slope = int(slope_input[0])
        
        ca = st.slider("Number of Major Vessels Colored by Flourosopy (ca)", 0, 4, 0)
        
        thal_input = st.selectbox("Thalassemia (thal)", 
                                  options=["0: Normal", "1: Fixed Defect", "2: Reversable Defect", "3: Unknown"], index=2)
        thal = int(thal_input[0])

    st.markdown("---")
    
    # Center the predict button
    _, btn_col, _ = st.columns([1, 1, 1])
    with btn_col:
        predict_button = st.button("🔍 Run Prediction Analysis", use_container_width=True, type="primary")

# --- Prediction Logic ---
if predict_button:
    # Add a small progress bar for visual effect during presentation
    with st.spinner("Analyzing patient data against the Random Forest model..."):
        time.sleep(1) # Artificial delay for app interaction feel
        
        # Format the input
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0]
        
        st.markdown("### Diagnosis Result")
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            if prediction[0] == 1:
                st.error("⚠️ **High Risk**")
                st.write("The model predicts a high likelihood of heart disease.")
            else:
                st.success("✅ **Low Risk**")
                st.write("The model predicts a low likelihood of heart disease.")
                
        with res_col2:
            st.write("**Model Confidence (Probability):**")
            disease_prob = probability[1] * 100
            
            # Show a visual progress bar for the probability
            st.progress(int(disease_prob))
            st.write(f"Probability of Heart Disease: **{disease_prob:.2f}%**")