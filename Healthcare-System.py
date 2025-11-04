# filepath: d:\A-Healthcare-System-using-Machine-Learning-Techniques-for-Disease-Prediction-with-Chatbot-Assistance-main\Healthcare-System.py
from streamlit_option_menu import option_menu
import pickle
import streamlit as st
import numpy as np
import pandas as pd
import os
from groq import Groq


# MUST BE FIRST Streamlit command
st.set_page_config(
    page_title="Healthcare System with Chatbot",
    page_icon="🏥",
    layout="wide"
)

# Enhanced CSS styling with modern healthcare color theme (no blur on titles)
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  :root{
    /* LIGHT THEME */
    --bg-grad-1:#667eea;         /* page gradient */
    --bg-grad-2:#764ba2;
    --text:#1a1a1a;
    --muted:#556;
    --panel:#89CFF0;             /* main header / chatbot bg */
    --card:rgba(255,255,255,0.9); /* input backgrounds */
    --border:rgba(102,126,234,0.25);
    --accent-1:#667eea;
    --accent-2:#764ba2;
    --good-1:#00b894; --good-2:#00a085;
    --bad-1:#ff6b6b;  --bad-2:#ee5a24;
    --note-bg:#ffffff; --note-border:#007bff;
    --sidebar:#5B68E1; --nav-selected:#E23F55; --nav-text:#ffffff;
  }

  @media (prefers-color-scheme: dark){
    :root{
      /* DARK THEME OVERRIDES */
      --bg-grad-1:#1f2335;
      --bg-grad-2:#151a2b;
      --text:#e8eaf0;
      --muted:#b6bdd6;
      --panel:#1f2a44;
      --card:rgba(28,32,48,0.9);
      --border:rgba(255,255,255,0.12);
      --accent-1:#7aa2f7;
      --accent-2:#a78bfa;
      --good-1:#10b981; --good-2:#059669;
      --bad-1:#f87171;  --bad-2:#ef4444;
      --note-bg:#22283b; --note-border:#60a5fa;
      --sidebar:#242b49; --nav-selected:#7c3aed; --nav-text:#e8eaf0;
    }
  }

  body {
    font-family: 'Inter','Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;
    color: var(--text);
    background: linear-gradient(135deg, var(--bg-grad-1) 0%, var(--bg-grad-2) 100%);
    margin:0; padding:0;
  }
  .stApp { min-height:100vh; }

  /* Hide Streamlit chrome (keep if you like) */
 /* Keep header visible; just hide menu/footer/deploy */
#MainMenu, footer, .stDeployButton { display:none !important; }


  /* Placeholder color */
  input::placeholder, textarea::placeholder { color:#888 !important; opacity:1 !important; }

  .main-header{
    background: var(--panel);
    border:1px solid rgba(255,255,255,0.15);
    padding:3rem 2rem; border-radius:25px; text-align:center; color:var(--text);
    margin-bottom:2.5rem;
  }
  .main-header h1{ font-size:3.2rem; color:var(--text); margin-bottom:.8rem; font-weight:700; }

  .prediction-container{ background:transparent; border:none; padding:0; margin:0; }

  /* Inputs */
  .input-section{ background:transparent; border:none; padding:0; border-radius:0; margin-bottom:2rem; }
  .stNumberInput > div > div > input,
  .stTextInput   > div > div > input{
    border-radius:12px; border:2px solid var(--border);
    padding:1rem 1.25rem; background:var(--card);
    font-size:1rem; color:var(--text);
  }

  /* Labels & radios now follow theme text color */
  .stNumberInput > label,
  .stTextInput   > label,
  .stRadio       > label{ color:var(--text) !important; font-weight:500; }
  .stRadio > div{ color:var(--text) !important; }

  /* Button */
  .stButton > button{
    background: linear-gradient(135deg, var(--accent-1) 0%, var(--accent-2) 100%);
    color:white; padding:1rem 2rem; border-radius:14px; border:none; font-weight:600;
  }

  .result-positive{
    background: linear-gradient(135deg,var(--bad-1),var(--bad-2));
    color:white; padding:1.5rem; border-radius:12px; margin-top:1rem;
  }
  .result-negative{
    background: linear-gradient(135deg,var(--good-1),var(--good-2));
    color:white; padding:1.5rem; border-radius:12px; margin-top:1rem;
  }

  .chatbot-container{
    padding:3rem; border-radius:25px; background:var(--panel); margin-bottom:1.5rem;
  }
  .chatbot-container h1, .chatbot-container p{ color:var(--text) !important; }

  .chat-response{ background:#fff; color:#2c3e50; padding:1.6rem; border-radius:10px; margin-top:1rem; }
  @media (prefers-color-scheme: dark){
    .chat-response{ background:#0f1424; color:var(--text); }
  }

  .info-note{
    background: var(--note-bg);
    border-left:5px solid var(--note-border);
    padding:1rem; border-radius:8px; margin-bottom:1rem; color:var(--text);
  }

  .medical-disclaimer{ background:#fff3cd; padding:1rem; border-radius:8px; margin-top:1rem; color:#333; }
  @media (prefers-color-scheme: dark){
    .medical-disclaimer{ background:#3b2f12; color:#f3f4f6; }
  }

  /* Sidebar option menu recolor to theme */
  section[data-testid="stSidebar"] { background-color: var(--sidebar) !important; }
  .css-1d391kg, .css-1d391kg * { color: var(--nav-text) !important; } /* fallback for older Streamlit */
</style>
""", unsafe_allow_html=True)



groq_api_key = st.secrets.get("API_KEYS", {}).get("GROQ_API_KEY")

# Load models
try:
    diabetes_model = pickle.load(open("Diabetes Model.pkl", 'rb'))
    heart_disease_model = pickle.load(open("Heart Disease Model.pkl", 'rb'))
    liver_model = pickle.load(open("Liver Disease Model.pkl", 'rb'))
    scaler = pickle.load(open("Scaler.pkl", 'rb'))
except Exception as e:
    st.error(f"Model files missing or not loaded properly: {e}. Check your files.")
    st.stop()

# Sidebar Menu
# Sidebar Menu
with st.sidebar:
    selected = option_menu(
        '🏥 Healthcare System',
        ['Diabetes Prediction',
         'Heart Disease Prediction',
         'Liver Disease Prediction',
         'Healthcare Chatbot'],
        icons=['droplet-fill', 'heart-fill', 'capsule', 'chat-dots-fill'],
        default_index=0,
        styles={  # --- ADDED THIS STYLES DICTIONARY ---
            "container": {"padding": "0!important", "background-color": "#5B68E1"}, # Set background to plum
            "icon": {"color": "white", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "5px", "--hover-color": "#eee", "color": "white"},
            "nav-link-selected": {"background-color": "#E23F55", "color": "white"}, # Hot pink for selected item
        }
    )

# Diabetes Prediction
if selected == 'Diabetes Prediction':
    st.markdown('<div class="main-header"><h1>🩺 Diabetes Prediction System</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="prediction-container">', unsafe_allow_html=True) 
    
    # --- UI FIX: Restored the "white box" ---
    st.markdown('<div class="info-note">ℹ️ <strong>Input Guidelines:</strong> Smoking History codes: never:0, No Info:1, current:2, former:3, ever:4, not current:5</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    col7, col8 = st.columns(2)
    with col1:
        gender_option = st.radio('👤 Gender', ['♂ Male', '♀ Female'], key="gender_d_radio", horizontal=True)
        gender = 1.0 if gender_option == '♂ Male' else 0.0
    with col2:
        age = st.number_input('📅 Age', min_value=0, max_value=120, value=30, step=1, key="age_d")
    with col3:
        hypertension = st.number_input('🩸 Hypertension', min_value=0.0, max_value=1.0, step=0.1, key="hyper_d")
    with col4:
        heart_disease = st.number_input('❤️ Heart Disease', min_value=0.0, max_value=1.0, step=0.1, key="hd_d")
    with col5:
        smoking_history = st.number_input('🚬 Smoking History', min_value=0.0, max_value=5.0, step=1.0, key="sh_d")
    with col6:
        BMI = st.number_input('⚖️ BMI', min_value=0.0, max_value=50.0, step=1.0, key="bmi_d")
    with col7:
        HbA1c_level = st.number_input('🔬 HbA1c Level', min_value=0.0, max_value=20.0, step=1.0, key="hba1c_d")
    with col8:
        blood_glucose_level = st.number_input('🩸 Blood Glucose Level', min_value=0.0, max_value=500.0, step=1.0, key="bgl_d")
    
    st.markdown('</div>', unsafe_allow_html=True) 
    
    if st.button('🔍 Get Diabetes Prediction', use_container_width=True, key="predict_diabetes"):
        input_data = np.array([[gender, age, hypertension, heart_disease, smoking_history, BMI, HbA1c_level, blood_glucose_level]], dtype=float)
        prediction = diabetes_model.predict(input_data)
        if int(prediction[0]) == 1:
            st.markdown('<div class="result-positive">⚠️ HIGH RISK: The analysis indicates potential diabetes risk. Please consult a healthcare professional.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-negative">✅ LOW RISK: The analysis suggests low diabetes risk. Maintain healthy lifestyle habits.</div>', unsafe_allow_html=True)
            
    st.markdown('</div>', unsafe_allow_html=True) 

# Heart Disease Prediction
elif selected == 'Heart Disease Prediction':
    st.markdown('<div class="main-header"><h1>❤️ Heart Disease Prediction System</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="prediction-container">', unsafe_allow_html=True) 
    
    # --- UI FIX: Restored the "white box" ---
    st.markdown('<div class="info-note">ℹ️ <strong>Input Guidelines:</strong> Thal (Normal:0, Fixed:1, Reversible:2)</div>', unsafe_allow_html=True)

    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    col7, col8, col9 = st.columns(3)
    col10, col11, col12 = st.columns(3)
    col13 = st.columns(1)[0]
    with col1:
        age = st.number_input('📅 Age', min_value=0, max_value=120, value=30, step=1, key="age_h")
    with col2:
        gender_option = st.radio('👤 Gender', ['♂ Male', '♀ Female'], key="gender_h_radio", horizontal=True)
        gender = 1.0 if gender_option == '♂ Male' else 0.0
    with col3:
        chest_pain = st.number_input('💔 Chest Pain Type', min_value=0.0, max_value=3.0, step=0.2, key="cp_h")
    with col4:
        resting_bp = st.number_input('🩸 Resting BP', min_value=0.0, max_value=250.0, step=20.0, key="rbp_h")
    with col5:
        cholesterol = st.number_input('🧪 Serum Cholesterol', min_value=0.0, max_value=600.0,step=50.0, key="chol_h")
    with col6:
        fasting_bs = st.number_input('🍽️ Fasting Blood Sugar', min_value=0.0, max_value=1.0, step=1.0, key="fbs_h")
    with col7:
        resting_ecg = st.number_input('📊 Resting ECG', min_value=0.0, max_value=2.0, step=0.1, key="recg_h")
    with col8:
        max_hr = st.number_input('💓 Max Heart Rate', min_value=0.0, max_value=250.0, step=10.0,key="mhr_h")
    with col9:
        exercise_angina = st.number_input('🏃 Exercise Induced Angina', min_value=0.0, max_value=1.0, step=0.1, key="eia_h")
    with col10:
        st_depression = st.number_input('📉 ST Depression', min_value=0.0, max_value=10.0, step=1.0,key="std_h")
    with col11:
        slope = st.number_input('📈 Slope', min_value=0.0, max_value=2.0, step=0.1, key="slope_h")
    with col12:
        major_vessels = st.number_input('🩸 Major Vessels Colored', min_value=0.0, max_value=4.0, step=1.0, key="mvc_h")
    with col13:
        thal = st.number_input('🔬 Thal', min_value=0.0, max_value=2.0, step=0.1, key="thal_h")
    
    st.markdown('</div>', unsafe_allow_html=True) 
    if st.button('🔍 Get Heart Disease Prediction', use_container_width=True, key="predict_heart"):
        input_data = np.array([[age, gender, chest_pain, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, st_depression, slope, major_vessels, thal]], dtype=float)
        prediction = heart_disease_model.predict(input_data)
        if int(prediction[0]) == 1:
            st.markdown('<div class="result-positive">⚠️ HIGH RISK: The analysis indicates potential heart disease risk. Immediate medical consultation recommended.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-negative">✅ LOW RISK: The analysis suggests low heart disease risk. Continue maintaining heart-healthy habits.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True) 

# Liver Disease Prediction
elif selected == 'Liver Disease Prediction':
    st.markdown('<div class="main-header"><h1>🫘 Liver Disease Prediction System</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="prediction-container">', unsafe_allow_html=True) 
    
    # --- UI FIX: Restored the "white box" ---
    st.markdown('<div class="info-note">ℹ️ <strong>Input Guidelines:</strong> Enter all laboratory values as per your medical reports</div>', unsafe_allow_html=True)

    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    with col1:
        age = st.number_input('📅 Age', min_value=0, max_value=120, value=30, step=1, key="age_l")
    with col2:
        gender_option = st.radio('👤 Gender', ['♂ Male', '♀ Female'], key="gender_l_radio", horizontal=True)
        gender = 1.0 if gender_option == '♂ Male' else 0.0
    with col3:
        total_bilirubin = st.number_input('🔬 Total Bilirubin', min_value=0.0, max_value=50.0, step=5.0,key="tb_l")
    with col4:
        alkaline_phosphotase = st.number_input('🧪 Alkaline Phosphatase', min_value=0.0, max_value=2000.0, step=100.0,key="ap_l")
    with col5:
        alamine_aminotransferase = st.number_input('🩸 Alanine Aminotransferase', min_value=0.0, max_value=500.0,step=50.0, key="alt_l")
    with col6:
        albumin_globulin_ratio = st.number_input('⚗️ Albumin/Globulin Ratio', min_value=0.0, max_value=5.0, step=1.0,key="agr_l")
    
    st.markdown('</div>', unsafe_allow_html=True) 
    
    if st.button("🔍 Get Liver Disease Prediction", use_container_width=True, key="predict_liver"):
        try:
            inputs = {
                'Age': age,
                'Gender': gender,
                'Total_Bilirubin': total_bilirubin,
                'Alkaline_Phosphotase': alkaline_phosphotase,
                'Alamine_Aminotransferase': alamine_aminotransferase,
                'Albumin_and_Globulin_Ratio': albumin_globulin_ratio
            }
            df = pd.DataFrame([inputs])
            skewed = ['Total_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Albumin_and_Globulin_Ratio']
            
            df[skewed] = np.log1p(df[skewed])
            df[df.columns] = scaler.transform(df[df.columns])
            
            prediction = liver_model.predict(df)
            
            if int(prediction[0]) == 1:
                st.markdown('<div class="result-positive">⚠️ HIGH RISK: The analysis indicates potential liver disease. Please consult a hepatologist immediately.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-negative">✅ LOW RISK: The analysis suggests healthy liver function. Maintain liver-friendly lifestyle.</div>', unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"⚠️ Prediction Failed: Please ensure all fields are filled correctly. Error: {e}")
            
    st.markdown('</div>', unsafe_allow_html=True) 

# Healthcare Chatbot
elif selected == 'Healthcare Chatbot':
    st.markdown('<div class="prediction-container">', unsafe_allow_html=True) 

    st.markdown("""
        <div class="main-header" style="background: #89CFF0; margin-bottom: 2rem;">
            <h1 style="color: #1a1a1a;">🤖 AI Healthcare Assistant</h1>
            <p style="color: #333; font-size: 1.1rem; margin-bottom: 0;">Get instant medical information about any disease or health condition</p>
        </div>
    """, unsafe_allow_html=True)

    disease_query = st.text_input(
        "🔍 Ask about any disease or health condition:",
        placeholder="e.g., What are the symptoms of diabetes? Tell me about hypertension...",
        help="Enter the name of any disease or ask specific health questions",
        key="disease_query"
    )

    def generate_disease_response_groq(disease_name):
        prompt = f"""
You are a professional medical assistant. Provide a concise but comprehensive response (10-15 lines) about: {disease_name}

Include the following key points in a well-structured format:
- Brief overview of the condition
- Main causes and risk factors
- Key symptoms to watch for
- Basic treatment approaches
- Important prevention tips
- When to seek medical help

Keep the response informative yet easy to understand for general public. Use clear, simple language and organize information with bullet points or short paragraphs.
"""
        if not groq_api_key:
            return "⚠️ GROQ API key not configured. Set GROQ_API_KEY in your .env file."
        try:
            client = Groq(api_key=groq_api_key)
            chat_completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a helpful medical information assistant. Provide accurate, concise medical information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"⚠️ Error connecting to medical database: {str(e)}"

    if disease_query:
        with st.spinner("🔍 Analyzing medical information..."):
            detailed_info = generate_disease_response_groq(disease_query)
            st.markdown(f'<div class="chat-response">{detailed_info}</div>', unsafe_allow_html=True)

        st.markdown("""
<style>
    :root {
        --disclaimer-text-light: #000000;
        --disclaimer-text-dark: #ffffff;
    }

    /* Light Theme */
    .medical-disclaimer {
        color: var(--disclaimer-text-light);
    }

    /* Dark Theme */
    @media (prefers-color-scheme: dark) {
        .medical-disclaimer {
            color: var(--disclaimer-text-dark);
        }
    }
</style>

<div class="medical-disclaimer">
    <strong>⚠️ Medical Disclaimer:</strong> This information is for educational purposes only and should not replace professional medical advice. 
    Always consult with qualified healthcare providers for diagnosis and treatment.
</div>
""", unsafe_allow_html=True)

    
    st.markdown('</div>', unsafe_allow_html=True)
