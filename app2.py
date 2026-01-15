import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ==============================
# Page Configuration
# ==============================
st.set_page_config(
    page_title="Breast Cancer 3-Year OS Prediction",
    page_icon="🎗️",
    layout="centered"
)

# ==============================
# Custom CSS (UI Polish)
# ==============================
st.markdown(
    """
    <style>
    body {
        background-color: #F5F7FA;
        font-family: "Helvetica Neue", Arial, sans-serif;
    }

    .main-title {
        font-size: 32px;
        font-weight: 700;
        color: #1F3A5F;
        text-align: center;
        margin-bottom: 8px;
    }

    .subtitle {
        font-size: 16px;
        color: #5D6D7E;
        text-align: center;
        margin-bottom: 30px;
    }

    .card {
        background-color: #FFFFFF;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }

    .metric-value {
        font-size: 36px;
        font-weight: 700;
        color: #2C3E50;
    }

    .risk-high {
        background-color: #FDEDEC;
        border-left: 6px solid #C0392B;
        padding: 20px;
        border-radius: 10px;
        margin-top: 15px;
    }

    .risk-low {
        background-color: #EAFAF1;
        border-left: 6px solid #1E8449;
        padding: 20px;
        border-radius: 10px;
        margin-top: 15px;
    }

    .footer {
        font-size: 13px;
        color: #7F8C8D;
        text-align: center;
        margin-top: 40px;
    }
    
    /* Small tweaks for radio button alignment */
    div.row-widget.stRadio > div{
        flex-direction: row;
        align-items: stretch;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==============================
# Load Model
# ==============================
# 注意：请确保目录下有 'svm_model.pkl' 文件
@st.cache_resource
def load_model():
    try:
        return joblib.load("svm_model.pkl")
    except FileNotFoundError:
        st.error("⚠️ Model file 'svm_model.pkl' not found. Please upload it.")
        return None

model = load_model()

# ==============================
# Title
# ==============================
st.markdown(
    """
    <div class="main-title">
    🎗️ Breast Cancer 3-Year OS Prediction
    </div>
    <div class="subtitle">
    🤖 An SVM-based Clinical Decision Support Tool
    </div>
    """,
    unsafe_allow_html=True
)

# ==============================
# Introduction
# ==============================
st.markdown(
    """
    <div class="card">
    <b>ℹ️ Introduction</b><br><br>
    This web-based calculator was developed using an optimized 
    <b>Support Vector Machine (SVM)</b> model to estimate the 
    <b>individualized 3-year overall survival (OS) probability</b> 
    for breast cancer patients.<br><br>
    
    <i>This tool integrates clinicopathological parameters to assist in identifying high-risk patients.</i>
    </div>
    """,
    unsafe_allow_html=True
)

# ==============================
# Sidebar Inputs
# ==============================
st.sidebar.header("📋 Patient Clinical Parameters")

# 1. Age
age = st.sidebar.selectbox(
    "📅 Age Group",
    options=[
        (1, "18–39 years"),
        (2, "40–69 years"),
        (3, "≥70 years")
    ],
    format_func=lambda x: x[1]
)[0]

# 2. Degree
degree = st.sidebar.selectbox(
    "🔬 Differentiation Grade",
    options=[
        (1, "Grade I (Well differentiated)"),
        (2, "Grade II (Moderately differentiated)"),
        (3, "Grade III (Poorly differentiated)")
    ],
    format_func=lambda x: x[1]
)[0]

# 3. T Stage
t_stage = st.sidebar.selectbox(
    "📏 T Stage (Tumor Size)",
    options=[
        (1, "T1 (≤20 mm)"),
        (2, "T2 (20–50 mm)"),
        (3, "T3 (>50 mm)"),
        (4, "T4 (Invasion)")
    ],
    format_func=lambda x: x[1]
)[0]

# 4. Lymph Node (Updated as requested)
ln_status = st.sidebar.radio(
    "🦠 Lymph Node Metastasis",
    options=[
        (0, "No"),
        (1, "Yes")
    ],
    format_func=lambda x: x[1],
    horizontal=True  # 横向排列，更像开关
)[0]

# 5. Molecular
molecular = st.sidebar.selectbox(
    "🧬 Molecular Subtype",
    options=[
        (1, "Luminal A"),
        (2, "Luminal B"),
        (3, "HER2-enriched"),
        (4, "Triple-negative")
    ],
    format_func=lambda x: x[1]
)[0]

# 6. NLR
nlr = st.sidebar.number_input(
    "🩸 NLR (Neutrophil-to-Lymphocyte Ratio)",
    min_value=0.1,
    max_value=50.0,
    value=2.5,
    step=0.1
)

# ==============================
# Prediction
# ==============================
st.markdown("<br>", unsafe_allow_html=True)
predict = st.button("🔍 Predict Survival Risk", use_container_width=True)

if predict and model:
    # DataFrame with feature names
    X = pd.DataFrame(
        [[age, degree, t_stage, ln_status, molecular, nlr]],
        columns=[
            "Age",
            "Degree_of_differentiation",
            "T_stage",
            "lymphaden_Status",
            "Molecular_typing",
            "NLR"
        ]
    )

    try:
        prob = model.predict_proba(X)[0, 1]
        prob_percent = prob * 100

        st.markdown(
            f"""
            <div class="card">
            <b>📈 Predicted Probability of Poor Prognosis (3-Year OS)</b>
            <div class="metric-value">{prob_percent:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if prob >= 0.5:
            st.markdown(
                """
                <div class="risk-high">
                <b>🚨 High-Risk Group (Poor Prognosis)</b><br><br>
                Patients in this group may benefit from:
                <ul>
                    <li>💊 Intensified adjuvant therapy</li>
                    <li>🏥 Closer clinical surveillance</li>
                    <li>📆 More rigorous follow-up schedules</li>
                </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div class="risk-low">
                <b>✅ Low-Risk Group (Favorable Prognosis)</b><br><br>
                Patients in this group generally have a favorable prognosis 
                under standard treatment.
                </div>
                """,
                unsafe_allow_html=True
            )
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")

# ==============================
# Model Interpretation
# ==============================
st.markdown(
    """
    <div class="card">
    <b>📊 Model Interpretation Summary</b><br><br>
    The SVM model identified the following risk factors:
    <ul>
        <li>👵 Older age</li>
        <li>🔬 Poor differentiation (Grade III)</li>
        <li>📏 Advanced T stage</li>
        <li>🦠 Positive lymph node status</li>
        <li>🧬 Triple-negative subtype</li>
        <li>🩸 Elevated NLR</li>
    </ul>
    </div>
    """,
    unsafe_allow_html=True
)

# ==============================
# Footer
# ==============================
st.markdown(
    """
    <div class="footer">
    © SVM-based Breast Cancer Survival Prediction Tool <br>
    For research and clinical decision support only
    </div>
    """,
    unsafe_allow_html=True
)
