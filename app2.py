import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ==============================
# Page Configuration
# ==============================
st.set_page_config(
    page_title="Breast Cancer 3-Year OS Prediction",
    page_icon="🧬",
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
    </style>
    """,
    unsafe_allow_html=True
)

# ==============================
# Load Model
# ==============================
@st.cache_resource
def load_model():
    return joblib.load("svm_model.pkl")

model = load_model()

# ==============================
# Title
# ==============================
st.markdown(
    """
    <div class="main-title">
    Breast Cancer 3-Year Overall Survival Prediction
    </div>
    <div class="subtitle">
    An SVM-based Clinical Decision Support Tool
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
    <b>Introduction</b><br><br>
    This web-based calculator was developed using an optimized 
    <b>Support Vector Machine (SVM)</b> model to estimate the 
    <b>individualized 3-year overall survival (OS) probability</b> 
    for breast cancer patients.<br><br>

    By integrating key clinicopathological parameters, this tool may assist 
    clinicians in identifying high-risk patients who could benefit from 
    intensified adjuvant therapies and closer surveillance.<br><br>

    <i>
    This calculator is intended for risk assessment and decision support only 
    and should not replace professional clinical judgment.
    </i>
    </div>
    """,
    unsafe_allow_html=True
)

# ==============================
# Sidebar Inputs
# ==============================
st.sidebar.header("Patient Clinical Parameters")

age = st.sidebar.selectbox(
    "Age",
    options=[
        (1, "18–39 years"),
        (2, "40–69 years"),
        (3, "≥70 years")
    ],
    format_func=lambda x: x[1]
)[0]

degree = st.sidebar.selectbox(
    "Degree of Differentiation",
    options=[
        (1, "Grade I (Well differentiated)"),
        (2, "Grade II (Moderately differentiated)"),
        (3, "Grade III (Poorly differentiated)")
    ],
    format_func=lambda x: x[1]
)[0]

t_stage = st.sidebar.selectbox(
    "T Stage",
    options=[
        (1, "T1 (≤20 mm)"),
        (2, "T2 (20–50 mm)"),
        (3, "T3 (>50 mm)"),
        (4, "T4 (Skin or chest wall invasion)")
    ],
    format_func=lambda x: x[1]
)[0]

ln_status = st.sidebar.radio(
    "Lymph Node Status",
    options=[
        (0, "Negative"),
        (1, "Positive")
    ],
    format_func=lambda x: x[1]
)[0]

molecular = st.sidebar.selectbox(
    "Molecular Subtype",
    options=[
        (1, "Luminal A"),
        (2, "Luminal B"),
        (3, "HER2-enriched"),
        (4, "Triple-negative")
    ],
    format_func=lambda x: x[1]
)[0]

nlr = st.sidebar.number_input(
    "Neutrophil-to-Lymphocyte Ratio (NLR)",
    min_value=0.1,
    max_value=50.0,
    value=2.5,
    step=0.1
)

# ==============================
# Prediction
# ==============================
st.markdown("<br>", unsafe_allow_html=True)
predict = st.button("🔍 Predict 3-Year Survival Risk", use_container_width=True)

if predict:

    # DataFrame with feature names (NO warnings)
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

    prob = model.predict_proba(X)[0, 1]
    prob_percent = prob * 100

    st.markdown(
        f"""
        <div class="card">
        <b>Predicted Probability of Poor Prognosis (3-Year OS)</b>
        <div class="metric-value">{prob_percent:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if prob >= 0.5:
        st.markdown(
            """
            <div class="risk-high">
            <b>High-Risk Group (Poor Prognosis)</b><br><br>
            Patients in this group may benefit from:
            <ul>
                <li>Intensified adjuvant therapy</li>
                <li>Closer clinical surveillance</li>
                <li>More rigorous follow-up schedules</li>
            </ul>
            Multidisciplinary evaluation is strongly recommended.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="risk-low">
            <b>Low-Risk Group (Favorable Prognosis)</b><br><br>
            Patients in this group generally have a favorable prognosis 
            under standard treatment and follow-up strategies.
            </div>
            """,
            unsafe_allow_html=True
        )

# ==============================
# Model Interpretation
# ==============================
st.markdown(
    """
    <div class="card">
    <b>Model Interpretation Summary</b><br><br>
    The SVM model identified the following factors as being associated with poorer outcomes:
    <ul>
        <li>Older age</li>
        <li>Higher degree of poor differentiation</li>
        <li>Advanced T stage</li>
        <li>Positive lymph node status</li>
        <li>Triple-negative molecular subtype</li>
        <li>Elevated neutrophil-to-lymphocyte ratio (NLR)</li>
    </ul>
    These findings are consistent with established clinical evidence.
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
