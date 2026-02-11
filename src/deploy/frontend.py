import streamlit as st
import httpx
from PIL import Image
import io
import base64

# --- Configuration ---
BACKEND_URL = "http://127.0.0.1:8000"

# --- Page Setup ---
st.set_page_config(
    page_title="Neuro Confidence Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Injection (Dark Professional Theme) ---
st.markdown("""
<style>
    /* Global Reset & Colors */
    .stApp {
        background-color: #0F172A;
        color: #E2E8F0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styling */
    h1, h2, h3 {
        color: #F8FAFC !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }
    
    /* Card Styling */
    div.stMetric {
        background-color: #1E293B;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    label[data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricValue"] {
        color: #F1F5F9 !important;
        font-size: 1.5rem !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1F2937;
    }
    
    /* Button Styling */
    div.stButton > button {
        background-color: #2563EB;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 500;
        width: 100%;
        transition: background-color 0.2s;
    }
    div.stButton > button:hover {
        background-color: #1D4ED8;
        border: none;
        color: white;
    }
    
    /* Warning/Success Elements */
    .decision-card {
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        text-align: center;
    }
    .decision-success {
        background-color: #064E3B; /* Dark Green */
        border: 1px solid #059669;
        color: #D1FAE5;
    }
    .decision-abstain {
        background-color: #451a03; /* Dark Amber */
        border: 1px solid #d97706;
        color: #fef3c7;
    }
    
    /* Utilities */
    .small-text {
        font-size: 0.75rem;
        color: #64748B;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
    <div style="margin-bottom: 2rem; border-bottom: 1px solid #334155; padding-bottom: 1rem;">
        <h1 style="font-size: 1.8rem; margin: 0;">NEURO CONFIDENCE LAB</h1>
        <p style="color: #94A3B8; margin: 0; font-size: 0.9rem;">RESEARCH INTERFACE V1.0 &nbsp;&bull;&nbsp; NOT FOR CLINICAL USE</p>
    </div>
""", unsafe_allow_html=True)

# --- Sidebar (Controls) ---
with st.sidebar:
    st.markdown("### Input Configuration")
    
    uploaded_file = st.file_uploader("Input MRI Scan", type=["png", "jpg", "jpeg"], help="Select a T1-weighted MRI slice.")
    
    st.markdown("---")
    st.markdown("### Safety Parameters")
    
    threshold = st.slider(
        "Abstention Threshold", 
        min_value=0.0, max_value=1.0, value=0.7, step=0.05,
        help="Minimum confidence required for a prediction."
    )
    
    temperature = st.slider(
        "Calibration Temperature (T)", 
        min_value=0.1, max_value=5.0, value=1.5, step=0.1,
        help="Post-hoc scaling factor. T>1 softens the distribution."
    )
    
    show_gradcam = st.checkbox("Enable Interpretability Overlay", value=False)
    
    st.markdown("---")
    run_btn = st.button("RUN ANALYSIS")
    
    st.markdown("""
        <div style="margin-top: 2rem; font-size: 0.7rem; color: #475569;">
            CONFIDENTIAL RESEARCH PREVIEW.<br>
            DO NOT RELY FOR DIAGNOSIS.
        </div>
    """, unsafe_allow_html=True)

# --- Main Area ---

if uploaded_file is None:
    # Empty State
    st.markdown("""
        <div style="text-align: center; color: #475569; margin-top: 5rem;">
            <h3>Ready for Analysis</h3>
            <p>Upload a scan in the sidebar to begin.</p>
        </div>
    """, unsafe_allow_html=True)

else:
    # Display Layout
    col_img, col_results = st.columns([1, 1.5], gap="large")
    
    with col_img:
        st.markdown("**Input Scan**")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
    
    if run_btn:
        with col_results:
            with st.spinner("Computing Inference & Uncertainty..."):
                try:
                    # Prepare payload
                    files = {"file": uploaded_file.getvalue()}
                    data = {"threshold": threshold, "temperature": temperature}
                    
                    # Call Backend
                    resp = httpx.post(f"{BACKEND_URL}/predict", files=files, data=data, timeout=30.0)
                    
                    if resp.status_code == 200:
                        result = resp.json()
                        
                        # Decision Card
                        if result["abstained"]:
                            st.markdown(f"""
                                <div class="decision-card decision-abstain">
                                    <h2 style="margin:0; color: #FCD34D;">ABSTAINED</h2>
                                    <p style="margin:0;">MODEL UNCERTAIN</p>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                                <div class="decision-card decision-success">
                                    <h2 style="margin:0; color: #A7F3D0;">{result['prediction'].upper()}</h2>
                                    <p style="margin:0;">PREDICTION ACCEPTED</p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        # Metrics Grid
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.metric("Confidence", f"{result['confidence']:.2f}")
                        with m2:
                            st.metric("Entropy", f"{result['entropy']:.2f}")
                        with m3:
                            st.metric("Threshold", f"{threshold:.2f}")

                        # Explanation Logic
                        if show_gradcam:
                            st.markdown("### Attention Map")
                            with st.spinner("Generating Explanation..."):
                                files_exp = {"file": uploaded_file.getvalue()}
                                resp_exp = httpx.post(f"{BACKEND_URL}/explain", files=files_exp, timeout=30.0)
                                
                                if resp_exp.status_code == 200:
                                    exp_data = resp_exp.json()
                                    img_data = base64.b64decode(exp_data["image"])
                                    gradcam_img = Image.open(io.BytesIO(img_data))
                                    st.image(gradcam_img, use_column_width=True, caption="Grad-CAM Overlay")
                                else:
                                    st.error("Visualization service unavailable.")
                    
                    else:
                        st.error(f"System Error: Backend returned {resp.status_code}")
                        
                except httpx.ConnectError:
                    st.error("Connection Failed: Backend service is not reachable.")
                except Exception as e:
                    st.error(f"Runtime Error: {str(e)}")
