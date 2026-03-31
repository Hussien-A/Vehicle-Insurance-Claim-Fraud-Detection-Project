import os
import requests
import streamlit as st

API_URL = os.environ.get("STREAMLIT_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Righteous&display=swap');
    
    html, body, [class*="css"] { 
        font-family: 'Space Grotesk', sans-serif; 
        background-color: #0b0f19; 
        color: #e2e8f0; 
    }
    .main .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1000px; }
    
    header[data-testid="stHeader"] {
        background: linear-gradient(90deg, #0f172a 0%, #020617 100%);
        position: relative;
        box-shadow: 0 4px 20px rgba(0, 242, 254, 0.15);
        border-bottom: 1px solid #1e293b;
    }
    header[data-testid="stHeader"]::after {
        content: "Vehicle Insurance Fraud Detection System";
        font-family: 'Righteous', cursive;
        position: absolute;
        left: 1.5rem;
        top: 50%;
        transform: translateY(-50%);
        background: -webkit-linear-gradient(45deg, #00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.4rem;
        letter-spacing: 0.05em;
    }
    
    .stDeployButton { display: none !important; }
    [data-testid="stDeployButton"] { display: none !important; }
    
    
    .stForm { 
        background: rgba(30, 41, 59, 0.4); 
        border-radius: 20px; 
        padding: 2.5rem; 
        border: 1px solid rgba(148, 163, 184, 0.1); 
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.5); 
    }
    
    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stMarkdown"]) { margin-bottom: 0.5rem; }
    
    
    .result-card { padding: 2rem; border-radius: 16px; text-align: center; margin: 1.5rem 0; font-size: 1.4rem; letter-spacing: 1px; flex: 1; text-transform: uppercase; font-weight: 700; box-shadow: 0 8px 32px rgba(0,0,0,0.3); }
    .result-fraud { background: rgba(220, 38, 38, 0.1); border: 1px solid #ef4444; color: #fca5a5; text-shadow: 0 0 10px rgba(239,68,68,0.5); }
    .result-ok { background: rgba(34, 197, 94, 0.1); border: 1px solid #22c55e; color: #86efac; text-shadow: 0 0 10px rgba(34,197,94,0.5); }
    
    
    .metric-box { background: rgba(15, 23, 42, 0.6); border-radius: 12px; padding: 1rem 1.5rem; display: inline-block; margin-top: 1.5rem; font-size: 1.1rem; border: 1px solid #334155; color: #cbd5e1; box-shadow: inset 0 2px 4px rgba(0,0,0,0.5); }
    
    
    .section-title { font-size: 1.1rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 1rem; border-bottom: 1px solid #334155; padding-bottom: 0.75rem; }
    
    footer { text-align: center; color: #64748b; font-size: 0.9rem; margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid #334155; }
    
    
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(0, 114, 255, 0.4);
        width: 100%;
        margin-top: 1.5rem;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px rgba(0, 114, 255, 0.6);
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Vehicle Insurance Claim Fraud Detection")
st.markdown("Submit claim details to get a **Fraud** or **Legitimate** prediction from the ML model.")
st.divider()

with st.form("claim_form"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-title">Policy & Insured</p>', unsafe_allow_html=True)
        policy_state = st.selectbox("Policy State", ["GA", "PA", "MI", "CA", "NY", "NC", "FL", "TX", "IL", "OH"], index=0)
        policy_deductible = st.number_input("Policy Deductible", min_value=0, value=500, step=100)
        policy_annual_premium = st.number_input("Annual Premium", min_value=0.0, value=1000.0, step=50.0)
        insured_age = st.number_input("Insured Age", min_value=18, max_value=100, value=35)
        insured_sex = st.selectbox("Insured Sex", ["MALE", "FEMALE", "OTHER"], index=0)
        insured_education_level = st.selectbox("Education", ["High School", "College", "Masters", "PhD"], index=1)
        insured_occupation = st.selectbox(
            "Occupation",
            ["Clerk", "Doctor", "Engineer", "Lawyer", "Manager", "Sales", "Teacher", "Technician", "Other"],
            index=4,
        )
        insured_hobbies = st.selectbox(
            "Hobbies", ["chess", "reading", "movies", "camping", "hiking", "yachting", "paintball", "other"], index=1
        )
    with col2:
        st.markdown('<p class="section-title">Incident & Claim</p>', unsafe_allow_html=True)
        incident_type = st.selectbox(
            "Incident Type",
            ["Multi-vehicle Collision", "Single Vehicle Collision", "Parked Car", "Vehicle Theft"],
            index=0,
        )
        collision_type = st.selectbox("Collision Type", ["Front", "Rear", "Side", "Unknown"], index=0)
        incident_severity = st.selectbox(
            "Incident Severity", ["Minor Damage", "Major Damage", "Total Loss"], index=1
        )
        authorities_contacted = st.selectbox(
            "Authorities Contacted", ["Police", "Fire", "Ambulance", "None"], index=0
        )
        incident_state = st.selectbox("Incident State", ["GA", "PA", "MI", "CA", "NY", "NC", "FL", "TX", "IL", "OH"], index=0)
        incident_date = st.date_input("Incident Date").isoformat()
        incident_hour_of_the_day = st.slider("Incident Hour", 0, 23, 12)
        number_of_vehicles_involved = st.number_input("Vehicles Involved", min_value=1, value=1)
        bodily_injuries = st.number_input("Bodily Injuries", min_value=0, value=0)
        witnesses = st.number_input("Witnesses", min_value=0, value=0)
        police_report_available = st.selectbox("Police Report Available", ["No", "Yes"], index=0)
        claim_amount = st.number_input("Claim Amount", min_value=0.0, value=5000.0, step=100.0)
        total_claim_amount = st.number_input("Total Claim Amount", min_value=0.0, value=6000.0, step=100.0)

    submitted = st.form_submit_button("🔍 Get Prediction")

if submitted:
    payload = {
        "policy_state": policy_state,
        "policy_deductible": float(policy_deductible),
        "policy_annual_premium": float(policy_annual_premium),
        "insured_age": int(insured_age),
        "insured_sex": insured_sex,
        "insured_education_level": insured_education_level,
        "insured_occupation": insured_occupation,
        "insured_hobbies": insured_hobbies,
        "incident_type": incident_type,
        "collision_type": collision_type,
        "incident_severity": incident_severity,
        "authorities_contacted": authorities_contacted,
        "incident_state": incident_state,
        "incident_date": incident_date,
        "incident_hour_of_the_day": int(incident_hour_of_the_day),
        "number_of_vehicles_involved": int(number_of_vehicles_involved),
        "bodily_injuries": int(bodily_injuries),
        "witnesses": int(witnesses),
        "police_report_available": police_report_available,
        "claim_amount": float(claim_amount),
        "total_claim_amount": float(total_claim_amount),
    }
    with st.spinner("Analyzing claim..."):
        try:
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            r.raise_for_status()
            out = r.json()
            pred = out.get("prediction", 0)
            label = out.get("label", "Legitimate")
            prob = out.get("probability_fraud")
            if pred == 1:
                st.markdown(
                    '<div class="result-card result-fraud"><strong>Prediction: Fraud</strong> — Claim flagged for review.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="result-card result-ok"><strong>Prediction: Legitimate</strong> — No fraud detected.</div>',
                    unsafe_allow_html=True,
                )
            if prob is not None:
                st.markdown(f'<div class="metric-box">Probability of fraud: <strong>{prob:.1%}</strong></div>', unsafe_allow_html=True)
                st.progress(float(prob))
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API. Start it with: `uvicorn main:app --port 8000` in the api folder.")
        except requests.exceptions.HTTPError as e:
            st.error(f"API error: {e.response.status_code}")
        except Exception as e:
            st.error(str(e))

st.markdown('<footer>Vehicle Insurance Claim Fraud Detection — ML project</footer>', unsafe_allow_html=True)
