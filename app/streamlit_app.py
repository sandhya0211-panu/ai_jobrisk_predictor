import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Job Risk Predictor", page_icon="🤖", layout="wide")

# -------------------------------
# HEADER
# -------------------------------
st.markdown("## 🤖 AI Job Risk Predictor")
st.caption("Smart AI-based job risk analysis")

st.markdown("---")

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("data/ai_job_replacement_2020_2026_v2.csv")
job_roles = sorted(df["job_role"].unique())

# -------------------------------
# LAYOUT (SIDE BY SIDE 🔥)
# -------------------------------
col1, col2 = st.columns([1, 1])

# -------------------------------
# LEFT SIDE (INPUTS)
# -------------------------------
with col1:
    st.subheader("📌 Job Details")

    job_role_input = st.text_input("Enter Job Role (optional)")
    job_role = st.selectbox("Select Job Role", job_roles)

    selected_job = job_role_input if job_role_input else job_role

    st.subheader("🧠 Questions")

    automation_level = st.selectbox("How repetitive is your job?", ["Low", "Medium", "High"])
    skill_level = st.selectbox("Do you have required skills?", ["Yes", "Somewhat", "No"])
    job_type = st.selectbox("Job type?", ["Government", "Private", "Startup"])
    learning = st.selectbox("Learning new skills?", ["Regularly", "Sometimes", "No"])

# -------------------------------
# VALUE MAPPING
# -------------------------------
automation_risk = 20 if automation_level == "Low" else 50 if automation_level == "Medium" else 80
skill_gap = 0.2 if skill_level == "Yes" else 0.5 if skill_level == "Somewhat" else 0.8
job_security = 80 if job_type == "Government" else 50 if job_type == "Private" else 30
reskilling = 30 if learning == "Regularly" else 60 if learning == "Sometimes" else 90

# -------------------------------
# RIGHT SIDE (RESULT)
# -------------------------------
with col2:
    st.subheader("📊 Result")

    if st.button("🚀 Predict AI Risk"):

        with st.spinner("Analyzing..."):
            url = "http://127.0.0.1:8000/predict"

            data = {
                "job_role": selected_job,
                "automation_risk_percent": automation_risk,
                "skill_gap_index": skill_gap,
                "job_security_score": job_security,
                "reskilling_urgency_score": reskilling
            }

            response = requests.post(url, json=data)
            result = response.json()
            prediction = result["ai_job_risk"]

        st.metric("AI Job Risk (%)", f"{prediction}%")

        if prediction < 40:
            st.success("🟢 Low Risk")
        elif prediction < 70:
            st.warning("🟡 Medium Risk")
        else:
            st.error("🔴 High Risk")

        # -------------------------------
        # CHART
        # -------------------------------
        st.subheader("📈 Analysis")

        labels = ["Automation", "Skill Gap", "Job Security", "Reskilling"]
        values = [automation_risk, skill_gap * 100, job_security, reskilling]

        fig, ax = plt.subplots()
        ax.bar(labels, values)

        st.pyplot(fig)