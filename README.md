# 🤖 AI Job Risk Predictor

An end-to-end Machine Learning web application that predicts the risk of job replacement due to AI and automation.

---

## 🌐 Live Demo

* 🔗 **Frontend (Streamlit):** https://aijobriskpredictor-hckzjx3cycwkqs5dbstngu.streamlit.app
* 🔗 **Backend (FastAPI):** https://ai-jobrisk-predictor.onrender.com

---

## 📌 Problem Statement

With rapid advancements in AI, many jobs are at risk of automation.
This project aims to analyze job roles and predict the likelihood of job replacement using Machine Learning.

---

## 🚀 Solution

This system takes user inputs such as job role, automation level, and skill gap, and predicts an **AI Job Risk Score (%)** using a trained ML model.

---

## 🏗️ System Architecture

```id="arch"
User (UI - Streamlit)
        ↓
Frontend sends request (HTTP)
        ↓
FastAPI Backend (/predict API)
        ↓
ML Model (Linear Regression)
        ↓
Prediction Response
        ↓
Displayed in Streamlit UI
```

---

## 🧠 Machine Learning Approach

### ✔️ Models Used

* Linear Regression
* Random Forest Regressor
* XGBoost Regressor

### ✅ Final Model Selected

**Linear Regression**

### 📊 Why Linear Regression?

* Simpler and interpretable
* Performed better on dataset
* Suitable for linear relationships between features and target

---

## 📊 Dataset

* 📍 Source: Kaggle
* 📁 Name: AI Job Replacement Dataset (2020–2026)
* 📌 Features include:

  * Job Role
  * Automation Risk
  * Skill Gap Index
  * Job Security Score
  * Reskilling Urgency

---

## 📈 Model Workflow

```id="workflow"
Data Collection → Data Cleaning → Feature Engineering → Model Training → Evaluation → Deployment
```

---

## ⚙️ Tech Stack

* **Frontend:** Streamlit
* **Backend:** FastAPI
* **ML Libraries:** Scikit-learn, XGBoost
* **Data Handling:** Pandas, NumPy
* **Visualization:** Matplotlib
* **Deployment:** Render & Streamlit Cloud

---

## 📂 Project Structure

```id="structure"
ai_jobrisk_predictor/
│
├── api/
│   └── main.py
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   └── ai_job_replacement_2020_2026_v2.csv
│
├── models/
│   ├── job_risk_model.pkl
│   └── features.pkl
│
├── notebooks/
│   └── eda.ipynb
│
├── requirements.txt
└── README.md
```

---

## 📊 Input Features

| Feature            | Description                  |
| ------------------ | ---------------------------- |
| Job Role           | Selected job type            |
| Automation Risk    | Level of repetitive work     |
| Skill Gap          | Skill deficiency level       |
| Job Security       | Stability of job             |
| Reskilling Urgency | Need for learning new skills |

---

## 📈 Output

* 🎯 AI Job Risk (%)
* 🟢 Low Risk (<40%)
* 🟡 Medium Risk (40–70%)
* 🔴 High Risk (>70%)

---

## ⚙️ How to Run Locally

```bash id="runlocal"
git clone https://github.com/sandhya0211-panu/ai_jobrisk_predictor.git
cd ai_jobrisk_predictor
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn api.main:app --reload
cd app
streamlit run streamlit_app.py
```

---

## 📌 Key Highlights

* 🔥 End-to-end ML pipeline
* 🌐 Fully deployed web application
* ⚡ Real-time predictions
* 🧠 Model comparison performed
* 🎯 Clean UI + API integration

---

## 🚧 Future Improvements

* 🔹 Add deep learning models
* 🔹 Improve prediction accuracy
* 🔹 Add user authentication
* 🔹 Deploy mobile-friendly UI
* 🔹 Integrate live job market data APIs

---

## 👩‍💻 Author

**Sandhyarani Panuganti**

---

## ⭐ Show your support

If you like this project, give it a ⭐ on GitHub!

---
