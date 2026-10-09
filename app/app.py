import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================

model = joblib.load('models/fraud_model.pkl')

# =========================
# DARK MODE CSS
# =========================

st.markdown("""
<style>

/* Main App */
.stApp {
    background-color: #0f172a;
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Text */
h1, h2, h3, h4, h5, h6, p, label {
    color: white !important;
}

/* Buttons */
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #1d4ed8;
    color: white;
}

/* Input boxes */
.stTextInput>div>div>input {
    background-color: #1e293b;
    color: #38bdf8 !important;   /* Light Blue */
    -webkit-text-fill-color: #38bdf8 !important;
    font-weight: bold;
}

/* Placeholder */
.stTextInput input::placeholder {
    color: #94a3b8 !important;
    opacity: 1;
}

/* Password input */
.stTextInput input {
    color: white !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background-color: #1e293b;
    border: 1px solid #334155;
    padding: 15px;
    border-radius: 12px;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background-color: white;
    border-radius: 10px;
}

/* Info boxes */
.stAlert {
    border-radius: 10px;
}

/* Footer */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOGIN SYSTEM
# =========================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    st.title("🔐 Secure Login")

    st.subheader("AI Fraud Detection System")

    username = st.text_input("Username")

    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button("Login"):

        if username == "JOSHNA" and password == "joshna71159":

            st.success("✅ Login Successful")

            time.sleep(1)

            st.session_state.logged_in = True

            st.rerun()

        else:

            st.error("❌ Invalid Username or Password")

    st.stop()

# =========================
# SIDEBAR
# =========================

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
    width=80
)

st.sidebar.title("💳 Navigation")

st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Go To",
    [
        "Home",
        "Upload Data",
        "Analytics",
        "Live Prediction",
        "About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.success("✅ Logged In")

if st.sidebar.button("🚪 Logout"):

    st.session_state.logged_in = False

    st.rerun()

# =========================
# HOME PAGE
# =========================

if menu == "Home":

    st.title("💳 AI Credit Card Fraud Detection System")

    st.write("""
    This system detects fraudulent credit card transactions using Machine Learning.
    """)

    st.header("🏠 Dashboard Overview")

    st.info("""
    Upload transaction CSV data to detect fraudulent transactions.
    """)

    st.markdown("""
    ## Features

    - Fraud Detection using AI
    - Random Forest Machine Learning Model
    - Fraud Analytics Dashboard
    - CSV Upload Support
    - Download Prediction Results
    - Real-Time Fraud Prediction
    - Secure Login Authentication
    - Dark Mode UI
    - Animated Metrics Dashboard
    """)

# =========================
# UPLOAD PAGE
# =========================

elif menu == "Upload Data":

    st.title("📂 Upload Transaction Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        data = pd.read_csv(uploaded_file)

        st.subheader("📄 Uploaded Dataset")
        st.dataframe(data.head())

        y_true = data["Class"]

        X = data.drop("Class", axis=1)

        predictions = model.predict(X)

        data["Prediction"] = predictions

        fraud_count = len(data[data["Prediction"] == 1])
        normal_count = len(data[data["Prediction"] == 0])

        # =========================
        # ANIMATED METRICS
        # =========================

        st.subheader("📊 Fraud Detection Summary")

        col1, col2, col3 = st.columns(3)

        total_placeholder = col1.empty()
        fraud_placeholder = col2.empty()
        normal_placeholder = col3.empty()

        for i in range(0, 101, 5):

            total_placeholder.metric(
                "Total Transactions",
                int(len(data) * i / 100)
            )

            fraud_placeholder.metric(
                "Fraud Transactions",
                int(fraud_count * i / 100)
            )

            normal_placeholder.metric(
                "Normal Transactions",
                int(normal_count * i / 100)
            )

            time.sleep(0.03)

        # =========================
        # PREDICTION RESULTS
        # =========================

        st.subheader("🔍 Prediction Results")

        st.dataframe(data.head(50))

        # =========================
        # FRAUD TRANSACTIONS
        # =========================

        st.subheader("🚨 Fraud Transactions")

        fraud_data = data[data["Prediction"] == 1]

        st.dataframe(fraud_data.head(20))

        # =========================
        # DOWNLOAD BUTTON
        # =========================

        csv = data.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 Download Prediction Results",
            data=csv,
            file_name='fraud_predictions.csv',
            mime='text/csv'
        )

# =========================
# ANALYTICS PAGE
# =========================

elif menu == "Analytics":

    st.title("📊 Fraud Analytics Dashboard")

    uploaded_file = st.file_uploader(
        "Upload CSV File For Analytics",
        type=["csv"]
    )

    if uploaded_file is not None:

        data = pd.read_csv(uploaded_file)

        y_true = data["Class"]

        X = data.drop("Class", axis=1)

        y_pred = model.predict(X)

        data["Prediction"] = y_pred

        # =========================
        # PIE CHART
        # =========================

        st.subheader("Fraud vs Normal Transactions")

        labels = ['Normal', 'Fraud']

        sizes = [
            len(data[data['Prediction'] == 0]),
            len(data[data['Prediction'] == 1])
        ]

        fig1, ax1 = plt.subplots(figsize=(8,8))

        ax1.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90
        )

        ax1.axis('equal')

        st.pyplot(fig1)

        # =========================
        # BAR CHART
        # =========================

        st.subheader("Transaction Distribution")

        fig2, ax2 = plt.subplots(figsize=(8,5))

        transaction_counts = data['Prediction'].value_counts()

        ax2.bar(
            ['Normal', 'Fraud'],
            transaction_counts
        )

        ax2.set_ylabel("Count")

        st.pyplot(fig2)

        # =========================
        # MODEL METRICS
        # =========================

        st.subheader("🤖 Model Performance Metrics")

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("Precision", f"{precision:.4f}")
        col3.metric("Recall", f"{recall:.4f}")
        col4.metric("F1 Score", f"{f1:.4f}")

        # =========================
        # CONFUSION MATRIX
        # =========================

        st.subheader("📌 Confusion Matrix")

        cm = confusion_matrix(y_true, y_pred)

        fig3, ax3 = plt.subplots(figsize=(6,4))

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud']
        )

        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        st.pyplot(fig3)

# =========================
# LIVE PREDICTION
# =========================

elif menu == "Live Prediction":

    st.title("⚡ Real-Time Fraud Prediction")

    st.write("""
    Enter transaction details manually to predict fraud.
    """)

    st.subheader("Transaction Input")

    time_value = st.number_input("Time", value=0.0)

    amount = st.number_input("Amount", value=0.0)

    feature_values = []

    for i in range(1, 29):

        value = st.number_input(
            f"V{i}",
            value=0.0
        )

        feature_values.append(value)

    if st.button("Predict Transaction"):

        input_data = pd.DataFrame(
            [[
                time_value,
                *feature_values,
                amount
            ]],
            columns=[
                "Time",
                "V1","V2","V3","V4","V5","V6","V7",
                "V8","V9","V10","V11","V12","V13",
                "V14","V15","V16","V17","V18","V19",
                "V20","V21","V22","V23","V24","V25",
                "V26","V27","V28",
                "Amount"
            ]
        )

        prediction = model.predict(input_data)[0]

        with st.spinner("Analyzing Transaction..."):

            time.sleep(2)

        if prediction == 1:

            st.error("🚨 Fraudulent Transaction Detected")

        else:

            st.success("✅ Normal Transaction")

# =========================
# ABOUT PAGE
# =========================

elif menu == "About Project":

    st.title("ℹ About Project")

    st.markdown("""
    ## AI Credit Card Fraud Detection System

    This project uses Machine Learning to detect fraudulent credit card transactions.

    ### Technologies Used

    - Python
    - Streamlit
    - Pandas
    - NumPy
    - Scikit-learn
    - Matplotlib
    - Seaborn

    ### Machine Learning Model

    - Random Forest Classifier

    ### Features

    - Upload CSV Dataset
    - Fraud Prediction
    - Analytics Dashboard
    - Confusion Matrix
    - Download Results
    - Interactive Visualizations
    - Real-Time Prediction
    - Secure Login Authentication
    - Dark Mode Dashboard
    - Animated Metrics

    ### Developed By

    AI & Machine Learning Project
    """)

# =========================
# FOOTER
# =========================

st.markdown("---")

st.caption("© 2026 AI Credit Card Fraud Detection System")