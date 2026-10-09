# 💳 AI Credit Card Fraud Detection System

An end-to-end machine learning project that detects potentially fraudulent credit card transactions using classification models and an interactive Streamlit dashboard.

## 🚀 Features

* Interactive Streamlit dashboard
* Credit card transaction data analysis
* Data preprocessing and exploratory data analysis
* Machine learning model training and evaluation
* Fraud detection predictions
* Confusion matrix and performance visualizations
* Jupyter notebooks documenting the project workflow

## 🛠️ Technologies Used

* Python
* Pandas and NumPy
* Scikit-learn
* Matplotlib and Seaborn
* Streamlit
* Jupyter Notebook
* Git and GitHub

## 📁 Project Structure

```text
AI-Credit-Card-Fraud-Detection/
├── app/
│   └── app.py
├── models/
│   └── fraud_model.pkl
├── notebooks/
├── reports/
│   ├── charts/
│   └── screenshots/
├── src/
│   ├── predict.py
│   ├── preprocess.py
│   ├── train_model.py
│   └── utils.py
├── data/
├── .gitignore
└── README.md
```

## ⚙️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/joshna0911-ops/AI-Credit-Card-Fraud-Detection.git
   ```

2. Navigate to the project directory:

   ```bash
   cd AI-Credit-Card-Fraud-Detection
   ```

3. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

4. Install the required packages:

   ```bash
   pip install streamlit pandas numpy scikit-learn matplotlib seaborn joblib
   ```

5. Obtain the credit card transaction dataset separately and place it in `data/raw/creditcard.csv`.

## ▶️ Run the Application

```bash
python -m streamlit run app/app.py
```

## 📊 Project Workflow

1. Load and understand transaction data.
2. Clean and preprocess the dataset.
3. Explore transaction patterns and class imbalance.
4. Train and evaluate machine learning models.
5. Use the dashboard to explore predictions and analytics.

## ⚠️ Disclaimer

This project is intended for educational and demonstration purposes. Predictions are not a substitute for a financial institution's fraud investigation or production risk controls.

## 👩‍💻 Author

**Joshna**

GitHub: [joshna0911-ops](https://github.com/joshna0911-ops)

