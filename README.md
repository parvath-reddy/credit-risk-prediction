# 💳 Credit Risk Prediction Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://credit-risk-prediction-xyz.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🚀 Live Demo
Experience the application: [https://credit-risk-prediction-xyz.streamlit.app/](https://credit-risk-prediction-xyz.streamlit.app/)

## 📋 Overview

An advanced machine learning platform that assesses credit risk for loan applications using the Extra Trees Classifier algorithm. The system analyzes 8 key financial and demographic indicators to predict creditworthiness with ~78% accuracy, processing over 1,000 historical credit records from the German Credit Dataset.

This production-ready application provides real-time risk assessment, confidence scoring, and visual analytics to support data-driven lending decisions in financial institutions.

## ✨ Key Features

- **Real-time Risk Assessment**: Instant credit risk evaluation with confidence scoring
- **Advanced ML Model**: Extra Trees Classifier trained on 1,000+ credit applications
- **Interactive Dashboard**: Visual analytics with Plotly charts and gauges
- **Risk Mitigation Insights**: Automated recommendations for high-risk applications
- **Responsive UI/UX**: Modern, mobile-friendly interface with intuitive design
- **Comprehensive Analysis**: Multi-factor evaluation including financial, demographic, and loan characteristics

## 🛠️ Technology Stack

- **Machine Learning**: Scikit-learn, Extra Trees Classifier
- **Frontend**: Streamlit, Plotly, HTML/CSS
- **Data Processing**: Pandas, NumPy
- **Model Persistence**: Joblib
- **Deployment**: Streamlit Cloud
- **Version Control**: Git/GitHub

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 78.5% |
| Precision | 82.1% |
| Recall | 75.3% |
| F1-Score | 78.6% |

## 🔍 Features Analyzed

1. **Financial Indicators**
   - Checking account status
   - Savings account balance
   - Credit amount requested

2. **Demographics**
   - Age
   - Gender
   - Employment category (4 levels)

3. **Loan Characteristics**
   - Duration (1-72 months)
   - Credit amount (€100-20,000)

4. **Stability Factors**
   - Housing status (own/rent/free)

## 📁 Project Structure

```
credit-risk-prediction/
├── 📂 data/
│   └── german_credit_data.csv
├── 📂 models/
│   ├── extra_trees_credit_model.pkl
│   ├── Checking_account_encoder.pkl
│   ├── Housing_encoder.pkl
│   ├── Saving_accounts_encoder.pkl
│   ├── Sex_encoder.pkl
│   └── target_encoder.pkl
├── 📂 notebooks/
│   └── Untitled.ipynb
├── 📓 app.py
├── 📋 requirements.txt
├── 📖 README.md
└── .gitignore
```

## 🚀 Installation & Setup

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/credit-risk-prediction.git
cd credit-risk-prediction
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Access the app**
Open browser and navigate to `http://localhost:8501`

## 📦 Requirements

```txt
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.3.0
joblib>=1.3.0
plotly>=5.17.0
```

## 🌐 Deployment

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy with one click

### Update Existing Deployment

```bash
# Make changes to your code
git add .
git commit -m "Update features"
git push origin main
# Streamlit Cloud will automatically redeploy
```

## 📈 Dataset Information

- **Source**: German Credit Dataset (UCI ML Repository)
- **Size**: 1,000 credit applications
- **Features**: 11 attributes (8 used for prediction)
- **Target**: Binary classification (Good/Bad credit risk)
- **Class Distribution**: 70% Good Risk, 30% Bad Risk


