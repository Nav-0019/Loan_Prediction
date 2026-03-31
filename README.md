# 🏦 Loan Risk Oracle

**Predict financial risk with precision using machine learning.**

[](https://www.python.org/)
[](https://streamlit.io/)
[](https://scikit-learn.org/)

## 📖 Overview

**Loan Risk Oracle** is an end-to-end machine learning solution designed to quantify the probability of loan defaults. By analyzing key metrics—including income, occupation, assets, and geographical data—the system provides a data-driven risk score to assist financial institutions in making informed lending decisions.

> [\!TIP]
> **Live Demo:** [View the Interactive Web App here](https://www.google.com/search?q=https://share.streamlit.io/nav-0019/loan-risk-predictor)

-----

## ✨ Key Features

  * **Real-Time Inference:** Optimized pre-trained models deliver risk assessments in milliseconds.
  * **Probabilistic Scoring:** Uses Logistic Regression to provide a nuanced risk percentage rather than a simple Yes/No.
  * **Interactive UI:** A sleek dashboard built with Streamlit for seamless data entry and visualization.
  * **Modular Pipeline:** Clean, decoupled code for data preprocessing, feature engineering, and model prediction.

-----

## 🛠️ Technology Stack

| Category | Tools |
| :--- | :--- |
| **Language** | Python 3.9+ |
| **Interface** | Streamlit |
| **Machine Learning** | Scikit-learn (Logistic Regression) |
| **Data Handling** | Pandas, NumPy |
| **Environment** | WSL2 / Ubuntu |

-----

## 🚀 Getting Started

### 1\. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Nav-0019/nav-0019.github.io.git
cd loan-risk-oracle
pip install -r requirements.txt
```

### 2\. Model Training

If you wish to retrain the model with fresh data:

```bash
python train_model.py
```

### 3\. Launch the App

Run the local development server:

```bash
streamlit run loan_risk_predictor.py
```

-----

## ⚙️ System Workflow

1.  **Data Ingestion:** Collects historical applicant data.
2.  **Preprocessing:** Handles missing values and scales features using Standard Scaler.
3.  **Feature Engineering:** Encodes categorical variables (Profession/City) for model compatibility.
4.  **Risk Analysis:** The Logistic Regression engine calculates the default probability.
5.  **Insight Delivery:** Displays a visual risk gauge and classification (Low/Medium/High Risk).

-----

## 🎯 Use Cases

  * **Fintech Startups:** Quick-hit prototype for automated credit scoring.
  * **Academic Research:** Demonstrating the impact of socio-economic factors on creditworthiness.
  * **Risk Management:** Supporting human loan officers with objective, data-backed second opinions.

-----

## 🤝 Connect

**Developer:** [Nav-0019](https://www.google.com/search?q=https://github.com/Nav-0019)  
**Project Link:** [Loan Risk Oracle Repository](https://github.com/Nav-0019/nav-0019.github.io/)

-----

*Developed as part of a commitment to utilizing Data Science for impactful decision-making.*
