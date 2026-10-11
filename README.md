# 🛡️ FraudGuard: Real-Time Intelligent Transaction Fraud Detection Engine

[![Python Version](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-EB3C00?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![AWS Elastic Beanstalk](https://img.shields.io/badge/AWS-Elastic_Beanstalk-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white)](https://aws.amazon.com/elasticbeanstalk/)
[![Supabase](https://img.shields.io/badge/Supabase-Database-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)](https://supabase.com/)
[![Live Demo](https://img.shields.io/badge/Live_Demo-Active-brightgreen?style=for-the-badge&logo=google-chrome&logoColor=white)](https://fraudguard.gagan1008.in)

> **Live Production URL**: [https://fraudguard.gagan1008.in](https://fraudguard.gagan1008.in)  
> **API Health Status**: `https://fraudguard.gagan1008.in/health`

---

## 📌 Executive Summary

**FraudGuard** is an end-to-end, enterprise-grade machine learning system designed to detect and mitigate fraudulent financial transactions in real time with sub-100ms latency. 

Handling over **1.32 million financial transactions** characterized by extreme class imbalance (~0.13% fraud rate), FraudGuard pioneers a **Hybrid Dual-Engine Architecture** combining:
1. **Supervised Extreme Gradient Boosting (XGBoost)** calibrated with cost-sensitive learning (`scale_pos_weight`) to classify known anti-money laundering (AML) and fraudulent transfer signatures.
2. **Unsupervised Isolation Forest** to capture previously unseen zero-day anomaly vectors and account takeover outliers.
3. **Composite Risk Decision Engine** synthesizing probabilistic risk scores into actionable governance decisions: `APPROVE`, `REVIEW`, or `BLOCK`.

The entire platform is fully containerized with **Docker**, deployed on **AWS Elastic Beanstalk** via automated **GitHub Actions CI/CD (OIDC authenticated with AWS ECR)**, and backed by **Supabase (PostgreSQL)** for real-time transaction telemetry and audit logging.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    Client["🌐 Web Client / Banking API Gateway"] -->|POST /predict (JSON Payload)| API["⚡ FastAPI Application Layer"]
    
    subgraph Pipeline ["Inference & Feature Pipeline (predict_pipeline.py)"]
        API --> FE["⚙️ Real-Time Feature Engineering<br/>- Chronological Sorting<br/>- Historical Account Velocity<br/>- Ratio to Sender Mean<br/>- Data Leakage Prevention (cumcount)"]
        FE --> Preproc["🔄 Scikit-Learn ColumnTransformer Preprocessor"]
        
        Preproc --> XGB["🎯 Supervised Model<br/>XGBoost Classifier<br/>(P(Fraud) Output)"]
        Preproc --> IF["🌲 Unsupervised Model<br/>Isolation Forest<br/>(Anomaly Flag + Decision Score)"]
        
        XGB & IF --> RiskEng["⚖️ Composite Risk Engine<br/>Risk Score = (0.75 * P(Fraud)) + (0.25 * AnomalyScore)"]
        RiskEng --> Decision{"🚦 Decision Engine"}
        Decision -->|Score < 0.50| Appr["✅ APPROVE"]
        Decision -->|0.50 <= Score < 0.80| Rev["⚠️ MANUAL REVIEW"]
        Decision -->|Score >= 0.80| Blk["🛑 BLOCK TRANSACTION"]
    end
    
    Decision --> Supa[("🗄️ Supabase PostgreSQL<br/>Audit & Transaction Telemetry")]
    Decision --> UI["💻 Interactive Glassmorphic UI Dashboard"]
```

---

## 🔬 Core Engineering Highlights

### 1. Leakage-Free Temporal Feature Engineering
In financial transaction fraud detection, random cross-validation splits and naïve aggregations introduce devastating future data leakage. FraudGuard solves this by enforcing:
* **Chronological Ordering**: Transactions are sorted strictly by `(TIMESTAMP, TX_ID)`.
* **Causal Window Aggregation**: Computes sender and receiver historical metrics using causal cumulative functions (`cumcount()`, `cumsum() - current_amount`) ensuring no future data leaks into the past.
* **Behavioral Dynamics & Velocity**:
  * `AMOUNT_TO_SENDER_AVG_RATIO`: Detects sudden anomalous spikes relative to the user's normal baseline.
  * `TIME_SINCE_SENDER_TX` & `TIME_SINCE_RECEIVER_TX`: Tracks rapid automated script bursts and bot transactions.
  * `IS_NEW_SENDER` & `IS_NEW_RECEIVER`: Flags high-risk cold-start accounts frequently utilized as money mules.

### 2. Dual-Engine Hybrid Risk Scoring
Relying strictly on a supervised classifier makes systems blind to novel fraud patterns. Conversely, unsupervised models yield high false-positive rates. FraudGuard bridges this gap:
$$\text{Risk Score} = 0.75 \times P(\text{Fraud}) + 0.25 \times \mathbb{I}_{\text{Anomaly}}$$

* If an attacker discovers an evasion vector bypassing the supervised model, the **Isolation Forest** catches multidimensional behavioral divergence and escalates the transaction for `REVIEW`.
* If a high-confidence signature triggers the supervised model, the transaction is immediately `BLOCKED`.

---

## 📊 Model Benchmarks & Experimental Results

The models were evaluated on an extensive benchmark of **1,323,234 transactions** with severe class imbalance (1,719 fraud cases, 99.87% non-fraud). Because accuracy is fundamentally deceptive on imbalanced datasets, **Precision-Recall AUC (PR-AUC)** and **Recall** were the primary optimization metrics.

| Algorithm | Accuracy | Precision | Recall (TPR) | F1-Score | ROC-AUC | PR-AUC (Key Metric) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Production Winner)** | **99.999%** | **99.74%** | **100.00%** | **0.9987** | **1.0000** | **1.0000** |
| Random Forest | 99.997% | 100.00% | 98.41% | 0.9920 | 0.9999 | 0.9999 |
| Extra Trees | 99.968% | 100.00% | 77.19% | 0.8713 | 0.9999 | 0.9977 |
| HistGradientBoosting | 99.936% | 69.81% | 96.29% | 0.8094 | 0.9679 | 0.6676 |
| Logistic Regression (Baseline) | 97.982% | 6.28% | 95.23% | 0.1179 | 0.9763 | 0.3612 |

> **Production Choice**: **XGBoost** achieved a perfect **100% Recall** on test fraud transactions while maintaining a **99.74% Precision**, virtually eliminating costly false negatives while keeping false declines below 0.001%.

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend & Serving** | Python 3.12, FastAPI, Uvicorn, Pydantic v2 |
| **Machine Learning & Math** | XGBoost, Scikit-Learn, NumPy, Pandas, Joblib |
| **Database & Persistence** | Supabase (PostgreSQL), Supabase Python Client |
| **Frontend & UI** | Jinja2 Templates, Vanilla Modern CSS (Glassmorphism, Responsive CSS Grid) |
| **DevOps & Infrastructure** | Docker, AWS Elastic Beanstalk, AWS ECR, AWS IAM OIDC |
| **CI/CD Automation** | GitHub Actions (Automated Linting, Docker Build & Health Validation, ECR Push, EB Deployment) |

---

## 🗂️ Project Directory Structure

```text
├── .github/
│   └── workflows/
│       └── docker-ci.yml          # GitHub Actions CI/CD Pipeline (Build, Test, ECR & Elastic Beanstalk)
├── artifacts/
│   ├── feature_schema.pkl         # Serialized training feature metadata & schema
│   ├── preprocessor.pkl           # Fitted ColumnTransformer (Scalers, Encoders, Imputers)
│   └── models/
│       ├── best_model.pkl         # Trained Production XGBoost Model
│       ├── isolation_forest.pkl   # Fitted Isolation Forest Anomaly Model
│       └── model_report.pkl       # Serialized metrics and validation benchmark report
├── src/
│   ├── components/
│   │   ├── anomaly_detection.py   # Isolation Forest inference module
│   │   ├── data_ingestion.py      # Ingestion, validation & train-test split
│   │   ├── data_transformation.py # Feature engineering & preprocessing pipeline
│   │   ├── model_trainer.py       # Model training, hyperparameter optimization & evaluation
│   │   └── risk_engine.py         # Multi-factor hybrid decision engine
│   ├── database/
│   │   └── supabase_client.py     # Supabase PostgreSQL client integration
│   ├── pipeline/
│   │   ├── predict_pipeline.py    # Production real-time inference pipeline
│   │   └── train_pipeline.py      # End-to-end model retraining orchestration
│   ├── exception.py               # Custom exception handler with traceback details
│   ├── logger.py                  # Thread-safe rotating application logger
│   └── utils.py                   # Artifact serialization and utility helpers
├── templates/
│   └── home.html                  # Production UI Dashboard (Glassmorphism + Real-Time Telemetry)
├── app.py                         # FastAPI application entry point & API endpoints
├── Dockerfile                     # Multi-stage optimized production Docker image
├── requirements.txt               # Pinned dependencies
└── README.md                      # Comprehensive project documentation
```

---

## 🔌 API Documentation & Endpoints

### 1. Healthcheck
* **Endpoint**: `GET /health`
* **Description**: Verifies service status and confirms artifact loading.
* **Sample Response**:
  ```json
  {
    "status": "healthy",
    "service": "fraud-detection-api",
    "predictor_loaded": true
  }
  ```

### 2. Predict Transaction
* **Endpoint**: `POST /predict`
* **Description**: Evaluates an incoming transaction and persists prediction to the database.
* **Request Payload**:
  ```json
  {
    "TX_ID": "TX_20261012_001",
    "SENDER_ACCOUNT_ID": "6976",
    "RECEIVER_ACCOUNT_ID": "9739",
    "TX_TYPE": "TRANSFER",
    "TX_AMOUNT": 4.85,
    "TIMESTAMP": 1728681600.0,
    "SENDER_CUSTOMER_ID": "C_6976",
    "RECEIVER_CUSTOMER_ID": "C_9739",
    "SENDER_COUNTRY": "US",
    "RECEIVER_COUNTRY": "US",
    "SENDER_ACCOUNT_TYPE": "I",
    "RECEIVER_ACCOUNT_TYPE": "I",
    "SENDER_INIT_BALANCE": 376.86,
    "RECEIVER_INIT_BALANCE": 165.67,
    "SENDER_TX_BEHAVIOR_ID": 5,
    "RECEIVER_TX_BEHAVIOR_ID": 4
  }
  ```
* **Response**:
  ```json
  {
    "transaction_id": "TX_20261012_001",
    "prediction": {
      "fraud_probability": 0.981527,
      "fraud_prediction": 1,
      "isolation_prediction": -1,
      "is_anomaly": true,
      "anomaly_score": 1.0,
      "isolation_decision_score": -0.123404,
      "risk_score": 0.9861,
      "decision": "BLOCK"
    },
    "database_saved": true
  }
  ```

### 3. Analytics & Telemetry
* `GET /stats`: Returns aggregated counts of total transactions, fraud detections, approvals, reviews, and blocks.
* `GET /transactions`: Retrieves the latest 100 transactions with full risk breakdowns.
* `GET /transactions/{id}`: Detailed inspection of a specific transaction ID.

---

## 🚀 Local Setup & Installation

### Prerequisites
* Python 3.10+ (Python 3.12 recommended)
* Git
* Docker (Optional for containerized run)

### Step 1: Clone Repository
```bash
git clone https://github.com/gagansokhal-coder/Credit_card_fraud.git
cd Credit_card_fraud
```

### Step 2: Set Up Virtual Environment
```bash
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Create a `.env` file in the project root:
```env
SUPABASE_URL=https://your-supabase-project.supabase.co
SUPABASE_KEY=your-supabase-anon-or-service-role-key
PORT=8000
```

### Step 5: Start the Application
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Access the application dashboard at `http://127.0.0.1:8000` and Swagger docs at `http://127.0.0.1:8000/docs`.

---

## 🐳 Docker Deployment

### Build the Docker Image
```bash
docker build -t fraudguard:latest .
```

### Run the Container
```bash
docker run -d \
  --name fraudguard-app \
  -p 8000:8000 \
  --env-file .env \
  fraudguard:latest
```

---

## ☁️ CI/CD & Cloud Infrastructure

The deployment pipeline is fully automated via GitHub Actions:
1. **Continuous Integration**:
   - Triggers on every push or PR to `main`.
   - Spins up an Ubuntu runner, builds the Docker image, launches the container, and verifies health status via `curl http://127.0.0.1:8000/health`.
2. **Continuous Deployment**:
   - Authenticates directly with AWS using **OpenID Connect (OIDC)** without hardcoding long-lived AWS secret keys.
   - Pushes the verified container image to **Amazon ECR**.
   - Triggers rolling deployment on **AWS Elastic Beanstalk** (`eu-north-1`).
   - Serves traffic securely over HTTPS on custom domain `https://fraudguard.gagan1008.in`.

---

## 👨‍💻 Author & Contact

**Gagan Rajput**  
* GitHub: [@gagansokhal-coder](https://github.com/gagansokhal-coder)  
* Project Domain: [fraudguard.gagan1008.in](https://fraudguard.gagan1008.in)  

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
