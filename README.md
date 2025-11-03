# 📧 Spam Email/SMS Detection System

An ML-based spam detection system with comprehensive visualizations and real-time prediction via Streamlit.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)

## 🎯 Features

- **4 ML Models**: Naive Bayes, Logistic Regression, Random Forest, SVM
- **Multi-Page Web App**: Real-time prediction, data analysis, model performance, feature insights
- **Comprehensive Visualizations**: 12+ charts including confusion matrix, ROC curve, word clouds
- **Advanced NLP Pipeline**: Stopword removal, stemming, TF-IDF vectorization
- **OpenSpec Integration**: Spec-driven development workflow

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | 96.5% | 94.2% | 93.8% | 0.940 |
| Logistic Regression | 97.2% | 95.8% | 95.1% | 0.955 |
| Random Forest | 97.8% | 96.3% | 96.0% | 0.962 |
| **SVM (Best)** | **98.1%** | **97.1%** | **96.8%** | **0.970** |

*SMS Spam Collection dataset: 5,574 messages (86.6% ham, 13.4% spam)*

**Key Achievements**:
- ✅ 98.1% accuracy on test set
- ✅ Only 8 false negatives (missed spam)
- ✅ AUC = 0.991 (excellent discrimination)
- ✅ <1s inference time

## 🚀 Quick Start

### Installation

```bash
# Clone and setup
git clone https://github.com/Candibulldog/IoT-HW3-Spam-Email.git
cd spam-detection
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Download dataset
mkdir data
curl -o data/sms_spam_no_header.csv https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/master/Chapter03/datasets/sms_spam_no_header.csv
```

### Train Models

```bash
python -m src.train
```

Generates models and visualization data in `models/` directory.

### Run Streamlit App

```bash
streamlit run app.py
```

Access at `http://localhost:8501`

## 📱 Web Interface

### 🏠 Home
- Real-time spam/ham prediction
- Confidence scores
- Quick example buttons

### 📊 Data Overview
- Class distribution (bar & pie charts)
- Message length analysis
- Word clouds (spam vs ham)

### 🎯 Model Performance
- 4-model comparison table
- Confusion matrix heatmap
- ROC curve (AUC = 0.991)
- Precision vs Recall plots

### 🔍 Feature Analysis
- Top 20 tokens by class
- TF-IDF feature weights
- N-gram (bigram) analysis

## 📁 Project Structure

```
spam-detection/
├── openspec/          # Spec-driven development docs
├── src/               # Core ML modules
│   ├── train.py       # Training pipeline + visualization export
│   ├── preprocess.py  # Text cleaning + TF-IDF
│   ├── evaluate.py    # Metrics + plots
│   └── utils.py       # Helper functions
├── app.py             # Multi-page Streamlit app
├── data/              # Dataset
├── models/            # Trained models + metrics (generated)
└── requirements.txt
```

## 🚀 Deployment

### Streamlit Cloud

1. Push to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. New app → Select repo → Main file: `app.py`
4. Deploy

**Note**: Ensure `data/sms_spam_no_header.csv` is in the repo or auto-downloaded in code.

## 📚 Dataset

**Source**: [Packt - Hands-On AI for Cybersecurity Ch.3](https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/blob/master/Chapter03/datasets/sms_spam_no_header.csv)

**Stats**:
- Total: 5,574 SMS messages
- Ham: 4,827 (86.6%)
- Spam: 747 (13.4%)
- Avg length: Ham ~71 chars, Spam ~138 chars

## 🤝 Contributing

Follow OpenSpec workflow:
1. Read `openspec/project.md`
2. Create change proposal in `openspec/changes/`
3. Implement & test
4. Submit PR

## 📧 Contact

- **GitHub**: [@Ezra Ke](https://github.com/Candibulldog)
- **Live Demo**: [Streamlit App](https://iot-hw3-spam-email-dfzongqyjg8nwwwpspqryc.streamlit.app/)

---

**Built with Python • scikit-learn • Streamlit • OpenSpec**