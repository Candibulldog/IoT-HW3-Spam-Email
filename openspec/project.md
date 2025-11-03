# Project Context

## Purpose
Build a machine learning-based spam email/SMS detection system that can accurately classify messages as spam or ham (legitimate). The system includes a complete training pipeline and an interactive web interface for real-time predictions.

## Project Overview
- **Type**: Machine Learning Classification Project
- **Domain**: Cybersecurity / Natural Language Processing
- **Dataset**: SMS Spam Collection (Packt Publishing - Chapter 3)
- **Goal**: Achieve >95% accuracy in spam detection
- **Deliverables**:
  1. Trained ML models with performance metrics
  2. Interactive Streamlit web application
  3. Complete documentation and analysis

## Tech Stack

### Core Framework
- **Language**: Python 3.9+
- **ML Libraries**: scikit-learn, nltk
- **Web Framework**: Streamlit
- **Data Processing**: pandas, numpy

### Key Libraries
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
nltk>=3.8
streamlit>=1.28.0
matplotlib>=3.7.0
seaborn>=0.12.0
wordcloud>=1.9.0
```

### Hardware Requirements
- GPU: NVIDIA GPU with CUDA support (optional, for faster training)
- CPU: Multi-core processor recommended
- RAM: Minimum 8GB

## Architecture

### Data Pipeline
```
Raw Data (CSV) 
  → Preprocessing (lowercase, remove punctuation, stopwords)
  → Feature Engineering (TF-IDF Vectorization)
  → Train/Test Split (80/20)
  → Model Training
  → Evaluation
  → Serialization (pickle)
```

### Model Strategy
Train and compare multiple classifiers:
1. **Multinomial Naive Bayes** (baseline)
2. **Logistic Regression**
3. **Random Forest**
4. **Support Vector Machine (SVM)**

Select best model based on:
- Accuracy
- Precision & Recall
- F1-Score
- ROC-AUC

### Deployment Architecture
```
User Input (Streamlit UI)
  → Text Preprocessing
  → TF-IDF Vectorization
  → Model Prediction
  → Result Display (Spam/Ham + Confidence)
```

## Project Standards

### Code Style
- Follow PEP 8 conventions
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use docstrings for all functions and classes

### File Organization
- Keep data preprocessing logic in `src/preprocess.py`
- Training logic in `src/train.py` (not notebooks for main training)
- Evaluation metrics in `src/evaluate.py`
- Utility functions in `src/utils.py`

### Naming Conventions
- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions/Variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`

### Documentation
- Every function must have a docstring
- Complex algorithms need inline comments
- README must include:
  - Installation instructions
  - Usage examples
  - Performance metrics
  - Screenshots

## Development Workflow

### 1. Data Preparation
- Download dataset from Packt GitHub
- Verify data integrity
- Perform exploratory data analysis (EDA)

### 2. Model Development
- Implement preprocessing pipeline
- Train multiple models
- Compare performance
- Select best model

### 3. Web Application
- Design Streamlit UI
- Integrate trained model
- Add real-time prediction
- Test user experience

### 4. Deployment
- Push to GitHub
- Deploy to Streamlit Cloud
- Test live application

## Quality Standards

### Performance Targets
- **Accuracy**: ≥95%
- **Precision**: ≥90% (minimize false positives)
- **Recall**: ≥90% (minimize false negatives)
- **Inference Time**: <1 second per prediction

### Testing Requirements
- Validate preprocessing on sample data
- Cross-validate model performance (5-fold CV)
- Test Streamlit app with various input lengths
- Verify model serialization/deserialization

## References
- Dataset: [Packt SMS Spam Dataset](https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/blob/master/Chapter03/datasets/sms_spam_no_header.csv)
- Book: "Hands-On Artificial Intelligence for Cybersecurity" - Chapter 3
- Framework: [OpenSpec Spec-Driven Development](https://github.com/Fission-AI/OpenSpec)
