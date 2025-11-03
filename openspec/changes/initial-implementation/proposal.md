# Proposal: Initial Spam Detection Implementation

## Summary
Implement a complete spam email/SMS detection system from scratch, including data preprocessing, model training, evaluation, and a Streamlit web interface for real-time predictions.

## Motivation
This is a greenfield project (0→1) to build a production-ready spam classifier. The goal is to demonstrate end-to-end machine learning workflow from raw data to deployed application.

## Scope

### In Scope
- ✅ Data preprocessing pipeline (text cleaning, TF-IDF vectorization)
- ✅ Training multiple classification models (Naive Bayes, Logistic Regression, Random Forest, SVM)
- ✅ Model evaluation with comprehensive metrics
- ✅ Streamlit web application for user interaction
- ✅ Model serialization and loading
- ✅ Complete documentation

### Out of Scope
- ❌ Deep learning models (LSTM, Transformers)
- ❌ Real-time email server integration
- ❌ Multi-language support (English only)
- ❌ User authentication/database

## Technical Approach

### Data Pipeline
1. Load SMS spam dataset (5,574 samples)
2. Clean text:
   - Convert to lowercase
   - Remove punctuation and numbers
   - Remove stopwords (NLTK)
   - Stemming/Lemmatization
3. Split data (80% train, 20% test)
4. Vectorize using TF-IDF (max 3000 features)

### Model Training
Train four models:
1. **Multinomial Naive Bayes** (baseline, fast)
2. **Logistic Regression** (interpretable)
3. **Random Forest** (ensemble, robust)
4. **Support Vector Machine** (high accuracy)

Compare using:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC Curve

Select best model based on F1-Score (balance of precision/recall).

### Deployment
- Build Streamlit app with:
  - Text input box
  - Predict button
  - Result display (Spam/Ham + confidence %)
  - Optional: display preprocessing steps
- Deploy to Streamlit Cloud (free tier)

## Success Criteria
- ✅ Model accuracy ≥95%
- ✅ F1-Score ≥0.90
- ✅ Inference time <1 second
- ✅ Streamlit app functional and user-friendly
- ✅ Complete README with usage instructions

## Timeline
- **Phase 1** (Day 1): Data preprocessing + EDA
- **Phase 2** (Day 2): Model training + evaluation
- **Phase 3** (Day 3): Streamlit app + deployment

## References
- Dataset: [Packt SMS Spam Dataset](https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/blob/master/Chapter03/datasets/sms_spam_no_header.csv)
- See `openspec/project.md` for full project context
