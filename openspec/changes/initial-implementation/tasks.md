# Tasks: Initial Spam Detection Implementation

## Phase 1: Data Preparation
- [ ] Task 1.1: Download SMS spam dataset from Packt GitHub
- [ ] Task 1.2: Create `src/preprocess.py` with text cleaning functions
  - [ ] Lowercase conversion
  - [ ] Punctuation/number removal
  - [ ] Stopword removal (NLTK)
  - [ ] TF-IDF vectorization
- [ ] Task 1.3: Create `src/utils.py` with helper functions
  - [ ] Data loading
  - [ ] Train/test split
  - [ ] Visualization utilities
- [ ] Task 1.4: Perform exploratory data analysis (optional notebook)
  - [ ] Class distribution
  - [ ] Text length statistics
  - [ ] Word clouds for spam/ham

## Phase 2: Model Development
- [ ] Task 2.1: Implement `src/train.py` training script
  - [ ] Support argparse for hyperparameters
  - [ ] Train Naive Bayes model
  - [ ] Train Logistic Regression
  - [ ] Train Random Forest
  - [ ] Train SVM
  - [ ] Save best model to `models/spam_classifier.pkl`
  - [ ] Save vectorizer to `models/vectorizer.pkl`
- [ ] Task 2.2: Implement `src/evaluate.py` evaluation functions
  - [ ] Calculate accuracy, precision, recall, F1
  - [ ] Generate confusion matrix
  - [ ] Plot ROC curve
  - [ ] Save metrics to JSON/text file
- [ ] Task 2.3: Run full training pipeline
  - [ ] Execute `train.py` with optimal hyperparameters
  - [ ] Validate model performance (≥95% accuracy)
  - [ ] Document best model selection reasoning

## Phase 3: Web Application
- [ ] Task 3.1: Create `app.py` Streamlit application
  - [ ] Load trained model and vectorizer
  - [ ] Create text input interface
  - [ ] Implement real-time prediction
  - [ ] Display result (Spam/Ham) with confidence
  - [ ] Add optional features:
    - [ ] Show preprocessing steps
    - [ ] Display model metrics
    - [ ] Provide example messages
- [ ] Task 3.2: Test application locally
  - [ ] Test with spam examples
  - [ ] Test with ham examples
  - [ ] Test edge cases (empty input, very long text)
- [ ] Task 3.3: Deploy to Streamlit Cloud
  - [ ] Push code to GitHub
  - [ ] Connect Streamlit Cloud account
  - [ ] Configure deployment settings
  - [ ] Test live application

## Phase 4: Documentation
- [ ] Task 4.1: Write comprehensive README.md
  - [ ] Project description
  - [ ] Installation instructions
  - [ ] Usage examples
  - [ ] Performance metrics table
  - [ ] Screenshots of Streamlit app
  - [ ] Links to live demo
- [ ] Task 4.2: Create requirements.txt
  - [ ] List all dependencies with versions
- [ ] Task 4.3: Add .gitignore
  - [ ] Exclude large files, cache, models (use Git LFS if needed)
- [ ] Task 4.4: Final code review
  - [ ] Check PEP 8 compliance
  - [ ] Verify docstrings
  - [ ] Test all functionality

## Checkpoints
- ✅ **Checkpoint 1**: Preprocessing pipeline works correctly
- ✅ **Checkpoint 2**: All models trained, best model selected
- ✅ **Checkpoint 3**: Streamlit app runs locally
- ✅ **Checkpoint 4**: Documentation complete, project ready for submission

## Dependencies
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

## Notes
- Use Python 3.9+ for compatibility
- Test on both CPU and GPU if available
- Keep model files under 100MB for GitHub (use Git LFS if larger)
- Follow OpenSpec workflow: proposal → implementation → validation
