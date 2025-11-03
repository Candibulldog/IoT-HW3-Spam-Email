# AI Agents Handoff

## Context
This project implements a spam email/SMS detection system using machine learning. The system trains multiple classification models and deploys the best performer via a Streamlit web interface.

**Always read `openspec/project.md` first** to understand:
- Project purpose and architecture
- Tech stack and standards
- Quality requirements

## When to Use This Guide
- Planning new features or changes
- Understanding project structure
- Implementing model improvements
- Adding new functionality

## Project Structure

### Source Code (`src/`)
- **`preprocess.py`**: Text preprocessing (lowercasing, removing punctuation, stopwords, TF-IDF)
- **`train.py`**: Model training script (supports GPU, argparse for hyperparameters)
- **`evaluate.py`**: Performance evaluation (metrics, confusion matrix, ROC curve)
- **`utils.py`**: Helper functions (logging, file I/O, visualization)

### Data (`data/`)
- **`sms_spam_no_header.csv`**: Raw SMS spam dataset (5,574 samples)

### Models (`models/`)
- **`spam_classifier.pkl`**: Best trained model (serialized)
- **`vectorizer.pkl`**: Fitted TF-IDF vectorizer

### Application (`app.py`)
- Streamlit web interface for real-time spam detection

## Development Guidelines

### Adding New Features
1. Read `openspec/project.md` to ensure alignment
2. Check if similar functionality exists
3. Follow project standards (PEP 8, type hints, docstrings)
4. Test thoroughly before committing

### Code Quality
- **Type Hints**: Use for function signatures
- **Docstrings**: Required for all public functions
- **Error Handling**: Use try-except with specific exceptions
- **Logging**: Use Python's logging module (not print statements)

### Performance
- Vectorization preferred over loops
- Use generators for large datasets
- Profile code if adding computationally intensive features

### Testing
- Validate preprocessing with sample inputs
- Cross-validate model performance
- Test edge cases (empty strings, very long messages, special characters)

## Common Tasks

### Training a New Model
```bash
python src/train.py --model naive_bayes --test_size 0.2 --random_state 42
```

### Evaluating Performance
```python
from src.evaluate import evaluate_model
metrics = evaluate_model(model, X_test, y_test)
```

### Running Streamlit App
```bash
streamlit run app.py
```

## OpenSpec Workflow

### Creating a Change Proposal
When adding significant functionality:
1. Create a new folder in `openspec/changes/<feature-name>/`
2. Write `proposal.md` (what and why)
3. Write `tasks.md` (how - actionable steps)
4. Add spec deltas (requirements changes)

### Example Change Structure
```
openspec/changes/add-model-comparison/
├── proposal.md      # Summary and motivation
├── tasks.md         # Implementation checklist
└── specs/
    └── spec.md      # Detailed requirements
```

### Validation Commands
```bash
# List all changes
ls openspec/changes/

# Validate a change proposal
cat openspec/changes/<feature-name>/proposal.md
```

## Important Notes

### Don't
- ❌ Modify `openspec/project.md` without discussion
- ❌ Change core architecture without proposal
- ❌ Skip testing for "small" changes
- ❌ Use notebooks for production training code

### Do
- ✅ Follow existing code patterns
- ✅ Add comments for complex logic
- ✅ Update documentation when changing behavior
- ✅ Use version control (meaningful commit messages)

## Integration Points

### Loading Trained Models
```python
import pickle

with open('models/spam_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
```

### Preprocessing New Text
```python
from src.preprocess import preprocess_text

clean_text = preprocess_text("Win FREE iPhone now!")
```

### Making Predictions
```python
# Vectorize input
features = vectorizer.transform([clean_text])

# Predict
prediction = model.predict(features)[0]  # 0=ham, 1=spam
probability = model.predict_proba(features)[0]
```

## References
- Main spec: `openspec/project.md`
- Dataset source: [Packt GitHub Repository](https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/blob/master/Chapter03/datasets/sms_spam_no_header.csv)
- OpenSpec guide: [GitHub - Fission-AI/OpenSpec](https://github.com/Fission-AI/OpenSpec)

---

**Last Updated**: 2025-01-03
**Maintained By**: Project Team
