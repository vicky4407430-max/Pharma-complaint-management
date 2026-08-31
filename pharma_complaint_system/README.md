# AI-Powered Pharmaceutical Complaint Management System

A comprehensive system for classifying, prioritizing, and managing pharmaceutical customer complaints using machine learning and rule-based analysis.

## Overview

This system provides:

- **AI Classification**: Machine learning models to automatically categorize complaints and assess severity
- **Rule-Based Analysis**: Business rules for product detection, batch extraction, sentiment analysis, and regulatory flagging
- **Priority Scoring**: Intelligent priority calculation based on severity, sentiment, customer type, and regulatory flags
- **CAPA Suggestions**: Automated Corrective and Preventive Action recommendations
- **Interactive Dashboard**: Gradio-based web interface for complaint submission and tracking

## Installation

```bash
pip install scikit-learn pandas numpy matplotlib gradio
```

## Quick Start

```python
from pharma_complaint_system import (
    ComplaintClassifier,
    ComplaintAnalyzer,
    generate_dataset,
    create_dashboard,
)

# Generate synthetic training data
df = generate_dataset(3000)

# Train the classifier
classifier = ComplaintClassifier()
metrics = classifier.train(df)
print(f"Category accuracy: {metrics['category_accuracy']:.2%}")
print(f"Severity accuracy: {metrics['severity_accuracy']:.2%}")

# Analyze a complaint
analyzer = ComplaintAnalyzer()
result = analyzer.analyze(
    text="Patient experienced severe rash after taking Amoxicillin LOT12345",
    severity="Critical",
    customer_type="Hospital",
    category="Adverse Event"
)

print(f"Priority: {result.priority_level} ({result.priority_score}/100)")
print(f"Owner: {result.owner}")
print(f"CAPA Actions: {result.capa_actions}")

# Launch the dashboard
demo = create_dashboard(classifier)
demo.launch()
```

## Project Structure

```
pharma_complaint_system/
├── __init__.py          # Package initialization and exports
├── config.py            # Configuration constants and business rules
├── data_generator.py    # Synthetic data generation
├── models.py           # ML classification models
├── analysis.py         # Rule-based analysis
└── dashboard.py        # Gradio web interface
```

## Module Details

### `config.py`
Contains all configuration constants:
- Product and customer type definitions
- Complaint categories and severity levels
- Text templates for data generation
- CAPA actions and ownership mappings
- Sentiment analysis keywords
- Regulatory flag terms

### `data_generator.py`
Generates synthetic complaint data for training:
```python
from pharma_complaint_system import generate_dataset

df = generate_dataset(n=3000, seed=42)
```

### `models.py`
Provides the `ComplaintClassifier` class:
```python
from pharma_complaint_system import ComplaintClassifier

classifier = ComplaintClassifier(max_features=20000)
metrics = classifier.train(df)

# Make predictions
category = classifier.predict_category("Tablets are broken")
top_3 = classifier.predict_category("Tablets are broken", n=3)
```

### `analysis.py`
Provides the `ComplaintAnalyzer` class:
```python
from pharma_complaint_system import ComplaintAnalyzer

analyzer = ComplaintAnalyzer()

# Individual methods
product = analyzer.detect_product("Amoxicillin LOT12345")
batch = analyzer.extract_batch("Batch number is LOT98765")
sentiment, score = analyzer.analyze_sentiment("Very disappointed!")
flags = analyzer.detect_regulatory_flags("Patient had allergic reaction")

# Comprehensive analysis
result = analyzer.analyze(
    text="Complaint text...",
    severity="Major",
    customer_type="Pharmacist",
    category="Product Quality"
)
```

### `dashboard.py`
Creates an interactive web interface with tabs for:
- New complaint submission with AI analysis
- Complaint queue viewing and filtering
- Analytics charts and model metrics
- System documentation

## Features

### AI Classification
- TF-IDF vectorization with n-grams
- Logistic Regression for multi-class classification
- Stratified train/test split for evaluation
- Probability-based top-N predictions

### Rule-Based Analysis
- Product name detection from keywords
- Batch/lot number extraction via regex
- Sentiment analysis (Positive/Neutral/Negative)
- Regulatory flag detection
- Priority scoring (0-100)
- Department owner assignment
- CAPA action suggestions

### Priority Calculation
Priority score is calculated as:
```
base_score (from severity) +
sentiment_adjustment +
customer_type_adjustment +
flag_bonus (capped at 15)
```

Priority levels:
- **Immediate**: ≥85
- **High**: ≥70
- **Medium**: ≥45
- **Low**: <45

## API Reference

### `ComplaintClassifier`

**Constructor parameters:**
- `max_features`: Maximum TF-IDF features (default: 20000)
- `ngram_range`: N-gram range for TF-IDF (default: (1, 2))
- `max_iter`: Max logistic regression iterations (default: 1000)
- `test_size`: Test split proportion (default: 0.2)
- `random_state`: Random seed (default: 42)

**Methods:**
- `train(df)`: Train both category and severity models
- `predict_category(text, n=1)`: Predict complaint category
- `predict_severity(text, n=1)`: Predict complaint severity
- `evaluate(verbose=True)`: Evaluate model performance
- `get_metrics_dataframe()`: Get metrics as DataFrame

### `ComplaintAnalyzer`

**Methods:**
- `detect_product(text)`: Extract product name
- `extract_batch(text)`: Extract batch/lot number
- `analyze_sentiment(text)`: Analyze sentiment
- `detect_regulatory_flags(text)`: Detect regulatory flags
- `calculate_priority(severity, sentiment, customer_type, flags)`: Calculate priority
- `get_suggested_owner(category)`: Get department owner
- `get_capa_actions(category)`: Get CAPA actions
- `analyze(text, severity, customer_type, category)`: Full analysis

## Example Use Cases

### 1. Batch Processing
```python
complaints = [
    "Broken tablets in Paracetamol batch LOT11111",
    "Missing leaflet in Amoxicillin pack",
    "Patient had allergic reaction to Metformin"
]

for complaint in complaints:
    category = classifier.predict_category(complaint)
    severity = classifier.predict_severity(complaint)
    print(f"{complaint[:50]}... -> {category}, {severity}")
```

### 2. Custom Priority Thresholds
```python
analyzer = ComplaintAnalyzer()
score, level = analyzer.calculate_priority(
    severity="Major",
    sentiment="Negative",
    customer_type="Hospital",
    flags=["Adverse event", "Contamination"]
)
```

### 3. Integration with External Systems
```python
# Process incoming complaint from API
def process_api_complaint(complaint_data):
    text = complaint_data['description']
    
    # AI classification
    category = classifier.predict_category(text)
    severity = classifier.predict_severity(text)
    
    # Rule-based enhancement
    result = analyzer.analyze(text, severity, complaint_data['customer_type'], category)
    
    # Return structured response
    return {
        'category': category,
        'severity': severity,
        'priority_score': result.priority_score,
        'priority_level': result.priority_level,
        'owner': result.owner,
        'capa_actions': result.capa_actions,
    }
```

## Notes

- This demo uses synthetic training data and in-memory storage
- For production use, implement:
  - Secure database with audit logs
  - User authentication and authorization
  - Model validation and versioning
  - Pharma compliance controls (21 CFR Part 11, etc.)
  - Proper error handling and logging

## License

MIT License
