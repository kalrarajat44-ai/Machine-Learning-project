# MindScan — Mental Health Prediction using Social Media

**MCA Project | Guru Jambheshwar University of Science and Technology, Hisar**

- **Student:** Rajat Kumar (240010120052)
- **Supervisor:** Dr. Sakshi Dhingra (Dept. of CSE)
- **Date:** March 2026

---

## Overview

MindScan is a machine learning-based web application that uses NLP to analyze
social media text and predict mental health conditions (Depression, Anxiety,
Stress, Normal).

## Features

- Text analysis with TF-IDF feature extraction
- Multiple ML models: Logistic Regression, Naïve Bayes, Random Forest, Ensemble
- Sentiment analysis (positive/negative/neutral)
- Interactive web dashboard with charts
- Model performance metrics and comparison

## Project Structure

```
mental_health_prediction/
├── app.py                  # Flask web application (main entry)
├── train_model.py          # Model training script
├── requirements.txt        # Python dependencies
├── models/                 # Saved trained models (auto-generated)
│   ├── vectorizer.pkl
│   ├── individual_models.pkl
│   ├── ensemble.pkl
│   └── results.pkl
├── utils/
│   └── preprocessor.py     # NLP text preprocessing module
└── templates/
    ├── base.html           # Base layout template
    ├── index.html          # Main analysis page
    ├── dashboard.html      # Model metrics dashboard
    └── about.html          # About / bibliography page
```

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the models
```bash
python train_model.py
```

### 3. Launch the web application
```bash
python app.py
```

Then open your browser at: **http://localhost:5000**

## Methodology

1. **Data Collection** — Labeled social media text dataset (500 samples, 4 classes)
2. **Preprocessing** — Lowercasing, URL/punctuation removal, tokenization, stop-word removal
3. **Feature Extraction** — TF-IDF vectorizer with bigrams (max 5000 features)
4. **Sentiment Analysis** — Lexicon-based positive/negative/neutral scoring
5. **Model Training** — LR, Naïve Bayes, Random Forest, Voting Ensemble
6. **Evaluation** — Accuracy, Precision, Recall, F1-score
7. **Deployment** — Flask web application with interactive UI

## Tools & Technologies

| Category         | Tools                                      |
|------------------|--------------------------------------------|
| Language         | Python 3, HTML, CSS, JavaScript            |
| ML               | Scikit-learn (LR, NB, RF, Voting)          |
| NLP              | TF-IDF, custom lexicon sentiment analysis  |
| Data Processing  | Pandas, NumPy                              |
| Visualization    | Matplotlib, Seaborn                        |
| Web Framework    | Flask + Jinja2                             |

## Disclaimer

This system is for **research and educational purposes only** and is **not a
substitute for professional medical diagnosis**. If you or someone you know
is experiencing mental health challenges, please consult a qualified mental
health professional.
