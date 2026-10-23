"""
Mental Health Prediction System - Flask Application
Author: Rajat Kumar (240010120052)
Guide: Dr. Sakshi Dhingra
Guru Jambheshwar University of Science and Technology, Hisar
"""

import os
import json
import pickle
import re
import string
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from flask import Flask, render_template, request, jsonify
from utils.preprocessor import clean_text, get_sentiment_score

app = Flask(__name__)

# ── Load Models ──────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

def load_models():
    vectorizer = pickle.load(open(os.path.join(MODEL_DIR, 'vectorizer.pkl'), 'rb'))
    individual = pickle.load(open(os.path.join(MODEL_DIR, 'individual_models.pkl'), 'rb'))
    ensemble   = pickle.load(open(os.path.join(MODEL_DIR, 'ensemble.pkl'), 'rb'))
    results    = pickle.load(open(os.path.join(MODEL_DIR, 'results.pkl'), 'rb'))
    return vectorizer, individual, ensemble, results

try:
    VECTORIZER, INDIVIDUAL_MODELS, ENSEMBLE, MODEL_RESULTS = load_models()
    MODELS_LOADED = True
except Exception as e:
    MODELS_LOADED = False
    print(f"[WARNING] Could not load models: {e}")

# ── Label Config ──────────────────────────────────────────────────────────────
LABEL_INFO = {
    'Depression': {
        'color': '#6366f1',
        'icon': '😔',
        'description': 'Signs of depressive patterns detected in the text.',
        'tips': [
            'Reach out to a mental health professional',
            'Connect with trusted friends or family',
            'Establish a daily routine with small achievable goals',
            'Practice self-compassion and be kind to yourself',
        ]
    },
    'Anxiety': {
        'color': '#f59e0b',
        'icon': '😰',
        'description': 'Signs of anxiety-related patterns detected in the text.',
        'tips': [
            'Try deep breathing exercises (4-7-8 technique)',
            'Limit caffeine and screen time',
            'Practice mindfulness or meditation daily',
            'Consider speaking with a therapist or counselor',
        ]
    },
    'Stress': {
        'color': '#ef4444',
        'icon': '😤',
        'description': 'Elevated stress indicators detected in the text.',
        'tips': [
            'Prioritize and break tasks into manageable steps',
            'Take regular short breaks during work',
            'Exercise regularly — even a 20-min walk helps',
            'Practice time management and set boundaries',
        ]
    },
    'Normal': {
        'color': '#10b981',
        'icon': '😊',
        'description': 'No significant mental health concerns detected.',
        'tips': [
            'Maintain your healthy routines',
            'Continue practicing self-care',
            'Stay connected with your support network',
            'Check in with yourself regularly',
        ]
    }
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def make_chart_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor='#0f172a', edgecolor='none')
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return data


def predict_text(text: str) -> dict:
    cleaned = clean_text(text)
    sentiment = get_sentiment_score(text)
    vec = VECTORIZER.transform([cleaned])

    # Individual predictions
    individual_preds = {}
    individual_probs = {}
    for name, model in INDIVIDUAL_MODELS.items():
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]
        classes = model.classes_
        individual_preds[name] = pred
        individual_probs[name] = {c: round(float(p) * 100, 1) for c, p in zip(classes, prob)}

    # Ensemble prediction
    ens_pred = ENSEMBLE.predict(vec)[0]
    ens_prob = ENSEMBLE.predict_proba(vec)[0]
    ens_classes = ENSEMBLE.classes_
    ens_probs = {c: round(float(p) * 100, 1) for c, p in zip(ens_classes, ens_prob)}

    label_info = LABEL_INFO[ens_pred]

    return {
        'prediction': ens_pred,
        'confidence': max(ens_probs.values()),
        'probabilities': ens_probs,
        'individual_predictions': individual_preds,
        'individual_probabilities': individual_probs,
        'sentiment': sentiment,
        'cleaned_text': cleaned,
        'word_count': len(text.split()),
        'color': label_info['color'],
        'icon': label_info['icon'],
        'description': label_info['description'],
        'tips': label_info['tips'],
    }


def make_probability_chart(probs: dict, predicted: str):
    labels = list(probs.keys())
    values = list(probs.values())
    colors = [LABEL_INFO[l]['color'] for l in labels]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    bars = ax.barh(labels, values, color=colors, height=0.55, 
                   edgecolor='none', zorder=2)
    for bar, val in zip(bars, values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', ha='left',
                color='white', fontsize=9, fontweight='bold')

    ax.set_xlim(0, 115)
    ax.set_xlabel('Probability (%)', color='#94a3b8', fontsize=9)
    ax.tick_params(colors='#cbd5e1', labelsize=10)
    ax.spines[:].set_visible(False)
    ax.xaxis.set_tick_params(color='#334155')
    ax.yaxis.set_tick_params(color='#334155')
    ax.grid(axis='x', color='#334155', linewidth=0.5, zorder=1)
    ax.set_title('Prediction Probabilities', color='#f1f5f9',
                 fontsize=11, fontweight='bold', pad=10)
    fig.tight_layout()
    return make_chart_b64(fig)


def make_sentiment_chart(sentiment: dict):
    labels = ['Positive', 'Negative', 'Neutral']
    values = [sentiment['positive'] * 100,
              sentiment['negative'] * 100,
              sentiment['neutral'] * 100]
    colors = ['#10b981', '#ef4444', '#64748b']

    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')

    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90,
        wedgeprops=dict(edgecolor='#0f172a', linewidth=2),
        textprops=dict(color='#cbd5e1', fontsize=9)
    )
    for at in autotexts:
        at.set_color('white')
        at.set_fontweight('bold')

    ax.set_title('Sentiment Distribution', color='#f1f5f9',
                 fontsize=11, fontweight='bold', pad=10)
    fig.tight_layout()
    return make_chart_b64(fig)


def make_model_comparison_chart():
    names = list(MODEL_RESULTS.keys())
    accs = [MODEL_RESULTS[n]['accuracy'] for n in names]
    colors = ['#6366f1', '#f59e0b', '#ef4444', '#10b981']

    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    bars = ax.bar(names, accs, color=colors[:len(names)],
                  edgecolor='none', width=0.55, zorder=2)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{acc}%', ha='center', va='bottom', color='white',
                fontsize=9, fontweight='bold')

    ax.set_ylim(0, 105)
    ax.set_ylabel('Accuracy (%)', color='#94a3b8', fontsize=9)
    ax.tick_params(colors='#cbd5e1', labelsize=8)
    ax.spines[:].set_visible(False)
    ax.grid(axis='y', color='#334155', linewidth=0.5, zorder=1)
    ax.set_title('Model Comparison — Accuracy', color='#f1f5f9',
                 fontsize=11, fontweight='bold', pad=10)
    plt.xticks(rotation=15, ha='right')
    fig.tight_layout()
    return make_chart_b64(fig)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', models_loaded=MODELS_LOADED)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = (data or {}).get('text', '').strip()

    if not text:
        return jsonify({'error': 'Please enter some text to analyze.'}), 400
    if len(text) < 10:
        return jsonify({'error': 'Text is too short. Please write at least a sentence.'}), 400
    if not MODELS_LOADED:
        return jsonify({'error': 'Models not loaded. Please train first.'}), 500

    result = predict_text(text)

    # Generate charts
    result['prob_chart'] = make_probability_chart(result['probabilities'], result['prediction'])
    result['sentiment_chart'] = make_sentiment_chart(result['sentiment'])

    return jsonify(result)


@app.route('/dashboard')
def dashboard():
    if not MODELS_LOADED:
        return render_template('dashboard.html', error='Models not loaded.')
    comparison_chart = make_model_comparison_chart()
    return render_template('dashboard.html',
                           results=MODEL_RESULTS,
                           comparison_chart=comparison_chart)


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True, port=5000)
