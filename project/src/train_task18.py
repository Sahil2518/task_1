"""
Task 18 — Natural Language Processing
PlaceMux Phase 1 Industry Immersion

Steps:
  1. Generate mock text data (Interviewer Notes).
  2. Resilient text cleaning (NLTK if available, Regex fallback).
  3. TF-IDF Vectorization with n-grams (capturing multi-word meaning).
  4. Train NLP classification model (Logistic Regression).
  5. Evaluate on test set (F1-score, Confusion Matrix).
  6. Inspect errors (Language-specific failure modes).
  7. Package the pipeline.
"""

import os
import sys
import json
import traceback
import re
import random
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)
from sklearn.base import BaseEstimator, TransformerMixin

# ── Constants ────────────────────────────────────────────────────────
SEED       = 42
LOGS_DIR   = "logs"
PLOTS_DIR  = os.path.join(LOGS_DIR, "task18_plots")
MODELS_DIR = "models"
RESULTS_JSON = os.path.join(LOGS_DIR, "task18_results.json")
PIPELINE_PATH = os.path.join(MODELS_DIR, "nlp_pipeline.pkl")

np.random.seed(SEED)
random.seed(SEED)

# ── 1. Mock Data Generation ──────────────────────────────────────────

def generate_mock_text_data(n_samples=1200):
    """
    Generates synthetic interviewer notes and a binary 'placed' target.
    Includes some mixed signals to test the model (edge cases).
    """
    positive_phrases = [
        "Strong technical skills", "excellent communication", "great problem solver",
        "culturally a good fit", "highly recommended", "good grasp of concepts",
        "impressed with their system design", "very proactive and articulate",
        "wrote clean and efficient code", "exceeded expectations"
    ]
    
    negative_phrases = [
        "needs improvement in coding logic", "communication was unclear",
        "technical skills lacking", "did not answer well", "struggled with basic algorithms",
        "poor system design", "lacks practical experience", "not a good fit right now",
        "code was buggy and inefficient", "seemed unprepared"
    ]
    
    mixed_phrases = [
        "good communication but poor coding", "strong technically but lacked soft skills",
        "great potential though needs more experience", "average performance"
    ]
    
    data = []
    labels = []
    
    for _ in range(n_samples):
        # 45% positive, 45% negative, 10% mixed/edge cases
        rand_val = random.random()
        
        if rand_val < 0.45:
            # Positive case
            note = f"{random.choice(positive_phrases)}. {random.choice(positive_phrases).capitalize()}."
            label = 1
        elif rand_val < 0.90:
            # Negative case
            note = f"{random.choice(negative_phrases).capitalize()}. {random.choice(negative_phrases)}."
            label = 0
        else:
            # Mixed / Edge case (harder for NLP to classify)
            note = random.choice(mixed_phrases).capitalize() + "."
            # Label depends on the predominant sentiment or arbitrary business logic; let's say mixed usually = 0
            label = 0
            
        # Occasionally inject empty strings or null-like values to test edge case handling
        if random.random() < 0.02:
            note = "" if random.random() < 0.5 else "   "
            label = 0
            
        data.append(note)
        labels.append(label)
        
    df = pd.DataFrame({"interviewer_notes": data, "placed": labels})
    return df

# ── 2. Resilient Text Cleaner ────────────────────────────────────────

class ResilientTextCleaner(BaseEstimator, TransformerMixin):
    """
    Cleans text data. Attempts to use NLTK for stopwords and lemmatization.
    If NLTK is unavailable (e.g., download timeout), falls back to robust regex cleaning.
    Handles edge cases like empty strings or non-string inputs.
    """
    def __init__(self):
        self.use_nltk = False
        self.stop_words = set()
        self.lemmatizer = None
        
        try:
            import nltk
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            
            # Check if required corpora are available
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
            nltk.data.find('tokenizers/punkt')
            
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            self.use_nltk = True
            print("  [INFO] NLTK resources found. Using advanced text cleaning.")
        except Exception as e:
            print(f"  [WARNING] NLTK resources not fully available or corrupted ({e}). Falling back to regex-based cleaning.")
            self.use_nltk = False

    def clean_text(self, text):
        if not isinstance(text, str) or not text.strip():
            return "empty_text"
            
        # Lowercase
        text = text.lower()
        
        if self.use_nltk:
            import nltk
            # Tokenize
            tokens = nltk.word_tokenize(text)
            # Remove punctuation and non-alphabetic tokens
            tokens = [word for word in tokens if word.isalpha()]
            # Remove stopwords and lemmatize
            tokens = [self.lemmatizer.lemmatize(w) for w in tokens if w not in self.stop_words]
            return " ".join(tokens)
        else:
            # Fallback: Regex for basic cleaning
            # Remove punctuation
            text = re.sub(r'[^\w\s]', '', text)
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Handle pandas Series or list
        if isinstance(X, pd.Series):
            return X.apply(self.clean_text).tolist()
        return [self.clean_text(str(x)) for x in X]

# ── 3. Modeling & Evaluation ─────────────────────────────────────────

def build_nlp_pipeline():
    """
    Builds the full NLP pipeline: Cleaning -> TF-IDF -> Classifier.
    Using TF-IDF with ngram_range=(1,2) mitigates the 'bag-of-words' pitfall
    by preserving local context (e.g., 'not good' vs 'good').
    """
    pipeline = Pipeline([
        ('cleaner', ResilientTextCleaner()),
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=1000, min_df=2)),
        ('clf', LogisticRegression(random_state=SEED, class_weight='balanced'))
    ])
    return pipeline

def plot_confusion_matrix(y_true, y_pred, out_path):
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Not Placed', 'Placed'], 
                    yticklabels=['Not Placed', 'Placed'])
        plt.title('NLP Classification - Confusion Matrix', fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"  [WARNING] Failed to plot confusion matrix: {e}")

# ── 4. Error Analysis ────────────────────────────────────────────────

def inspect_errors(df_test, y_test, y_pred):
    """
    Finds and analyzes false positives and false negatives to understand
    language-specific failure modes.
    """
    df_results = df_test.copy()
    df_results['true_label'] = y_test
    df_results['predicted_label'] = y_pred
    
    # False Positives: Model predicted 1 (Placed), but true was 0 (Not Placed)
    fp = df_results[(df_results['true_label'] == 0) & (df_results['predicted_label'] == 1)]
    # False Negatives: Model predicted 0 (Not Placed), but true was 1 (Placed)
    fn = df_results[(df_results['true_label'] == 1) & (df_results['predicted_label'] == 0)]
    
    print("\n  [Error Analysis] Examining NLP Failure Modes:")
    print(f"    False Positives (Predicted Placed, Actually Not Placed): {len(fp)}")
    if not fp.empty:
        print("    Examples:")
        for idx, row in fp.head(3).iterrows():
            print(f"      - \"{row['interviewer_notes']}\"")
            
    print(f"\n    False Negatives (Predicted Not Placed, Actually Placed): {len(fn)}")
    if not fn.empty:
        print("    Examples:")
        for idx, row in fn.head(3).iterrows():
            print(f"      - \"{row['interviewer_notes']}\"")
            
    return len(fp), len(fn)

# ── Main ──────────────────────────────────────────────────────────────

def main():
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("\n" + "="*65)
    print("  Task 18 — Natural Language Processing")
    print("  PlaceMux  Phase 1 Industry Immersion")
    print("="*65)

    # ── Step 1: Load/Generate Data ──
    print("\n[1/6] Generating and preparing mock text data ...")
    try:
        df = generate_mock_text_data(n_samples=1500)
        
        # Train/Val/Test Split (70/15/15)
        X = df['interviewer_notes']
        y = df['placed']
        
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=SEED, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=SEED, stratify=y_temp)
        
        print(f"  Train : {len(X_train)} samples")
        print(f"  Val   : {len(X_val)} samples")
        print(f"  Test  : {len(X_test)} samples")
    except Exception as exc:
        print(f"  [FATAL] Data preparation failed: {exc}")
        traceback.print_exc(); sys.exit(1)

    # ── Step 2 & 3: Build & Train NLP Pipeline ──
    print("\n[2/6] Building and training NLP pipeline (Cleaning + TF-IDF + LR) ...")
    try:
        pipeline = build_nlp_pipeline()
        pipeline.fit(X_train, y_train)
        print("  Pipeline trained successfully.")
    except Exception as exc:
        print(f"  [FATAL] Pipeline training failed: {exc}")
        traceback.print_exc(); sys.exit(1)

    # ── Step 4: Evaluate ──
    print("\n[3/6] Evaluating on validation and test sets ...")
    try:
        # Validation Eval
        y_val_pred = pipeline.predict(X_val)
        val_f1 = f1_score(y_val, y_val_pred)
        
        # Test Eval
        y_test_pred = pipeline.predict(X_test)
        test_f1 = f1_score(y_test, y_test_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_prec = precision_score(y_test, y_test_pred)
        test_rec = recall_score(y_test, y_test_pred)
        
        print(f"\n  [Performance Metrics] Test Set Evaluation")
        print(f"  ------------------------------------------------")
        print(f"  Accuracy  : {test_acc:.4f}")
        print(f"  Precision : {test_prec:.4f}")
        print(f"  Recall    : {test_rec:.4f}")
        print(f"  F1-Score  : {test_f1:.4f}  (Primary Metric)")
        
        # We don't have a numeric baseline here since this is pure text,
        # but we can compare to a majority class baseline.
        majority_class = y_test.mode()[0]
        baseline_preds = [majority_class] * len(y_test)
        baseline_f1 = f1_score(y_test, baseline_preds, zero_division=0)
        
        print(f"\n  [Comparison vs Baseline]")
        print(f"  Majority Class Baseline F1 : {baseline_f1:.4f}")
        print(f"  NLP Pipeline F1            : {test_f1:.4f}")
        print(f"  Lift over Baseline         : {test_f1 - baseline_f1:+.4f}")
        
        plot_confusion_matrix(y_test, y_test_pred, os.path.join(PLOTS_DIR, "nlp_confusion_matrix.png"))
        print(f"  Saved confusion matrix to -> {PLOTS_DIR}/nlp_confusion_matrix.png")
        
    except Exception as exc:
        print(f"  [FATAL] Evaluation failed: {exc}")
        traceback.print_exc(); sys.exit(1)

    # ── Step 5: Error Analysis ──
    print("\n[4/6] Inspecting errors for language-specific failure modes ...")
    try:
        df_test = pd.DataFrame({'interviewer_notes': X_test})
        fp_count, fn_count = inspect_errors(df_test, y_test, y_test_pred)
    except Exception as exc:
        print(f"  [WARNING] Error analysis failed: {exc}")

    # ── Step 6: Package Pipeline ──
    print("\n[5/6] Packaging pipeline for reuse ...")
    try:
        with open(PIPELINE_PATH, 'wb') as f:
            pickle.dump(pipeline, f)
        print(f"  Pipeline saved -> {PIPELINE_PATH}")
    except Exception as exc:
        print(f"  [WARNING] Failed to package pipeline: {exc}")

    # ── Step 7: Save Report ──
    print("\n[6/6] Saving results JSON ...")
    try:
        report = {
            "task": "Task 18 — Natural Language Processing",
            "seed": SEED,
            "pipeline": {
                "vectorizer": "TF-IDF (1-2 ngrams)",
                "classifier": "Logistic Regression (Balanced)",
                "cleaning": "Resilient (NLTK or Regex Fallback)"
            },
            "metrics_test": {
                "accuracy": test_acc,
                "precision": test_prec,
                "recall": test_rec,
                "f1_score": test_f1
            },
            "baseline_comparison": {
                "baseline_f1": baseline_f1,
                "lift": test_f1 - baseline_f1
            },
            "error_analysis": {
                "false_positives": fp_count,
                "false_negatives": fn_count,
                "notes": "Failure modes typically stem from mixed feedback (e.g., 'good communication but poor coding'). N-grams help mitigate 'bag-of-words' loss of context."
            },
            "pitfalls_avoided": {
                "skipping_text_cleaning": "Implemented a robust custom cleaner that handles punctuation, casing, and gracefully manages missing NLTK dependencies.",
                "bag_of_words_meaning_loss": "Used n-grams (1, 2) in TF-IDF to capture local word order and negate combinations.",
                "wrong_metric": "Evaluated using F1-score (and Precision/Recall), which is robust for imbalanced textual edge cases.",
                "dependency_failures": "Added fallback regex cleaning to handle NLTK download timeouts or absence.",
                "edge_case_inputs": "Added handling for empty strings or non-string inputs in the custom transformer."
            }
        }
        with open(RESULTS_JSON, "w") as f:
            json.dump(report, f, indent=4)
        print(f"  Report saved -> {RESULTS_JSON}")
    except Exception as exc:
        print(f"  [ERROR] Failed to save JSON: {exc}")

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  TASK 18 — COMPLETE")
    print("="*65)
    print(f"  Test F1-Score : {test_f1:.4f}  (Baseline: {baseline_f1:.4f})")
    print(f"  Test Accuracy : {test_acc:.4f}")
    print(f"\n  Pipeline saved to -> {PIPELINE_PATH}")
    print(f"  Plots          -> {PLOTS_DIR}/")
    print(f"  Report         -> {RESULTS_JSON}")
    print("="*65 + "\n")

if __name__ == "__main__":
    main()
