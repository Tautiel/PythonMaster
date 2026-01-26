#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MACHINE LEARNING - MODULE 4                               ║
║                    MODEL TUNING & VALIDATION                                  ║
║                    ML-M4: 15% del percorso ML                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

SYLLABUS ML MODULE 4:
├── ML 4.1 - Cross-Validation (K-Fold, Stratified, Leave-One-Out)
├── ML 4.2 - Hyperparameter Tuning (GridSearchCV, RandomizedSearchCV)
├── ML 4.3 - Overfitting vs Underfitting (bias-variance tradeoff)
├── ML 4.4 - Learning Curves & Validation Curves
├── ML 4.5 - Model Selection & Comparison
└── ML 4.6 - Pipelines (sklearn.pipeline)

PREREQUISITI: ML Module 1-3 completati
TEMPO STIMATO: 2 settimane (2-3 ore/giorno)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (cross_val_score, KFold, StratifiedKFold, 
                                      GridSearchCV, RandomizedSearchCV,
                                      learning_curve, validation_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# ══════════════════════════════════════════════════════════════════════════════
# ML 4.1 - CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("ML 4.1 - CROSS-VALIDATION")
print("=" * 70)

print("""
📋 PERCHÉ CROSS-VALIDATION?

Un solo train/test split può dare risultati fortunati/sfortunati.
CV usa MULTIPLE splits per una stima più robusta.

TIPI:
┌─────────────────────┬─────────────────────────────────────────┐
│ K-Fold              │ Divide in K parti, usa K-1 per train    │
│                     │ Ogni parte è test una volta             │
├─────────────────────┼─────────────────────────────────────────┤
│ Stratified K-Fold   │ Come K-Fold ma mantiene proporzioni     │
│                     │ USARE per classification!               │
├─────────────────────┼─────────────────────────────────────────┤
│ Leave-One-Out       │ K = numero campioni                     │
│                     │ Costoso, per dataset piccoli            │
├─────────────────────┼─────────────────────────────────────────┤
│ Time Series Split   │ Per dati temporali (no shuffle)         │
└─────────────────────┴─────────────────────────────────────────┘
""")

# Generate data
np.random.seed(42)
X = np.random.randn(200, 5)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

print("🔹 BASIC CROSS-VALIDATION:")
model = LogisticRegression(random_state=42)
scores = cross_val_score(model, X, y, cv=5)
print(f"  5-Fold CV scores: {scores.round(4)}")
print(f"  Mean: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

print("\n🔹 STRATIFIED K-FOLD (manuale):")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))
print(f"  Scores: {np.array(scores).round(4)}")

print("""
📋 SCORING PARAMETER:

cross_val_score(model, X, y, cv=5, scoring='accuracy')

CLASSIFICAZIONE: 'accuracy', 'precision', 'recall', 'f1', 'roc_auc'
REGRESSIONE: 'r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'

Nota: 'neg_' perché sklearn massimizza sempre
""")

# ══════════════════════════════════════════════════════════════════════════════
# ML 4.2 - HYPERPARAMETER TUNING
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ML 4.2 - HYPERPARAMETER TUNING")
print("=" * 70)

print("""
📋 HYPERPARAMETERS vs PARAMETERS

Parameters: appresi dal modello (coefficienti, weights)
Hyperparameters: impostati PRIMA del training (learning rate, max_depth)

METODI DI TUNING:
┌─────────────────────┬─────────────────────────────────────────┐
│ GridSearchCV        │ Prova TUTTE le combinazioni             │
│                     │ Esaustivo ma costoso                    │
├─────────────────────┼─────────────────────────────────────────┤
│ RandomizedSearchCV  │ Prova N combinazioni random             │
│                     │ Più veloce, buono per molti HP          │
├─────────────────────┼─────────────────────────────────────────┤
│ Bayesian Optim.     │ Usa risultati precedenti per guidare    │
│ (Optuna, etc.)      │ Più efficiente                          │
└─────────────────────┴─────────────────────────────────────────┘
""")

print("🔹 GRID SEARCH CV:")
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1  # Usa tutti i CPU
)
grid_search.fit(X, y)

print(f"  Best params: {grid_search.best_params_}")
print(f"  Best score: {grid_search.best_score_:.4f}")
print(f"  Total fits: {len(param_grid['C']) * len(param_grid['kernel']) * len(param_grid['gamma']) * 5}")

print("\n🔹 RANDOMIZED SEARCH CV:")
from scipy.stats import uniform, randint

param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(3, 15),
    'min_samples_split': randint(2, 10)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=20,  # Solo 20 combinazioni random
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)
random_search.fit(X, y)

print(f"  Best params: {random_search.best_params_}")
print(f"  Best score: {random_search.best_score_:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# ML 4.3 - OVERFITTING VS UNDERFITTING
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ML 4.3 - OVERFITTING VS UNDERFITTING")
print("=" * 70)

print("""
📋 BIAS-VARIANCE TRADEOFF

UNDERFITTING (High Bias):
  - Modello troppo semplice
  - Train error ALTO, Test error ALTO
  - Soluzioni: più features, modello più complesso

OVERFITTING (High Variance):
  - Modello memorizza training data
  - Train error BASSO, Test error ALTO
  - Soluzioni: più dati, regularization, modello più semplice

SWEET SPOT:
  - Train error e Test error simili e bassi

DIAGNOSI:
┌─────────────────────┬───────────────┬───────────────┐
│                     │ Train Error   │ Test Error    │
├─────────────────────┼───────────────┼───────────────┤
│ Underfitting        │ Alto          │ Alto          │
│ Overfitting         │ Basso         │ Alto          │
│ Good Fit            │ Basso         │ Basso         │
└─────────────────────┴───────────────┴───────────────┘
""")

# Demonstrate overfitting
from sklearn.tree import DecisionTreeClassifier

print("🔹 ESEMPIO OVERFITTING:")
X_train, X_test = X[:160], X[160:]
y_train, y_test = y[:160], y[160:]

# Overfit model
dt_overfit = DecisionTreeClassifier(max_depth=None, random_state=42)
dt_overfit.fit(X_train, y_train)
print(f"  Deep Tree - Train acc: {dt_overfit.score(X_train, y_train):.4f}")
print(f"  Deep Tree - Test acc: {dt_overfit.score(X_test, y_test):.4f}")

# Regularized model
dt_regular = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_regular.fit(X_train, y_train)
print(f"  Shallow Tree - Train acc: {dt_regular.score(X_train, y_train):.4f}")
print(f"  Shallow Tree - Test acc: {dt_regular.score(X_test, y_test):.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# ML 4.4 - LEARNING CURVES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ML 4.4 - LEARNING & VALIDATION CURVES")
print("=" * 70)

print("""
📋 LEARNING CURVE

Mostra come performance cambia con SIZE del training set.

INTERPRETAZIONE:
- Gap grande tra train/test → Overfitting (serve più dati)
- Entrambi alti → Underfitting (serve modello diverso)
- Convergono in basso → Good fit

📋 VALIDATION CURVE

Mostra come performance cambia con un HYPERPARAMETER.
Aiuta a trovare il valore ottimale.
""")

print("🔹 LEARNING CURVE:")
train_sizes, train_scores, test_scores = learning_curve(
    LogisticRegression(random_state=42),
    X, y,
    train_sizes=np.linspace(0.1, 1.0, 5),
    cv=5,
    scoring='accuracy'
)

print(f"  Train sizes: {train_sizes}")
print(f"  Train scores (mean): {train_scores.mean(axis=1).round(4)}")
print(f"  Test scores (mean): {test_scores.mean(axis=1).round(4)}")

print("\n🔹 VALIDATION CURVE:")
param_range = [1, 3, 5, 7, 10]
train_scores, test_scores = validation_curve(
    DecisionTreeClassifier(random_state=42),
    X, y,
    param_name='max_depth',
    param_range=param_range,
    cv=5,
    scoring='accuracy'
)

print(f"  max_depth values: {param_range}")
print(f"  Train scores: {train_scores.mean(axis=1).round(4)}")
print(f"  Test scores: {test_scores.mean(axis=1).round(4)}")

# ══════════════════════════════════════════════════════════════════════════════
# ML 4.5 - MODEL SELECTION
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ML 4.5 - MODEL SELECTION & COMPARISON")
print("=" * 70)

print("""
📋 COME SCEGLIERE IL MODELLO?

1. Cross-validation su più modelli
2. Confronta mean score E standard deviation
3. Considera anche: tempo training, interpretabilità, requisiti

WORKFLOW:
1. Prova modelli base con default hyperparameters
2. Seleziona top 2-3 modelli
3. Tune hyperparameters dei migliori
4. Valutazione finale su test set tenuto da parte
""")

print("🔹 CONFRONTO MODELLI:")
models = {
    'Logistic': LogisticRegression(random_state=42),
    'SVM': SVC(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42)
}

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    results[name] = {'mean': scores.mean(), 'std': scores.std()}
    print(f"  {name:15s}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

# ══════════════════════════════════════════════════════════════════════════════
# ML 4.6 - PIPELINES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ML 4.6 - PIPELINES")
print("=" * 70)

print("""
📋 SKLEARN PIPELINES

Catena di trasformazioni + modello finale.
ESSENZIALE per evitare data leakage!

VANTAGGI:
- Codice più pulito
- Evita data leakage (fit solo su train)
- Facile da usare con GridSearchCV
- Riproducibilità
""")

print("🔹 BASIC PIPELINE:")
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(random_state=42))
])

scores = cross_val_score(pipeline, X, y, cv=5)
print(f"  Pipeline CV: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

print("\n🔹 PIPELINE + GRID SEARCH:")
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(random_state=42))
])

# Nota: usa 'clf__C' per accedere a parametri del classifier
param_grid = {
    'clf__C': [0.1, 1, 10],
    'clf__kernel': ['linear', 'rbf']
}

grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X, y)
print(f"  Best params: {grid.best_params_}")
print(f"  Best score: {grid.best_score_:.4f}")

print("""
📋 COLUMN TRANSFORMER (per mixed data):

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(), categorical_features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])
""")

# ══════════════════════════════════════════════════════════════════════════════
# QUIZ DI VERIFICA
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("QUIZ DI VERIFICA - ML MODULE 4")
print("=" * 70)

print("""
Q1. 5-Fold CV divide i dati in?
    A) 5 train sets  B) 5 parti, ognuna è test una volta  C) 50%  D) Random
    → RISPOSTA: B

Q2. Stratified K-Fold serve per?
    A) Velocità  B) Mantenere proporzioni classi  C) Più dati  D) Meno folds
    → RISPOSTA: B

Q3. GridSearchCV prova?
    A) Combinazioni random  B) Tutte le combinazioni  C) Solo le migliori  D) Una
    → RISPOSTA: B

Q4. Train error basso, Test error alto indica?
    A) Underfitting  B) Overfitting  C) Good fit  D) Errore
    → RISPOSTA: B

Q5. Per overfitting, cosa aiuta?
    A) Più features  B) Modello più complesso  C) Regularization  D) Meno dati
    → RISPOSTA: C

Q6. Learning curve mostra performance vs?
    A) Hyperparameter  B) Training size  C) Tempo  D) Features
    → RISPOSTA: B

Q7. Pipeline evita?
    A) Overfitting  B) Data leakage  C) Underfitting  D) Tutte
    → RISPOSTA: B

Q8. 'clf__C' in GridSearch accede a?
    A) Parametro del pipeline  B) Parametro del classificatore  C) CV  D) Errore
    → RISPOSTA: B

Q9. RandomizedSearchCV è preferibile quando?
    A) Pochi HP  B) Molti HP  C) Mai  D) Sempre
    → RISPOSTA: B

Q10. cross_val_score restituisce?
    A) Un numero  B) Array di scores  C) Modello  D) Best params
    → RISPOSTA: B
""")

print("\n" + "=" * 70)
print("ML MODULE 4 - MODEL TUNING COMPLETATO!")
print("Prossimo: ML MODULE 5 - DEEP LEARNING BASICS")
print("=" * 70)
