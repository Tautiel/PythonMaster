#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MACHINE LEARNING - MODULE 3                               ║
║                    SUPERVISED LEARNING                                        ║
║                    ML-M3: 30% del percorso ML (BIGGEST!)                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

SYLLABUS ML MODULE 3:
├── ML 3.1 - Linear Regression (simple, multiple, polynomial)
├── ML 3.2 - Logistic Regression (binary, multiclass)
├── ML 3.3 - Decision Trees (classification, regression)
├── ML 3.4 - Random Forest & Ensemble Methods
├── ML 3.5 - Support Vector Machines (SVM)
├── ML 3.6 - K-Nearest Neighbors (KNN)
└── ML 3.7 - Model Evaluation Metrics

PREREQUISITI: ML Module 1-2 completati
TEMPO STIMATO: 3-4 settimane (2-3 ore/giorno)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            confusion_matrix, classification_report,
                            mean_squared_error, mean_absolute_error, r2_score)

# ══════════════════════════════════════════════════════════════════════════════
# ML 3.1 - LINEAR REGRESSION
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("ML 3.1 - LINEAR REGRESSION")
print("=" * 70)

print("""
📋 LINEAR REGRESSION

Predice un valore CONTINUO (regressione).
Formula: y = β₀ + β₁x₁ + β₂x₂ + ... + ε

TIPI:
- Simple: una feature
- Multiple: multiple features
- Polynomial: features non-lineari

ASSUNZIONI:
1. Linearità
2. Indipendenza errori
3. Omoschedasticità (varianza costante)
4. Normalità residui
""")

# Generate sample data
np.random.seed(42)
X = np.random.randn(200, 3)
y = 3*X[:, 0] + 2*X[:, 1] - X[:, 2] + np.random.randn(200)*0.5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🔹 LINEAR REGRESSION:")
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print(f"  Coefficients: {lr.coef_.round(3)}")
print(f"  Intercept: {lr.intercept_:.3f}")
print(f"  R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"  MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"  MAE: {mean_absolute_error(y_test, y_pred):.4f}")

print("\n🔹 REGULARIZATION:")
print("""
Ridge (L2): penalizza coefficienti grandi
  → Previene overfitting, mantiene tutte le features
  
Lasso (L1): può azzerare coefficienti
  → Feature selection automatica

ElasticNet: combina L1 e L2
""")

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
print(f"  Ridge R²: {ridge.score(X_test, y_test):.4f}")

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
print(f"  Lasso R²: {lasso.score(X_test, y_test):.4f}")
print(f"  Lasso coefficients: {lasso.coef_.round(3)}")

# ══════════════════════════════════════════════════════════════════════════════
# ML 3.2 - LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ML 3.2 - LOGISTIC REGRESSION")
print("=" * 70)

print("""
📋 LOGISTIC REGRESSION

Predice PROBABILITÀ di appartenenza a una classe.
Output: P(y=1|X) tramite funzione sigmoid.

TIPI:
- Binary: 2 classi
- Multiclass: >2 classi (one-vs-rest o multinomial)

NON è regressione, è CLASSIFICATION!
""")

# Generate classification data
np.random.seed(42)
X_clf = np.random.randn(300, 2)
y_clf = ((X_clf[:, 0] + X_clf[:, 1]) > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

print("🔹 LOGISTIC REGRESSION:")
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
y_prob = log_reg.predict_proba(X_test)[:, 1]

print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"  Coefficients: {log_reg.coef_.round(3)}")
print(f"  Sample probabilities: {y_prob[:5].round(3)}")

# ══════════════════════════════════════════════════════════════════════════════
# ML 3.3 - DECISION TREES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ML 3.3 - DECISION TREES")
print("=" * 70)

print("""
📋 DECISION TREES

Struttura ad albero che fa decisioni binarie.
Interpretabile e non richiede scaling!

PARAMETRI CHIAVE:
- max_depth: profondità massima (previene overfitting)
- min_samples_split: min campioni per split
- min_samples_leaf: min campioni per foglia
- criterion: 'gini' o 'entropy' (classification)

PRO: interpretabile, no scaling, features non-lineari
CON: overfitting facile, instabile
""")

print("🔹 DECISION TREE CLASSIFIER:")
dt_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_clf.fit(X_train, y_train)
y_pred = dt_clf.predict(X_test)
print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"  Feature importances: {dt_clf.feature_importances_.round(3)}")

print("\n🔹 DECISION TREE REGRESSOR:")
dt_reg = DecisionTreeRegressor(max_depth=5, random_state=42)
# Using regression data from before
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y, test_size=0.2, random_state=42)
dt_reg.fit(X_train_r, y_train_r)
print(f"  R² Score: {dt_reg.score(X_test_r, y_test_r):.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# ML 3.4 - RANDOM FOREST & ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ML 3.4 - RANDOM FOREST & ENSEMBLE METHODS")
print("=" * 70)

print("""
📋 ENSEMBLE METHODS

Combinano multiple modelli per performance migliori.

TIPI:
┌─────────────────────┬─────────────────────────────────────────┐
│ Bagging             │ Training parallelo su subset dati      │
│ (Random Forest)     │ Riduce varianza                        │
├─────────────────────┼─────────────────────────────────────────┤
│ Boosting            │ Training sequenziale                   │
│ (GradientBoosting,  │ Corregge errori precedenti             │
│  XGBoost, LightGBM) │ Riduce bias                            │
├─────────────────────┼─────────────────────────────────────────┤
│ Stacking            │ Meta-model su predizioni base          │
└─────────────────────┴─────────────────────────────────────────┘
""")

print("🔹 RANDOM FOREST:")
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)
print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"  Feature importances: {rf_clf.feature_importances_.round(3)}")

print("\n🔹 GRADIENT BOOSTING:")
gb_clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
gb_clf.fit(X_train, y_train)
y_pred = gb_clf.predict(X_test)
print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("""
📋 PARAMETRI RANDOM FOREST:

n_estimators: numero di alberi (più = meglio, ma più lento)
max_depth: profondità max alberi
max_features: features per split ('sqrt', 'log2', int, float)
min_samples_split: min campioni per split
bootstrap: usa bootstrap sampling (True per RF)
""")

# ══════════════════════════════════════════════════════════════════════════════
# ML 3.5 - SUPPORT VECTOR MACHINES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ML 3.5 - SUPPORT VECTOR MACHINES (SVM)")
print("=" * 70)

print("""
📋 SVM

Trova l'iperpiano che massimizza il margine tra classi.
Può usare kernel per dati non-lineari.

KERNEL:
- 'linear': dati linearmente separabili
- 'rbf': Radial Basis Function (default, versatile)
- 'poly': polinomiale

PARAMETRI:
- C: regularization (alto = meno regolarizzazione)
- gamma: influenza di singoli punti (rbf, poly)

⚠️ RICHIEDE FEATURE SCALING!
""")

print("🔹 SVM CLASSIFIER:")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_clf = SVC(kernel='rbf', C=1.0, random_state=42)
svm_clf.fit(X_train_scaled, y_train)
y_pred = svm_clf.predict(X_test_scaled)
print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\n🔹 SVM con probability:")
svm_prob = SVC(kernel='rbf', probability=True, random_state=42)
svm_prob.fit(X_train_scaled, y_train)
print(f"  Probabilities available: {hasattr(svm_prob, 'predict_proba')}")

# ══════════════════════════════════════════════════════════════════════════════
# ML 3.6 - K-NEAREST NEIGHBORS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ML 3.6 - K-NEAREST NEIGHBORS (KNN)")
print("=" * 70)

print("""
📋 KNN

Classifica basandosi sui K vicini più prossimi.
"Lazy learner": non costruisce modello, memorizza dati.

PARAMETRI:
- n_neighbors (K): numero di vicini
- weights: 'uniform' o 'distance'
- metric: 'euclidean', 'manhattan', etc.

⚠️ RICHIEDE FEATURE SCALING!
⚠️ Lento su grandi dataset (deve calcolare distanze)
""")

print("🔹 KNN CLASSIFIER:")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
print(f"  Accuracy (K=5): {accuracy_score(y_test, y_pred):.4f}")

# Test different K values
print("\n🔹 TUNING K:")
for k in [3, 5, 7, 11]:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
    print(f"  K={k}: CV accuracy = {scores.mean():.4f} (+/- {scores.std():.4f})")

# ══════════════════════════════════════════════════════════════════════════════
# ML 3.7 - MODEL EVALUATION METRICS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ML 3.7 - MODEL EVALUATION METRICS")
print("=" * 70)

print("""
📋 CLASSIFICATION METRICS:

┌─────────────────────┬─────────────────────────────────────────┐
│ Accuracy            │ (TP+TN) / Total                         │
│                     │ Quando: classi bilanciate               │
├─────────────────────┼─────────────────────────────────────────┤
│ Precision           │ TP / (TP+FP)                            │
│                     │ "Dei predetti positivi, quanti corretti"│
├─────────────────────┼─────────────────────────────────────────┤
│ Recall (Sensitivity)│ TP / (TP+FN)                            │
│                     │ "Dei veri positivi, quanti trovati"     │
├─────────────────────┼─────────────────────────────────────────┤
│ F1 Score            │ 2 * (Precision*Recall)/(Precision+Recall)│
│                     │ Media armonica                          │
├─────────────────────┼─────────────────────────────────────────┤
│ AUC-ROC             │ Area under ROC curve                    │
│                     │ Performance su tutte le soglie          │
└─────────────────────┴─────────────────────────────────────────┘
""")

# Final model evaluation
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)

print("🔹 CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

print("🔹 CONFUSION MATRIX:")
cm = confusion_matrix(y_test, y_pred)
print(f"  [[TN, FP],\n   [FN, TP]] =\n{cm}")

print("""
📋 REGRESSION METRICS:

┌─────────────────────┬─────────────────────────────────────────┐
│ MSE                 │ Mean Squared Error                      │
│                     │ Penalizza errori grandi                 │
├─────────────────────┼─────────────────────────────────────────┤
│ RMSE                │ Root MSE                                │
│                     │ Stessa unità del target                 │
├─────────────────────┼─────────────────────────────────────────┤
│ MAE                 │ Mean Absolute Error                     │
│                     │ Robusto a outliers                      │
├─────────────────────┼─────────────────────────────────────────┤
│ R²                  │ Coefficient of determination            │
│                     │ 1 = perfetto, 0 = media, <0 = peggio    │
└─────────────────────┴─────────────────────────────────────────┘
""")

# ══════════════════════════════════════════════════════════════════════════════
# QUIZ DI VERIFICA
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("QUIZ DI VERIFICA - ML MODULE 3")
print("=" * 70)

print("""
Q1. Linear Regression predice?
    A) Classi  B) Probabilità  C) Valori continui  D) Cluster
    → RISPOSTA: C

Q2. Logistic Regression è per?
    A) Regressione  B) Classification  C) Clustering  D) Tutte
    → RISPOSTA: B

Q3. Ridge regression usa regularization?
    A) L1  B) L2  C) L1+L2  D) Nessuna
    → RISPOSTA: B

Q4. Random Forest è un esempio di?
    A) Boosting  B) Bagging  C) Stacking  D) Single model
    → RISPOSTA: B

Q5. SVM richiede feature scaling?
    A) Sì sempre  B) No mai  C) Solo per kernel lineare  D) Opzionale
    → RISPOSTA: A

Q6. KNN è un?
    A) Eager learner  B) Lazy learner  C) Deep learner  D) Reinforcement
    → RISPOSTA: B

Q7. Precision misura?
    A) TP/(TP+FN)  B) TP/(TP+FP)  C) (TP+TN)/Total  D) TP/Total
    → RISPOSTA: B

Q8. F1 Score è?
    A) Media semplice  B) Media armonica  C) Media geometrica  D) Massimo
    → RISPOSTA: B

Q9. R² = 0 significa?
    A) Perfetto  B) Peggio della media  C) Come la media  D) Errore
    → RISPOSTA: C

Q10. Per dati sbilanciati, preferire?
    A) Accuracy  B) F1/Recall  C) MSE  D) R²
    → RISPOSTA: B
""")

print("\n" + "=" * 70)
print("ML MODULE 3 - SUPERVISED LEARNING COMPLETATO!")
print("Prossimo: ML MODULE 4 - MODEL TUNING & VALIDATION")
print("=" * 70)
