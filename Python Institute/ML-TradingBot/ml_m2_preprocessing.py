#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MACHINE LEARNING - MODULE 2                               ║
║                    DATA PREPROCESSING & FEATURE ENGINEERING                   ║
║                    ML-M2: 20% del percorso ML                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

SYLLABUS ML MODULE 2:
├── ML 2.1 - Feature Scaling (StandardScaler, MinMaxScaler, Normalizer)
├── ML 2.2 - Encoding Categoricals (LabelEncoder, OneHotEncoder, get_dummies)
├── ML 2.3 - Feature Engineering (creation, transformation, binning)
├── ML 2.4 - Train/Test Split (stratification, random_state)
├── ML 2.5 - Handling Imbalanced Data (oversampling, undersampling, SMOTE)
└── ML 2.6 - Feature Selection (correlation, variance, SelectKBest)

PREREQUISITI: ML Module 1 completato
TEMPO STIMATO: 2 settimane (2-3 ore/giorno)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold

# ══════════════════════════════════════════════════════════════════════════════
# ML 2.1 - FEATURE SCALING
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("ML 2.1 - FEATURE SCALING")
print("=" * 70)

print("""
📋 PERCHÉ SCALARE?

Molti algoritmi ML sono sensibili alla scala delle features:
- Gradient Descent converge più velocemente
- Distance-based algorithms (KNN, SVM) funzionano meglio
- Regularization è più equa

METODI PRINCIPALI:
┌─────────────────────┬─────────────────────────────────────────┐
│ StandardScaler      │ z = (x - mean) / std                    │
│                     │ Media=0, Std=1                          │
├─────────────────────┼─────────────────────────────────────────┤
│ MinMaxScaler        │ x' = (x - min) / (max - min)            │
│                     │ Range [0, 1]                            │
├─────────────────────┼─────────────────────────────────────────┤
│ RobustScaler        │ Usa mediana e IQR                       │
│                     │ Resistente agli outliers                │
└─────────────────────┴─────────────────────────────────────────┘
""")

# Create sample data
np.random.seed(42)
data = pd.DataFrame({
    'age': np.random.randint(18, 70, 100),
    'salary': np.random.randint(20000, 150000, 100),
    'experience': np.random.randint(0, 40, 100)
})

print("🔹 DATI ORIGINALI:")
print(data.describe().round(2))

# StandardScaler
print("\n🔹 STANDARD SCALER (z-score):")
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
df_scaled = pd.DataFrame(data_scaled, columns=data.columns)
print(df_scaled.describe().round(2))

# MinMaxScaler
print("\n🔹 MINMAX SCALER (0-1):")
minmax = MinMaxScaler()
data_minmax = minmax.fit_transform(data)
df_minmax = pd.DataFrame(data_minmax, columns=data.columns)
print(df_minmax.describe().round(2))

print("""
⚠️ IMPORTANTE: fit_transform vs transform

# Durante TRAINING:
scaler.fit_transform(X_train)  # Calcola parametri E trasforma

# Durante TESTING:
scaler.transform(X_test)  # USA parametri già calcolati, solo trasforma

# MAI fare fit su test data! Causa data leakage.
""")

# ══════════════════════════════════════════════════════════════════════════════
# ML 2.2 - ENCODING CATEGORICALS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ML 2.2 - ENCODING CATEGORICALS")
print("=" * 70)

print("""
📋 PERCHÉ ENCODARE?

ML algorithms lavorano con numeri, non stringhe.
Dobbiamo convertire variabili categoriche in numeriche.

METODI:
┌─────────────────────┬─────────────────────────────────────────┐
│ LabelEncoder        │ Converte in interi (0, 1, 2, ...)       │
│                     │ Usare per target o variabili ordinali   │
├─────────────────────┼─────────────────────────────────────────┤
│ OneHotEncoder       │ Crea colonne binarie per ogni categoria │
│                     │ Usare per features nominali             │
├─────────────────────┼─────────────────────────────────────────┤
│ pd.get_dummies()    │ Versione Pandas di OneHot               │
│                     │ Più semplice da usare                   │
└─────────────────────┴─────────────────────────────────────────┘
""")

# Sample data with categories
df_cat = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red', 'blue'],
    'size': ['S', 'M', 'L', 'M', 'S'],
    'price': [10, 20, 30, 15, 25]
})
print("🔹 DATI ORIGINALI:")
print(df_cat)

# LabelEncoder
print("\n🔹 LABEL ENCODER:")
le = LabelEncoder()
df_cat['color_encoded'] = le.fit_transform(df_cat['color'])
print(df_cat[['color', 'color_encoded']])
print(f"Mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")

# get_dummies (OneHot semplificato)
print("\n🔹 ONE-HOT ENCODING (pd.get_dummies):")
df_onehot = pd.get_dummies(df_cat[['color', 'size']], prefix=['color', 'size'])
print(df_onehot)

print("""
⚠️ QUANDO USARE COSA:

LabelEncoder:
  - Target variable (y)
  - Variabili ORDINALI (S < M < L)

OneHotEncoder / get_dummies:
  - Features NOMINALI (no ordine: red, blue, green)
  - Evita che il modello pensi che 2 > 1 > 0

drop_first=True:
  - Evita multicollinearità (dummy variable trap)
  - pd.get_dummies(df, drop_first=True)
""")

# ══════════════════════════════════════════════════════════════════════════════
# ML 2.3 - FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ML 2.3 - FEATURE ENGINEERING")
print("=" * 70)

print("""
📋 FEATURE ENGINEERING

L'arte di creare nuove features dai dati esistenti.
Spesso più importante della scelta del modello!
""")

# Sample dataset
df_fe = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=100),
    'price': np.random.uniform(100, 200, 100),
    'quantity': np.random.randint(1, 50, 100)
})

print("🔹 FEATURE CREATION:")
# Combine features
df_fe['revenue'] = df_fe['price'] * df_fe['quantity']
print(f"  revenue = price * quantity")

# Extract from datetime
df_fe['year'] = df_fe['date'].dt.year
df_fe['month'] = df_fe['date'].dt.month
df_fe['day_of_week'] = df_fe['date'].dt.dayofweek
df_fe['is_weekend'] = df_fe['day_of_week'].isin([5, 6]).astype(int)
print(f"  Estratto: year, month, day_of_week, is_weekend")

print("\n🔹 BINNING (Discretization):")
df_fe['price_category'] = pd.cut(df_fe['price'], 
                                  bins=[0, 120, 160, 200],
                                  labels=['low', 'medium', 'high'])
print(df_fe[['price', 'price_category']].head(10))

print("\n🔹 LOG TRANSFORMATION:")
# Per distribuzioni skewed
df_fe['log_revenue'] = np.log1p(df_fe['revenue'])  # log(1+x) per gestire 0
print(f"  log_revenue = log(1 + revenue)")

print("\n🔹 POLYNOMIAL FEATURES:")
print("""
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
# Crea: x1, x2, x1², x2², x1*x2
""")

# ══════════════════════════════════════════════════════════════════════════════
# ML 2.4 - TRAIN/TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ML 2.4 - TRAIN/TEST SPLIT")
print("=" * 70)

print("""
📋 PERCHÉ DIVIDERE?

- Training set: per addestrare il modello
- Test set: per valutare le performance su dati MAI visti
- Evita OVERFITTING (memorizzare invece di generalizzare)

SPLIT TIPICI:
- 80/20 (più comune)
- 70/30
- 60/20/20 (train/validation/test)
""")

# Create sample data
X = np.random.randn(1000, 5)
y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Binary classification

print("🔹 BASIC SPLIT:")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  X_train shape: {X_train.shape}")
print(f"  X_test shape: {X_test.shape}")
print(f"  y_train distribution: {np.bincount(y_train)}")
print(f"  y_test distribution: {np.bincount(y_test)}")

print("\n🔹 STRATIFIED SPLIT (per classification):")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  y_train distribution: {np.bincount(y_train)}")
print(f"  y_test distribution: {np.bincount(y_test)}")
print("  (Proporzioni mantenute!)")

print("""
⚠️ PARAMETRI IMPORTANTI:

test_size: proporzione test set (0.2 = 20%)
random_state: seed per riproducibilità (SEMPRE impostarlo!)
stratify: mantiene proporzioni classi (USARE per classification)
shuffle: mescola dati prima di dividere (default=True)
""")

# ══════════════════════════════════════════════════════════════════════════════
# ML 2.5 - HANDLING IMBALANCED DATA
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ML 2.5 - HANDLING IMBALANCED DATA")
print("=" * 70)

print("""
📋 PROBLEMA: DATI SBILANCIATI

Esempio: Fraud detection (99% normale, 1% frode)
Il modello può predire sempre "normale" e avere 99% accuracy!

SOLUZIONI:
┌─────────────────────┬─────────────────────────────────────────┐
│ Undersampling       │ Riduci classe maggioritaria             │
│                     │ Pro: veloce. Con: perdi informazioni    │
├─────────────────────┼─────────────────────────────────────────┤
│ Oversampling        │ Duplica classe minoritaria              │
│                     │ Pro: mantieni dati. Con: overfitting    │
├─────────────────────┼─────────────────────────────────────────┤
│ SMOTE               │ Crea dati sintetici per classe minore   │
│                     │ Pro: dati nuovi. Con: può creare noise  │
├─────────────────────┼─────────────────────────────────────────┤
│ Class weights       │ Penalizza errori su classe minore       │
│                     │ Pro: no modifica dati                   │
└─────────────────────┴─────────────────────────────────────────┘
""")

# Imbalanced example
y_imb = np.array([0]*950 + [1]*50)
print(f"🔹 Dataset sbilanciato: {np.bincount(y_imb)}")

print("""
# SMOTE (Synthetic Minority Over-sampling Technique)
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Class weights in models
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(class_weight='balanced')
# Oppure: class_weight={0: 1, 1: 10}
""")

# ══════════════════════════════════════════════════════════════════════════════
# ML 2.6 - FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ML 2.6 - FEATURE SELECTION")
print("=" * 70)

print("""
📋 PERCHÉ SELEZIONARE FEATURES?

- Ridurre overfitting
- Migliorare performance
- Ridurre tempo training
- Interpretabilità

METODI:
┌─────────────────────┬─────────────────────────────────────────┐
│ Correlation         │ Rimuovi features altamente correlate    │
├─────────────────────┼─────────────────────────────────────────┤
│ VarianceThreshold   │ Rimuovi features con bassa varianza     │
├─────────────────────┼─────────────────────────────────────────┤
│ SelectKBest         │ Seleziona K migliori per score          │
├─────────────────────┼─────────────────────────────────────────┤
│ RFE                 │ Recursive Feature Elimination           │
├─────────────────────┼─────────────────────────────────────────┤
│ Feature Importance  │ Da modelli tree-based                   │
└─────────────────────┴─────────────────────────────────────────┘
""")

# Generate sample data
np.random.seed(42)
X_sel = np.random.randn(100, 10)
y_sel = (X_sel[:, 0] + X_sel[:, 1] * 2 + X_sel[:, 2] * 0.5 > 0).astype(int)

print("🔹 VARIANCE THRESHOLD:")
selector = VarianceThreshold(threshold=0.5)
X_var = selector.fit_transform(X_sel)
print(f"  Features prima: {X_sel.shape[1]}")
print(f"  Features dopo: {X_var.shape[1]}")

print("\n🔹 SELECT K BEST:")
selector = SelectKBest(f_classif, k=5)
X_best = selector.fit_transform(X_sel, y_sel)
print(f"  Features prima: {X_sel.shape[1]}")
print(f"  Features dopo: {X_best.shape[1]}")
print(f"  Scores: {selector.scores_.round(2)}")

print("""
🔹 CORRELATION-BASED:

corr_matrix = df.corr().abs()
upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
df_reduced = df.drop(columns=to_drop)
""")

# ══════════════════════════════════════════════════════════════════════════════
# QUIZ DI VERIFICA
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("QUIZ DI VERIFICA - ML MODULE 2")
print("=" * 70)

print("""
Q1. StandardScaler produce dati con?
    A) Range 0-1  B) Media=0, Std=1  C) Mediana=0  D) Min=-1, Max=1
    → RISPOSTA: B

Q2. Quando usare LabelEncoder?
    A) Features nominali  B) Target variable  C) Sempre  D) Mai
    → RISPOSTA: B

Q3. get_dummies(drop_first=True) serve per?
    A) Velocità  B) Evitare multicollinearità  C) Ridurre memoria  D) Nulla
    → RISPOSTA: B

Q4. random_state in train_test_split serve per?
    A) Velocità  B) Riproducibilità  C) Stratificazione  D) Shuffle
    → RISPOSTA: B

Q5. stratify=y serve per?
    A) Velocità  B) Shuffle  C) Mantenere proporzioni classi  D) Random
    → RISPOSTA: C

Q6. SMOTE serve per?
    A) Feature scaling  B) Encoding  C) Dati sbilanciati  D) Feature selection
    → RISPOSTA: C

Q7. SelectKBest seleziona features basandosi su?
    A) Varianza  B) Correlazione  C) Statistical tests  D) Random
    → RISPOSTA: C

Q8. fit_transform su test data è?
    A) Corretto  B) Data leakage  C) Necessario  D) Opzionale
    → RISPOSTA: B (ERRORE! Mai fit su test)

Q9. MinMaxScaler produce range?
    A) -1 a 1  B) 0 a 1  C) Media 0  D) Qualsiasi
    → RISPOSTA: B

Q10. Polynomial features degree=2 su 2 features produce?
    A) 2 features  B) 4 features  C) 5 features  D) 6 features
    → RISPOSTA: C (x1, x2, x1², x2², x1*x2)
""")

# ══════════════════════════════════════════════════════════════════════════════
# ESERCIZI PRATICI
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ESERCIZI PRATICI")
print("=" * 70)

print("""
📝 ESERCIZIO 1: Pipeline Preprocessing
Crea una pipeline che:
1. Scala features numeriche (StandardScaler)
2. Encoda features categoriche (OneHot)
3. Seleziona top 10 features (SelectKBest)

📝 ESERCIZIO 2: Feature Engineering
Dato un dataset con colonna 'datetime':
- Estrai: anno, mese, giorno, ora, giorno_settimana
- Crea: is_weekend, is_morning, quarter

📝 ESERCIZIO 3: Imbalanced Data
Dato un dataset con 95% classe 0 e 5% classe 1:
- Applica SMOTE
- Confronta distribuzione prima/dopo
- Allena modello con class_weight='balanced'

📝 ESERCIZIO 4: Feature Selection
Dato un dataset con 50 features:
- Rimuovi features con correlazione > 0.9
- Seleziona top 20 con SelectKBest
- Confronta performance con tutte le features
""")

print("\n" + "=" * 70)
print("ML MODULE 2 - PREPROCESSING COMPLETATO!")
print("Prossimo: ML MODULE 3 - SUPERVISED LEARNING")
print("=" * 70)
