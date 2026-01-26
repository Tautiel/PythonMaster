#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MACHINE LEARNING - MODULE 1                               ║
║                    DATA FOUNDATIONS                                           ║
║                    ML-M1: 25% del percorso ML                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

SYLLABUS ML MODULE 1:
├── ML 1.1 - NumPy Fundamentals (arrays, operations, broadcasting)
├── ML 1.2 - Pandas DataFrames (creation, selection, filtering)
├── ML 1.3 - Data Loading (CSV, Excel, JSON, API)
├── ML 1.4 - Data Inspection (info, describe, dtypes, missing values)
├── ML 1.5 - Data Cleaning (handling nulls, duplicates, outliers)
└── ML 1.6 - Basic Statistics (mean, median, std, correlation)

PREREQUISITI: PCAP completato (OOP, file I/O)
TEMPO STIMATO: 2-3 settimane (2-3 ore/giorno)
"""

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# ML 1.1 - NUMPY FUNDAMENTALS
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("ML 1.1 - NUMPY FUNDAMENTALS")
print("=" * 70)

print("""
📋 NUMPY: Numerical Python

NumPy è la libreria fondamentale per:
- Array multidimensionali (ndarray)
- Operazioni matematiche veloci
- Broadcasting
- Linear algebra
""")

# Array creation
print("\n🔹 CREAZIONE ARRAY:")
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.zeros((3, 4))           # 3x4 di zeri
arr3 = np.ones((2, 3))            # 2x3 di uni
arr4 = np.arange(0, 10, 2)        # [0, 2, 4, 6, 8]
arr5 = np.linspace(0, 1, 5)       # 5 numeri tra 0 e 1
arr6 = np.random.randn(3, 3)      # 3x3 random normal

print(f"np.array([1,2,3,4,5]): {arr1}")
print(f"np.zeros((3,4)).shape: {arr2.shape}")
print(f"np.arange(0,10,2): {arr4}")
print(f"np.linspace(0,1,5): {arr5}")

# Array attributes
print("\n🔹 ATTRIBUTI ARRAY:")
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"arr.shape: {arr.shape}")      # (2, 3)
print(f"arr.ndim: {arr.ndim}")        # 2
print(f"arr.size: {arr.size}")        # 6
print(f"arr.dtype: {arr.dtype}")      # int64

# Indexing and slicing
print("\n🔹 INDEXING E SLICING:")
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"arr[0, 0]: {arr[0, 0]}")      # 1
print(f"arr[1, :]: {arr[1, :]}")      # [4, 5, 6] (seconda riga)
print(f"arr[:, 1]: {arr[:, 1]}")      # [2, 5, 8] (seconda colonna)
print(f"arr[0:2, 1:3]: \n{arr[0:2, 1:3]}")  # Subarray

# Operations
print("\n🔹 OPERAZIONI:")
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(f"a + b: {a + b}")              # Element-wise
print(f"a * b: {a * b}")
print(f"a.dot(b): {a.dot(b)}")        # Dot product
print(f"np.sum(a): {np.sum(a)}")
print(f"np.mean(a): {np.mean(a)}")
print(f"np.std(a): {np.std(a):.4f}")

# Broadcasting
print("\n🔹 BROADCASTING:")
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"arr + 10:\n{arr + 10}")       # Aggiunge 10 a tutti
print(f"arr * 2:\n{arr * 2}")

# ══════════════════════════════════════════════════════════════════════════════
# ML 1.2 - PANDAS DATAFRAMES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ML 1.2 - PANDAS DATAFRAMES")
print("=" * 70)

print("""
📋 PANDAS: Data Analysis Library

Strutture principali:
- Series: array 1D con indice
- DataFrame: tabella 2D (righe e colonne)
""")

# Series
print("\n🔹 SERIES:")
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print(f"Series:\n{s}")
print(f"s['b']: {s['b']}")

# DataFrame creation
print("\n🔹 CREAZIONE DATAFRAME:")
data = {
    'name': ['Marco', 'Anna', 'Luca', 'Sara'],
    'age': [25, 30, 22, 28],
    'city': ['Milan', 'Rome', 'Naples', 'Milan'],
    'salary': [35000, 45000, 28000, 42000]
}
df = pd.DataFrame(data)
print(df)

# Selection
print("\n🔹 SELEZIONE COLONNE:")
print(f"df['name']:\n{df['name']}")
print(f"\ndf[['name', 'age']]:\n{df[['name', 'age']]}")

# Row selection
print("\n🔹 SELEZIONE RIGHE:")
print(f"df.iloc[0] (prima riga):\n{df.iloc[0]}")
print(f"\ndf.iloc[1:3] (righe 1-2):\n{df.iloc[1:3]}")
print(f"\ndf.loc[df['age'] > 25] (filtering):\n{df.loc[df['age'] > 25]}")

# Adding columns
print("\n🔹 AGGIUNGERE COLONNE:")
df['bonus'] = df['salary'] * 0.1
df['senior'] = df['age'] > 25
print(df)

# ══════════════════════════════════════════════════════════════════════════════
# ML 1.3 - DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ML 1.3 - DATA LOADING")
print("=" * 70)

print("""
📋 CARICAMENTO DATI:

# CSV
df = pd.read_csv('file.csv')
df = pd.read_csv('file.csv', sep=';', encoding='utf-8')

# Excel
df = pd.read_excel('file.xlsx', sheet_name='Sheet1')

# JSON
df = pd.read_json('file.json')

# Da dizionario
df = pd.DataFrame(dict_data)

# Da URL
df = pd.read_csv('https://example.com/data.csv')

# Salvare
df.to_csv('output.csv', index=False)
df.to_excel('output.xlsx', index=False)
df.to_json('output.json')
""")

# ══════════════════════════════════════════════════════════════════════════════
# ML 1.4 - DATA INSPECTION
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ML 1.4 - DATA INSPECTION")
print("=" * 70)

print("\n🔹 METODI DI ISPEZIONE:")
print(f"df.head():\n{df.head(2)}")
print(f"\ndf.tail(2):\n{df.tail(2)}")
print(f"\ndf.shape: {df.shape}")
print(f"\ndf.columns: {list(df.columns)}")
print(f"\ndf.dtypes:\n{df.dtypes}")
print(f"\ndf.info():")
df.info()
print(f"\ndf.describe():\n{df.describe()}")

# Missing values
print("\n🔹 VALORI MANCANTI:")
df_with_null = df.copy()
df_with_null.loc[0, 'salary'] = np.nan
print(f"df.isnull().sum():\n{df_with_null.isnull().sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# ML 1.5 - DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ML 1.5 - DATA CLEANING")
print("=" * 70)

print("""
📋 GESTIONE VALORI MANCANTI:

# Verificare null
df.isnull().sum()

# Rimuovere righe con null
df.dropna()
df.dropna(subset=['column'])

# Riempire null
df.fillna(0)
df.fillna(df.mean())
df['col'].fillna(df['col'].median(), inplace=True)

# Interpolazione
df.interpolate()
""")

print("\n🔹 ESEMPIO PRATICO:")
df_dirty = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [10, np.nan, 30, np.nan, 50],
    'C': ['x', 'y', 'x', 'y', 'x']
})
print(f"DataFrame con null:\n{df_dirty}")
print(f"\nNull per colonna:\n{df_dirty.isnull().sum()}")
print(f"\nFillna con media:\n{df_dirty.fillna(df_dirty.mean(numeric_only=True))}")

print("""
📋 GESTIONE DUPLICATI:

# Trovare duplicati
df.duplicated()
df.duplicated().sum()

# Rimuovere duplicati
df.drop_duplicates()
df.drop_duplicates(subset=['col1', 'col2'])
""")

print("""
📋 GESTIONE OUTLIERS:

# Metodo IQR (Interquartile Range)
Q1 = df['col'].quantile(0.25)
Q3 = df['col'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Filtrare outliers
df_clean = df[(df['col'] >= lower) & (df['col'] <= upper)]

# Oppure: Z-score
from scipy import stats
z_scores = np.abs(stats.zscore(df['col']))
df_clean = df[z_scores < 3]
""")

# ══════════════════════════════════════════════════════════════════════════════
# ML 1.6 - BASIC STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ML 1.6 - BASIC STATISTICS")
print("=" * 70)

print("\n🔹 STATISTICHE DESCRITTIVE:")
print(f"df['salary'].mean(): {df['salary'].mean()}")
print(f"df['salary'].median(): {df['salary'].median()}")
print(f"df['salary'].std(): {df['salary'].std():.2f}")
print(f"df['salary'].min(): {df['salary'].min()}")
print(f"df['salary'].max(): {df['salary'].max()}")
print(f"df['salary'].quantile(0.75): {df['salary'].quantile(0.75)}")

print("\n🔹 VALUE COUNTS:")
print(f"df['city'].value_counts():\n{df['city'].value_counts()}")

print("\n🔹 GROUPBY:")
grouped = df.groupby('city')['salary'].mean()
print(f"Salary medio per città:\n{grouped}")

print("\n🔹 CORRELAZIONE:")
numeric_df = df[['age', 'salary', 'bonus']]
print(f"Correlation matrix:\n{numeric_df.corr()}")

# ══════════════════════════════════════════════════════════════════════════════
# QUIZ DI VERIFICA
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("QUIZ DI VERIFICA - ML MODULE 1")
print("=" * 70)

print("""
Q1. np.zeros((3, 4)).shape restituisce?
    A) (4, 3)  B) (3, 4)  C) 12  D) (12,)
    → RISPOSTA: B

Q2. Come selezionare la colonna 'name' da un DataFrame df?
    A) df.name  B) df['name']  C) df.get('name')  D) Tutte corrette
    → RISPOSTA: D (ma B è la più comune)

Q3. df.iloc[0] restituisce?
    A) Prima colonna  B) Prima riga  C) Primo elemento  D) Errore
    → RISPOSTA: B

Q4. df.isnull().sum() restituisce?
    A) Numero totale null  B) Null per colonna  C) Boolean  D) DataFrame
    → RISPOSTA: B

Q5. df.dropna() fa cosa?
    A) Riempie null con 0  B) Rimuove righe con null  C) Conta null  D) Nulla
    → RISPOSTA: B

Q6. df.fillna(df.mean()) fa cosa?
    A) Errore  B) Riempie con media colonna  C) Calcola media  D) Rimuove null
    → RISPOSTA: B

Q7. df.groupby('city')['salary'].mean() restituisce?
    A) DataFrame  B) Series  C) Float  D) List
    → RISPOSTA: B

Q8. np.array([1,2,3]) + 10 restituisce?
    A) Errore  B) [11, 12, 13]  C) 10  D) [1, 2, 3, 10]
    → RISPOSTA: B (broadcasting)

Q9. df.corr() calcola?
    A) Covarianza  B) Correlazione  C) Media  D) Varianza
    → RISPOSTA: B

Q10. pd.read_csv() può leggere da URL?
    A) Sì  B) No  C) Solo con requests  D) Solo locale
    → RISPOSTA: A
""")

# ══════════════════════════════════════════════════════════════════════════════
# ESERCIZI PRATICI
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ESERCIZI PRATICI")
print("=" * 70)

print("""
📝 ESERCIZIO 1: Array NumPy
Crea un array 5x5 di numeri casuali tra 0 e 100.
Calcola: media, max, min, somma di ogni riga.

📝 ESERCIZIO 2: DataFrame
Crea un DataFrame con 100 righe e colonne:
- id (1-100)
- value (random)
- category (A, B, C random)
Calcola la media di 'value' per ogni 'category'.

📝 ESERCIZIO 3: Data Cleaning
Dato un DataFrame con valori mancanti:
- Conta i null per colonna
- Riempi i null numerici con la mediana
- Riempi i null categorici con la moda

📝 ESERCIZIO 4: Analisi
Carica un CSV (puoi usare dati fake) e:
- Mostra le prime 5 righe
- Descrivi le statistiche
- Trova la correlazione tra colonne numeriche
- Raggruppa per una colonna categorica
""")

print("\n" + "=" * 70)
print("ML MODULE 1 - DATA FOUNDATIONS COMPLETATO!")
print("Prossimo: ML MODULE 2 - DATA PREPROCESSING")
print("=" * 70)
