"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    🐍 PYTHON MASTER - SCHEDA ESERCIZI COMPLETA               ║
║                                                                              ║
║                    PARTE 8: NUMPY                                            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np

# ==============================================================================
# SEZIONE 23: NUMPY BASICS
# ==============================================================================

print("=" * 70)
print("SEZIONE 23: NUMPY BASICS")
print("=" * 70)

# ------------------------------------------------------------------------------
# ESERCIZIO 23.1: Array Creation
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Crea array NumPy in vari modi:
1. Da liste Python
2. Con funzioni built-in (zeros, ones, arange, linspace)
3. Con random

💡 TEORIA:
NumPy array sono più efficienti delle liste per operazioni numeriche.
Sono omogenei (stesso tipo) e hanno dimensione fissa.

🎯 SKILLS: np.array, np.zeros, np.ones, np.arange, np.linspace
⏱️ TEMPO: 10 minuti
🔢 LIVELLO: Principiante
"""

def esercizio_23_1():
    """Array Creation"""
    
    # 1. DA LISTE
    print("--- DA LISTE ---")
    
    prices = np.array([100.0, 101.5, 99.8, 102.3, 103.1])
    print(f"  prices: {prices}")
    print(f"  dtype: {prices.dtype}")
    print(f"  shape: {prices.shape}")
    
    # 2D array
    ohlc = np.array([
        [100, 105, 98, 103],
        [103, 108, 101, 106],
        [106, 107, 104, 105],
    ])
    print(f"\n  OHLC shape: {ohlc.shape}")
    
    # 2. FUNZIONI BUILT-IN
    print("\n--- FUNZIONI BUILT-IN ---")
    
    zeros = np.zeros(5)
    ones = np.ones((2, 3))
    print(f"  zeros(5): {zeros}")
    print(f"  ones((2,3)):\n{ones}")
    
    indices = np.arange(0, 10, 2)
    print(f"  arange(0, 10, 2): {indices}")
    
    percentiles = np.linspace(0, 100, 5)
    print(f"  linspace(0, 100, 5): {percentiles}")
    
    # 3. RANDOM
    print("\n--- RANDOM ---")
    
    np.random.seed(42)
    
    uniform = np.random.random(5)
    print(f"  random(5): {uniform}")
    
    returns = np.random.normal(0.001, 0.02, 5)
    print(f"  normal(0.001, 0.02, 5): {returns}")
    
    # 4. APPLICAZIONE: Simula prezzi
    print("\n--- SIMULAZIONE PREZZI ---")
    
    initial_price = 100.0
    n_days = 10
    daily_returns = np.random.normal(0.001, 0.02, n_days)
    prices = initial_price * np.cumprod(1 + daily_returns)
    print(f"  Simulated prices: {prices}")

if __name__ == "__main__":
    esercizio_23_1()
    print("✅ Esercizio 23.1 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 23.2: Indexing e Slicing
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Pratica indexing e slicing:
1. Basic indexing
2. Fancy indexing (con array di indici)
3. Boolean indexing
4. Slicing multidimensionale

💡 TEORIA:
NumPy supporta indexing avanzato oltre a quello Python base.
Boolean indexing è potentissimo per filtrare dati.

🎯 SKILLS: indexing, slicing, boolean masks
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Intermedio
"""

def esercizio_23_2():
    """Indexing e Slicing"""
    
    # 1. BASIC INDEXING
    print("--- BASIC INDEXING ---")
    
    prices = np.array([100, 102, 98, 105, 103, 108, 107, 110])
    
    print(f"  prices: {prices}")
    print(f"  prices[0]: {prices[0]}")
    print(f"  prices[-1]: {prices[-1]}")
    print(f"  prices[2:5]: {prices[2:5]}")
    print(f"  prices[::2]: {prices[::2]}")
    
    # 2. 2D INDEXING
    print("\n--- 2D INDEXING ---")
    
    ohlc = np.array([
        [100, 105, 98, 103],
        [103, 108, 101, 106],
        [106, 110, 104, 109],
    ])
    
    print(f"  Day 0: {ohlc[0]}")
    print(f"  All Opens (col 0): {ohlc[:, 0]}")
    print(f"  All Closes (col 3): {ohlc[:, 3]}")
    
    # 3. BOOLEAN INDEXING
    print("\n--- BOOLEAN INDEXING ---")
    
    mask = prices > 105
    print(f"  prices > 105: {mask}")
    print(f"  prices[prices > 105]: {prices[mask]}")
    
    # Trova indici
    indices_high = np.where(prices > 105)[0]
    print(f"  Indici dove > 105: {indices_high}")

if __name__ == "__main__":
    esercizio_23_2()
    print("✅ Esercizio 23.2 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 23.3: Operazioni Vettoriali
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Esegui operazioni vettoriali:
1. Operazioni elemento per elemento
2. Broadcasting
3. Funzioni universali (ufuncs)
4. Aggregazioni

💡 TEORIA:
Le operazioni vettoriali sono molto più veloci dei loop Python.
Broadcasting estende automaticamente array di forme compatibili.

🎯 SKILLS: vectorization, broadcasting, ufuncs
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Intermedio
"""

def esercizio_23_3():
    """Operazioni Vettoriali"""
    
    # 1. OPERAZIONI ELEMENTO PER ELEMENTO
    print("--- OPERAZIONI ELEMENTO PER ELEMENTO ---")
    
    prices = np.array([100, 102, 98, 105, 103])
    quantities = np.array([10, 20, 15, 25, 30])
    
    values = prices * quantities
    print(f"  prices * quantities = {values}")
    
    returns = (prices[1:] - prices[:-1]) / prices[:-1] * 100
    print(f"  Returns %: {returns}")
    
    # 2. BROADCASTING
    print("\n--- BROADCASTING ---")
    
    normalized = prices - prices.mean()
    print(f"  prices - mean: {normalized}")
    
    # 3. UFUNCS
    print("\n--- UFUNCS ---")
    
    print(f"  np.sqrt(prices): {np.sqrt(prices)}")
    print(f"  np.log(prices): {np.log(prices)}")
    
    # 4. AGGREGAZIONI
    print("\n--- AGGREGAZIONI ---")
    
    print(f"  sum: {prices.sum()}")
    print(f"  mean: {prices.mean():.2f}")
    print(f"  std: {prices.std():.2f}")
    print(f"  min: {prices.min()}, max: {prices.max()}")
    
    # 5. SHARPE RATIO
    print("\n--- SHARPE RATIO ---")
    
    np.random.seed(42)
    daily_returns = np.random.normal(0.001, 0.02, 252)
    
    mean_return = daily_returns.mean()
    std_return = daily_returns.std()
    sharpe = (mean_return / std_return) * np.sqrt(252)
    
    print(f"  Sharpe Ratio: {sharpe:.2f}")

if __name__ == "__main__":
    esercizio_23_3()
    print("✅ Esercizio 23.3 completato!\n")


# ==============================================================================
# SEZIONE 24: NUMPY AVANZATO
# ==============================================================================

print("\n" + "=" * 70)
print("SEZIONE 24: NUMPY AVANZATO")
print("=" * 70)

# ------------------------------------------------------------------------------
# ESERCIZIO 24.1: Reshaping e Stacking
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Manipola shape di array:
1. reshape, ravel, flatten
2. vstack, hstack, concatenate
3. split, array_split

💡 TEORIA:
reshape cambia la forma senza copiare dati (se possibile).
vstack/hstack uniscono array verticalmente/orizzontalmente.

🎯 SKILLS: reshape, stack, concatenate
⏱️ TEMPO: 10 minuti
🔢 LIVELLO: Intermedio
"""

def esercizio_24_1():
    """Reshaping e Stacking"""
    
    # 1. RESHAPE
    print("--- RESHAPE ---")
    
    arr = np.arange(12)
    print(f"  Original: {arr}")
    
    matrix = arr.reshape(3, 4)
    print(f"  reshape(3,4):\n{matrix}")
    
    auto = arr.reshape(4, -1)
    print(f"  reshape(4,-1):\n{auto}")
    
    # 2. STACKING
    print("\n--- STACKING ---")
    
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    
    vstacked = np.vstack([a, b])
    print(f"  vstack:\n{vstacked}")
    
    hstacked = np.hstack([a, b])
    print(f"  hstack: {hstacked}")
    
    # 3. SPLITTING
    print("\n--- SPLITTING ---")
    
    arr = np.arange(12)
    splits = np.split(arr, 3)
    print(f"  split in 3: {splits}")

if __name__ == "__main__":
    esercizio_24_1()
    print("✅ Esercizio 24.1 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 24.2: Statistical Operations
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Esegui operazioni statistiche:
1. Correlazione e covarianza
2. Percentili e quantili
3. Rolling statistics (manuale)

💡 TEORIA:
NumPy fornisce funzioni statistiche ottimizzate.
Per rolling statistics più complesse, usa pandas.

🎯 SKILLS: correlation, percentile, statistics
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Intermedio-Avanzato
"""

def esercizio_24_2():
    """Statistical Operations"""
    
    np.random.seed(42)
    
    # 1. STATISTICHE BASE
    print("--- STATISTICHE BASE ---")
    
    returns = np.random.normal(0.001, 0.02, 100)
    
    print(f"  Mean: {np.mean(returns):.6f}")
    print(f"  Std: {np.std(returns):.6f}")
    print(f"  Median: {np.median(returns):.6f}")
    
    # 2. PERCENTILI
    print("\n--- PERCENTILI ---")
    
    var_95 = np.percentile(returns, 5)
    print(f"  VaR 95%: {var_95:.4f}")
    
    # 3. CORRELAZIONE
    print("\n--- CORRELAZIONE ---")
    
    returns_aapl = np.random.normal(0.001, 0.02, 100)
    returns_googl = returns_aapl * 0.7 + np.random.normal(0, 0.01, 100)
    
    corr = np.corrcoef(returns_aapl, returns_googl)[0, 1]
    print(f"  Correlation AAPL-GOOGL: {corr:.2f}")
    
    # 4. PORTFOLIO METRICS
    print("\n--- PORTFOLIO METRICS ---")
    
    returns_matrix = np.vstack([returns_aapl, returns_googl])
    weights = np.array([0.6, 0.4])
    
    portfolio_returns = returns_matrix.T @ weights
    port_sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
    
    print(f"  Portfolio Sharpe: {port_sharpe:.2f}")

if __name__ == "__main__":
    esercizio_24_2()
    print("✅ Esercizio 24.2 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 24.3: Linear Algebra
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Operazioni di algebra lineare:
1. Prodotto matriciale
2. Inverse e determinante
3. Least squares

💡 TEORIA:
np.linalg contiene funzioni di algebra lineare.
Utile per ottimizzazione portfolio e regressione.

🎯 SKILLS: np.dot, np.linalg, matrix operations
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Avanzato
"""

def esercizio_24_3():
    """Linear Algebra"""
    
    # 1. PRODOTTO MATRICIALE
    print("--- PRODOTTO MATRICIALE ---")
    
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    
    matmul = A @ B
    print(f"  A @ B:\n{matmul}")
    
    # 2. INVERSE
    print("\n--- INVERSE ---")
    
    print(f"  det(A) = {np.linalg.det(A):.2f}")
    
    A_inv = np.linalg.inv(A)
    print(f"  A^(-1):\n{A_inv}")
    
    # 3. LEAST SQUARES
    print("\n--- LEAST SQUARES ---")
    
    np.random.seed(42)
    x = np.linspace(0, 10, 20)
    y_noisy = 2 * x + 5 + np.random.normal(0, 1, 20)
    
    X = np.vstack([x, np.ones(len(x))]).T
    coeffs, _, _, _ = np.linalg.lstsq(X, y_noisy, rcond=None)
    m, b = coeffs
    
    print(f"  True: y = 2x + 5")
    print(f"  Fitted: y = {m:.2f}x + {b:.2f}")
    
    # 4. PORTFOLIO OPTIMIZATION
    print("\n--- PORTFOLIO OPTIMIZATION ---")
    
    cov = np.array([
        [0.04, 0.02, 0.01],
        [0.02, 0.03, 0.005],
        [0.01, 0.005, 0.02],
    ])
    
    ones = np.ones(3)
    cov_inv = np.linalg.inv(cov)
    weights_mv = cov_inv @ ones
    weights_mv = weights_mv / weights_mv.sum()
    
    print(f"  Min Variance Weights: {np.round(weights_mv, 3)}")

if __name__ == "__main__":
    esercizio_24_3()
    print("✅ Esercizio 24.3 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 24.4: Performance e Best Practices
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Ottimizza codice NumPy:
1. Evita loop Python
2. Pre-allocazione
3. Views vs copies

💡 TEORIA:
NumPy è veloce quando usi operazioni vettoriali.
Loop Python su array sono lenti.

🎯 SKILLS: vectorization, performance optimization
⏱️ TEMPO: 10 minuti
🔢 LIVELLO: Avanzato
"""

def esercizio_24_4():
    """Performance e Best Practices"""
    
    import time
    
    # 1. LOOP vs VECTORIZED
    print("--- LOOP vs VECTORIZED ---")
    
    size = 100000
    a = np.random.random(size)
    b = np.random.random(size)
    
    # Loop Python
    start = time.perf_counter()
    result_loop = np.empty(size)
    for i in range(size):
        result_loop[i] = a[i] + b[i]
    time_loop = time.perf_counter() - start
    
    # Vectorized
    start = time.perf_counter()
    result_vec = a + b
    time_vec = time.perf_counter() - start
    
    print(f"  Loop: {time_loop:.4f}s")
    print(f"  Vectorized: {time_vec:.6f}s")
    print(f"  Speedup: {time_loop/time_vec:.0f}x")
    
    # 2. VIEW vs COPY
    print("\n--- VIEW vs COPY ---")
    
    arr = np.arange(10)
    view = arr[2:5]
    view[0] = 999
    print(f"  Original dopo modifica view: {arr}")
    
    arr = np.arange(10)
    copy = arr[2:5].copy()
    copy[0] = 999
    print(f"  Original dopo modifica copy: {arr}")
    
    # 3. BEST PRACTICES
    print("\n--- BEST PRACTICES ---")
    print("""
  ✅ Usa operazioni vettoriali invece di loop
  ✅ Pre-alloca array quando possibile
  ✅ Usa view invece di copy quando appropriato
  ✅ Scegli dtype appropriato (float32 vs float64)
  ✅ Usa np.where invece di if/else in loop
    """)

if __name__ == "__main__":
    esercizio_24_4()
    print("✅ Esercizio 24.4 completato!\n")


# ==============================================================================
# RIEPILOGO
# ==============================================================================

print("\n" + "=" * 70)
print("RIEPILOGO: ESERCIZI NUMPY COMPLETATI")
print("=" * 70)

print("""
ESERCIZI COMPLETATI IN QUESTA PARTE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SEZIONE 23 - NumPy Basics:
  ✅ 23.1 Array Creation
  ✅ 23.2 Indexing e Slicing
  ✅ 23.3 Operazioni Vettoriali

SEZIONE 24 - NumPy Avanzato:
  ✅ 24.1 Reshaping e Stacking
  ✅ 24.2 Statistical Operations
  ✅ 24.3 Linear Algebra
  ✅ 24.4 Performance e Best Practices

TOTALE QUESTA PARTE: 7 esercizi
TOTALE CUMULATIVO: 73 esercizi
""")

if __name__ == "__main__":
    print("\n🎉 TUTTI GLI ESERCIZI DELLA PARTE 8 COMPLETATI!")
