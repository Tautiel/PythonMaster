"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    🐍 PYTHON MASTER - SCHEDA ESERCIZI COMPLETA               ║
║                                                                              ║
║                    PARTE 9: PANDAS                                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np

# ==============================================================================
# SEZIONE 25: PANDAS BASICS
# ==============================================================================

print("=" * 70)
print("SEZIONE 25: PANDAS BASICS")
print("=" * 70)

# ------------------------------------------------------------------------------
# ESERCIZIO 25.1: Series e DataFrame Creation
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Crea Series e DataFrame:
1. Da liste, dict, array
2. Con indici custom
3. Con tipi specificati

💡 TEORIA:
Series = array 1D con indice
DataFrame = tabella 2D con indici righe/colonne
Index può essere qualsiasi tipo (date, stringhe, ecc.)

🎯 SKILLS: pd.Series, pd.DataFrame, index, dtypes
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Principiante
"""

def esercizio_25_1():
    """Series e DataFrame Creation"""
    
    # 1. SERIES
    print("--- SERIES ---")
    
    prices = pd.Series([100, 102, 98, 105, 103])
    print(f"  Da lista:\n{prices}\n")
    
    prices_idx = pd.Series(
        [100, 102, 98, 105, 103],
        index=['Mon', 'Tue', 'Wed', 'Thu', 'Fri'],
        name='price'
    )
    print(f"  Con indice:\n{prices_idx}\n")
    
    portfolio = pd.Series({'AAPL': 150.0, 'GOOGL': 140.0, 'MSFT': 380.0})
    print(f"  Da dict:\n{portfolio}\n")
    
    # 2. DATAFRAME
    print("--- DATAFRAME ---")
    
    df = pd.DataFrame({
        'symbol': ['AAPL', 'GOOGL', 'MSFT'],
        'price': [150.0, 140.0, 380.0],
        'quantity': [100, 50, 75],
    })
    print(f"  Da dict:\n{df}\n")
    
    dates = pd.date_range('2024-01-01', periods=3)
    df_dated = pd.DataFrame({
        'open': [100, 102, 101],
        'high': [105, 108, 106],
        'low': [98, 100, 99],
        'close': [103, 105, 104],
    }, index=dates)
    print(f"  Con DateIndex:\n{df_dated}\n")
    
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")

if __name__ == "__main__":
    esercizio_25_1()
    print("✅ Esercizio 25.1 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 25.2: Selection e Indexing
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Seleziona dati in vari modi:
1. [] selection
2. loc[] e iloc[]
3. Boolean indexing
4. Query()

💡 TEORIA:
loc = label-based (usa nomi)
iloc = integer-based (usa posizioni)
[] su DataFrame seleziona colonne

🎯 SKILLS: loc, iloc, boolean indexing, query
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Intermedio
"""

def esercizio_25_2():
    """Selection e Indexing"""
    
    df = pd.DataFrame({
        'symbol': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'],
        'price': [150, 140, 380, 175, 250],
        'volume': [1000000, 500000, 750000, 800000, 1200000],
        'sector': ['Tech', 'Tech', 'Tech', 'Retail', 'Auto']
    })
    
    print("DataFrame:")
    print(df)
    
    # LOC
    print("\n--- LOC ---")
    print(f"  df.loc[0]: {dict(df.loc[0])}")
    print(f"  df.loc[1:3, 'price']: {df.loc[1:3, 'price'].values}")
    
    # ILOC
    print("\n--- ILOC ---")
    print(f"  df.iloc[0]: {dict(df.iloc[0])}")
    print(f"  df.iloc[-1]: {dict(df.iloc[-1])}")
    
    # BOOLEAN
    print("\n--- BOOLEAN ---")
    high_price = df[df['price'] > 200]
    print(f"  price > 200:\n{high_price}")
    
    tech_stocks = df[(df['sector'] == 'Tech') & (df['price'] < 200)]
    print(f"\n  Tech AND price < 200:\n{tech_stocks}")
    
    # QUERY
    print("\n--- QUERY ---")
    result = df.query('sector == "Tech" and volume > 600000')
    print(f"  query result:\n{result}")

if __name__ == "__main__":
    esercizio_25_2()
    print("✅ Esercizio 25.2 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 25.3: Data Manipulation
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Manipola dati:
1. Aggiungere/rimuovere colonne
2. Apply e transform
3. Sorting
4. Handling missing data

💡 TEORIA:
apply() applica funzione a righe/colonne
fillna/dropna per gestire NaN

🎯 SKILLS: apply, sorting, missing data
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Intermedio
"""

def esercizio_25_3():
    """Data Manipulation"""
    
    df = pd.DataFrame({
        'symbol': ['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
        'price': [150.0, 140.0, 380.0, 175.0],
        'quantity': [100, 50, 75, 80],
    })
    
    # AGGIUNGERE COLONNE
    print("--- AGGIUNGERE COLONNE ---")
    df['value'] = df['price'] * df['quantity']
    df['weight'] = df['value'] / df['value'].sum() * 100
    print(df)
    
    # APPLY
    print("\n--- APPLY ---")
    df['price_category'] = df['price'].apply(
        lambda x: 'HIGH' if x > 200 else 'MEDIUM' if x > 150 else 'LOW'
    )
    print(df)
    
    # SORTING
    print("\n--- SORTING ---")
    by_value = df.sort_values('value', ascending=False)
    print(f"  By value desc:\n{by_value[['symbol', 'value']]}")
    
    # MISSING DATA
    print("\n--- MISSING DATA ---")
    df_missing = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': [np.nan, 2, 3, 4],
    })
    print(f"  Original:\n{df_missing}")
    print(f"  fillna(0):\n{df_missing.fillna(0)}")
    print(f"  dropna():\n{df_missing.dropna()}")

if __name__ == "__main__":
    esercizio_25_3()
    print("✅ Esercizio 25.3 completato!\n")


# ==============================================================================
# SEZIONE 26: PANDAS PER FINANZA
# ==============================================================================

print("\n" + "=" * 70)
print("SEZIONE 26: PANDAS PER FINANZA")
print("=" * 70)

# ------------------------------------------------------------------------------
# ESERCIZIO 26.1: Time Series
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Lavora con time series:
1. DatetimeIndex
2. Resampling
3. Rolling windows
4. Shift e pct_change

💡 TEORIA:
pandas ha supporto nativo per time series.
resample() aggrega per periodo.
rolling() calcola statistiche mobili.

🎯 SKILLS: DatetimeIndex, resample, rolling
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Intermedio
"""

def esercizio_26_1():
    """Time Series"""
    
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    prices = 100 * (1 + np.random.normal(0.001, 0.02, 30)).cumprod()
    ts = pd.Series(prices, index=dates, name='price')
    
    print("--- TIME SERIES ---")
    print(f"  First 5:\n{ts.head()}")
    
    # RESAMPLE
    print("\n--- RESAMPLE ---")
    weekly_ohlc = ts.resample('W').ohlc()
    print(f"  Weekly OHLC:\n{weekly_ohlc.head()}")
    
    # ROLLING
    print("\n--- ROLLING ---")
    df = pd.DataFrame({'price': ts})
    df['SMA_5'] = df['price'].rolling(5).mean()
    df['SMA_10'] = df['price'].rolling(10).mean()
    print(f"  Rolling stats:\n{df.tail(10)}")
    
    # PCT_CHANGE
    print("\n--- PCT_CHANGE ---")
    df['returns'] = df['price'].pct_change() * 100
    print(f"  With returns:\n{df[['price', 'returns']].tail()}")

if __name__ == "__main__":
    esercizio_26_1()
    print("✅ Esercizio 26.1 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 26.2: GroupBy e Aggregazione
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Usa groupby per:
1. Aggregazioni per gruppo
2. Multi aggregation
3. Pivot tables

💡 TEORIA:
groupby() split-apply-combine
agg() per aggregazioni multiple
pivot_table per tabelle riassuntive

🎯 SKILLS: groupby, agg, pivot_table
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Intermedio-Avanzato
"""

def esercizio_26_2():
    """GroupBy e Aggregazione"""
    
    np.random.seed(42)
    trades = pd.DataFrame({
        'symbol': np.tile(['AAPL', 'GOOGL', 'MSFT'], 10),
        'side': np.random.choice(['BUY', 'SELL'], 30),
        'quantity': np.random.randint(10, 100, 30),
        'price': np.random.uniform(100, 200, 30).round(2)
    })
    trades['value'] = trades['quantity'] * trades['price']
    
    print("Trades sample:")
    print(trades.head())
    
    # GROUPBY
    print("\n--- GROUPBY ---")
    by_symbol = trades.groupby('symbol')['value'].sum()
    print(f"  Total value by symbol:\n{by_symbol}")
    
    # MULTI AGGREGATION
    print("\n--- MULTI AGGREGATION ---")
    agg_result = trades.groupby('symbol').agg({
        'quantity': ['sum', 'mean'],
        'value': ['sum', 'count'],
    })
    print(f"  Multiple aggs:\n{agg_result}")
    
    # PIVOT TABLE
    print("\n--- PIVOT TABLE ---")
    pivot = pd.pivot_table(
        trades, values='value', index='symbol',
        columns='side', aggfunc='sum', fill_value=0
    )
    print(f"  Pivot:\n{pivot}")

if __name__ == "__main__":
    esercizio_26_2()
    print("✅ Esercizio 26.2 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 26.3: Merge e Join
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Combina DataFrame:
1. merge (SQL-like join)
2. concat
3. join su index

💡 TEORIA:
merge() è come SQL JOIN
concat() impila DataFrames

🎯 SKILLS: merge, concat, join
⏱️ TEMPO: 10 minuti
🔢 LIVELLO: Intermedio
"""

def esercizio_26_3():
    """Merge e Join"""
    
    prices = pd.DataFrame({
        'symbol': ['AAPL', 'GOOGL', 'MSFT'],
        'price': [150, 140, 380]
    })
    
    info = pd.DataFrame({
        'symbol': ['AAPL', 'GOOGL', 'AMZN'],
        'sector': ['Tech', 'Tech', 'Retail'],
    })
    
    print("prices:")
    print(prices)
    print("\ninfo:")
    print(info)
    
    # INNER
    print("\n--- INNER MERGE ---")
    inner = pd.merge(prices, info, on='symbol')
    print(inner)
    
    # LEFT
    print("\n--- LEFT MERGE ---")
    left = pd.merge(prices, info, on='symbol', how='left')
    print(left)
    
    # OUTER
    print("\n--- OUTER MERGE ---")
    outer = pd.merge(prices, info, on='symbol', how='outer')
    print(outer)
    
    # CONCAT
    print("\n--- CONCAT ---")
    df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
    vertical = pd.concat([df1, df2], ignore_index=True)
    print(f"  Vertical:\n{vertical}")

if __name__ == "__main__":
    esercizio_26_3()
    print("✅ Esercizio 26.3 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 26.4: Financial Analysis
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Analisi finanziaria con pandas:
1. Calcolo rendimenti
2. Drawdown
3. Performance metrics
4. Correlation analysis

💡 TEORIA:
pandas semplifica l'analisi finanziaria con metodi
built-in per returns, rolling stats, ecc.

🎯 SKILLS: Financial analysis, metrics
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Avanzato
"""

def esercizio_26_4():
    """Financial Analysis"""
    
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=252, freq='B')
    
    prices = pd.DataFrame({
        'AAPL': 150 * (1 + np.random.normal(0.001, 0.02, 252)).cumprod(),
        'GOOGL': 140 * (1 + np.random.normal(0.0008, 0.018, 252)).cumprod(),
        'MSFT': 380 * (1 + np.random.normal(0.0012, 0.022, 252)).cumprod(),
    }, index=dates)
    
    print("Prices sample:")
    print(prices.head())
    
    # RETURNS
    print("\n--- RETURNS ---")
    returns = prices.pct_change().dropna()
    cumulative = (1 + returns).cumprod() - 1
    print(f"  Cumulative returns (last):\n{cumulative.iloc[-1].round(4)}")
    
    # DRAWDOWN
    print("\n--- DRAWDOWN ---")
    def calculate_drawdown(prices):
        peak = prices.expanding().max()
        return (prices - peak) / peak * 100
    
    drawdown = prices.apply(calculate_drawdown)
    max_dd = drawdown.min()
    print(f"  Max Drawdown:\n{max_dd.round(2)}")
    
    # PERFORMANCE METRICS
    print("\n--- PERFORMANCE METRICS ---")
    annual_return = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol
    
    metrics = pd.DataFrame({
        'Return%': (annual_return * 100).round(2),
        'Vol%': (annual_vol * 100).round(2),
        'Sharpe': sharpe.round(2),
        'MaxDD%': max_dd.round(2)
    })
    print(metrics)
    
    # CORRELATION
    print("\n--- CORRELATION ---")
    corr = returns.corr()
    print(f"  Correlation matrix:\n{corr.round(3)}")

if __name__ == "__main__":
    esercizio_26_4()
    print("✅ Esercizio 26.4 completato!\n")


# ==============================================================================
# RIEPILOGO
# ==============================================================================

print("\n" + "=" * 70)
print("RIEPILOGO: ESERCIZI PANDAS COMPLETATI")
print("=" * 70)

print("""
ESERCIZI COMPLETATI IN QUESTA PARTE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SEZIONE 25 - Pandas Basics:
  ✅ 25.1 Series e DataFrame Creation
  ✅ 25.2 Selection e Indexing
  ✅ 25.3 Data Manipulation

SEZIONE 26 - Pandas per Finanza:
  ✅ 26.1 Time Series
  ✅ 26.2 GroupBy e Aggregazione
  ✅ 26.3 Merge e Join
  ✅ 26.4 Financial Analysis

TOTALE QUESTA PARTE: 7 esercizi
TOTALE CUMULATIVO: 80 esercizi
""")

if __name__ == "__main__":
    print("\n🎉 TUTTI GLI ESERCIZI DELLA PARTE 9 COMPLETATI!")
