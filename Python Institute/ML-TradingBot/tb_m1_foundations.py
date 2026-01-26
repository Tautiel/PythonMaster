#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    TRADING BOT - MODULE 1                                    ║
║                    TRADING FOUNDATIONS                                        ║
║                    TB-M1: 20% del percorso Trading                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

SYLLABUS TB MODULE 1:
├── TB 1.1 - Market Concepts (OHLCV, timeframes, order types)
├── TB 1.2 - Trading Terminology (spread, slippage, leverage, margin)
├── TB 1.3 - OHLCV Data Handling (pandas for candles)
├── TB 1.4 - Returns & Performance (daily returns, cumulative, drawdown)
├── TB 1.5 - Basic Technical Indicators (SMA, EMA)
└── TB 1.6 - Pine Script to Python Mapping

PREREQUISITI: PCEP completato
TEMPO STIMATO: 2 settimane (2-3 ore/giorno)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ══════════════════════════════════════════════════════════════════════════════
# TB 1.1 - MARKET CONCEPTS
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TB 1.1 - MARKET CONCEPTS")
print("=" * 70)

print("""
📋 OHLCV - CANDLESTICK DATA

Ogni candela contiene:
┌─────────────────────┬─────────────────────────────────────────┐
│ Open (O)            │ Prezzo all'apertura del periodo         │
│ High (H)            │ Prezzo massimo nel periodo              │
│ Low (L)             │ Prezzo minimo nel periodo               │
│ Close (C)           │ Prezzo alla chiusura del periodo        │
│ Volume (V)          │ Quantità scambiata nel periodo          │
└─────────────────────┴─────────────────────────────────────────┘

TIMEFRAMES:
- 1m, 5m, 15m, 30m  → Scalping, Day trading
- 1h, 4h            → Swing trading
- 1d, 1w            → Position trading

📋 ORDER TYPES:

┌─────────────────────┬─────────────────────────────────────────┐
│ Market Order        │ Esegui SUBITO al miglior prezzo         │
│                     │ Pro: veloce. Con: slippage              │
├─────────────────────┼─────────────────────────────────────────┤
│ Limit Order         │ Esegui SOLO a prezzo specificato o      │
│                     │ migliore. Pro: controllo. Con: può non  │
│                     │ eseguire                                │
├─────────────────────┼─────────────────────────────────────────┤
│ Stop Loss           │ Vendi se prezzo scende sotto X          │
│                     │ Protegge da perdite                     │
├─────────────────────┼─────────────────────────────────────────┤
│ Take Profit         │ Vendi se prezzo sale sopra X            │
│                     │ Blocca profitti                         │
├─────────────────────┼─────────────────────────────────────────┤
│ Stop Limit          │ Stop che diventa Limit order            │
│                     │ Più controllo ma può non eseguire       │
└─────────────────────┴─────────────────────────────────────────┘
""")

# ══════════════════════════════════════════════════════════════════════════════
# TB 1.2 - TRADING TERMINOLOGY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TB 1.2 - TRADING TERMINOLOGY")
print("=" * 70)

print("""
📋 TERMINOLOGIA ESSENZIALE

┌─────────────────────┬─────────────────────────────────────────┐
│ Bid                 │ Prezzo a cui puoi VENDERE               │
│ Ask                 │ Prezzo a cui puoi COMPRARE              │
│ Spread              │ Ask - Bid (costo implicito)             │
├─────────────────────┼─────────────────────────────────────────┤
│ Slippage            │ Differenza tra prezzo atteso e reale    │
│                     │ Comune in mercati volatili              │
├─────────────────────┼─────────────────────────────────────────┤
│ Leverage            │ Moltiplicatore del capitale             │
│                     │ 10x = controlli 10€ con 1€              │
│                     │ ⚠️ Amplifica guadagni E perdite         │
├─────────────────────┼─────────────────────────────────────────┤
│ Margin              │ Capitale richiesto come garanzia        │
│                     │ Margin call = devi aggiungere fondi     │
├─────────────────────┼─────────────────────────────────────────┤
│ Position Size       │ Quantità di asset nella posizione       │
│ Lot Size            │ Unità standard di trading               │
├─────────────────────┼─────────────────────────────────────────┤
│ Long                │ Compri, guadagni se prezzo sale         │
│ Short               │ Vendi allo scoperto, guadagni se scende │
└─────────────────────┴─────────────────────────────────────────┘

📋 CRYPTO SPECIFICO:

- Spot: compri/vendi l'asset reale
- Futures: contratti derivati con scadenza
- Perpetual: futures senza scadenza, con funding rate
- Funding Rate: pagamento periodico tra long e short
""")

# ══════════════════════════════════════════════════════════════════════════════
# TB 1.3 - OHLCV DATA HANDLING
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TB 1.3 - OHLCV DATA HANDLING")
print("=" * 70)

# Create sample OHLCV data
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=100, freq='1h')
base_price = 100

# Simulate price movement
returns = np.random.randn(100) * 0.02
prices = base_price * np.exp(np.cumsum(returns))

df = pd.DataFrame({
    'timestamp': dates,
    'open': prices,
    'high': prices * (1 + np.abs(np.random.randn(100) * 0.01)),
    'low': prices * (1 - np.abs(np.random.randn(100) * 0.01)),
    'close': prices * (1 + np.random.randn(100) * 0.005),
    'volume': np.random.randint(1000, 10000, 100)
})
df.set_index('timestamp', inplace=True)

print("🔹 STRUTTURA DATAFRAME OHLCV:")
print(df.head())
print(f"\nShape: {df.shape}")
print(f"Columns: {list(df.columns)}")

print("\n🔹 ACCESSO AI DATI:")
print(f"Ultimo close: {df['close'].iloc[-1]:.2f}")
print(f"Close precedente: {df['close'].iloc[-2]:.2f}")  # Pine: close[1]
print(f"Close 5 barre fa: {df['close'].iloc[-6]:.2f}")  # Pine: close[5]

print("\n🔹 SHIFT (equivalente a [] in Pine):")
df['close_prev'] = df['close'].shift(1)  # Pine: close[1]
df['close_5_ago'] = df['close'].shift(5)  # Pine: close[5]
print(df[['close', 'close_prev', 'close_5_ago']].tail())

# ══════════════════════════════════════════════════════════════════════════════
# TB 1.4 - RETURNS & PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TB 1.4 - RETURNS & PERFORMANCE")
print("=" * 70)

print("🔹 CALCOLO RETURNS:")

# Simple returns
df['return'] = df['close'].pct_change()  # (close - close[1]) / close[1]
print(f"Simple returns:\n{df['return'].tail()}")

# Log returns (più accurate per analisi)
df['log_return'] = np.log(df['close'] / df['close'].shift(1))
print(f"\nLog returns:\n{df['log_return'].tail()}")

# Cumulative returns
df['cum_return'] = (1 + df['return']).cumprod() - 1
print(f"\nCumulative return finale: {df['cum_return'].iloc[-1]*100:.2f}%")

print("\n🔹 DRAWDOWN:")
# Rolling maximum
df['rolling_max'] = df['close'].cummax()
# Drawdown
df['drawdown'] = (df['close'] - df['rolling_max']) / df['rolling_max']
print(f"Max Drawdown: {df['drawdown'].min()*100:.2f}%")

print("""
📋 METRICHE DI PERFORMANCE:

┌─────────────────────┬─────────────────────────────────────────┐
│ Total Return        │ (finale - iniziale) / iniziale         │
│ Annualized Return   │ (1 + total)^(365/days) - 1             │
│ Sharpe Ratio        │ (return - risk_free) / std             │
│ Sortino Ratio       │ Come Sharpe ma solo downside std       │
│ Max Drawdown        │ Massima perdita da picco               │
│ Win Rate            │ Trades vincenti / totali               │
│ Risk/Reward         │ Avg win / Avg loss                     │
└─────────────────────┴─────────────────────────────────────────┘
""")

# ══════════════════════════════════════════════════════════════════════════════
# TB 1.5 - BASIC TECHNICAL INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TB 1.5 - BASIC TECHNICAL INDICATORS")
print("=" * 70)

print("🔹 SMA (Simple Moving Average):")
print("Pine: sma(close, 20)")
print("Python:")

def sma(series, period):
    """Simple Moving Average"""
    return series.rolling(window=period).mean()

df['sma_20'] = sma(df['close'], 20)
df['sma_50'] = sma(df['close'], 50)
print(df[['close', 'sma_20', 'sma_50']].tail())

print("\n🔹 EMA (Exponential Moving Average):")
print("Pine: ema(close, 20)")
print("Python:")

def ema(series, period):
    """Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()

df['ema_20'] = ema(df['close'], 20)
print(df[['close', 'sma_20', 'ema_20']].tail())

print("\n🔹 CROSSOVER DETECTION:")
print("Pine: ta.crossover(fast, slow)")
print("Python:")

def crossover(fast, slow):
    """True when fast crosses above slow"""
    return (fast > slow) & (fast.shift(1) <= slow.shift(1))

def crossunder(fast, slow):
    """True when fast crosses below slow"""
    return (fast < slow) & (fast.shift(1) >= slow.shift(1))

df['golden_cross'] = crossover(df['sma_20'], df['sma_50'])
df['death_cross'] = crossunder(df['sma_20'], df['sma_50'])
print(f"Golden crosses: {df['golden_cross'].sum()}")
print(f"Death crosses: {df['death_cross'].sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# TB 1.6 - PINE SCRIPT TO PYTHON MAPPING
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TB 1.6 - PINE SCRIPT TO PYTHON MAPPING")
print("=" * 70)

print("""
📋 PINE SCRIPT → PYTHON REFERENCE

┌────────────────────────┬────────────────────────────────────────┐
│ PINE SCRIPT            │ PYTHON (con pandas df)                 │
├────────────────────────┼────────────────────────────────────────┤
│ close                  │ df['close']                            │
│ close[1]               │ df['close'].shift(1)                   │
│ close[n]               │ df['close'].shift(n)                   │
├────────────────────────┼────────────────────────────────────────┤
│ ta.sma(close, 20)      │ df['close'].rolling(20).mean()         │
│ ta.ema(close, 20)      │ df['close'].ewm(span=20).mean()        │
├────────────────────────┼────────────────────────────────────────┤
│ ta.highest(high, 20)   │ df['high'].rolling(20).max()           │
│ ta.lowest(low, 20)     │ df['low'].rolling(20).min()            │
├────────────────────────┼────────────────────────────────────────┤
│ ta.crossover(a, b)     │ (a > b) & (a.shift(1) <= b.shift(1))   │
│ ta.crossunder(a, b)    │ (a < b) & (a.shift(1) >= b.shift(1))   │
├────────────────────────┼────────────────────────────────────────┤
│ ta.rsi(close, 14)      │ (vedi implementazione sotto)           │
│ ta.macd(close,12,26,9) │ (vedi implementazione sotto)           │
├────────────────────────┼────────────────────────────────────────┤
│ ta.atr(14)             │ (vedi implementazione sotto)           │
│ ta.stoch(14, 3, 3)     │ (vedi implementazione sotto)           │
├────────────────────────┼────────────────────────────────────────┤
│ strategy.entry("L",    │ # Signal generation                    │
│   strategy.long)       │ df['signal'] = 1                       │
│ strategy.close("L")    │ df['signal'] = 0                       │
└────────────────────────┴────────────────────────────────────────┘

🔹 RSI IMPLEMENTATION:

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

🔹 MACD IMPLEMENTATION:

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

🔹 ATR IMPLEMENTATION:

def atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()
""")

# ══════════════════════════════════════════════════════════════════════════════
# QUIZ DI VERIFICA
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("QUIZ DI VERIFICA - TB MODULE 1")
print("=" * 70)

print("""
Q1. OHLCV: la 'V' sta per?
    A) Value  B) Volume  C) Volatility  D) Variance
    → RISPOSTA: B

Q2. Spread è?
    A) High - Low  B) Ask - Bid  C) Close - Open  D) Return
    → RISPOSTA: B

Q3. Un Market Order garantisce?
    A) Prezzo  B) Esecuzione  C) Entrambi  D) Nessuno
    → RISPOSTA: B

Q4. df['close'].shift(1) equivale a Pine Script?
    A) close  B) close[0]  C) close[1]  D) close[-1]
    → RISPOSTA: C

Q5. pct_change() calcola?
    A) Differenza  B) Return percentuale  C) Media  D) Somma
    → RISPOSTA: B

Q6. SMA in Python si calcola con?
    A) ewm()  B) rolling().mean()  C) cumsum()  D) diff()
    → RISPOSTA: B

Q7. EMA in Python si calcola con?
    A) ewm()  B) rolling().mean()  C) cumsum()  D) diff()
    → RISPOSTA: A

Q8. Max Drawdown misura?
    A) Profitto massimo  B) Perdita massima da picco  C) Volatilità  D) Sharpe
    → RISPOSTA: B

Q9. Leverage 10x significa?
    A) Guadagni 10x  B) Controlli 10x il capitale  C) Perdi 10x  D) 10 trade
    → RISPOSTA: B

Q10. Crossover(fast, slow) è True quando?
    A) fast > slow  B) fast < slow  C) fast incrocia sopra slow  D) Sempre
    → RISPOSTA: C
""")

print("\n" + "=" * 70)
print("TB MODULE 1 - FOUNDATIONS COMPLETATO!")
print("Prossimo: TB MODULE 2 - STRATEGY DEVELOPMENT")
print("=" * 70)
