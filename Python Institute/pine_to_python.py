"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    PINE SCRIPT → PYTHON CONVERSION GUIDE                     ║
║                                                                              ║
║                 Traduci le tue strategie TradingView in Python               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Questa guida ti mostra come convertire elementi Pine Script in Python.
Ogni sezione mostra l'equivalente Pine Script e Python.

═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


# ══════════════════════════════════════════════════════════════════════════════
#                    VARIABILI BUILT-IN
# ══════════════════════════════════════════════════════════════════════════════

"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         VARIABILI PINE → PYTHON                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

PINE SCRIPT:
    open, high, low, close, volume
    bar_index
    time
    
PYTHON (con DataFrame df):
    df['open'], df['high'], df['low'], df['close'], df['volume']
    df.index (o range(len(df)))
    df.index (se datetime index)
"""

def pine_variables_example(df: pd.DataFrame):
    """Equivalenti delle variabili Pine."""
    
    # Pine: close
    close = df['close']
    
    # Pine: close[1] (valore precedente)
    close_prev = df['close'].shift(1)
    
    # Pine: close[5] (5 barre fa)
    close_5_ago = df['close'].shift(5)
    
    # Pine: bar_index
    bar_index = pd.Series(range(len(df)), index=df.index)
    
    # Pine: time
    time = df.index  # se datetime index
    
    return close, close_prev, bar_index


# ══════════════════════════════════════════════════════════════════════════════
#                    INDICATORI TECNICI
# ══════════════════════════════════════════════════════════════════════════════

"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              SMA - Simple Moving Average                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

PINE SCRIPT:
    sma(close, 20)
    ta.sma(close, 20)  // Pine v5

PYTHON:
    df['close'].rolling(20).mean()
"""

def sma(series: pd.Series, period: int) -> pd.Series:
    """
    Simple Moving Average.
    Pine: sma(source, length)
    """
    return series.rolling(window=period).mean()


"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              EMA - Exponential Moving Average                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

PINE SCRIPT:
    ema(close, 20)
    ta.ema(close, 20)  // Pine v5

PYTHON:
    df['close'].ewm(span=20, adjust=False).mean()
"""

def ema(series: pd.Series, period: int) -> pd.Series:
    """
    Exponential Moving Average.
    Pine: ema(source, length)
    """
    return series.ewm(span=period, adjust=False).mean()


"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              RSI - Relative Strength Index                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

PINE SCRIPT:
    rsi(close, 14)
    ta.rsi(close, 14)  // Pine v5

PYTHON:
    (vedi funzione sotto)
"""

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index.
    Pine: rsi(source, length)
    """
    delta = series.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi_values = 100 - (100 / (1 + rs))
    
    return rsi_values


"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              MACD                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

PINE SCRIPT:
    [macdLine, signalLine, histLine] = macd(close, 12, 26, 9)
    [macdLine, signalLine, histLine] = ta.macd(close, 12, 26, 9)  // Pine v5

PYTHON:
    (vedi funzione sotto)
"""

def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD Indicator.
    Pine: macd(source, fastlen, slowlen, siglen)
    
    Returns:
        macd_line, signal_line, histogram
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              BOLLINGER BANDS                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

PINE SCRIPT:
    [middle, upper, lower] = bb(close, 20, 2)
    [middle, upper, lower] = ta.bb(close, 20, 2)  // Pine v5

PYTHON:
    (vedi funzione sotto)
"""

def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.
    Pine: bb(source, length, mult)
    
    Returns:
        middle, upper, lower
    """
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    return middle, upper, lower


"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              ATR - Average True Range                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

PINE SCRIPT:
    atr(14)
    ta.atr(14)  // Pine v5

PYTHON:
    (vedi funzione sotto)
"""

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range.
    Pine: atr(length)
    
    Requires DataFrame with high, low, close columns.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    return true_range.rolling(window=period).mean()


"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              STOCHASTIC                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

PINE SCRIPT:
    stoch(close, high, low, 14)
    ta.stoch(close, high, low, 14, 3, 3)  // Pine v5

PYTHON:
    (vedi funzione sotto)
"""

def stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator.
    Pine: stoch(close, high, low, length)
    
    Returns:
        %K, %D
    """
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    
    k = 100 * (df['close'] - low_min) / (high_max - low_min)
    d = k.rolling(window=d_period).mean()
    
    return k, d


"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              SUPERTREND                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

PINE SCRIPT:
    [supertrend, direction] = supertrend(3, 10)
    [supertrend, direction] = ta.supertrend(3, 10)  // Pine v5

PYTHON:
    (vedi funzione sotto)
"""

def supertrend(
    df: pd.DataFrame,
    multiplier: float = 3.0,
    period: int = 10
) -> Tuple[pd.Series, pd.Series]:
    """
    Supertrend Indicator.
    Pine: supertrend(factor, atrPeriod)
    
    Returns:
        supertrend_line, direction (1 = up, -1 = down)
    """
    hl2 = (df['high'] + df['low']) / 2
    atr_val = atr(df, period)
    
    upper_band = hl2 + (multiplier * atr_val)
    lower_band = hl2 - (multiplier * atr_val)
    
    supertrend_line = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    
    for i in range(period, len(df)):
        if df['close'].iloc[i] > upper_band.iloc[i-1]:
            direction.iloc[i] = 1
        elif df['close'].iloc[i] < lower_band.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]
        
        if direction.iloc[i] == 1:
            supertrend_line.iloc[i] = lower_band.iloc[i]
        else:
            supertrend_line.iloc[i] = upper_band.iloc[i]
    
    return supertrend_line, direction


# ══════════════════════════════════════════════════════════════════════════════
#                    CONDIZIONI E LOGICA
# ══════════════════════════════════════════════════════════════════════════════

"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              CROSSOVER / CROSSUNDER                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

PINE SCRIPT:
    crossover(fast, slow)   // fast incrocia sopra slow
    crossunder(fast, slow)  // fast incrocia sotto slow
    ta.crossover(fast, slow)  // Pine v5
    ta.crossunder(fast, slow)  // Pine v5

PYTHON:
    (vedi funzioni sotto)
"""

def crossover(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """
    Crossover detection.
    Pine: crossover(source1, source2)
    
    Returns True when series1 crosses above series2.
    """
    return (series1 > series2) & (series1.shift(1) <= series2.shift(1))


def crossunder(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """
    Crossunder detection.
    Pine: crossunder(source1, source2)
    
    Returns True when series1 crosses below series2.
    """
    return (series1 < series2) & (series1.shift(1) >= series2.shift(1))


"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              HIGHEST / LOWEST                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

PINE SCRIPT:
    highest(high, 20)
    lowest(low, 20)
    ta.highest(high, 20)  // Pine v5
    ta.lowest(low, 20)    // Pine v5

PYTHON:
    df['high'].rolling(20).max()
    df['low'].rolling(20).min()
"""

def highest(series: pd.Series, period: int) -> pd.Series:
    """Pine: highest(source, length)"""
    return series.rolling(window=period).max()


def lowest(series: pd.Series, period: int) -> pd.Series:
    """Pine: lowest(source, length)"""
    return series.rolling(window=period).min()


"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              CHANGE / ROC                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

PINE SCRIPT:
    change(close)      // close - close[1]
    change(close, 5)   // close - close[5]
    roc(close, 10)     // % change

PYTHON:
    df['close'].diff()
    df['close'].diff(5)
    df['close'].pct_change(10) * 100
"""

def change(series: pd.Series, period: int = 1) -> pd.Series:
    """Pine: change(source, length)"""
    return series.diff(period)


def roc(series: pd.Series, period: int) -> pd.Series:
    """Pine: roc(source, length) - Rate of Change in %"""
    return series.pct_change(period) * 100


# ══════════════════════════════════════════════════════════════════════════════
#                    SEGNALI DI TRADING
# ══════════════════════════════════════════════════════════════════════════════

"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              STRATEGY ENTRIES                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

PINE SCRIPT:
    strategy.entry("Long", strategy.long, when=buyCondition)
    strategy.entry("Short", strategy.short, when=sellCondition)
    strategy.close("Long", when=exitCondition)

PYTHON:
    (vedi classe sotto)
"""

@dataclass
class Signal:
    """Rappresenta un segnale di trading."""
    timestamp: pd.Timestamp
    signal_type: str  # 'long', 'short', 'close_long', 'close_short'
    price: float
    reason: str = ""


class PineStrategyConverter:
    """
    Converte logica Pine Script in Python.
    
    Esempio Pine:
    ```
    //@version=5
    strategy("My Strategy", overlay=true)
    
    fast = ta.ema(close, 9)
    slow = ta.ema(close, 21)
    
    longCondition = ta.crossover(fast, slow)
    shortCondition = ta.crossunder(fast, slow)
    
    if (longCondition)
        strategy.entry("Long", strategy.long)
    if (shortCondition)
        strategy.entry("Short", strategy.short)
    ```
    
    Equivalente Python:
    ```
    converter = PineStrategyConverter(df)
    converter.add_indicator('fast', ema(df['close'], 9))
    converter.add_indicator('slow', ema(df['close'], 21))
    converter.add_entry_condition('long', crossover(converter.indicators['fast'], converter.indicators['slow']))
    converter.add_entry_condition('short', crossunder(converter.indicators['fast'], converter.indicators['slow']))
    signals = converter.generate_signals()
    ```
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.indicators = {}
        self.entry_conditions = {}
        self.exit_conditions = {}
    
    def add_indicator(self, name: str, series: pd.Series):
        """Aggiungi indicatore."""
        self.indicators[name] = series
    
    def add_entry_condition(self, name: str, condition: pd.Series):
        """Aggiungi condizione di entry."""
        self.entry_conditions[name] = condition
    
    def add_exit_condition(self, name: str, condition: pd.Series):
        """Aggiungi condizione di exit."""
        self.exit_conditions[name] = condition
    
    def generate_signals(self) -> List[Signal]:
        """Genera lista di segnali."""
        signals = []
        
        for i in range(len(self.df)):
            timestamp = self.df.index[i]
            price = self.df['close'].iloc[i]
            
            for name, condition in self.entry_conditions.items():
                if condition.iloc[i]:
                    signals.append(Signal(
                        timestamp=timestamp,
                        signal_type=name,
                        price=price,
                        reason=f"Entry condition: {name}"
                    ))
            
            for name, condition in self.exit_conditions.items():
                if condition.iloc[i]:
                    signals.append(Signal(
                        timestamp=timestamp,
                        signal_type=f"close_{name}",
                        price=price,
                        reason=f"Exit condition: {name}"
                    ))
        
        return signals


# ══════════════════════════════════════════════════════════════════════════════
#                    ESEMPIO COMPLETO: SCALPING STRATEGY
# ══════════════════════════════════════════════════════════════════════════════

"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ESEMPIO: SCALPING STRATEGY CONVERSION                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

PINE SCRIPT ORIGINALE:
```pinescript
//@version=5
strategy("Scalping EMA RSI", overlay=true)

// Inputs
emaFast = input.int(9, "Fast EMA")
emaSlow = input.int(21, "Slow EMA")
rsiLength = input.int(14, "RSI Length")
rsiOversold = input.int(30, "RSI Oversold")
rsiOverbought = input.int(70, "RSI Overbought")

// Indicators
fast = ta.ema(close, emaFast)
slow = ta.ema(close, emaSlow)
rsiValue = ta.rsi(close, rsiLength)

// Conditions
emaBullish = fast > slow
emaBearish = fast < slow
rsiOversoldCond = rsiValue < rsiOversold
rsiOverboughtCond = rsiValue > rsiOverbought

// Entry
longCondition = emaBullish and rsiOversoldCond
shortCondition = emaBearish and rsiOverboughtCond

// Execute
if (longCondition)
    strategy.entry("Long", strategy.long)
if (shortCondition)
    strategy.entry("Short", strategy.short)

// Exit
if (rsiOverboughtCond and strategy.position_size > 0)
    strategy.close("Long")
if (rsiOversoldCond and strategy.position_size < 0)
    strategy.close("Short")
```
"""

class ScalpingEmaRsiStrategy:
    """
    PYTHON CONVERSION della strategia Pine Script sopra.
    """
    
    def __init__(
        self,
        ema_fast: int = 9,
        ema_slow: int = 21,
        rsi_length: int = 14,
        rsi_oversold: int = 30,
        rsi_overbought: int = 70
    ):
        # Inputs (equivalente input.int in Pine)
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_length = rsi_length
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcola tutti gli indicatori."""
        result = df.copy()
        
        # Indicators (equivalente ta.ema, ta.rsi in Pine)
        result['fast_ema'] = ema(df['close'], self.ema_fast)
        result['slow_ema'] = ema(df['close'], self.ema_slow)
        result['rsi'] = rsi(df['close'], self.rsi_length)
        
        return result
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Genera segnali di trading."""
        result = self.calculate_indicators(df)
        
        # Conditions (equivalente variabili bool in Pine)
        ema_bullish = result['fast_ema'] > result['slow_ema']
        ema_bearish = result['fast_ema'] < result['slow_ema']
        rsi_oversold_cond = result['rsi'] < self.rsi_oversold
        rsi_overbought_cond = result['rsi'] > self.rsi_overbought
        
        # Entry conditions (equivalente if + strategy.entry in Pine)
        result['long_entry'] = ema_bullish & rsi_oversold_cond
        result['short_entry'] = ema_bearish & rsi_overbought_cond
        
        # Exit conditions (equivalente strategy.close in Pine)
        result['long_exit'] = rsi_overbought_cond
        result['short_exit'] = rsi_oversold_cond
        
        # Signal column (1=long, -1=short, 0=hold)
        result['signal'] = 0
        result.loc[result['long_entry'], 'signal'] = 1
        result.loc[result['short_entry'], 'signal'] = -1
        
        return result
    
    def backtest(self, df: pd.DataFrame, initial_capital: float = 10000) -> dict:
        """Backtest semplificato."""
        signals = self.generate_signals(df)
        
        capital = initial_capital
        position = 0
        trades = []
        
        for i in range(1, len(signals)):
            price = signals['close'].iloc[i]
            
            # Long entry
            if signals['long_entry'].iloc[i] and position == 0:
                position = capital / price
                entry_price = price
                capital = 0
            
            # Long exit
            elif signals['long_exit'].iloc[i] and position > 0:
                capital = position * price
                pnl = (price - entry_price) / entry_price * 100
                trades.append({'type': 'long', 'pnl_pct': pnl})
                position = 0
        
        # Close final position
        if position > 0:
            capital = position * signals['close'].iloc[-1]
        
        return {
            'final_capital': capital,
            'total_return': (capital - initial_capital) / initial_capital * 100,
            'total_trades': len(trades),
            'trades': trades
        }


# ══════════════════════════════════════════════════════════════════════════════
#                    REFERENCE TABLE
# ══════════════════════════════════════════════════════════════════════════════

PINE_TO_PYTHON_REFERENCE = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PINE SCRIPT → PYTHON QUICK REFERENCE                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────┬────────────────────────────────────────────────┐
│ PINE SCRIPT                 │ PYTHON                                         │
├─────────────────────────────┼────────────────────────────────────────────────┤
│ close                       │ df['close']                                    │
│ close[1]                    │ df['close'].shift(1)                           │
│ close[n]                    │ df['close'].shift(n)                           │
├─────────────────────────────┼────────────────────────────────────────────────┤
│ ta.sma(close, 20)           │ df['close'].rolling(20).mean()                 │
│ ta.ema(close, 20)           │ df['close'].ewm(span=20, adjust=False).mean()  │
│ ta.rsi(close, 14)           │ rsi(df['close'], 14)  # custom function        │
│ ta.macd(close, 12, 26, 9)   │ macd(df['close'], 12, 26, 9)                   │
│ ta.bb(close, 20, 2)         │ bollinger_bands(df['close'], 20, 2)            │
│ ta.atr(14)                  │ atr(df, 14)                                    │
│ ta.stoch(close,high,low,14) │ stochastic(df, 14)                             │
├─────────────────────────────┼────────────────────────────────────────────────┤
│ ta.crossover(a, b)          │ (a > b) & (a.shift(1) <= b.shift(1))           │
│ ta.crossunder(a, b)         │ (a < b) & (a.shift(1) >= b.shift(1))           │
│ ta.highest(high, 20)        │ df['high'].rolling(20).max()                   │
│ ta.lowest(low, 20)          │ df['low'].rolling(20).min()                    │
│ ta.change(close)            │ df['close'].diff()                             │
├─────────────────────────────┼────────────────────────────────────────────────┤
│ input.int(14, "Length")     │ length = 14  # or argparse/config              │
│ input.float(0.1, "Factor")  │ factor = 0.1                                   │
│ input.bool(true, "Use X")   │ use_x = True                                   │
├─────────────────────────────┼────────────────────────────────────────────────┤
│ if (condition)              │ if condition:  OR  df.loc[condition]           │
│ condition ? a : b           │ a if condition else b  OR  np.where(cond,a,b)  │
│ and                         │ and (scalar) / & (series)                      │
│ or                          │ or (scalar) / | (series)                       │
│ not                         │ not (scalar) / ~ (series)                      │
├─────────────────────────────┼────────────────────────────────────────────────┤
│ strategy.entry("L", long)   │ signals.append(Signal('long', price))          │
│ strategy.close("L")         │ signals.append(Signal('close_long', price))    │
│ strategy.position_size      │ self.position  # track manually                │
└─────────────────────────────┴────────────────────────────────────────────────┘

NOTA: Per Series (colonne DataFrame), usa operatori bitwise (&, |, ~)
      Per valori scalari, usa operatori logici (and, or, not)
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    DEMO
# ══════════════════════════════════════════════════════════════════════════════

def demo():
    """Demo della conversione Pine → Python."""
    print("=" * 70)
    print("          PINE SCRIPT → PYTHON CONVERSION DEMO")
    print("=" * 70)
    
    # Genera dati di esempio
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='1h')
    prices = 50000 + np.cumsum(np.random.randn(100) * 100)
    
    df = pd.DataFrame({
        'open': prices * 0.999,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'close': prices,
        'volume': np.random.randint(100, 1000, 100)
    }, index=dates)
    
    print("\n1. INDICATORI")
    print("-" * 40)
    
    # SMA
    df['sma_20'] = sma(df['close'], 20)
    print(f"SMA(20) ultimo valore: {df['sma_20'].iloc[-1]:.2f}")
    
    # EMA
    df['ema_9'] = ema(df['close'], 9)
    print(f"EMA(9) ultimo valore: {df['ema_9'].iloc[-1]:.2f}")
    
    # RSI
    df['rsi_14'] = rsi(df['close'], 14)
    print(f"RSI(14) ultimo valore: {df['rsi_14'].iloc[-1]:.2f}")
    
    # MACD
    macd_line, signal_line, hist = macd(df['close'])
    print(f"MACD line ultimo valore: {macd_line.iloc[-1]:.2f}")
    
    print("\n2. CROSSOVER DETECTION")
    print("-" * 40)
    
    cross_up = crossover(df['ema_9'], df['sma_20'])
    cross_down = crossunder(df['ema_9'], df['sma_20'])
    print(f"Crossover count: {cross_up.sum()}")
    print(f"Crossunder count: {cross_down.sum()}")
    
    print("\n3. STRATEGY BACKTEST")
    print("-" * 40)
    
    strategy = ScalpingEmaRsiStrategy(
        ema_fast=9,
        ema_slow=21,
        rsi_length=14,
        rsi_oversold=30,
        rsi_overbought=70
    )
    
    results = strategy.backtest(df)
    print(f"Final capital: ${results['final_capital']:.2f}")
    print(f"Total return: {results['total_return']:.2f}%")
    print(f"Total trades: {results['total_trades']}")
    
    print("\n4. REFERENCE TABLE")
    print(PINE_TO_PYTHON_REFERENCE)


if __name__ == "__main__":
    demo()
