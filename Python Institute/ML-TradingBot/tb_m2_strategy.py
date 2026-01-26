#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    TRADING BOT - MODULE 2                                    ║
║                    STRATEGY DEVELOPMENT                                       ║
║                    TB-M2: 25% del percorso Trading (BIGGEST!)                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

SYLLABUS TB MODULE 2:
├── TB 2.1 - Strategy Class Architecture (OOP from PCAP)
├── TB 2.2 - Signal Generation (entry/exit logic)
├── TB 2.3 - Advanced Indicators (RSI, MACD, Bollinger, ATR)
├── TB 2.4 - Multiple Timeframe Analysis
├── TB 2.5 - Risk Management (position sizing, stop loss, take profit)
└── TB 2.6 - Strategy Examples (trend following, mean reversion, breakout)

PREREQUISITI: TB-M1, PCAP (OOP) completati
TEMPO STIMATO: 3 settimane (2-3 ore/giorno)
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum

# ══════════════════════════════════════════════════════════════════════════════
# TB 2.1 - STRATEGY CLASS ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TB 2.1 - STRATEGY CLASS ARCHITECTURE")
print("=" * 70)

print("""
📋 ARCHITETTURA OOP PER STRATEGIE

Usiamo le competenze PCAP (OOP, ABC, inheritance) per creare
un framework riutilizzabile per le strategie.

PRINCIPI:
- Strategy base class astratta
- Ogni strategia implementa generate_signals()
- Separazione tra logica e esecuzione
""")

class Signal(Enum):
    """Enum per segnali di trading"""
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class TradeSignal:
    """Dataclass per segnale di trading"""
    timestamp: pd.Timestamp
    signal: Signal
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = 1.0


class BaseStrategy(ABC):
    """
    Classe base astratta per tutte le strategie.
    Ogni strategia deve implementare generate_signals().
    """
    
    def __init__(self, name: str):
        self.name = name
        self.parameters = {}
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Genera segnali di trading.
        
        Args:
            df: DataFrame con OHLCV data
            
        Returns:
            Series con segnali (1=buy, -1=sell, 0=hold)
        """
        pass
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Override per calcolare indicatori specifici"""
        return df
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Valida che il DataFrame abbia le colonne necessarie"""
        required = ['open', 'high', 'low', 'close', 'volume']
        return all(col in df.columns for col in required)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


print("🔹 ESEMPIO IMPLEMENTAZIONE:")
print("""
class SMACrossStrategy(BaseStrategy):
    def __init__(self, fast_period=10, slow_period=30):
        super().__init__("SMA Cross")
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def generate_signals(self, df):
        df = df.copy()
        df['sma_fast'] = df['close'].rolling(self.fast_period).mean()
        df['sma_slow'] = df['close'].rolling(self.slow_period).mean()
        
        signals = pd.Series(0, index=df.index)
        signals[(df['sma_fast'] > df['sma_slow']) & 
                (df['sma_fast'].shift(1) <= df['sma_slow'].shift(1))] = 1
        signals[(df['sma_fast'] < df['sma_slow']) & 
                (df['sma_fast'].shift(1) >= df['sma_slow'].shift(1))] = -1
        
        return signals
""")

# ══════════════════════════════════════════════════════════════════════════════
# TB 2.2 - SIGNAL GENERATION
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TB 2.2 - SIGNAL GENERATION")
print("=" * 70)

print("""
📋 LOGICA DEI SEGNALI

ENTRY CONDITIONS (quando entrare):
- Crossover di medie mobili
- RSI oversold/overbought
- Breakout di livelli
- Pattern recognition

EXIT CONDITIONS (quando uscire):
- Stop loss raggiunto
- Take profit raggiunto
- Segnale opposto
- Time-based exit
- Trailing stop

SIGNAL VALUES:
- 1 = BUY (apri long)
- -1 = SELL (apri short o chiudi long)
- 0 = HOLD (nessuna azione)
""")

# Helper functions
def sma(series, period):
    return series.rolling(window=period).mean()

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def crossover(fast, slow):
    return (fast > slow) & (fast.shift(1) <= slow.shift(1))

def crossunder(fast, slow):
    return (fast < slow) & (fast.shift(1) >= slow.shift(1))

# Create sample data
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=200, freq='1h')
returns = np.random.randn(200) * 0.02
prices = 100 * np.exp(np.cumsum(returns))

df = pd.DataFrame({
    'open': prices,
    'high': prices * (1 + np.abs(np.random.randn(200) * 0.01)),
    'low': prices * (1 - np.abs(np.random.randn(200) * 0.01)),
    'close': prices * (1 + np.random.randn(200) * 0.005),
    'volume': np.random.randint(1000, 10000, 200)
}, index=dates)

print("🔹 ESEMPIO: SMA CROSSOVER SIGNALS")

class SMACrossStrategy(BaseStrategy):
    def __init__(self, fast_period=10, slow_period=30):
        super().__init__("SMA Cross")
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def generate_signals(self, df):
        df = df.copy()
        df['sma_fast'] = sma(df['close'], self.fast_period)
        df['sma_slow'] = sma(df['close'], self.slow_period)
        
        signals = pd.Series(0, index=df.index)
        signals[crossover(df['sma_fast'], df['sma_slow'])] = 1
        signals[crossunder(df['sma_fast'], df['sma_slow'])] = -1
        
        return signals

strategy = SMACrossStrategy(fast_period=10, slow_period=30)
signals = strategy.generate_signals(df)
print(f"Total signals: {(signals != 0).sum()}")
print(f"Buy signals: {(signals == 1).sum()}")
print(f"Sell signals: {(signals == -1).sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# TB 2.3 - ADVANCED INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TB 2.3 - ADVANCED INDICATORS")
print("=" * 70)

print("🔹 RSI (Relative Strength Index):")

def rsi(series, period=14):
    """
    RSI: misura momentum, range 0-100
    Oversold: < 30
    Overbought: > 70
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

df['rsi'] = rsi(df['close'], 14)
print(f"RSI ultimo: {df['rsi'].iloc[-1]:.2f}")
print(f"RSI < 30 (oversold): {(df['rsi'] < 30).sum()} volte")
print(f"RSI > 70 (overbought): {(df['rsi'] > 70).sum()} volte")

print("\n🔹 MACD (Moving Average Convergence Divergence):")

def macd(series, fast=12, slow=26, signal=9):
    """
    MACD: differenza tra EMA veloce e lenta
    Signal line: EMA del MACD
    Histogram: MACD - Signal
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

df['macd'], df['macd_signal'], df['macd_hist'] = macd(df['close'])
print(f"MACD ultimo: {df['macd'].iloc[-1]:.4f}")
print(f"Signal ultimo: {df['macd_signal'].iloc[-1]:.4f}")

print("\n🔹 BOLLINGER BANDS:")

def bollinger_bands(series, period=20, std_dev=2):
    """
    Bollinger Bands: SMA +/- n standard deviations
    """
    middle = sma(series, period)
    std = series.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower

df['bb_upper'], df['bb_middle'], df['bb_lower'] = bollinger_bands(df['close'])
print(f"BB Upper: {df['bb_upper'].iloc[-1]:.2f}")
print(f"BB Middle: {df['bb_middle'].iloc[-1]:.2f}")
print(f"BB Lower: {df['bb_lower'].iloc[-1]:.2f}")

print("\n🔹 ATR (Average True Range):")

def atr(df, period=14):
    """
    ATR: misura volatilità media
    Utile per stop loss dinamici
    """
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

df['atr'] = atr(df, 14)
print(f"ATR ultimo: {df['atr'].iloc[-1]:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# TB 2.4 - MULTIPLE TIMEFRAME ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TB 2.4 - MULTIPLE TIMEFRAME ANALYSIS")
print("=" * 70)

print("""
📋 ANALISI MULTI-TIMEFRAME

Principio: timeframe maggiore = direzione, minore = entry

ESEMPIO:
- Daily: determina trend (up/down)
- 4H: conferma trend
- 1H: entry point

IMPLEMENTAZIONE:
""")

def resample_ohlcv(df, timeframe):
    """
    Resample OHLCV a timeframe superiore
    
    Args:
        df: DataFrame con OHLCV
        timeframe: '4h', '1d', etc.
    """
    return df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

print("🔹 ESEMPIO RESAMPLE:")
df_4h = resample_ohlcv(df, '4h')
print(f"Original shape (1h): {df.shape}")
print(f"Resampled shape (4h): {df_4h.shape}")

print("""
🔹 STRATEGIA MULTI-TIMEFRAME:

class MultiTimeframeStrategy(BaseStrategy):
    def generate_signals(self, df_1h, df_4h, df_1d):
        # Trend dal daily
        df_1d['trend'] = df_1d['close'] > df_1d['close'].rolling(20).mean()
        
        # Conferma da 4H
        df_4h['sma_fast'] = df_4h['close'].rolling(10).mean()
        df_4h['sma_slow'] = df_4h['close'].rolling(30).mean()
        df_4h['confirm'] = df_4h['sma_fast'] > df_4h['sma_slow']
        
        # Entry da 1H (solo se trend e conferma OK)
        signals = pd.Series(0, index=df_1h.index)
        # ... logica entry
        
        return signals
""")

# ══════════════════════════════════════════════════════════════════════════════
# TB 2.5 - RISK MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TB 2.5 - RISK MANAGEMENT")
print("=" * 70)

print("""
📋 RISK MANAGEMENT ESSENZIALE

⚠️ REGOLA D'ORO: Mai rischiare più del 1-2% del capitale per trade!

┌─────────────────────┬─────────────────────────────────────────┐
│ CONCETTO            │ DESCRIZIONE                             │
├─────────────────────┼─────────────────────────────────────────┤
│ Risk per Trade      │ % capitale che puoi perdere (1-2%)      │
│ Position Size       │ Quanto comprare basato sul risk         │
│ Stop Loss           │ Prezzo a cui chiudi in perdita          │
│ Take Profit         │ Prezzo a cui chiudi in profitto         │
│ Risk/Reward Ratio   │ TP distance / SL distance (min 1:2)     │
└─────────────────────┴─────────────────────────────────────────┘
""")

print("🔹 POSITION SIZING:")

def calculate_position_size(
    account_balance: float,
    risk_percent: float,
    entry_price: float,
    stop_loss: float
) -> float:
    """
    Calcola position size basato sul risk.
    
    Args:
        account_balance: Capitale totale
        risk_percent: % da rischiare (es: 0.01 = 1%)
        entry_price: Prezzo di entry
        stop_loss: Prezzo stop loss
    
    Returns:
        Numero di unità da comprare
    """
    risk_amount = account_balance * risk_percent
    risk_per_unit = abs(entry_price - stop_loss)
    position_size = risk_amount / risk_per_unit
    return position_size

# Esempio
balance = 10000
risk = 0.01  # 1%
entry = 100
stop = 95  # 5% sotto entry

size = calculate_position_size(balance, risk, entry, stop)
print(f"Account: ${balance}")
print(f"Risk: {risk*100}% = ${balance * risk}")
print(f"Entry: ${entry}, Stop: ${stop}")
print(f"Position size: {size:.2f} unità")
print(f"Valore posizione: ${size * entry:.2f}")

print("\n🔹 ATR-BASED STOP LOSS:")

def atr_stop_loss(df, multiplier=2):
    """
    Stop loss basato su ATR (più dinamico)
    """
    current_atr = df['atr'].iloc[-1]
    current_price = df['close'].iloc[-1]
    
    stop_long = current_price - (current_atr * multiplier)
    stop_short = current_price + (current_atr * multiplier)
    
    return stop_long, stop_short

stop_long, stop_short = atr_stop_loss(df, multiplier=2)
print(f"\nPrezzo attuale: {df['close'].iloc[-1]:.2f}")
print(f"ATR: {df['atr'].iloc[-1]:.4f}")
print(f"Stop Loss (long): {stop_long:.2f}")
print(f"Stop Loss (short): {stop_short:.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# TB 2.6 - STRATEGY EXAMPLES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TB 2.6 - STRATEGY EXAMPLES")
print("=" * 70)

print("🔹 STRATEGY 1: RSI MEAN REVERSION")

class RSIMeanReversionStrategy(BaseStrategy):
    """
    Compra quando RSI < oversold
    Vendi quando RSI > overbought
    """
    def __init__(self, rsi_period=14, oversold=30, overbought=70):
        super().__init__("RSI Mean Reversion")
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, df):
        df = df.copy()
        df['rsi'] = rsi(df['close'], self.rsi_period)
        
        signals = pd.Series(0, index=df.index)
        signals[df['rsi'] < self.oversold] = 1   # Buy oversold
        signals[df['rsi'] > self.overbought] = -1  # Sell overbought
        
        return signals

rsi_strategy = RSIMeanReversionStrategy()
rsi_signals = rsi_strategy.generate_signals(df)
print(f"RSI Strategy - Buy signals: {(rsi_signals == 1).sum()}")
print(f"RSI Strategy - Sell signals: {(rsi_signals == -1).sum()}")

print("\n🔹 STRATEGY 2: BOLLINGER BAND BREAKOUT")

class BollingerBreakoutStrategy(BaseStrategy):
    """
    Compra quando chiude sopra upper band
    Vendi quando chiude sotto lower band
    """
    def __init__(self, period=20, std_dev=2):
        super().__init__("Bollinger Breakout")
        self.period = period
        self.std_dev = std_dev
    
    def generate_signals(self, df):
        df = df.copy()
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = bollinger_bands(
            df['close'], self.period, self.std_dev
        )
        
        signals = pd.Series(0, index=df.index)
        signals[df['close'] > df['bb_upper']] = 1   # Breakout up
        signals[df['close'] < df['bb_lower']] = -1  # Breakout down
        
        return signals

bb_strategy = BollingerBreakoutStrategy()
bb_signals = bb_strategy.generate_signals(df)
print(f"BB Strategy - Buy signals: {(bb_signals == 1).sum()}")
print(f"BB Strategy - Sell signals: {(bb_signals == -1).sum()}")

print("\n🔹 STRATEGY 3: MACD + RSI COMBO")

class MACDRSIStrategy(BaseStrategy):
    """
    Combina MACD crossover con RSI filter
    Buy: MACD cross up AND RSI < 70
    Sell: MACD cross down AND RSI > 30
    """
    def __init__(self):
        super().__init__("MACD + RSI")
    
    def generate_signals(self, df):
        df = df.copy()
        df['macd'], df['macd_signal'], _ = macd(df['close'])
        df['rsi'] = rsi(df['close'])
        
        macd_cross_up = crossover(df['macd'], df['macd_signal'])
        macd_cross_down = crossunder(df['macd'], df['macd_signal'])
        
        signals = pd.Series(0, index=df.index)
        signals[macd_cross_up & (df['rsi'] < 70)] = 1
        signals[macd_cross_down & (df['rsi'] > 30)] = -1
        
        return signals

combo_strategy = MACDRSIStrategy()
combo_signals = combo_strategy.generate_signals(df)
print(f"Combo Strategy - Buy signals: {(combo_signals == 1).sum()}")
print(f"Combo Strategy - Sell signals: {(combo_signals == -1).sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# QUIZ DI VERIFICA
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("QUIZ DI VERIFICA - TB MODULE 2")
print("=" * 70)

print("""
Q1. Una strategia dovrebbe ereditare da?
    A) object  B) ABC  C) BaseStrategy  D) dict
    → RISPOSTA: C (o B per la base)

Q2. RSI oversold è tipicamente?
    A) < 30  B) > 70  C) = 50  D) < 0
    → RISPOSTA: A

Q3. MACD è la differenza tra?
    A) SMA  B) Due EMA  C) High e Low  D) Open e Close
    → RISPOSTA: B

Q4. Risk per trade consigliato è?
    A) 10%  B) 50%  C) 1-2%  D) 100%
    → RISPOSTA: C

Q5. Position size dipende da?
    A) Solo capitale  B) Solo stop loss  C) Entrambi + risk%  D) Nulla
    → RISPOSTA: C

Q6. ATR misura?
    A) Trend  B) Volatilità  C) Volume  D) Momentum
    → RISPOSTA: B

Q7. Bollinger upper band è?
    A) SMA + std  B) SMA - std  C) EMA  D) High
    → RISPOSTA: A

Q8. Multi-timeframe: timeframe maggiore indica?
    A) Entry  B) Exit  C) Trend direction  D) Volume
    → RISPOSTA: C

Q9. Risk/Reward 1:2 significa?
    A) Rischio 2x reward  B) Reward 2x rischio  C) Uguali  D) Nessuno
    → RISPOSTA: B

Q10. crossover(a, b) è True quando?
    A) a > b  B) a < b  C) a incrocia sopra b  D) a incrocia sotto b
    → RISPOSTA: C
""")

print("\n" + "=" * 70)
print("TB MODULE 2 - STRATEGY DEVELOPMENT COMPLETATO!")
print("Prossimo: TB MODULE 3 - BACKTESTING")
print("=" * 70)
