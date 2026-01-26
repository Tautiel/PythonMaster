"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║              SPECIALIZZAZIONE TRADING - MODULE 2                             ║
║                  Data Analysis con NumPy & Pandas                            ║
║                                                                              ║
║                  Analisi Dati per Trading & Backtesting                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Prerequisiti: PE1, PE2 completati + spec_trading_foundations

STRUTTURA:
├── Section 2.1: NumPy Fundamentals
├── Section 2.2: Pandas per Time Series
├── Section 2.3: Technical Indicators
├── Section 2.4: Data Pipeline per Trading
└── Esercizi Pratici

═══════════════════════════════════════════════════════════════════════════════
"""

# pip install numpy pandas

# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 2.1: NUMPY FUNDAMENTALS
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    2.1 TEORIA: NUMPY                                         │
└──────────────────────────────────────────────────────────────────────────────┘

NumPy: Numerical Python
- Array N-dimensionali efficienti
- Operazioni vettorizzate (NO loop Python!)
- Fondamento di Pandas, scikit-learn, etc.
"""

NUMPY_BASICS = '''
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# CREAZIONE ARRAY
# ═══════════════════════════════════════════════════════════════════════════

# Da lista
prices = np.array([100.0, 101.5, 99.8, 102.3, 101.0])

# Array speciali
zeros = np.zeros(10)           # [0, 0, 0, ...]
ones = np.ones(10)             # [1, 1, 1, ...]
range_arr = np.arange(0, 10, 0.5)  # [0, 0.5, 1.0, ...]
linspace = np.linspace(0, 1, 100)  # 100 valori da 0 a 1

# ═══════════════════════════════════════════════════════════════════════════
# OPERAZIONI VETTORIZZATE (veloci!)
# ═══════════════════════════════════════════════════════════════════════════

# Aritmetica
returns = np.diff(prices) / prices[:-1]  # Returns percentuali
log_returns = np.log(prices[1:] / prices[:-1])  # Log returns

# Statistiche
mean = np.mean(prices)
std = np.std(prices)
max_price = np.max(prices)
min_price = np.min(prices)

# Cumulative
cumsum = np.cumsum(returns)    # Somma cumulativa
cumprod = np.cumprod(1 + returns)  # Prodotto cumulativo (equity curve)

# ═══════════════════════════════════════════════════════════════════════════
# SLICING E INDEXING
# ═══════════════════════════════════════════════════════════════════════════

prices[-1]      # Ultimo elemento
prices[-5:]     # Ultimi 5 elementi
prices[::2]     # Ogni 2 elementi
prices[prices > 100]  # Boolean indexing

# ═══════════════════════════════════════════════════════════════════════════
# ROLLING WINDOWS (manuale)
# ═══════════════════════════════════════════════════════════════════════════

def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple Moving Average."""
    result = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        result[i] = np.mean(arr[i - window + 1:i + 1])
    return result

# Più efficiente con convolve
def sma_fast(arr: np.ndarray, window: int) -> np.ndarray:
    """SMA con convolve (più veloce)."""
    weights = np.ones(window) / window
    return np.convolve(arr, weights, mode='valid')
'''


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 2.2: PANDAS PER TIME SERIES
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    2.2 TEORIA: PANDAS                                        │
└──────────────────────────────────────────────────────────────────────────────┘

Pandas: Data Analysis Library
- DataFrame: Tabella 2D con indice
- Series: Colonna singola con indice
- Perfetto per time series finanziarie
"""

PANDAS_TRADING = '''
import pandas as pd
import numpy as np
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════
# CREARE DATAFRAME DA OHLCV
# ═══════════════════════════════════════════════════════════════════════════

# Dati da ccxt (lista di liste)
ohlcv_data = [
    [1704067200000, 42000, 42500, 41800, 42300, 1234.5],
    [1704070800000, 42300, 42800, 42200, 42700, 1456.7],
    # ...
]

# Converti in DataFrame
df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# Converti timestamp in datetime e imposta come indice
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

# ═══════════════════════════════════════════════════════════════════════════
# ANALISI BASE
# ═══════════════════════════════════════════════════════════════════════════

# Statistiche
df.describe()           # Statistiche descrittive
df['close'].mean()      # Media
df['close'].std()       # Deviazione standard
df['volume'].sum()      # Volume totale

# Returns
df['returns'] = df['close'].pct_change()
df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

# ═══════════════════════════════════════════════════════════════════════════
# ROLLING OPERATIONS (FONDAMENTALE per indicatori!)
# ═══════════════════════════════════════════════════════════════════════════

# Simple Moving Average
df['sma_10'] = df['close'].rolling(window=10).mean()
df['sma_20'] = df['close'].rolling(window=20).mean()

# Exponential Moving Average
df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()

# Rolling standard deviation (per Bollinger Bands)
df['std_20'] = df['close'].rolling(window=20).std()

# Rolling max/min (per Support/Resistance)
df['high_20'] = df['high'].rolling(window=20).max()
df['low_20'] = df['low'].rolling(window=20).min()

# ═══════════════════════════════════════════════════════════════════════════
# RESAMPLING (cambiare timeframe)
# ═══════════════════════════════════════════════════════════════════════════

# Da 1H a 4H
df_4h = df.resample('4H').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
})

# Da 1H a 1D
df_daily = df.resample('1D').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
})

# ═══════════════════════════════════════════════════════════════════════════
# SHIFT E LAG (per segnali e confronti)
# ═══════════════════════════════════════════════════════════════════════════

# Valori precedenti
df['prev_close'] = df['close'].shift(1)  # Close di ieri

# Valori futuri (per calcolare target in backtest)
df['next_close'] = df['close'].shift(-1)  # Close di domani

# ═══════════════════════════════════════════════════════════════════════════
# FILTERING E QUERY
# ═══════════════════════════════════════════════════════════════════════════

# Boolean filtering
bullish_days = df[df['close'] > df['open']]
high_volume = df[df['volume'] > df['volume'].mean()]

# Query string
big_moves = df.query('returns > 0.02 or returns < -0.02')

# Multiple conditions
signals = df[(df['sma_10'] > df['sma_20']) & (df['volume'] > 1000)]
'''


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 2.3: TECHNICAL INDICATORS
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    2.3 TEORIA: INDICATORI TECNICI                            │
└──────────────────────────────────────────────────────────────────────────────┘
"""

TECHNICAL_INDICATORS = '''
import pandas as pd
import numpy as np

class TechnicalIndicators:
    """
    Libreria di indicatori tecnici per trading.
    Tutti i metodi sono statici e lavorano su Series/DataFrame.
    """
    
    # ═══════════════════════════════════════════════════════════════════════
    # TREND INDICATORS
    # ═══════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return series.rolling(window=period).mean()
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        MACD (Moving Average Convergence Divergence).
        
        Returns:
            DataFrame con colonne: macd, signal, histogram
        """
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        })
    
    # ═══════════════════════════════════════════════════════════════════════
    # MOMENTUM INDICATORS
    # ═══════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """
        RSI (Relative Strength Index).
        
        0-30: Oversold
        70-100: Overbought
        """
        delta = close.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                   k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """
        Stochastic Oscillator.
        
        Returns:
            DataFrame con colonne: k, d
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_period).mean()
        
        return pd.DataFrame({'k': k, 'd': d})
    
    # ═══════════════════════════════════════════════════════════════════════
    # VOLATILITY INDICATORS
    # ═══════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Bollinger Bands.
        
        Returns:
            DataFrame con colonne: middle, upper, lower, bandwidth
        """
        middle = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        bandwidth = (upper - lower) / middle * 100
        
        return pd.DataFrame({
            'middle': middle,
            'upper': upper,
            'lower': lower,
            'bandwidth': bandwidth
        })
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        ATR (Average True Range).
        Misura la volatilità.
        """
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(alpha=1/period, min_periods=period).mean()
        
        return atr
    
    # ═══════════════════════════════════════════════════════════════════════
    # VOLUME INDICATORS
    # ═══════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        VWAP (Volume Weighted Average Price).
        Reset giornaliero se vuoi il VWAP tradizionale.
        """
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        OBV (On Balance Volume).
        Trend di accumulazione/distribuzione.
        """
        direction = np.where(close > close.shift(1), 1, 
                    np.where(close < close.shift(1), -1, 0))
        return (volume * direction).cumsum()


# ═══════════════════════════════════════════════════════════════════════════
# ESEMPIO DI USO
# ═══════════════════════════════════════════════════════════════════════════

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge tutti gli indicatori al DataFrame OHLCV."""
    
    ti = TechnicalIndicators
    
    # Trend
    df['sma_10'] = ti.sma(df['close'], 10)
    df['sma_20'] = ti.sma(df['close'], 20)
    df['ema_10'] = ti.ema(df['close'], 10)
    
    # MACD
    macd_df = ti.macd(df['close'])
    df = pd.concat([df, macd_df], axis=1)
    
    # Momentum
    df['rsi'] = ti.rsi(df['close'])
    stoch_df = ti.stochastic(df['high'], df['low'], df['close'])
    df = pd.concat([df, stoch_df.add_prefix('stoch_')], axis=1)
    
    # Volatility
    bb_df = ti.bollinger_bands(df['close'])
    df = pd.concat([df, bb_df.add_prefix('bb_')], axis=1)
    df['atr'] = ti.atr(df['high'], df['low'], df['close'])
    
    # Volume
    df['obv'] = ti.obv(df['close'], df['volume'])
    
    return df
'''


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 2.4: DATA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    2.4 TEORIA: DATA PIPELINE                                 │
└──────────────────────────────────────────────────────────────────────────────┘
"""

DATA_PIPELINE = '''
import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime, timedelta


class TradingDataPipeline:
    """
    Pipeline per preparazione dati trading.
    Fetch -> Clean -> Transform -> Indicators -> Signals
    """
    
    def __init__(self, exchange_client):
        self.exchange = exchange_client
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', 
                    limit: int = 500) -> pd.DataFrame:
        """Fetch e converti OHLCV in DataFrame."""
        
        raw = self.exchange.get_ohlcv(symbol, timeframe, limit)
        
        df = pd.DataFrame(raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pulisci dati (NaN, outliers, gaps)."""
        
        # Rimuovi duplicati
        df = df[~df.index.duplicated(keep='first')]
        
        # Ordina per tempo
        df = df.sort_index()
        
        # Forward fill per piccoli gap
        df = df.fillna(method='ffill', limit=3)
        
        # Rimuovi righe con NaN rimanenti
        df = df.dropna()
        
        # Rimuovi outliers (>5 std)
        for col in ['open', 'high', 'low', 'close']:
            mean = df[col].mean()
            std = df[col].std()
            df = df[(df[col] > mean - 5*std) & (df[col] < mean + 5*std)]
        
        return df
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggiungi features per ML/analisi."""
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatility
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Price position
        df['price_vs_sma20'] = df['close'] / df['close'].rolling(20).mean() - 1
        
        # Momentum
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Time features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        
        # Target (per ML - next candle direction)
        df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame, strategy: str = 'sma_cross') -> pd.DataFrame:
        """Genera segnali di trading."""
        
        if strategy == 'sma_cross':
            df['sma_fast'] = df['close'].rolling(10).mean()
            df['sma_slow'] = df['close'].rolling(20).mean()
            
            # Segnale: 1 = buy, -1 = sell, 0 = hold
            df['signal'] = 0
            df.loc[df['sma_fast'] > df['sma_slow'], 'signal'] = 1
            df.loc[df['sma_fast'] < df['sma_slow'], 'signal'] = -1
            
            # Trade solo su cambio segnale
            df['trade'] = df['signal'].diff().fillna(0)
        
        return df
    
    def prepare_data(self, symbol: str, timeframe: str = '1h') -> pd.DataFrame:
        """Pipeline completa."""
        
        df = self.fetch_ohlcv(symbol, timeframe)
        df = self.clean_data(df)
        df = self.add_features(df)
        df = self.generate_signals(df)
        
        # Rimuovi prime righe con NaN (da rolling)
        df = df.dropna()
        
        return df
'''


# ══════════════════════════════════════════════════════════════════════════════
#                    ESERCIZI PRATICI
# ══════════════════════════════════════════════════════════════════════════════

EXERCISES = """
══════════════════════════════════════════════════════════════════════════════
                    ESERCIZI PRATICI
══════════════════════════════════════════════════════════════════════════════

ESERCIZIO 1: NumPy Performance
──────────────────────────────
Confronta il tempo di esecuzione per calcolare SMA(20) su 10000 elementi:
a) Con loop Python
b) Con np.convolve
c) Con pandas rolling

ESERCIZIO 2: DataFrame OHLCV
────────────────────────────
Crea un DataFrame con 1000 candele simulate e:
a) Calcola il range medio giornaliero (high-low)
b) Identifica le candele con volume > 2x media
c) Trova i giorni con gap > 1% rispetto alla chiusura precedente

ESERCIZIO 3: Indicatori Custom
──────────────────────────────
Implementa:
a) Supertrend indicator
b) Ichimoku Cloud (tenkan, kijun, senkou A/B)
c) Williams %R

ESERCIZIO 4: Signal Generation
──────────────────────────────
Crea una classe che generi segnali basati su:
a) RSI oversold/overbought con conferma MACD
b) Bollinger Band squeeze + breakout
c) Volume spike + price action

ESERCIZIO 5: Data Quality
─────────────────────────
Scrivi funzioni per:
a) Detectare gap temporali nei dati
b) Identificare candele anomale (volume zero, OHLC inconsistenti)
c) Riempire gap con interpolazione appropriata


══════════════════════════════════════════════════════════════════════════════
"""


if __name__ == "__main__":
    print("=" * 70)
    print("SPECIALIZZAZIONE TRADING - Module 2: Data Analysis")
    print("=" * 70)
    print("""
    Contenuti:
    print(NUMPY_BASICS)          # NumPy fondamentali
    print(PANDAS_TRADING)        # Pandas per trading
    print(TECHNICAL_INDICATORS)  # Indicatori tecnici
    print(DATA_PIPELINE)         # Pipeline dati
    print(EXERCISES)             # Esercizi pratici
    """)
