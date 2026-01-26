"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    SPECIALIZATION MODULE: AI & ML                            ║
║                   Machine Learning per Trading Systems                       ║
║                                                                              ║
║                   Applicazione Pratica delle Certificazioni                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Questo modulo introduce AI/ML nel contesto trading.
Prerequisiti: Certificazioni PCEP + PCAP completate.

STRUTTURA:
├── Part 1: NumPy Essentials (Data manipulation)
├── Part 2: Pandas for Financial Data
├── Part 3: Scikit-Learn Basics
├── Part 4: ML Models for Trading
├── Part 5: Feature Engineering per Trading
├── Part 6: Backtesting con ML
└── Part 7: Neural Networks Preview

LIBRERIE DA INSTALLARE:
pip install numpy pandas scikit-learn matplotlib

═══════════════════════════════════════════════════════════════════════════════
"""

# ══════════════════════════════════════════════════════════════════════════════
#                    PART 1: NUMPY ESSENTIALS
# ══════════════════════════════════════════════════════════════════════════════
"""
NumPy: Fondamento di tutto il ML in Python.
Array multidimensionali + operazioni vettorizzate.
"""

NUMPY_ESSENTIALS = '''
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# CREAZIONE ARRAY
# ═══════════════════════════════════════════════════════════════════════════

# Da lista
arr = np.array([1, 2, 3, 4, 5])

# Array speciali
zeros = np.zeros((3, 4))        # Matrice 3x4 di zeri
ones = np.ones((2, 3))          # Matrice 2x3 di uni
identity = np.eye(3)            # Matrice identità 3x3
range_arr = np.arange(0, 10, 2) # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5) # 5 valori equidistanti tra 0 e 1

# Random (importante per ML!)
np.random.seed(42)  # Riproducibilità
random_arr = np.random.rand(3, 3)       # Uniforme [0,1)
normal_arr = np.random.randn(3, 3)      # Normale standard
randint_arr = np.random.randint(0, 10, (3, 3))  # Interi

# ═══════════════════════════════════════════════════════════════════════════
# PROPRIETÀ E RESHAPE
# ═══════════════════════════════════════════════════════════════════════════

arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.shape)    # (2, 3)
print(arr.ndim)     # 2 (dimensioni)
print(arr.size)     # 6 (elementi totali)
print(arr.dtype)    # int64 (tipo dati)

# Reshape
reshaped = arr.reshape(3, 2)    # Nuova forma
flattened = arr.flatten()       # 1D
transposed = arr.T              # Trasposta

# ═══════════════════════════════════════════════════════════════════════════
# INDEXING E SLICING
# ═══════════════════════════════════════════════════════════════════════════

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Basic
print(arr[0, 1])     # 2 (riga 0, colonna 1)
print(arr[1])        # [4, 5, 6] (riga 1)
print(arr[:, 1])     # [2, 5, 8] (colonna 1)
print(arr[0:2, 1:3]) # Sotto-matrice

# Boolean indexing (MOLTO usato in ML)
mask = arr > 5
print(arr[mask])     # [6, 7, 8, 9]
arr[arr < 3] = 0     # Sostituisci valori

# Fancy indexing
indices = [0, 2]
print(arr[indices])  # Righe 0 e 2

# ═══════════════════════════════════════════════════════════════════════════
# OPERAZIONI VETTORIZZATE (Broadcasting)
# ═══════════════════════════════════════════════════════════════════════════

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Element-wise
print(a + b)         # [5, 7, 9]
print(a * b)         # [4, 10, 18]
print(a ** 2)        # [1, 4, 9]
print(np.sqrt(a))    # [1., 1.41, 1.73]

# Broadcasting
matrix = np.array([[1, 2], [3, 4]])
vector = np.array([10, 20])
print(matrix + vector)  # Aggiunge a ogni riga

# ═══════════════════════════════════════════════════════════════════════════
# FUNZIONI STATISTICHE (Trading!)
# ═══════════════════════════════════════════════════════════════════════════

prices = np.array([100, 102, 98, 105, 103, 107, 110])

print(np.mean(prices))      # Media
print(np.median(prices))    # Mediana
print(np.std(prices))       # Deviazione standard
print(np.var(prices))       # Varianza
print(np.min(prices), np.max(prices))  # Min/Max
print(np.argmin(prices))    # Indice del minimo
print(np.percentile(prices, 95))  # 95° percentile

# Returns
returns = np.diff(prices) / prices[:-1]
print(f"Returns: {returns}")
print(f"Cumulative return: {np.prod(1 + returns) - 1}")

# Rolling operations (simulazione)
def rolling_mean(arr, window):
    return np.convolve(arr, np.ones(window)/window, mode='valid')

# ═══════════════════════════════════════════════════════════════════════════
# ALGEBRA LINEARE
# ═══════════════════════════════════════════════════════════════════════════

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(np.dot(A, B))     # Prodotto matriciale
print(A @ B)            # Stessa cosa (Python 3.5+)
print(np.linalg.inv(A)) # Inversa
print(np.linalg.det(A)) # Determinante
eigenvalues, eigenvectors = np.linalg.eig(A)
'''


# ══════════════════════════════════════════════════════════════════════════════
#                    PART 2: PANDAS FOR FINANCIAL DATA
# ══════════════════════════════════════════════════════════════════════════════

PANDAS_ESSENTIALS = '''
import pandas as pd
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# SERIES E DATAFRAME
# ═══════════════════════════════════════════════════════════════════════════

# Series
prices = pd.Series([100, 102, 98, 105], 
                   index=['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
                   name='BTC_Price')

# DataFrame
data = {
    'open': [100, 102, 98, 105],
    'high': [103, 105, 102, 108],
    'low': [99, 101, 97, 104],
    'close': [102, 98, 105, 107],
    'volume': [1000, 1200, 800, 1500]
}
df = pd.DataFrame(data)
df.index = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'])

# ═══════════════════════════════════════════════════════════════════════════
# SELEZIONE DATI
# ═══════════════════════════════════════════════════════════════════════════

# Colonne
print(df['close'])           # Series
print(df[['open', 'close']]) # DataFrame

# Righe con loc (label) e iloc (indice)
print(df.loc['2024-01-02'])
print(df.iloc[1])
print(df.loc['2024-01-01':'2024-01-03', ['open', 'close']])

# Boolean filtering
print(df[df['volume'] > 1000])
print(df[(df['close'] > 100) & (df['volume'] > 800)])

# ═══════════════════════════════════════════════════════════════════════════
# OPERAZIONI FINANZIARIE
# ═══════════════════════════════════════════════════════════════════════════

# Returns
df['returns'] = df['close'].pct_change()
df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1

# Moving averages
df['SMA_5'] = df['close'].rolling(window=5).mean()
df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()

# Volatility
df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)

# Bollinger Bands
df['BB_middle'] = df['close'].rolling(20).mean()
df['BB_upper'] = df['BB_middle'] + 2 * df['close'].rolling(20).std()
df['BB_lower'] = df['BB_middle'] - 2 * df['close'].rolling(20).std()

# ═══════════════════════════════════════════════════════════════════════════
# AGGREGAZIONI
# ═══════════════════════════════════════════════════════════════════════════

# Statistiche
print(df.describe())
print(df['close'].mean())
print(df.groupby(df.index.month)['volume'].sum())

# Resample (cambio timeframe)
df_weekly = df.resample('W').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
})

# ═══════════════════════════════════════════════════════════════════════════
# I/O DATI
# ═══════════════════════════════════════════════════════════════════════════

# CSV
df.to_csv('prices.csv')
df = pd.read_csv('prices.csv', index_col=0, parse_dates=True)

# Da API (esempio con yfinance)
# import yfinance as yf
# btc = yf.download('BTC-USD', start='2023-01-01', end='2024-01-01')
'''


# ══════════════════════════════════════════════════════════════════════════════
#                    PART 3: SCIKIT-LEARN BASICS
# ══════════════════════════════════════════════════════════════════════════════

SKLEARN_BASICS = '''
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# WORKFLOW ML STANDARD
# ═══════════════════════════════════════════════════════════════════════════

# 1. Prepara dati
X = np.random.randn(1000, 5)  # 1000 samples, 5 features
y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Target binario

# 2. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Preprocessing (FONDAMENTALE!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit + transform
X_test_scaled = scaler.transform(X_test)         # Solo transform!

# 4. Train model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Predict
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)  # Probabilità

# 6. Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# 7. Cross-validation (più robusto)
scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")

# ═══════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════

# StandardScaler: media=0, std=1
scaler = StandardScaler()

# MinMaxScaler: range [0, 1]
scaler = MinMaxScaler()

# Handling missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent'

# Feature selection
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=3)
X_selected = selector.fit_transform(X, y)

# ═══════════════════════════════════════════════════════════════════════════
# MODELLI PRINCIPALI
# ═══════════════════════════════════════════════════════════════════════════

# CLASSIFICATION
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

# REGRESSION
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# Esempio regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(f"R2: {r2_score(y_test, y_pred)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")
'''


# ══════════════════════════════════════════════════════════════════════════════
#                    PART 4: ML MODELS FOR TRADING
# ══════════════════════════════════════════════════════════════════════════════

ML_TRADING_MODELS = '''
"""
ML per Trading: approcci comuni.

TASK COMUNI:
1. Direction prediction (classification): UP/DOWN
2. Return prediction (regression): quanto si muove
3. Volatility prediction (regression): quanto può muoversi
4. Signal generation (classification): BUY/SELL/HOLD
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

# ═══════════════════════════════════════════════════════════════════════════
# PREPARAZIONE DATI TEMPORALI
# ═══════════════════════════════════════════════════════════════════════════

def prepare_ml_features(df):
    """
    Prepara features per ML da dati OHLCV.
    ATTENZIONE: no data leakage!
    """
    features = pd.DataFrame(index=df.index)
    
    # Returns
    features['return_1'] = df['close'].pct_change(1)
    features['return_5'] = df['close'].pct_change(5)
    features['return_20'] = df['close'].pct_change(20)
    
    # Technical indicators (lagged!)
    features['sma_ratio'] = df['close'] / df['close'].rolling(20).mean()
    features['volatility'] = df['close'].pct_change().rolling(20).std()
    features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # Price patterns
    features['high_low_ratio'] = df['high'] / df['low']
    features['close_open_ratio'] = df['close'] / df['open']
    
    # Momentum
    features['rsi'] = calculate_rsi(df['close'], 14)
    
    # Target: direzione FUTURA (shift negativo = futuro)
    features['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    return features.dropna()


def calculate_rsi(prices, period=14):
    """Calcola RSI."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# ═══════════════════════════════════════════════════════════════════════════
# TIME SERIES CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def walk_forward_validation(X, y, model, n_splits=5):
    """
    Walk-forward validation per time series.
    CRUCIALE: mai usare dati futuri per training!
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
        print(f"Fold score: {score:.4f}")
    
    return np.mean(scores), np.std(scores)


# ═══════════════════════════════════════════════════════════════════════════
# ESEMPIO COMPLETO
# ═══════════════════════════════════════════════════════════════════════════

def train_direction_model(df):
    """
    Esempio: predire direzione prezzo.
    """
    # Prepara features
    features = prepare_ml_features(df)
    
    # Separa X e y
    feature_cols = [c for c in features.columns if c != 'target']
    X = features[feature_cols]
    y = features['target']
    
    # Walk-forward validation
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    mean_score, std_score = walk_forward_validation(X, y, model)
    
    print(f"Mean accuracy: {mean_score:.4f} (+/- {std_score*2:.4f})")
    
    # Train final model su tutti i dati
    model.fit(X, y)
    
    # Feature importance
    importance = pd.Series(
        model.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)
    print("\\nFeature importance:")
    print(importance)
    
    return model


# ═══════════════════════════════════════════════════════════════════════════
# TRADING SIGNALS DA ML
# ═══════════════════════════════════════════════════════════════════════════

def generate_ml_signals(model, df, threshold=0.6):
    """
    Genera segnali trading da modello ML.
    """
    features = prepare_ml_features(df)
    feature_cols = [c for c in features.columns if c != 'target']
    X = features[feature_cols]
    
    # Predici probabilità
    proba = model.predict_proba(X)[:, 1]  # Probabilità classe 1 (UP)
    
    signals = pd.Series(index=X.index, data=0)  # 0 = HOLD
    signals[proba > threshold] = 1   # BUY
    signals[proba < (1 - threshold)] = -1  # SELL
    
    return signals
'''


# ══════════════════════════════════════════════════════════════════════════════
#                    PART 5: FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_ENGINEERING = '''
"""
Feature Engineering per Trading.
La qualità delle features è PIÙ importante del modello!
"""

import pandas as pd
import numpy as np

class TradingFeatureEngineer:
    """
    Classe per generare features per trading ML.
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.features = pd.DataFrame(index=df.index)
    
    def add_returns(self, periods=[1, 5, 10, 20]):
        """Returns su diversi periodi."""
        for p in periods:
            self.features[f'return_{p}'] = self.df['close'].pct_change(p)
        return self
    
    def add_moving_averages(self, windows=[5, 10, 20, 50]):
        """SMA e EMA."""
        for w in windows:
            self.features[f'sma_{w}'] = self.df['close'].rolling(w).mean()
            self.features[f'ema_{w}'] = self.df['close'].ewm(span=w).mean()
            # Ratio price/MA
            self.features[f'price_sma_{w}_ratio'] = self.df['close'] / self.features[f'sma_{w}']
        return self
    
    def add_volatility(self, windows=[5, 10, 20]):
        """Volatilità realizzata."""
        for w in windows:
            self.features[f'volatility_{w}'] = (
                self.df['close'].pct_change().rolling(w).std() * np.sqrt(252)
            )
        return self
    
    def add_momentum(self):
        """Indicatori di momentum."""
        # RSI
        delta = self.df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        self.features['rsi'] = 100 - (100 / (1 + gain / loss))
        
        # MACD
        ema12 = self.df['close'].ewm(span=12).mean()
        ema26 = self.df['close'].ewm(span=26).mean()
        self.features['macd'] = ema12 - ema26
        self.features['macd_signal'] = self.features['macd'].ewm(span=9).mean()
        
        # Stochastic
        low_14 = self.df['low'].rolling(14).min()
        high_14 = self.df['high'].rolling(14).max()
        self.features['stoch_k'] = (self.df['close'] - low_14) / (high_14 - low_14) * 100
        self.features['stoch_d'] = self.features['stoch_k'].rolling(3).mean()
        
        return self
    
    def add_volume_features(self):
        """Features basate su volume."""
        self.features['volume_sma'] = self.df['volume'].rolling(20).mean()
        self.features['volume_ratio'] = self.df['volume'] / self.features['volume_sma']
        
        # OBV (On-Balance Volume)
        obv = (np.sign(self.df['close'].diff()) * self.df['volume']).fillna(0).cumsum()
        self.features['obv'] = obv
        self.features['obv_sma'] = obv.rolling(20).mean()
        
        return self
    
    def add_price_patterns(self):
        """Pattern di prezzo."""
        # Candlestick features
        self.features['body_size'] = abs(self.df['close'] - self.df['open'])
        self.features['upper_shadow'] = self.df['high'] - np.maximum(self.df['close'], self.df['open'])
        self.features['lower_shadow'] = np.minimum(self.df['close'], self.df['open']) - self.df['low']
        
        # Support/Resistance
        self.features['distance_from_high_20'] = self.df['close'] / self.df['high'].rolling(20).max()
        self.features['distance_from_low_20'] = self.df['close'] / self.df['low'].rolling(20).min()
        
        return self
    
    def add_lagged_features(self, feature_cols, lags=[1, 2, 3]):
        """Aggiungi versioni lagged delle features."""
        for col in feature_cols:
            if col in self.features.columns:
                for lag in lags:
                    self.features[f'{col}_lag_{lag}'] = self.features[col].shift(lag)
        return self
    
    def get_features(self, dropna=True):
        """Restituisci features finali."""
        if dropna:
            return self.features.dropna()
        return self.features


# USO:
# engineer = TradingFeatureEngineer(df)
# features = (engineer
#     .add_returns()
#     .add_moving_averages()
#     .add_volatility()
#     .add_momentum()
#     .add_volume_features()
#     .get_features())
'''


# ══════════════════════════════════════════════════════════════════════════════
#                    PART 6: BACKTESTING CON ML
# ══════════════════════════════════════════════════════════════════════════════

BACKTESTING_ML = '''
"""
Backtesting con modelli ML.
ATTENZIONE al data leakage e look-ahead bias!
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class MLBacktester:
    """
    Backtester semplice per strategie ML.
    """
    
    def __init__(self, initial_capital=10000, transaction_cost=0.001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
    
    def backtest(self, df, signals, price_col='close'):
        """
        Backtest con segnali.
        signals: 1 = long, -1 = short, 0 = flat
        """
        results = pd.DataFrame(index=df.index)
        results['price'] = df[price_col]
        results['signal'] = signals
        results['position'] = signals.shift(1).fillna(0)  # Esegui giorno dopo!
        
        # Returns
        results['market_return'] = results['price'].pct_change()
        results['strategy_return'] = results['position'] * results['market_return']
        
        # Transaction costs
        results['trades'] = results['position'].diff().abs()
        results['costs'] = results['trades'] * self.transaction_cost
        results['strategy_return_net'] = results['strategy_return'] - results['costs']
        
        # Cumulative
        results['cumulative_market'] = (1 + results['market_return']).cumprod()
        results['cumulative_strategy'] = (1 + results['strategy_return_net']).cumprod()
        
        # Equity curve
        results['equity'] = self.initial_capital * results['cumulative_strategy']
        
        return results
    
    def calculate_metrics(self, results):
        """Calcola metriche di performance."""
        metrics = {}
        
        # Total return
        metrics['total_return'] = results['cumulative_strategy'].iloc[-1] - 1
        metrics['market_return'] = results['cumulative_market'].iloc[-1] - 1
        
        # Annualized return (assumendo 252 giorni)
        n_days = len(results)
        metrics['annual_return'] = (1 + metrics['total_return']) ** (252/n_days) - 1
        
        # Volatility
        metrics['volatility'] = results['strategy_return_net'].std() * np.sqrt(252)
        
        # Sharpe ratio (assumendo rf=0)
        metrics['sharpe_ratio'] = metrics['annual_return'] / metrics['volatility']
        
        # Max drawdown
        cummax = results['equity'].cummax()
        drawdown = (results['equity'] - cummax) / cummax
        metrics['max_drawdown'] = drawdown.min()
        
        # Win rate
        winning_trades = (results['strategy_return_net'] > 0).sum()
        total_trades = (results['position'] != 0).sum()
        metrics['win_rate'] = winning_trades / total_trades if total_trades > 0 else 0
        
        # Number of trades
        metrics['n_trades'] = results['trades'].sum() / 2  # Round-trip
        
        return metrics
    
    def print_report(self, metrics):
        """Stampa report."""
        print("=" * 50)
        print("BACKTEST RESULTS")
        print("=" * 50)
        print(f"Total Return:    {metrics['total_return']*100:.2f}%")
        print(f"Market Return:   {metrics['market_return']*100:.2f}%")
        print(f"Annual Return:   {metrics['annual_return']*100:.2f}%")
        print(f"Volatility:      {metrics['volatility']*100:.2f}%")
        print(f"Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:    {metrics['max_drawdown']*100:.2f}%")
        print(f"Win Rate:        {metrics['win_rate']*100:.2f}%")
        print(f"Number of Trades: {metrics['n_trades']:.0f}")
        print("=" * 50)
'''


# ══════════════════════════════════════════════════════════════════════════════
#                    PART 7: NEURAL NETWORKS PREVIEW
# ══════════════════════════════════════════════════════════════════════════════

NEURAL_NETWORKS_PREVIEW = '''
"""
Preview: Deep Learning per Trading.
Richiede: TensorFlow o PyTorch

pip install tensorflow
# oppure
pip install torch
"""

# ═══════════════════════════════════════════════════════════════════════════
# LSTM per Time Series (TensorFlow/Keras)
# ═══════════════════════════════════════════════════════════════════════════

LSTM_EXAMPLE = """
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Prepara dati
seq_length = 60  # 60 timesteps
X, y = create_sequences(prices_normalized, seq_length)

# Reshape per LSTM: (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Modello LSTM
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)
"""

# ═══════════════════════════════════════════════════════════════════════════
# Transformer per Trading (concetto)
# ═══════════════════════════════════════════════════════════════════════════

TRANSFORMER_CONCEPT = """
I Transformer (come in GPT) stanno diventando popolari per trading:
- Attention mechanism cattura dipendenze temporali
- Migliore per sequenze lunghe rispetto a LSTM
- Librerie: pytorch-transformers, huggingface

Architettura base:
1. Embedding layer (per features)
2. Positional encoding (per sequenza temporale)
3. Multi-head attention layers
4. Feed-forward layers
5. Output layer (classification/regression)

Applicazioni:
- Price prediction
- Pattern recognition
- Sentiment analysis (da news/social)
"""

# ═══════════════════════════════════════════════════════════════════════════
# REINFORCEMENT LEARNING (concetto avanzato)
# ═══════════════════════════════════════════════════════════════════════════

RL_CONCEPT = """
Reinforcement Learning per Trading:
L'agente impara una policy ottimale interagendo con l'ambiente (mercato).

Components:
- State: features di mercato
- Action: buy, sell, hold
- Reward: profit/loss
- Policy: strategia (neural network)

Algoritmi comuni:
- DQN (Deep Q-Network)
- PPO (Proximal Policy Optimization)
- A2C (Advantage Actor-Critic)

Librerie:
- stable-baselines3
- ray[rllib]
- tf-agents

ATTENZIONE:
- Molto complesso da implementare correttamente
- Rischio di overfitting al backtest
- Richiede simulatore di mercato realistico
"""
'''


# ══════════════════════════════════════════════════════════════════════════════
#                    ROADMAP AI/ML
# ══════════════════════════════════════════════════════════════════════════════

AI_ML_ROADMAP = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ROADMAP AI/ML PER TRADING                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

FASE 1: Fondamenta (dopo PCAP) - 2 settimane
├── NumPy: array, operazioni vettorizzate, statistica
├── Pandas: DataFrame, time series, I/O
└── Matplotlib/Seaborn: visualizzazione dati

FASE 2: ML Classico (dopo PCPP1) - 3 settimane
├── Scikit-learn: workflow, preprocessing, modelli
├── Feature engineering per trading
├── Cross-validation per time series
└── Backtesting con ML signals

FASE 3: Deep Learning (dopo PCPP2) - 4 settimane
├── TensorFlow/Keras basics
├── LSTM per time series
├── CNN per pattern recognition
└── Transformer basics

FASE 4: Avanzato (opzionale) - ongoing
├── Reinforcement Learning
├── NLP per sentiment analysis
├── AutoML
└── MLOps (deployment, monitoring)

RISORSE:
─────────
1. Kaggle (datasets, notebooks, competitions)
2. Coursera: ML di Andrew Ng
3. Fast.ai: Practical Deep Learning
4. Quantitative Trading with ML (libri)

NOTA IMPORTANTE:
────────────────
ML per trading è MOLTO difficile:
- I mercati sono efficienti (hard to beat)
- Overfitting è il rischio principale
- Paper trading PRIMA di soldi veri
- Inizia semplice, complessità graduale

══════════════════════════════════════════════════════════════════════════════
"""


if __name__ == "__main__":
    print("=" * 70)
    print("SPECIALIZATION MODULE: AI & ML for Trading")
    print("=" * 70)
    print("""
    Contenuti disponibili (print per visualizzare):
    
    print(NUMPY_ESSENTIALS)        # NumPy fondamentali
    print(PANDAS_ESSENTIALS)       # Pandas per finanza
    print(SKLEARN_BASICS)          # Scikit-learn
    print(ML_TRADING_MODELS)       # ML per trading
    print(FEATURE_ENGINEERING)     # Feature engineering
    print(BACKTESTING_ML)          # Backtesting
    print(NEURAL_NETWORKS_PREVIEW) # Deep Learning preview
    print(AI_ML_ROADMAP)           # Roadmap completa
    """)
