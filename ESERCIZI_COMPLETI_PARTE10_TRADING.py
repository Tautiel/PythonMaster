"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    🐍 PYTHON MASTER - SCHEDA ESERCIZI COMPLETA               ║
║                                                                              ║
║                    PARTE 10: TRADING SPECIFICO                               ║
║                    (Indicatori, Backtesting, Bot Architecture)               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
from abc import ABC, abstractmethod

# ==============================================================================
# SEZIONE 27: INDICATORI TECNICI
# ==============================================================================

print("=" * 70)
print("SEZIONE 27: INDICATORI TECNICI")
print("=" * 70)

# ------------------------------------------------------------------------------
# ESERCIZIO 27.1: Moving Averages
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Implementa medie mobili:
1. SMA (Simple Moving Average)
2. EMA (Exponential Moving Average)
3. WMA (Weighted Moving Average)
4. Crossover detection

💡 TEORIA:
SMA = media semplice degli ultimi N periodi
EMA = media esponenziale con peso decrescente
Crossover: quando una MA veloce attraversa una lenta

🎯 SKILLS: Moving averages, signal generation
⏱️ TEMPO: 20 minuti
🔢 LIVELLO: Intermedio
"""

def esercizio_27_1():
    """Moving Averages"""
    
    # Setup data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    prices = pd.Series(
        100 * (1 + np.random.normal(0.001, 0.02, 100)).cumprod(),
        index=dates,
        name='close'
    )
    
    df = pd.DataFrame({'close': prices})
    
    # 1. SMA
    print("--- SMA ---")
    
    def sma(series, period):
        """Simple Moving Average."""
        return series.rolling(window=period).mean()
    
    df['SMA_10'] = sma(df['close'], 10)
    df['SMA_20'] = sma(df['close'], 20)
    
    print(f"  SMA_10 last 5: {df['SMA_10'].tail().values.round(2)}")
    print(f"  SMA_20 last 5: {df['SMA_20'].tail().values.round(2)}")
    
    # 2. EMA
    print("\n--- EMA ---")
    
    def ema(series, period):
        """Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()
    
    df['EMA_10'] = ema(df['close'], 10)
    df['EMA_20'] = ema(df['close'], 20)
    
    print(f"  EMA_10 last 5: {df['EMA_10'].tail().values.round(2)}")
    
    # 3. WMA
    print("\n--- WMA ---")
    
    def wma(series, period):
        """Weighted Moving Average."""
        weights = np.arange(1, period + 1)
        return series.rolling(period).apply(
            lambda x: np.sum(weights * x) / weights.sum(), raw=True
        )
    
    df['WMA_10'] = wma(df['close'], 10)
    print(f"  WMA_10 last 5: {df['WMA_10'].tail().values.round(2)}")
    
    # 4. CROSSOVER
    print("\n--- CROSSOVER DETECTION ---")
    
    def detect_crossover(fast_ma, slow_ma):
        """Rileva crossover tra MA veloce e lenta."""
        signals = pd.Series(index=fast_ma.index, dtype='object')
        
        # Fast sopra slow = bullish
        fast_above = fast_ma > slow_ma
        
        # Crossover: cambio di stato
        crossover_up = fast_above & ~fast_above.shift(1)   # Golden cross
        crossover_down = ~fast_above & fast_above.shift(1)  # Death cross
        
        signals[crossover_up] = 'BUY'
        signals[crossover_down] = 'SELL'
        
        return signals
    
    df['signal'] = detect_crossover(df['EMA_10'], df['EMA_20'])
    
    # Mostra segnali
    signals = df[df['signal'].notna()][['close', 'EMA_10', 'EMA_20', 'signal']]
    print(f"  Crossover signals:\n{signals}")
    
    return df

if __name__ == "__main__":
    esercizio_27_1()
    print("✅ Esercizio 27.1 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 27.2: Momentum Indicators
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Implementa indicatori momentum:
1. RSI (Relative Strength Index)
2. MACD
3. Stochastic Oscillator
4. Signal generation

💡 TEORIA:
RSI: misura velocità e cambio dei movimenti di prezzo (0-100)
MACD: differenza tra EMA veloce e lenta
Stochastic: posizione del prezzo rispetto al range

🎯 SKILLS: RSI, MACD, Stochastic
⏱️ TEMPO: 25 minuti
🔢 LIVELLO: Intermedio-Avanzato
"""

def esercizio_27_2():
    """Momentum Indicators"""
    
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    prices = pd.Series(
        100 * (1 + np.random.normal(0.001, 0.02, 100)).cumprod(),
        index=dates
    )
    
    df = pd.DataFrame({'close': prices})
    
    # 1. RSI
    print("--- RSI ---")
    
    def rsi(series, period=14):
        """Relative Strength Index."""
        delta = series.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    df['RSI'] = rsi(df['close'], 14)
    
    print(f"  RSI last 5: {df['RSI'].tail().values.round(2)}")
    print(f"  RSI min: {df['RSI'].min():.2f}, max: {df['RSI'].max():.2f}")
    
    # RSI signals
    df['RSI_signal'] = np.where(df['RSI'] < 30, 'OVERSOLD',
                        np.where(df['RSI'] > 70, 'OVERBOUGHT', 'NEUTRAL'))
    
    # 2. MACD
    print("\n--- MACD ---")
    
    def macd(series, fast=12, slow=26, signal=9):
        """MACD indicator."""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = macd(df['close'])
    
    print(f"  MACD last 5: {df['MACD'].tail().values.round(4)}")
    print(f"  Signal last 5: {df['MACD_signal'].tail().values.round(4)}")
    
    # MACD crossover
    df['MACD_cross'] = np.where(
        (df['MACD'] > df['MACD_signal']) & (df['MACD'].shift(1) <= df['MACD_signal'].shift(1)),
        'BUY',
        np.where(
            (df['MACD'] < df['MACD_signal']) & (df['MACD'].shift(1) >= df['MACD_signal'].shift(1)),
            'SELL',
            None
        )
    )
    
    # 3. STOCHASTIC
    print("\n--- STOCHASTIC ---")
    
    def stochastic(high, low, close, k_period=14, d_period=3):
        """Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_period).mean()
        
        return k, d
    
    # Simula high/low
    df['high'] = df['close'] * (1 + np.abs(np.random.normal(0, 0.01, 100)))
    df['low'] = df['close'] * (1 - np.abs(np.random.normal(0, 0.01, 100)))
    
    df['STOCH_K'], df['STOCH_D'] = stochastic(df['high'], df['low'], df['close'])
    
    print(f"  %K last 5: {df['STOCH_K'].tail().values.round(2)}")
    print(f"  %D last 5: {df['STOCH_D'].tail().values.round(2)}")
    
    # 4. COMBINED SIGNALS
    print("\n--- COMBINED SIGNALS ---")
    
    def generate_signal(row):
        """Genera segnale combinato."""
        score = 0
        
        # RSI
        if row['RSI'] < 30:
            score += 1
        elif row['RSI'] > 70:
            score -= 1
        
        # MACD
        if row['MACD'] > row['MACD_signal']:
            score += 1
        else:
            score -= 1
        
        # Stochastic
        if row['STOCH_K'] < 20:
            score += 1
        elif row['STOCH_K'] > 80:
            score -= 1
        
        if score >= 2:
            return 'STRONG_BUY'
        elif score == 1:
            return 'BUY'
        elif score <= -2:
            return 'STRONG_SELL'
        elif score == -1:
            return 'SELL'
        return 'HOLD'
    
    df['combined_signal'] = df.apply(generate_signal, axis=1)
    
    signal_counts = df['combined_signal'].value_counts()
    print(f"  Signal distribution:\n{signal_counts}")
    
    return df

if __name__ == "__main__":
    esercizio_27_2()
    print("✅ Esercizio 27.2 completato!\n")


# ==============================================================================
# SEZIONE 28: BACKTESTING
# ==============================================================================

print("\n" + "=" * 70)
print("SEZIONE 28: BACKTESTING")
print("=" * 70)

# ------------------------------------------------------------------------------
# ESERCIZIO 28.1: Simple Backtest Engine
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Crea un semplice engine di backtesting:
1. Trade tracking
2. Position management
3. P&L calculation
4. Performance metrics

💡 TEORIA:
Backtesting simula una strategia su dati storici.
Traccia trades, calcola P&L, misura performance.

🎯 SKILLS: Backtesting, position tracking, metrics
⏱️ TEMPO: 30 minuti
🔢 LIVELLO: Avanzato
"""

def esercizio_28_1():
    """Simple Backtest Engine"""
    
    class Side(Enum):
        LONG = "LONG"
        SHORT = "SHORT"
    
    @dataclass
    class Trade:
        entry_date: pd.Timestamp
        entry_price: float
        exit_date: Optional[pd.Timestamp] = None
        exit_price: Optional[float] = None
        side: Side = Side.LONG
        quantity: int = 1
        
        @property
        def pnl(self):
            if self.exit_price is None:
                return 0
            if self.side == Side.LONG:
                return (self.exit_price - self.entry_price) * self.quantity
            else:
                return (self.entry_price - self.exit_price) * self.quantity
        
        @property
        def pnl_percent(self):
            if self.exit_price is None:
                return 0
            return self.pnl / (self.entry_price * self.quantity) * 100
    
    class SimpleBacktester:
        def __init__(self, initial_capital=10000):
            self.initial_capital = initial_capital
            self.capital = initial_capital
            self.position = None
            self.trades: List[Trade] = []
            self.equity_curve = []
        
        def buy(self, date, price, quantity=1):
            if self.position is not None:
                return  # Already in position
            
            cost = price * quantity
            if cost > self.capital:
                return  # Not enough capital
            
            self.capital -= cost
            self.position = Trade(
                entry_date=date,
                entry_price=price,
                side=Side.LONG,
                quantity=quantity
            )
        
        def sell(self, date, price):
            if self.position is None:
                return  # No position
            
            self.position.exit_date = date
            self.position.exit_price = price
            
            self.capital += price * self.position.quantity
            self.trades.append(self.position)
            self.position = None
        
        def get_equity(self, current_price):
            equity = self.capital
            if self.position:
                equity += current_price * self.position.quantity
            return equity
        
        def run(self, data, signal_column='signal'):
            """Esegue backtest su DataFrame con segnali."""
            for date, row in data.iterrows():
                price = row['close']
                signal = row.get(signal_column)
                
                if signal == 'BUY' and self.position is None:
                    qty = int(self.capital * 0.95 / price)  # 95% del capitale
                    if qty > 0:
                        self.buy(date, price, qty)
                
                elif signal == 'SELL' and self.position is not None:
                    self.sell(date, price)
                
                self.equity_curve.append({
                    'date': date,
                    'equity': self.get_equity(price),
                    'price': price
                })
            
            # Chiudi posizione finale
            if self.position:
                self.sell(data.index[-1], data['close'].iloc[-1])
        
        def get_metrics(self):
            """Calcola metriche di performance."""
            if not self.trades:
                return {}
            
            pnls = [t.pnl for t in self.trades]
            pnl_pcts = [t.pnl_percent for t in self.trades]
            
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            
            total_pnl = sum(pnls)
            win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0
            
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            
            profit_factor = abs(sum(wins) / sum(losses)) if losses else float('inf')
            
            equity_df = pd.DataFrame(self.equity_curve)
            max_equity = equity_df['equity'].expanding().max()
            drawdown = (equity_df['equity'] - max_equity) / max_equity * 100
            max_drawdown = drawdown.min()
            
            return {
                'total_trades': len(self.trades),
                'winning_trades': len(wins),
                'losing_trades': len(losses),
                'win_rate': round(win_rate, 2),
                'total_pnl': round(total_pnl, 2),
                'avg_win': round(avg_win, 2),
                'avg_loss': round(avg_loss, 2),
                'profit_factor': round(profit_factor, 2),
                'max_drawdown': round(max_drawdown, 2),
                'final_capital': round(self.capital, 2),
                'return_pct': round((self.capital - self.initial_capital) / self.initial_capital * 100, 2)
            }
    
    # Test
    print("--- BACKTEST ENGINE ---")
    
    # Genera dati con segnali
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    prices = 100 * (1 + np.random.normal(0.001, 0.02, 100)).cumprod()
    
    df = pd.DataFrame({'close': prices}, index=dates)
    
    # SMA crossover signals
    df['SMA_fast'] = df['close'].rolling(5).mean()
    df['SMA_slow'] = df['close'].rolling(20).mean()
    
    df['signal'] = np.where(
        (df['SMA_fast'] > df['SMA_slow']) & (df['SMA_fast'].shift(1) <= df['SMA_slow'].shift(1)),
        'BUY',
        np.where(
            (df['SMA_fast'] < df['SMA_slow']) & (df['SMA_fast'].shift(1) >= df['SMA_slow'].shift(1)),
            'SELL',
            None
        )
    )
    
    # Run backtest
    bt = SimpleBacktester(initial_capital=10000)
    bt.run(df)
    
    # Results
    metrics = bt.get_metrics()
    print("\n  Performance Metrics:")
    for key, value in metrics.items():
        print(f"    {key}: {value}")
    
    # Trades
    print(f"\n  Sample trades:")
    for trade in bt.trades[:5]:
        print(f"    {trade.entry_date.date()} → {trade.exit_date.date()}: "
              f"${trade.entry_price:.2f} → ${trade.exit_price:.2f} = ${trade.pnl:.2f}")
    
    return SimpleBacktester, bt

if __name__ == "__main__":
    esercizio_28_1()
    print("✅ Esercizio 28.1 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 28.2: Strategy Framework
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Crea un framework per strategie:
1. Interfaccia Strategy astratta
2. Implementazioni concrete
3. Backtester generico

💡 TEORIA:
Un framework permette di testare strategie diverse
con lo stesso engine di backtesting.

🎯 SKILLS: Strategy pattern, framework design
⏱️ TEMPO: 25 minuti
🔢 LIVELLO: Avanzato
"""

def esercizio_28_2():
    """Strategy Framework"""
    
    # 1. STRATEGY INTERFACE
    class Strategy(ABC):
        """Interfaccia base per strategie."""
        
        @property
        @abstractmethod
        def name(self) -> str:
            pass
        
        @abstractmethod
        def generate_signals(self, data: pd.DataFrame) -> pd.Series:
            """Genera segnali BUY/SELL/None."""
            pass
    
    # 2. CONCRETE STRATEGIES
    class SMACrossover(Strategy):
        def __init__(self, fast_period=10, slow_period=20):
            self.fast_period = fast_period
            self.slow_period = slow_period
        
        @property
        def name(self):
            return f"SMA({self.fast_period}/{self.slow_period})"
        
        def generate_signals(self, data):
            df = data.copy()
            df['fast'] = df['close'].rolling(self.fast_period).mean()
            df['slow'] = df['close'].rolling(self.slow_period).mean()
            
            signals = pd.Series(index=df.index, dtype='object')
            
            buy_signal = (df['fast'] > df['slow']) & (df['fast'].shift(1) <= df['slow'].shift(1))
            sell_signal = (df['fast'] < df['slow']) & (df['fast'].shift(1) >= df['slow'].shift(1))
            
            signals[buy_signal] = 'BUY'
            signals[sell_signal] = 'SELL'
            
            return signals
    
    class RSIStrategy(Strategy):
        def __init__(self, period=14, oversold=30, overbought=70):
            self.period = period
            self.oversold = oversold
            self.overbought = overbought
        
        @property
        def name(self):
            return f"RSI({self.period}, {self.oversold}/{self.overbought})"
        
        def generate_signals(self, data):
            df = data.copy()
            
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(self.period).mean()
            loss = (-delta).where(delta < 0, 0).rolling(self.period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            signals = pd.Series(index=df.index, dtype='object')
            
            # Buy quando RSI sale sopra oversold
            buy_signal = (rsi > self.oversold) & (rsi.shift(1) <= self.oversold)
            # Sell quando RSI scende sotto overbought
            sell_signal = (rsi < self.overbought) & (rsi.shift(1) >= self.overbought)
            
            signals[buy_signal] = 'BUY'
            signals[sell_signal] = 'SELL'
            
            return signals
    
    class MomentumStrategy(Strategy):
        def __init__(self, lookback=10, threshold=0.03):
            self.lookback = lookback
            self.threshold = threshold
        
        @property
        def name(self):
            return f"Momentum({self.lookback}, {self.threshold})"
        
        def generate_signals(self, data):
            df = data.copy()
            
            returns = df['close'].pct_change(self.lookback)
            
            signals = pd.Series(index=df.index, dtype='object')
            
            buy_signal = returns > self.threshold
            sell_signal = returns < -self.threshold
            
            signals[buy_signal] = 'BUY'
            signals[sell_signal] = 'SELL'
            
            return signals
    
    # 3. STRATEGY COMPARISON
    print("--- STRATEGY COMPARISON ---")
    
    # Test data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='D')
    prices = 100 * (1 + np.random.normal(0.001, 0.02, 200)).cumprod()
    data = pd.DataFrame({'close': prices}, index=dates)
    
    strategies = [
        SMACrossover(10, 30),
        SMACrossover(5, 20),
        RSIStrategy(14, 30, 70),
        MomentumStrategy(10, 0.02),
    ]
    
    results = []
    
    for strategy in strategies:
        # Backtest semplificato
        signals = strategy.generate_signals(data)
        
        capital = 10000
        position = 0
        trades = []
        entry_price = 0
        
        for date, price in data['close'].items():
            signal = signals.get(date)
            
            if signal == 'BUY' and position == 0:
                position = int(capital * 0.95 / price)
                entry_price = price
                capital -= position * price
            
            elif signal == 'SELL' and position > 0:
                pnl = (price - entry_price) * position
                trades.append(pnl)
                capital += position * price
                position = 0
        
        # Close final position
        if position > 0:
            pnl = (data['close'].iloc[-1] - entry_price) * position
            trades.append(pnl)
            capital += position * data['close'].iloc[-1]
        
        total_pnl = sum(trades)
        win_rate = len([t for t in trades if t > 0]) / len(trades) * 100 if trades else 0
        
        results.append({
            'strategy': strategy.name,
            'trades': len(trades),
            'win_rate': round(win_rate, 1),
            'total_pnl': round(total_pnl, 2),
            'return_pct': round((capital - 10000) / 10000 * 100, 2)
        })
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    return Strategy, results_df

if __name__ == "__main__":
    esercizio_28_2()
    print("✅ Esercizio 28.2 completato!\n")


# ==============================================================================
# SEZIONE 29: BOT ARCHITECTURE
# ==============================================================================

print("\n" + "=" * 70)
print("SEZIONE 29: BOT ARCHITECTURE")
print("=" * 70)

# ------------------------------------------------------------------------------
# ESERCIZIO 29.1: Trading Bot Structure
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Progetta architettura di un trading bot:
1. Componenti principali
2. Event-driven design
3. Risk management
4. Order management

💡 TEORIA:
Un bot di trading ha componenti:
- Data feed
- Strategy
- Risk manager
- Order executor
- Position tracker

🎯 SKILLS: Bot architecture, event-driven design
⏱️ TEMPO: 30 minuti
🔢 LIVELLO: Avanzato
"""

def esercizio_29_1():
    """Trading Bot Structure"""
    
    from enum import Enum
    from dataclasses import dataclass, field
    from typing import Callable
    from datetime import datetime
    
    # ENUMS
    class OrderSide(Enum):
        BUY = "BUY"
        SELL = "SELL"
    
    class OrderType(Enum):
        MARKET = "MARKET"
        LIMIT = "LIMIT"
        STOP = "STOP"
    
    class OrderStatus(Enum):
        PENDING = "PENDING"
        FILLED = "FILLED"
        CANCELLED = "CANCELLED"
    
    # DATA CLASSES
    @dataclass
    class Order:
        symbol: str
        side: OrderSide
        quantity: int
        order_type: OrderType = OrderType.MARKET
        price: Optional[float] = None
        status: OrderStatus = OrderStatus.PENDING
        order_id: str = field(default_factory=lambda: f"ORD-{datetime.now().strftime('%H%M%S%f')}")
    
    @dataclass
    class Position:
        symbol: str
        quantity: int
        entry_price: float
        current_price: float = 0
        
        @property
        def pnl(self):
            return (self.current_price - self.entry_price) * self.quantity
        
        @property
        def pnl_percent(self):
            return (self.pnl / (self.entry_price * self.quantity)) * 100
    
    @dataclass
    class Signal:
        symbol: str
        action: str  # BUY, SELL, HOLD
        strength: float = 1.0  # 0-1
        reason: str = ""
    
    # COMPONENTS
    class RiskManager:
        """Gestisce il rischio."""
        
        def __init__(self, max_position_pct=0.1, max_loss_pct=0.02, max_positions=5):
            self.max_position_pct = max_position_pct
            self.max_loss_pct = max_loss_pct
            self.max_positions = max_positions
        
        def check_order(self, order: Order, capital: float, positions: Dict) -> tuple:
            """Valida un ordine."""
            
            # Check max positions
            if order.side == OrderSide.BUY and len(positions) >= self.max_positions:
                return False, "Max positions reached"
            
            # Check position size
            if order.price:
                order_value = order.quantity * order.price
                if order_value > capital * self.max_position_pct:
                    return False, f"Order exceeds max position size ({self.max_position_pct*100}%)"
            
            return True, "OK"
        
        def calculate_position_size(self, capital, price, stop_loss_pct):
            """Calcola size basato su rischio."""
            risk_amount = capital * self.max_loss_pct
            risk_per_share = price * stop_loss_pct
            return int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
    
    class OrderManager:
        """Gestisce gli ordini."""
        
        def __init__(self):
            self.orders: List[Order] = []
            self.order_callbacks: List[Callable] = []
        
        def submit_order(self, order: Order):
            """Sottomette un ordine."""
            self.orders.append(order)
            print(f"    📤 Order submitted: {order.order_id} {order.side.value} {order.quantity} {order.symbol}")
            return order
        
        def fill_order(self, order: Order, fill_price: float):
            """Esegue un ordine."""
            order.status = OrderStatus.FILLED
            order.price = fill_price
            print(f"    ✅ Order filled: {order.order_id} @ ${fill_price}")
            
            for callback in self.order_callbacks:
                callback(order)
        
        def on_fill(self, callback: Callable):
            """Registra callback per fill."""
            self.order_callbacks.append(callback)
    
    class PositionManager:
        """Gestisce le posizioni."""
        
        def __init__(self):
            self.positions: Dict[str, Position] = {}
        
        def update_position(self, symbol: str, quantity: int, price: float):
            """Aggiorna o crea posizione."""
            if symbol in self.positions:
                pos = self.positions[symbol]
                pos.quantity += quantity
                if pos.quantity == 0:
                    del self.positions[symbol]
            elif quantity > 0:
                self.positions[symbol] = Position(symbol, quantity, price, price)
        
        def update_prices(self, prices: Dict[str, float]):
            """Aggiorna prezzi correnti."""
            for symbol, price in prices.items():
                if symbol in self.positions:
                    self.positions[symbol].current_price = price
        
        def get_total_pnl(self):
            return sum(p.pnl for p in self.positions.values())
    
    class TradingBot:
        """Bot di trading principale."""
        
        def __init__(self, initial_capital=10000):
            self.capital = initial_capital
            self.risk_manager = RiskManager()
            self.order_manager = OrderManager()
            self.position_manager = PositionManager()
            self.strategies: List = []
            
            # Register callbacks
            self.order_manager.on_fill(self._on_order_fill)
        
        def _on_order_fill(self, order: Order):
            """Callback quando ordine viene eseguito."""
            qty = order.quantity if order.side == OrderSide.BUY else -order.quantity
            self.position_manager.update_position(order.symbol, qty, order.price)
            
            # Update capital
            if order.side == OrderSide.BUY:
                self.capital -= order.quantity * order.price
            else:
                self.capital += order.quantity * order.price
        
        def add_strategy(self, strategy):
            """Aggiunge una strategia."""
            self.strategies.append(strategy)
        
        def process_signal(self, signal: Signal, current_price: float):
            """Processa un segnale di trading."""
            print(f"\n  📊 Processing signal: {signal.action} {signal.symbol}")
            
            if signal.action == 'BUY':
                # Calculate position size
                size = self.risk_manager.calculate_position_size(
                    self.capital, current_price, 0.02
                )
                
                if size > 0:
                    order = Order(
                        symbol=signal.symbol,
                        side=OrderSide.BUY,
                        quantity=size,
                        order_type=OrderType.MARKET
                    )
                    
                    # Risk check
                    ok, reason = self.risk_manager.check_order(
                        order, self.capital, self.position_manager.positions
                    )
                    
                    if ok:
                        self.order_manager.submit_order(order)
                        # Simulate fill
                        self.order_manager.fill_order(order, current_price)
                    else:
                        print(f"    ❌ Order rejected: {reason}")
            
            elif signal.action == 'SELL':
                if signal.symbol in self.position_manager.positions:
                    pos = self.position_manager.positions[signal.symbol]
                    order = Order(
                        symbol=signal.symbol,
                        side=OrderSide.SELL,
                        quantity=pos.quantity,
                        order_type=OrderType.MARKET
                    )
                    self.order_manager.submit_order(order)
                    self.order_manager.fill_order(order, current_price)
        
        def get_status(self):
            """Ritorna stato del bot."""
            return {
                'capital': round(self.capital, 2),
                'positions': len(self.position_manager.positions),
                'total_pnl': round(self.position_manager.get_total_pnl(), 2),
                'orders': len(self.order_manager.orders)
            }
    
    # TEST
    print("--- TRADING BOT TEST ---")
    
    bot = TradingBot(initial_capital=10000)
    
    # Simulate signals
    signals = [
        Signal('AAPL', 'BUY', 0.8, 'SMA crossover'),
        Signal('GOOGL', 'BUY', 0.7, 'RSI oversold'),
        Signal('AAPL', 'SELL', 0.9, 'Take profit'),
    ]
    
    prices = {'AAPL': 150.0, 'GOOGL': 140.0}
    
    for signal in signals:
        price = prices.get(signal.symbol, 100)
        bot.process_signal(signal, price)
    
    print(f"\n  Bot status: {bot.get_status()}")
    
    return TradingBot

if __name__ == "__main__":
    esercizio_29_1()
    print("✅ Esercizio 29.1 completato!\n")


# ==============================================================================
# RIEPILOGO FINALE
# ==============================================================================

print("\n" + "=" * 70)
print("RIEPILOGO: ESERCIZI TRADING COMPLETATI")
print("=" * 70)

print("""
ESERCIZI COMPLETATI IN QUESTA PARTE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SEZIONE 27 - Indicatori Tecnici:
  ✅ 27.1 Moving Averages (SMA, EMA, WMA, Crossover)
  ✅ 27.2 Momentum Indicators (RSI, MACD, Stochastic)

SEZIONE 28 - Backtesting:
  ✅ 28.1 Simple Backtest Engine
  ✅ 28.2 Strategy Framework

SEZIONE 29 - Bot Architecture:
  ✅ 29.1 Trading Bot Structure

TOTALE QUESTA PARTE: 5 esercizi
TOTALE CUMULATIVO: 85 esercizi

═══════════════════════════════════════════════════════════════════════
                    🎉 CORSO COMPLETO! 🎉
═══════════════════════════════════════════════════════════════════════

Hai completato tutti gli 85 esercizi del Python Master Course!

Riepilogo per sezioni:
• Parte 1-3: Fundamentals, Control Flow, Functions (34 esercizi)
• Parte 4: OOP (11 esercizi)
• Parte 5: Collections Avanzate (8 esercizi)
• Parte 6: Error Handling & File I/O (7 esercizi)
• Parte 7: Testing (6 esercizi)
• Parte 8: NumPy (7 esercizi)
• Parte 9: Pandas (7 esercizi)
• Parte 10: Trading Specifico (5 esercizi)

Prossimi passi consigliati:
1. Rivedi gli esercizi che ti hanno dato difficoltà
2. Implementa il bot trading completo integrando tutti i componenti
3. Testa su dati reali (paper trading)
4. Aggiungi connessione a exchange via API
5. Implementa logging e monitoring
""")

if __name__ == "__main__":
    print("\n🚀 SEI PRONTO PER COSTRUIRE IL TUO TRADING BOT!")
