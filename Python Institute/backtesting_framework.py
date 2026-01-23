"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    BACKTESTING FRAMEWORK                                     ║
║                                                                              ║
║                 Test strategie su dati storici                               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

PREREQUISITI:
pip install pandas numpy matplotlib

OPZIONALE (per dati):
pip install yfinance ccxt

═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
#                    DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

class SignalType(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class Trade:
    """Rappresenta un singolo trade."""
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    size: float
    side: SignalType  # BUY or SELL
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    
    def close(self, exit_time: datetime, exit_price: float):
        """Chiudi il trade."""
        self.exit_time = exit_time
        self.exit_price = exit_price
        
        if self.side == SignalType.BUY:
            self.pnl = (exit_price - self.entry_price) * self.size
            self.pnl_pct = (exit_price - self.entry_price) / self.entry_price
        else:
            self.pnl = (self.entry_price - exit_price) * self.size
            self.pnl_pct = (self.entry_price - exit_price) / self.entry_price


@dataclass
class BacktestResult:
    """Risultati del backtest."""
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration: timedelta
    equity_curve: pd.Series
    trades: List[Trade]


# ══════════════════════════════════════════════════════════════════════════════
#                    BASE STRATEGY
# ══════════════════════════════════════════════════════════════════════════════

class Strategy(ABC):
    """
    Base class per strategie.
    Estendi questa classe per creare la tua strategia.
    """
    
    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
        self.data = None
        self.signals = None
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Genera segnali di trading.
        
        Args:
            data: DataFrame con colonne OHLCV
            
        Returns:
            Series con segnali: 1 (buy), -1 (sell), 0 (hold)
        """
        pass
    
    def set_data(self, data: pd.DataFrame):
        """Imposta dati e genera segnali."""
        self.data = data
        self.signals = self.generate_signals(data)
        return self


# ══════════════════════════════════════════════════════════════════════════════
#                    STRATEGIE ESEMPIO
# ══════════════════════════════════════════════════════════════════════════════

class SMACrossStrategy(Strategy):
    """
    Strategia SMA Crossover.
    Buy quando fast SMA incrocia sopra slow SMA.
    Sell quando fast SMA incrocia sotto slow SMA.
    """
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        super().__init__(f"SMA_{fast_period}_{slow_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(index=data.index, data=0)
        
        # Calcola SMA
        sma_fast = data['close'].rolling(self.fast_period).mean()
        sma_slow = data['close'].rolling(self.slow_period).mean()
        
        # Genera segnali sui crossover
        signals[sma_fast > sma_slow] = 1
        signals[sma_fast < sma_slow] = -1
        
        # Solo ai cambi di direzione
        signals = signals.diff()
        signals = signals.replace({2: 1, -2: -1})
        signals = signals.fillna(0)
        
        return signals


class RSIMeanReversionStrategy(Strategy):
    """
    Strategia Mean Reversion con RSI.
    Buy quando RSI < oversold.
    Sell quando RSI > overbought.
    """
    
    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70):
        super().__init__(f"RSI_{period}_{oversold}_{overbought}")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
    
    def _calculate_rsi(self, prices: pd.Series) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(index=data.index, data=0)
        
        rsi = self._calculate_rsi(data['close'])
        
        # Buy su oversold
        signals[rsi < self.oversold] = 1
        # Sell su overbought
        signals[rsi > self.overbought] = -1
        
        return signals


class BollingerBandsStrategy(Strategy):
    """
    Strategia Bollinger Bands.
    Buy quando prezzo tocca banda inferiore.
    Sell quando prezzo tocca banda superiore.
    """
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__(f"BB_{period}_{std_dev}")
        self.period = period
        self.std_dev = std_dev
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(index=data.index, data=0)
        
        middle = data['close'].rolling(self.period).mean()
        std = data['close'].rolling(self.period).std()
        upper = middle + (std * self.std_dev)
        lower = middle - (std * self.std_dev)
        
        signals[data['close'] < lower] = 1
        signals[data['close'] > upper] = -1
        
        return signals


class MACDStrategy(Strategy):
    """
    Strategia MACD Crossover.
    """
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__(f"MACD_{fast}_{slow}_{signal}")
        self.fast = fast
        self.slow = slow
        self.signal_period = signal
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(index=data.index, data=0)
        
        ema_fast = data['close'].ewm(span=self.fast).mean()
        ema_slow = data['close'].ewm(span=self.slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period).mean()
        
        signals[macd_line > signal_line] = 1
        signals[macd_line < signal_line] = -1
        
        # Solo ai crossover
        signals = signals.diff()
        signals = signals.replace({2: 1, -2: -1})
        signals = signals.fillna(0)
        
        return signals


# ══════════════════════════════════════════════════════════════════════════════
#                    BACKTESTER
# ══════════════════════════════════════════════════════════════════════════════

class Backtester:
    """
    Engine per backtesting strategie.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000,
        commission: float = 0.001,  # 0.1%
        slippage: float = 0.0005,   # 0.05%
        position_sizing: str = 'fixed',  # 'fixed' o 'percent'
        position_size: float = 1.0  # 1 unità o 100% del capitale
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_sizing = position_sizing
        self.position_size = position_size
    
    def run(self, strategy: Strategy, data: pd.DataFrame) -> BacktestResult:
        """
        Esegue backtest.
        
        Args:
            strategy: Strategia da testare
            data: DataFrame con OHLCV (deve avere colonne: open, high, low, close, volume)
        """
        # Genera segnali
        strategy.set_data(data)
        signals = strategy.signals
        
        # Inizializza
        capital = self.initial_capital
        position = 0
        trades: List[Trade] = []
        current_trade: Optional[Trade] = None
        equity = [capital]
        
        for i in range(1, len(data)):
            date = data.index[i]
            price = data['close'].iloc[i]
            signal = signals.iloc[i]
            
            # Applica slippage
            buy_price = price * (1 + self.slippage)
            sell_price = price * (1 - self.slippage)
            
            # Calcola size
            if self.position_sizing == 'fixed':
                size = self.position_size
            else:
                size = (capital * self.position_size) / price
            
            # BUY Signal
            if signal == 1 and position == 0:
                # Commissione
                cost = buy_price * size * self.commission
                
                if capital >= buy_price * size + cost:
                    position = size
                    capital -= buy_price * size + cost
                    
                    current_trade = Trade(
                        entry_time=date,
                        exit_time=None,
                        entry_price=buy_price,
                        exit_price=None,
                        size=size,
                        side=SignalType.BUY
                    )
            
            # SELL Signal (chiudi posizione long)
            elif signal == -1 and position > 0:
                cost = sell_price * position * self.commission
                capital += sell_price * position - cost
                
                if current_trade:
                    current_trade.close(date, sell_price)
                    trades.append(current_trade)
                    current_trade = None
                
                position = 0
            
            # Calcola equity
            current_equity = capital + position * price
            equity.append(current_equity)
        
        # Chiudi posizione finale
        if position > 0:
            final_price = data['close'].iloc[-1]
            cost = final_price * position * self.commission
            capital += final_price * position - cost
            
            if current_trade:
                current_trade.close(data.index[-1], final_price)
                trades.append(current_trade)
        
        # Calcola metriche
        equity_series = pd.Series(equity, index=data.index[:len(equity)])
        
        return self._calculate_metrics(equity_series, trades, data)
    
    def _calculate_metrics(
        self,
        equity: pd.Series,
        trades: List[Trade],
        data: pd.DataFrame
    ) -> BacktestResult:
        """Calcola tutte le metriche di performance."""
        
        # Returns
        returns = equity.pct_change().dropna()
        
        # Total return
        total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]
        
        # Annual return (assumendo 252 giorni)
        days = (data.index[-1] - data.index[0]).days
        annual_return = (1 + total_return) ** (365 / max(days, 1)) - 1
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe = annual_return / volatility if volatility > 0 else 0
        
        # Max drawdown
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        max_drawdown = drawdown.min()
        
        # Trade statistics
        if trades:
            pnls = [t.pnl for t in trades if t.pnl is not None]
            winning = [p for p in pnls if p > 0]
            losing = [p for p in pnls if p <= 0]
            
            win_rate = len(winning) / len(pnls) if pnls else 0
            avg_win = np.mean(winning) if winning else 0
            avg_loss = np.mean(losing) if losing else 0
            
            profit_factor = abs(sum(winning) / sum(losing)) if losing and sum(losing) != 0 else float('inf')
            
            largest_win = max(pnls) if pnls else 0
            largest_loss = min(pnls) if pnls else 0
            
            # Duration
            durations = [(t.exit_time - t.entry_time) for t in trades if t.exit_time]
            avg_duration = np.mean(durations) if durations else timedelta(0)
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            largest_win = 0
            largest_loss = 0
            avg_duration = timedelta(0)
            winning = []
            losing = []
        
        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_trade_duration=avg_duration,
            equity_curve=equity,
            trades=trades
        )


# ══════════════════════════════════════════════════════════════════════════════
#                    REPORT E VISUALIZZAZIONE
# ══════════════════════════════════════════════════════════════════════════════

def print_report(result: BacktestResult, strategy_name: str = "Strategy"):
    """Stampa report dettagliato."""
    print("\n" + "=" * 70)
    print(f"               BACKTEST REPORT: {strategy_name}")
    print("=" * 70)
    
    print("\n📈 PERFORMANCE METRICS")
    print("-" * 40)
    print(f"Total Return:        {result.total_return * 100:>10.2f}%")
    print(f"Annual Return:       {result.annual_return * 100:>10.2f}%")
    print(f"Volatility:          {result.volatility * 100:>10.2f}%")
    print(f"Sharpe Ratio:        {result.sharpe_ratio:>10.2f}")
    print(f"Max Drawdown:        {result.max_drawdown * 100:>10.2f}%")
    
    print("\n📊 TRADE STATISTICS")
    print("-" * 40)
    print(f"Total Trades:        {result.total_trades:>10}")
    print(f"Winning Trades:      {result.winning_trades:>10}")
    print(f"Losing Trades:       {result.losing_trades:>10}")
    print(f"Win Rate:            {result.win_rate * 100:>10.2f}%")
    print(f"Profit Factor:       {result.profit_factor:>10.2f}")
    
    print("\n💰 PNL STATISTICS")
    print("-" * 40)
    print(f"Average Win:         ${result.avg_win:>10.2f}")
    print(f"Average Loss:        ${result.avg_loss:>10.2f}")
    print(f"Largest Win:         ${result.largest_win:>10.2f}")
    print(f"Largest Loss:        ${result.largest_loss:>10.2f}")
    print(f"Avg Trade Duration:  {result.avg_trade_duration}")
    
    print("\n" + "=" * 70)


def plot_results(result: BacktestResult, data: pd.DataFrame, strategy_name: str = "Strategy"):
    """Visualizza risultati con matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Installa matplotlib per i grafici: pip install matplotlib")
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 1. Price + Signals
    ax1 = axes[0]
    ax1.plot(data.index, data['close'], label='Price', alpha=0.7)
    ax1.set_title(f'{strategy_name} - Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Equity Curve
    ax2 = axes[1]
    ax2.plot(result.equity_curve.index, result.equity_curve.values, label='Equity', color='green')
    ax2.set_title('Equity Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Drawdown
    ax3 = axes[2]
    cummax = result.equity_curve.cummax()
    drawdown = (result.equity_curve - cummax) / cummax * 100
    ax3.fill_between(drawdown.index, drawdown.values, 0, alpha=0.5, color='red')
    ax3.set_title('Drawdown (%)')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('backtest_results.png', dpi=150)
    plt.show()
    print("Grafico salvato in backtest_results.png")


# ══════════════════════════════════════════════════════════════════════════════
#                    DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def generate_sample_data(
    days: int = 365,
    start_price: float = 100,
    volatility: float = 0.02
) -> pd.DataFrame:
    """Genera dati simulati per testing."""
    np.random.seed(42)
    
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Random walk con drift
    returns = np.random.normal(0.0001, volatility, days)
    prices = start_price * np.exp(np.cumsum(returns))
    
    # Genera OHLCV
    data = pd.DataFrame(index=dates)
    data['close'] = prices
    data['open'] = data['close'].shift(1).fillna(start_price)
    data['high'] = data[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.01, days))
    data['low'] = data[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.01, days))
    data['volume'] = np.random.randint(1000, 10000, days)
    
    return data


def load_crypto_data(symbol: str = 'BTC/USDT', timeframe: str = '1d', limit: int = 365):
    """
    Carica dati crypto da exchange via ccxt.
    """
    try:
        import ccxt
    except ImportError:
        print("Installa ccxt: pip install ccxt")
        return generate_sample_data()
    
    exchange = ccxt.binance()
    
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    return df


# ══════════════════════════════════════════════════════════════════════════════
#                    ESEMPIO COMPLETO
# ══════════════════════════════════════════════════════════════════════════════

def run_example():
    """Esegui esempio completo di backtest."""
    print("\n" + "=" * 70)
    print("                    BACKTESTING EXAMPLE")
    print("=" * 70)
    
    # 1. Genera/carica dati
    print("\n1. Generando dati di esempio...")
    data = generate_sample_data(days=500, start_price=50000, volatility=0.03)
    print(f"   Dati: {len(data)} giorni, da {data.index[0]} a {data.index[-1]}")
    
    # 2. Crea strategie
    strategies = [
        SMACrossStrategy(fast_period=10, slow_period=30),
        RSIMeanReversionStrategy(period=14, oversold=30, overbought=70),
        BollingerBandsStrategy(period=20, std_dev=2.0),
        MACDStrategy(fast=12, slow=26, signal=9)
    ]
    
    # 3. Backtest
    print("\n2. Eseguendo backtest...")
    backtester = Backtester(
        initial_capital=10000,
        commission=0.001,
        slippage=0.0005
    )
    
    results = []
    for strategy in strategies:
        result = backtester.run(strategy, data)
        results.append((strategy.name, result))
        print(f"   ✓ {strategy.name}: Return={result.total_return*100:.2f}%, Sharpe={result.sharpe_ratio:.2f}")
    
    # 4. Report migliore strategia
    best = max(results, key=lambda x: x[1].sharpe_ratio)
    print(f"\n3. Migliore strategia: {best[0]}")
    print_report(best[1], best[0])
    
    # 5. Grafico
    try:
        plot_results(best[1], data, best[0])
    except:
        print("(Grafici non disponibili - installa matplotlib)")
    
    return results


# ══════════════════════════════════════════════════════════════════════════════
#                    MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_example()
