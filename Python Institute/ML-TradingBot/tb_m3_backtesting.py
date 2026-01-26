#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    TRADING BOT - MODULE 3                                    ║
║                    BACKTESTING                                                ║
║                    TB-M3: 20% del percorso Trading                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

SYLLABUS TB MODULE 3:
├── TB 3.1 - Backtesting Concepts (walk-forward, overfitting, look-ahead bias)
├── TB 3.2 - Backtester Engine (trade execution simulation)
├── TB 3.3 - Performance Metrics (Sharpe, Sortino, Max DD, Win Rate)
├── TB 3.4 - Trade Analysis (entry/exit log, PnL)
├── TB 3.5 - Visualization (equity curve, drawdown chart)
└── TB 3.6 - Optimization & Avoiding Overfitting

PREREQUISITI: TB-M1, TB-M2 completati
TEMPO STIMATO: 2-3 settimane (2-3 ore/giorno)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
from datetime import datetime

# ══════════════════════════════════════════════════════════════════════════════
# TB 3.1 - BACKTESTING CONCEPTS
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TB 3.1 - BACKTESTING CONCEPTS")
print("=" * 70)

print("""
📋 COS'È IL BACKTESTING?

Simulare la strategia su DATI STORICI per valutare le performance.

⚠️ ATTENZIONE AI BIAS:

┌─────────────────────┬─────────────────────────────────────────┐
│ Look-Ahead Bias     │ Usare dati futuri nella decisione       │
│                     │ Es: usare close prima che sia formato   │
├─────────────────────┼─────────────────────────────────────────┤
│ Survivorship Bias   │ Testare solo su asset che esistono oggi │
│                     │ Es: ignorare crypto fallite             │
├─────────────────────┼─────────────────────────────────────────┤
│ Overfitting         │ Ottimizzare troppo sui dati storici     │
│                     │ La strategia non funzionerà in futuro   │
├─────────────────────┼─────────────────────────────────────────┤
│ Data Snooping       │ Testare troppe varianti                 │
│                     │ Per fortuna qualcuna funzionerà         │
└─────────────────────┴─────────────────────────────────────────┘

📋 METODOLOGIE:

1. In-Sample / Out-of-Sample Split
   - Ottimizza su 70% dati
   - Valida su 30% dati MAI visti

2. Walk-Forward Analysis
   - Train su periodo 1, test su periodo 2
   - Train su periodo 1+2, test su periodo 3
   - Simula condizioni reali

3. Cross-Validation per Time Series
   - Non random! Rispetta ordine temporale
""")

# ══════════════════════════════════════════════════════════════════════════════
# TB 3.2 - BACKTESTER ENGINE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TB 3.2 - BACKTESTER ENGINE")
print("=" * 70)

class Position(Enum):
    FLAT = 0
    LONG = 1
    SHORT = -1

@dataclass
class Trade:
    """Rappresenta un singolo trade"""
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    position: Position = Position.LONG
    size: float = 1.0
    pnl: float = 0.0
    pnl_percent: float = 0.0
    
    def close(self, exit_time, exit_price):
        self.exit_time = exit_time
        self.exit_price = exit_price
        
        if self.position == Position.LONG:
            self.pnl = (self.exit_price - self.entry_price) * self.size
            self.pnl_percent = (self.exit_price - self.entry_price) / self.entry_price
        else:  # SHORT
            self.pnl = (self.entry_price - self.exit_price) * self.size
            self.pnl_percent = (self.entry_price - self.exit_price) / self.entry_price


@dataclass
class BacktestResult:
    """Risultati del backtest"""
    initial_capital: float
    final_capital: float
    total_return: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = None


class Backtester:
    """
    Engine per backtesting strategie.
    Simula esecuzione ordini su dati storici.
    """
    
    def __init__(self, initial_capital: float = 10000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission  # 0.1% per trade
    
    def run(self, df: pd.DataFrame, signals: pd.Series) -> BacktestResult:
        """
        Esegue backtest.
        
        Args:
            df: DataFrame OHLCV
            signals: Series con segnali (1=buy, -1=sell, 0=hold)
        
        Returns:
            BacktestResult con tutte le metriche
        """
        capital = self.initial_capital
        position = Position.FLAT
        current_trade = None
        trades = []
        equity = [capital]
        
        for i in range(1, len(df)):
            timestamp = df.index[i]
            price = df['close'].iloc[i]
            signal = signals.iloc[i]
            
            # Close existing position on opposite signal
            if position != Position.FLAT and signal != 0:
                if (position == Position.LONG and signal == -1) or \
                   (position == Position.SHORT and signal == 1):
                    # Close position
                    current_trade.close(timestamp, price)
                    capital += current_trade.pnl
                    capital -= abs(current_trade.pnl) * self.commission
                    trades.append(current_trade)
                    position = Position.FLAT
                    current_trade = None
            
            # Open new position
            if position == Position.FLAT and signal != 0:
                position = Position.LONG if signal == 1 else Position.SHORT
                size = capital / price  # All-in per semplicità
                current_trade = Trade(
                    entry_time=timestamp,
                    entry_price=price,
                    position=position,
                    size=size
                )
                capital -= capital * self.commission
            
            # Track equity
            if current_trade:
                if position == Position.LONG:
                    unrealized = (price - current_trade.entry_price) * current_trade.size
                else:
                    unrealized = (current_trade.entry_price - price) * current_trade.size
                equity.append(capital + unrealized)
            else:
                equity.append(capital)
        
        # Close any open position at end
        if current_trade:
            current_trade.close(df.index[-1], df['close'].iloc[-1])
            capital += current_trade.pnl
            trades.append(current_trade)
        
        # Calculate metrics
        return self._calculate_metrics(trades, equity)
    
    def _calculate_metrics(self, trades: List[Trade], equity: List[float]) -> BacktestResult:
        """Calcola metriche di performance"""
        if not trades:
            return BacktestResult(
                initial_capital=self.initial_capital,
                final_capital=equity[-1],
                total_return=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                avg_win=0,
                avg_loss=0,
                profit_factor=0,
                max_drawdown=0,
                sharpe_ratio=0,
                trades=trades,
                equity_curve=pd.Series(equity)
            )
        
        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        equity_series = pd.Series(equity)
        returns = equity_series.pct_change().dropna()
        
        # Max Drawdown
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_dd = drawdown.min()
        
        # Sharpe Ratio (annualizzato, assumendo dati orari)
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 24) if returns.std() > 0 else 0
        
        return BacktestResult(
            initial_capital=self.initial_capital,
            final_capital=equity[-1],
            total_return=(equity[-1] - self.initial_capital) / self.initial_capital,
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=len(wins) / len(trades) if trades else 0,
            avg_win=np.mean(wins) if wins else 0,
            avg_loss=np.mean(losses) if losses else 0,
            profit_factor=abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            trades=trades,
            equity_curve=equity_series
        )


# Demo
print("🔹 ESEMPIO BACKTEST:")

# Create sample data
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=500, freq='1h')
returns = np.random.randn(500) * 0.02
prices = 100 * np.exp(np.cumsum(returns))

df = pd.DataFrame({
    'open': prices,
    'high': prices * (1 + np.abs(np.random.randn(500) * 0.01)),
    'low': prices * (1 - np.abs(np.random.randn(500) * 0.01)),
    'close': prices * (1 + np.random.randn(500) * 0.005),
    'volume': np.random.randint(1000, 10000, 500)
}, index=dates)

# Simple SMA cross signals
sma_fast = df['close'].rolling(10).mean()
sma_slow = df['close'].rolling(30).mean()
signals = pd.Series(0, index=df.index)
signals[(sma_fast > sma_slow) & (sma_fast.shift(1) <= sma_slow.shift(1))] = 1
signals[(sma_fast < sma_slow) & (sma_fast.shift(1) >= sma_slow.shift(1))] = -1

# Run backtest
backtester = Backtester(initial_capital=10000, commission=0.001)
result = backtester.run(df, signals)

print(f"Initial Capital: ${result.initial_capital:,.2f}")
print(f"Final Capital: ${result.final_capital:,.2f}")
print(f"Total Return: {result.total_return*100:.2f}%")
print(f"Total Trades: {result.total_trades}")
print(f"Win Rate: {result.win_rate*100:.1f}%")
print(f"Max Drawdown: {result.max_drawdown*100:.2f}%")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# TB 3.3 - PERFORMANCE METRICS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TB 3.3 - PERFORMANCE METRICS")
print("=" * 70)

print("""
📋 METRICHE PRINCIPALI

┌─────────────────────┬─────────────────────────────────────────┐
│ Total Return        │ (finale - iniziale) / iniziale         │
│                     │ Guadagno totale in %                    │
├─────────────────────┼─────────────────────────────────────────┤
│ Win Rate            │ trades vincenti / totali                │
│                     │ > 50% non significa profittevole!       │
├─────────────────────┼─────────────────────────────────────────┤
│ Profit Factor       │ gross profit / gross loss               │
│                     │ > 1.5 è buono, > 2 è eccellente         │
├─────────────────────┼─────────────────────────────────────────┤
│ Sharpe Ratio        │ (return - risk_free) / std(returns)     │
│                     │ Risk-adjusted return. > 1 buono, > 2 ★  │
├─────────────────────┼─────────────────────────────────────────┤
│ Sortino Ratio       │ Come Sharpe ma solo downside deviation  │
│                     │ Penalizza solo volatilità negativa      │
├─────────────────────┼─────────────────────────────────────────┤
│ Max Drawdown        │ Massima perdita da picco                │
│                     │ < 20% accettabile, < 10% ottimo         │
├─────────────────────┼─────────────────────────────────────────┤
│ Calmar Ratio        │ Annual return / Max Drawdown            │
│                     │ Rendimento per unità di drawdown        │
├─────────────────────┼─────────────────────────────────────────┤
│ Expectancy          │ (win_rate * avg_win) - (loss_rate * avg_loss) │
│                     │ Profitto atteso per trade               │
└─────────────────────┴─────────────────────────────────────────┘
""")

def calculate_sharpe_ratio(returns: pd.Series, risk_free: float = 0.02, periods_per_year: int = 252*24):
    """Calcola Sharpe Ratio annualizzato"""
    excess_returns = returns - risk_free / periods_per_year
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()

def calculate_sortino_ratio(returns: pd.Series, risk_free: float = 0.02, periods_per_year: int = 252*24):
    """Calcola Sortino Ratio (solo downside deviation)"""
    excess_returns = returns - risk_free / periods_per_year
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std()
    return np.sqrt(periods_per_year) * excess_returns.mean() / downside_std if downside_std > 0 else 0

def calculate_calmar_ratio(total_return: float, max_drawdown: float, years: float):
    """Calcola Calmar Ratio"""
    annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    return abs(annual_return / max_drawdown) if max_drawdown != 0 else 0

def calculate_expectancy(win_rate: float, avg_win: float, avg_loss: float):
    """Calcola Expectancy per trade"""
    return (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))

# Demo metrics
equity_returns = result.equity_curve.pct_change().dropna()
print(f"🔹 METRICHE CALCOLATE:")
print(f"Sharpe Ratio: {calculate_sharpe_ratio(equity_returns):.2f}")
print(f"Sortino Ratio: {calculate_sortino_ratio(equity_returns):.2f}")
print(f"Expectancy: ${calculate_expectancy(result.win_rate, result.avg_win, result.avg_loss):.2f} per trade")

# ══════════════════════════════════════════════════════════════════════════════
# TB 3.4 - TRADE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TB 3.4 - TRADE ANALYSIS")
print("=" * 70)

print("🔹 TRADE LOG:")
if result.trades:
    trade_data = []
    for i, t in enumerate(result.trades[:5]):  # Prime 5 trade
        trade_data.append({
            '#': i+1,
            'Entry': t.entry_time.strftime('%Y-%m-%d %H:%M') if t.entry_time else '',
            'Exit': t.exit_time.strftime('%Y-%m-%d %H:%M') if t.exit_time else '',
            'Entry $': f"{t.entry_price:.2f}",
            'Exit $': f"{t.exit_price:.2f}" if t.exit_price else '',
            'PnL': f"${t.pnl:.2f}",
            'PnL %': f"{t.pnl_percent*100:.2f}%"
        })
    
    trade_df = pd.DataFrame(trade_data)
    print(trade_df.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# TB 3.5 - VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TB 3.5 - VISUALIZATION")
print("=" * 70)

print("""
📋 GRAFICI ESSENZIALI

import matplotlib.pyplot as plt

🔹 EQUITY CURVE:

fig, ax = plt.subplots(figsize=(12, 6))
result.equity_curve.plot(ax=ax)
ax.set_title('Equity Curve')
ax.set_ylabel('Capital ($)')
ax.axhline(y=initial_capital, color='r', linestyle='--', label='Initial')
plt.legend()
plt.show()

🔹 DRAWDOWN CHART:

rolling_max = result.equity_curve.cummax()
drawdown = (result.equity_curve - rolling_max) / rolling_max * 100

fig, ax = plt.subplots(figsize=(12, 4))
drawdown.plot(ax=ax, color='red', fill=True, alpha=0.3)
ax.set_title('Drawdown')
ax.set_ylabel('Drawdown (%)')
plt.show()

🔹 MONTHLY RETURNS HEATMAP:

# Resample to monthly returns
monthly = result.equity_curve.resample('M').last().pct_change()
monthly_matrix = monthly.values.reshape(-1, 12)  # 12 months

import seaborn as sns
sns.heatmap(monthly_matrix, annot=True, fmt='.1%', cmap='RdYlGn')
plt.title('Monthly Returns')
plt.show()
""")

# ══════════════════════════════════════════════════════════════════════════════
# TB 3.6 - OPTIMIZATION & AVOIDING OVERFITTING
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TB 3.6 - OPTIMIZATION & AVOIDING OVERFITTING")
print("=" * 70)

print("""
📋 OTTIMIZZAZIONE PARAMETRI

⚠️ PERICOLO: Ottimizzare troppo = OVERFITTING!

METODO CORRETTO:
1. Dividi dati: 70% train, 30% test
2. Ottimizza SOLO su train
3. Valida su test (MAI toccare prima!)
4. Se buono su entrambi → OK

TECNICHE ANTI-OVERFITTING:

1. Walk-Forward Optimization
   - Ottimizza su periodo 1
   - Testa su periodo 2
   - Ri-ottimizza includendo periodo 2
   - Testa su periodo 3
   - ...

2. Robustness Testing
   - Cambia leggermente i parametri
   - La strategia funziona ancora?
   - Se sì = robusta, se no = overfitted

3. Out-of-Sample Testing
   - Tieni 20-30% dati completamente separati
   - Mai usarli per ottimizzazione
   - Test finale solo su questi

4. Monte Carlo Simulation
   - Randomizza ordine dei trade
   - Risultati simili? = robusto
""")

def walk_forward_optimization(df, strategy_class, param_grid, train_window, test_window):
    """
    Walk-Forward Optimization Framework
    
    Args:
        df: Full DataFrame
        strategy_class: Classe della strategia
        param_grid: Dict con parametri da testare
        train_window: Numero di barre per training
        test_window: Numero di barre per testing
    """
    results = []
    
    for start in range(0, len(df) - train_window - test_window, test_window):
        train_end = start + train_window
        test_end = train_end + test_window
        
        df_train = df.iloc[start:train_end]
        df_test = df.iloc[train_end:test_end]
        
        # Optimize on train
        best_params = None
        best_score = -np.inf
        
        # Grid search (simplified)
        for params in param_grid:
            strategy = strategy_class(**params)
            signals = strategy.generate_signals(df_train)
            bt = Backtester()
            result = bt.run(df_train, signals)
            
            if result.sharpe_ratio > best_score:
                best_score = result.sharpe_ratio
                best_params = params
        
        # Test on out-of-sample with best params
        strategy = strategy_class(**best_params)
        signals = strategy.generate_signals(df_test)
        bt = Backtester()
        test_result = bt.run(df_test, signals)
        
        results.append({
            'train_sharpe': best_score,
            'test_sharpe': test_result.sharpe_ratio,
            'params': best_params
        })
    
    return results

print("""
🔹 ESEMPIO WALK-FORWARD:

param_grid = [
    {'fast_period': 5, 'slow_period': 20},
    {'fast_period': 10, 'slow_period': 30},
    {'fast_period': 15, 'slow_period': 50},
]

results = walk_forward_optimization(
    df, SMACrossStrategy, param_grid,
    train_window=200, test_window=50
)

# Analizza: train_sharpe vs test_sharpe dovrebbero essere simili
# Se train >> test → overfitting!
""")

# ══════════════════════════════════════════════════════════════════════════════
# QUIZ DI VERIFICA
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("QUIZ DI VERIFICA - TB MODULE 3")
print("=" * 70)

print("""
Q1. Look-ahead bias è?
    A) Usare dati futuri  B) Ignorare commissioni  C) Overfitting  D) Slippage
    → RISPOSTA: A

Q2. Profit Factor > 1 significa?
    A) Perdita  B) Break-even  C) Profitto  D) Errore
    → RISPOSTA: C

Q3. Sharpe Ratio misura?
    A) Solo return  B) Risk-adjusted return  C) Solo risk  D) Trades
    → RISPOSTA: B

Q4. Max Drawdown del 20% significa?
    A) +20% max  B) -20% da picco  C) 20 trades  D) 20% win rate
    → RISPOSTA: B

Q5. Walk-forward optimization serve per?
    A) Velocità  B) Evitare overfitting  C) Più trades  D) Meno codice
    → RISPOSTA: B

Q6. Win rate 40% può essere profittevole?
    A) Mai  B) Sì, se avg_win > avg_loss  C) Sempre  D) Solo con leverage
    → RISPOSTA: B

Q7. Out-of-sample data è?
    A) Dati di training  B) Dati mai usati per ottimizzare  C) Dati futuri  D) Random
    → RISPOSTA: B

Q8. Sortino vs Sharpe: differenza?
    A) Nessuna  B) Sortino usa solo downside  C) Sharpe è meglio  D) Sortino è annuale
    → RISPOSTA: B

Q9. Commission 0.1% per trade significa?
    A) 0.1% totale  B) 0.1% entry + 0.1% exit  C) 0.1% al giorno  D) Gratis
    → RISPOSTA: B (0.2% round-trip)

Q10. Equity curve mostra?
    A) Prezzo asset  B) Capitale nel tempo  C) Drawdown  D) Trades
    → RISPOSTA: B
""")

print("\n" + "=" * 70)
print("TB MODULE 3 - BACKTESTING COMPLETATO!")
print("Prossimo: TB MODULE 4 - EXCHANGE CONNECTION")
print("=" * 70)
