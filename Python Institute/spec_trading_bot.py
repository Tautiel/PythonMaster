"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    SPECIALIZATION MODULE: TRADING BOT                        ║
║                         From Python to Production                            ║
║                                                                              ║
║                   Applicazione Pratica delle Certificazioni                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Questo modulo collega le competenze Python Institute al tuo obiettivo:
Trading Bot per Crypto Scalping.

STRUTTURA:
├── Part 1: Architecture Overview
├── Part 2: Data Layer (PCEP + Database)
├── Part 3: Strategy Engine (PCAP OOP)
├── Part 4: Exchange Integration (PCPP1 Network)
├── Part 5: Execution Engine (PCPP2 Concurrency)
├── Part 6: Risk Management
└── Part 7: Complete Bot Skeleton

═══════════════════════════════════════════════════════════════════════════════
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Callable
from enum import Enum
import time
import logging

# ══════════════════════════════════════════════════════════════════════════════
#                    PART 1: ARCHITECTURE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    TRADING BOT ARCHITECTURE                                  │
└──────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────┐
                    │   Config/Env    │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  Data Layer   │   │   Strategy    │   │    Risk       │
│  (Market Data)│──▶│   Engine      │──▶│  Management   │
└───────────────┘   └───────────────┘   └───────┬───────┘
        │                    │                  │
        │                    │                  │
        ▼                    ▼                  ▼
┌───────────────────────────────────────────────────────┐
│                  Execution Engine                     │
│              (Order Management System)                │
└───────────────────────────┬───────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   Exchange    │
                    │  (ccxt/API)   │
                    └───────────────┘


COMPONENTI E COMPETENZE PYTHON:
───────────────────────────────
1. Data Layer       → PCEP (types, collections) + PCPP2 (database)
2. Strategy Engine  → PCAP (OOP, inheritance)
3. Risk Management  → PCAP (exceptions) + PCPP1 (logging)
4. Execution Engine → PCPP2 (concurrency, asyncio)
5. Exchange API     → PCPP1 (network, REST)
6. Configuration    → PCPP1 (file processing, JSON)
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    PART 2: DATA LAYER
# ══════════════════════════════════════════════════════════════════════════════
"""
Competenze: PCEP (data types, collections) + PCPP2 (database)
"""

class OrderSide(Enum):
    """Enum per direzione ordine."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Tipo di ordine."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderStatus(Enum):
    """Stato dell'ordine."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Candle:
    """
    OHLCV Candle data.
    Usa dataclass (PCAP) per struttura dati immutabile.
    """
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    @property
    def body_size(self) -> float:
        """Dimensione corpo candela."""
        return abs(self.close - self.open)
    
    @property
    def is_bullish(self) -> bool:
        """Candela rialzista?"""
        return self.close > self.open
    
    @property
    def upper_wick(self) -> float:
        """Ombra superiore."""
        return self.high - max(self.open, self.close)
    
    @property
    def lower_wick(self) -> float:
        """Ombra inferiore."""
        return min(self.open, self.close) - self.low


@dataclass
class Order:
    """Rappresenta un ordine."""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None


@dataclass
class Position:
    """Posizione aperta."""
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    @property
    def unrealized_pnl(self) -> Callable[[float], float]:
        """Ritorna funzione per calcolare PnL dato prezzo corrente."""
        def calculate(current_price: float) -> float:
            if self.side == OrderSide.BUY:
                return (current_price - self.entry_price) * self.quantity
            else:
                return (self.entry_price - current_price) * self.quantity
        return calculate


# ══════════════════════════════════════════════════════════════════════════════
#                    PART 3: STRATEGY ENGINE
# ══════════════════════════════════════════════════════════════════════════════
"""
Competenze: PCAP (OOP, inheritance, ABC)
"""

@dataclass
class Signal:
    """Segnale di trading generato dalla strategia."""
    symbol: str
    side: OrderSide
    strength: float  # 0.0 - 1.0
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)


class Strategy(ABC):
    """
    Base class per tutte le strategie.
    Usa ABC (PCAP) per definire interfaccia.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"strategy.{name}")
    
    @abstractmethod
    def analyze(self, candles: List[Candle]) -> Optional[Signal]:
        """
        Analizza candele e genera segnale.
        Deve essere implementato dalle sottoclassi.
        """
        pass
    
    @abstractmethod
    def get_required_candles(self) -> int:
        """Numero minimo di candele necessarie."""
        pass


class ScalpingStrategy(Strategy):
    """
    Esempio strategia scalping base.
    Eredita da Strategy (PCAP inheritance).
    """
    
    def __init__(
        self,
        name: str = "BasicScalping",
        ema_fast: int = 9,
        ema_slow: int = 21,
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70
    ):
        super().__init__(name)
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
    
    def get_required_candles(self) -> int:
        return max(self.ema_slow, self.rsi_period) + 10
    
    def _calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """Calcola EMA (Exponential Moving Average)."""
        if len(prices) < period:
            return []
        
        multiplier = 2 / (period + 1)
        ema = [sum(prices[:period]) / period]  # SMA iniziale
        
        for price in prices[period:]:
            ema.append((price - ema[-1]) * multiplier + ema[-1])
        
        return ema
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calcola RSI (Relative Strength Index)."""
        if len(prices) < period + 1:
            return 50.0  # Neutro
        
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [c if c > 0 else 0 for c in changes[-period:]]
        losses = [-c if c < 0 else 0 for c in changes[-period:]]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def analyze(self, candles: List[Candle]) -> Optional[Signal]:
        """
        Analizza e genera segnale.
        Logica: EMA crossover + RSI confirmation
        """
        if len(candles) < self.get_required_candles():
            return None
        
        closes = [c.close for c in candles]
        
        # Calcola indicatori
        ema_fast = self._calculate_ema(closes, self.ema_fast)
        ema_slow = self._calculate_ema(closes, self.ema_slow)
        rsi = self._calculate_rsi(closes, self.rsi_period)
        
        if not ema_fast or not ema_slow:
            return None
        
        # Segnale BUY: EMA fast > EMA slow + RSI non overbought
        if ema_fast[-1] > ema_slow[-1] and rsi < self.rsi_overbought:
            if ema_fast[-2] <= ema_slow[-2]:  # Crossover appena avvenuto
                return Signal(
                    symbol=candles[-1].timestamp.isoformat(),  # placeholder
                    side=OrderSide.BUY,
                    strength=min((self.rsi_overbought - rsi) / 40, 1.0),
                    reason=f"EMA crossover UP, RSI={rsi:.1f}"
                )
        
        # Segnale SELL: EMA fast < EMA slow + RSI non oversold
        if ema_fast[-1] < ema_slow[-1] and rsi > self.rsi_oversold:
            if ema_fast[-2] >= ema_slow[-2]:  # Crossover appena avvenuto
                return Signal(
                    symbol=candles[-1].timestamp.isoformat(),
                    side=OrderSide.SELL,
                    strength=min((rsi - self.rsi_oversold) / 40, 1.0),
                    reason=f"EMA crossover DOWN, RSI={rsi:.1f}"
                )
        
        return None


# ══════════════════════════════════════════════════════════════════════════════
#                    PART 4: EXCHANGE INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════
"""
Competenze: PCPP1 (Network Programming, REST APIs)

In produzione userai: ccxt library
pip install ccxt
"""

class ExchangeInterface(ABC):
    """
    Interfaccia astratta per exchange.
    Permette di supportare multiple exchange.
    """
    
    @abstractmethod
    def get_balance(self, asset: str) -> float:
        pass
    
    @abstractmethod
    def get_ticker(self, symbol: str) -> Dict:
        pass
    
    @abstractmethod
    def get_ohlcv(self, symbol: str, timeframe: str, limit: int) -> List[Candle]:
        pass
    
    @abstractmethod
    def create_order(self, order: Order) -> Order:
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        pass


class MockExchange(ExchangeInterface):
    """
    Exchange simulato per testing.
    Implementa l'interfaccia senza connessione reale.
    """
    
    def __init__(self):
        self.balances = {"USDT": 10000.0, "BTC": 0.0}
        self.orders: Dict[str, Order] = {}
        self._order_counter = 0
    
    def get_balance(self, asset: str) -> float:
        return self.balances.get(asset, 0.0)
    
    def get_ticker(self, symbol: str) -> Dict:
        # Simula ticker BTC/USDT
        import random
        base_price = 50000
        return {
            "symbol": symbol,
            "bid": base_price - random.uniform(10, 50),
            "ask": base_price + random.uniform(10, 50),
            "last": base_price + random.uniform(-30, 30)
        }
    
    def get_ohlcv(self, symbol: str, timeframe: str, limit: int) -> List[Candle]:
        """Genera candele simulate."""
        import random
        candles = []
        base_price = 50000
        current_time = datetime.now()
        
        for i in range(limit):
            open_price = base_price + random.uniform(-100, 100)
            close_price = open_price + random.uniform(-50, 50)
            high_price = max(open_price, close_price) + random.uniform(0, 30)
            low_price = min(open_price, close_price) - random.uniform(0, 30)
            
            candles.append(Candle(
                timestamp=current_time,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=random.uniform(100, 1000)
            ))
            
            base_price = close_price
        
        return candles
    
    def create_order(self, order: Order) -> Order:
        self._order_counter += 1
        order.id = f"MOCK-{self._order_counter}"
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.now()
        order.filled_price = order.price or 50000  # Simula fill
        self.orders[order.id] = order
        return order
    
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False


CCXT_EXAMPLE = '''
# Esempio con ccxt (libreria reale)
import ccxt

# Inizializzare exchange
exchange = ccxt.binance({
    'apiKey': 'YOUR_API_KEY',
    'secret': 'YOUR_SECRET',
    'sandbox': True,  # Testnet!
})

# Fetch balance
balance = exchange.fetch_balance()
print(balance['USDT']['free'])

# Fetch ticker
ticker = exchange.fetch_ticker('BTC/USDT')
print(ticker['last'])

# Fetch OHLCV
ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1m', limit=100)
# [[timestamp, open, high, low, close, volume], ...]

# Create order
order = exchange.create_limit_buy_order('BTC/USDT', 0.001, 50000)

# Cancel order
exchange.cancel_order(order['id'], 'BTC/USDT')
'''


# ══════════════════════════════════════════════════════════════════════════════
#                    PART 5: EXECUTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
"""
Competenze: PCPP2 (Concurrency, AsyncIO)
"""

class ExecutionEngine:
    """
    Gestisce esecuzione ordini.
    In produzione userebbe asyncio per operazioni non-blocking.
    """
    
    def __init__(self, exchange: ExchangeInterface):
        self.exchange = exchange
        self.pending_orders: List[Order] = []
        self.filled_orders: List[Order] = []
        self.logger = logging.getLogger("execution")
    
    def submit_order(self, order: Order) -> Order:
        """Sottomette ordine all'exchange."""
        self.logger.info(f"Submitting order: {order}")
        
        try:
            filled_order = self.exchange.create_order(order)
            
            if filled_order.status == OrderStatus.FILLED:
                self.filled_orders.append(filled_order)
                self.logger.info(f"Order filled: {filled_order.id}")
            else:
                self.pending_orders.append(filled_order)
            
            return filled_order
            
        except Exception as e:
            self.logger.error(f"Order submission failed: {e}")
            order.status = OrderStatus.REJECTED
            return order
    
    def cancel_all_pending(self, symbol: str) -> int:
        """Cancella tutti gli ordini pending per un simbolo."""
        cancelled = 0
        for order in self.pending_orders[:]:  # Copia per iterare
            if order.symbol == symbol:
                if self.exchange.cancel_order(order.id, symbol):
                    self.pending_orders.remove(order)
                    cancelled += 1
        return cancelled


# ══════════════════════════════════════════════════════════════════════════════
#                    PART 6: RISK MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
"""
Competenze: PCAP (Exceptions), PCPP1 (Logging)
"""

class RiskLimitExceeded(Exception):
    """Eccezione custom per limiti di rischio."""
    pass


class RiskManager:
    """
    Gestisce risk management.
    CRITICO per trading reale!
    """
    
    def __init__(
        self,
        max_position_size: float = 0.1,      # 10% del capitale
        max_daily_loss: float = 0.02,        # 2% del capitale
        max_drawdown: float = 0.05,          # 5% drawdown massimo
        max_orders_per_minute: int = 10
    ):
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.max_orders_per_minute = max_orders_per_minute
        
        self.daily_pnl = 0.0
        self.peak_balance = 0.0
        self.order_timestamps: List[datetime] = []
        self.logger = logging.getLogger("risk")
    
    def check_order(
        self,
        order: Order,
        current_balance: float,
        current_position: Optional[Position]
    ) -> bool:
        """
        Verifica se ordine rispetta limiti di rischio.
        Solleva RiskLimitExceeded se violato.
        """
        # Rate limiting
        self._check_rate_limit()
        
        # Position size
        order_value = order.quantity * (order.price or 0)
        if order_value > current_balance * self.max_position_size:
            raise RiskLimitExceeded(
                f"Order size {order_value} exceeds {self.max_position_size*100}% limit"
            )
        
        # Daily loss
        if self.daily_pnl < -current_balance * self.max_daily_loss:
            raise RiskLimitExceeded(
                f"Daily loss limit exceeded: {self.daily_pnl}"
            )
        
        # Drawdown
        if self.peak_balance > 0:
            current_drawdown = (self.peak_balance - current_balance) / self.peak_balance
            if current_drawdown > self.max_drawdown:
                raise RiskLimitExceeded(
                    f"Max drawdown exceeded: {current_drawdown*100:.1f}%"
                )
        
        return True
    
    def _check_rate_limit(self):
        """Verifica rate limit ordini."""
        now = datetime.now()
        # Rimuovi timestamp vecchi (> 1 minuto)
        self.order_timestamps = [
            ts for ts in self.order_timestamps
            if (now - ts).total_seconds() < 60
        ]
        
        if len(self.order_timestamps) >= self.max_orders_per_minute:
            raise RiskLimitExceeded("Order rate limit exceeded")
        
        self.order_timestamps.append(now)
    
    def update_pnl(self, pnl: float):
        """Aggiorna PnL giornaliero."""
        self.daily_pnl += pnl
        self.logger.info(f"Daily PnL updated: {self.daily_pnl}")
    
    def update_balance(self, balance: float):
        """Aggiorna peak balance per drawdown."""
        if balance > self.peak_balance:
            self.peak_balance = balance
    
    def reset_daily(self):
        """Reset contatori giornalieri."""
        self.daily_pnl = 0.0
        self.logger.info("Daily risk counters reset")


# ══════════════════════════════════════════════════════════════════════════════
#                    PART 7: COMPLETE BOT SKELETON
# ══════════════════════════════════════════════════════════════════════════════

class TradingBot:
    """
    Bot completo che integra tutti i componenti.
    Questo è lo scheletro - aggiungerai logica nel tuo percorso.
    """
    
    def __init__(
        self,
        exchange: ExchangeInterface,
        strategy: Strategy,
        risk_manager: RiskManager,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m"
    ):
        self.exchange = exchange
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.execution = ExecutionEngine(exchange)
        
        self.symbol = symbol
        self.timeframe = timeframe
        self.running = False
        self.current_position: Optional[Position] = None
        
        self.logger = logging.getLogger("bot")
    
    def start(self):
        """Avvia il bot."""
        self.running = True
        self.logger.info(f"Bot started on {self.symbol}")
        
        while self.running:
            try:
                self._tick()
                time.sleep(1)  # In produzione: event-driven
            except KeyboardInterrupt:
                self.stop()
            except Exception as e:
                self.logger.error(f"Error in tick: {e}")
    
    def stop(self):
        """Ferma il bot."""
        self.running = False
        self.logger.info("Bot stopped")
    
    def _tick(self):
        """Singolo ciclo del bot."""
        # 1. Fetch market data
        candles = self.exchange.get_ohlcv(
            self.symbol,
            self.timeframe,
            self.strategy.get_required_candles()
        )
        
        # 2. Analyze with strategy
        signal = self.strategy.analyze(candles)
        
        if signal:
            self.logger.info(f"Signal received: {signal}")
            
            # 3. Create order
            order = Order(
                id="",
                symbol=self.symbol,
                side=signal.side,
                order_type=OrderType.MARKET,
                quantity=0.001,  # Placeholder - calcola in base a risk
            )
            
            # 4. Risk check
            try:
                balance = self.exchange.get_balance("USDT")
                self.risk_manager.check_order(order, balance, self.current_position)
                
                # 5. Execute
                filled = self.execution.submit_order(order)
                self.logger.info(f"Order executed: {filled.id}")
                
            except RiskLimitExceeded as e:
                self.logger.warning(f"Risk limit: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#                    ESEMPIO USO
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Esempio di avvio bot."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Inizializza componenti
    exchange = MockExchange()
    strategy = ScalpingStrategy()
    risk_manager = RiskManager(
        max_position_size=0.05,
        max_daily_loss=0.01
    )
    
    # Crea bot
    bot = TradingBot(
        exchange=exchange,
        strategy=strategy,
        risk_manager=risk_manager,
        symbol="BTC/USDT",
        timeframe="1m"
    )
    
    # Test singolo tick (non avviare loop infinito in questo esempio)
    print("Testing single tick...")
    candles = exchange.get_ohlcv("BTC/USDT", "1m", 50)
    signal = strategy.analyze(candles)
    print(f"Signal: {signal}")
    
    print("\nBot skeleton ready!")
    print("Prossimi passi:")
    print("1. Completa le certificazioni Python Institute")
    print("2. Integra ccxt per exchange reale")
    print("3. Implementa la tua strategia Pine Script in Python")
    print("4. Aggiungi backtesting")
    print("5. Test su paper trading")
    print("6. Deploy in produzione")


if __name__ == "__main__":
    main()
