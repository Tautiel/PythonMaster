"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║              SPECIALIZZAZIONE TRADING - MODULE 1                             ║
║                    Trading Foundations & ccxt                                ║
║                                                                              ║
║                  Dal Corso Certificazioni al Trading Bot                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Questo modulo collega le competenze Python Institute alla pratica del trading.
Prerequisiti: PE1, PE2 completati (PCEP, PCAP level)

STRUTTURA:
├── Section 1.1: Concetti Base Trading
├── Section 1.2: ccxt - Exchange Connectivity  
├── Section 1.3: Order Types & Execution
├── Section 1.4: Rate Limiting & Error Handling
├── Section 1.5: Primo Bot Base
└── Esercizi Pratici

═══════════════════════════════════════════════════════════════════════════════
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import time

# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 1.1: CONCETTI BASE TRADING
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    1.1 TEORIA: FONDAMENTI TRADING                            │
└──────────────────────────────────────────────────────────────────────────────┘

TERMINOLOGIA ESSENZIALE:
────────────────────────
Symbol/Pair: BTC/USDT (Base/Quote)
  - Base currency: BTC (cosa compri/vendi)
  - Quote currency: USDT (con cosa paghi)

Bid: Prezzo più alto che qualcuno paga per comprare
Ask: Prezzo più basso che qualcuno accetta per vendere
Spread: Ask - Bid (costo implicito)

Order Book: Lista di ordini bid/ask
  - Depth: Quantità a ogni livello di prezzo

Candlestick (OHLCV):
  - Open: Prezzo apertura
  - High: Massimo
  - Low: Minimo  
  - Close: Prezzo chiusura
  - Volume: Quantità scambiata

Timeframe: 1m, 5m, 15m, 1h, 4h, 1d, 1w


TIPI DI ORDINE:
───────────────
Market Order: Esegui subito al miglior prezzo disponibile
Limit Order: Esegui solo se prezzo raggiunge il target
Stop Order: Diventa market quando prezzo raggiunge trigger
Stop-Limit: Diventa limit quando prezzo raggiunge trigger


SCALPING SPECIFICO:
───────────────────
- Timeframe bassi (1m-15m)
- Molte operazioni al giorno
- Profitti piccoli per trade
- Alta frequenza
- Spread e commissioni CRITICI
"""

# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES PER TRADING
# ═══════════════════════════════════════════════════════════════════════════

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class OHLCV:
    """Candlestick data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    @property
    def is_bullish(self) -> bool:
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        return self.close < self.open
    
    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)
    
    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)
    
    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low


@dataclass
class Order:
    """Order representation."""
    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    amount: float
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class Position:
    """Position representation."""
    symbol: str
    side: OrderSide
    entry_price: float
    amount: float
    current_price: float = 0.0
    
    @property
    def pnl(self) -> float:
        """Profit/Loss non realizzato."""
        if self.side == OrderSide.BUY:
            return (self.current_price - self.entry_price) * self.amount
        else:
            return (self.entry_price - self.current_price) * self.amount
    
    @property
    def pnl_percent(self) -> float:
        """PnL in percentuale."""
        return (self.pnl / (self.entry_price * self.amount)) * 100


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 1.2: CCXT BASICS
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    1.2 TEORIA: CCXT LIBRARY                                  │
└──────────────────────────────────────────────────────────────────────────────┘

ccxt: CryptoCurrency eXchange Trading Library
- Supporta 100+ exchange
- API unificata
- Gestione rate limiting
- Async support

pip install ccxt

STRUTTURA CCXT:
───────────────
exchange.load_markets()      # Carica info mercati
exchange.fetch_ticker()      # Prezzo corrente
exchange.fetch_ohlcv()       # Candlestick
exchange.fetch_order_book()  # Order book
exchange.create_order()      # Crea ordine
exchange.fetch_balance()     # Saldo
"""

# ═══════════════════════════════════════════════════════════════════════════
# EXCHANGE WRAPPER (Pattern per il tuo bot)
# ═══════════════════════════════════════════════════════════════════════════

CCXT_WRAPPER_CODE = '''
import ccxt
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExchangeClient:
    """
    Wrapper ccxt con error handling e logging.
    Questo pattern sarà la base del tuo trading bot.
    """
    
    def __init__(self, exchange_id: str, api_key: str = None, secret: str = None, sandbox: bool = True):
        """
        Inizializza client exchange.
        
        Args:
            exchange_id: 'binance', 'bybit', 'kraken', etc.
            api_key: API key (opzionale per dati pubblici)
            secret: API secret
            sandbox: True per testnet (SEMPRE iniziare qui!)
        """
        exchange_class = getattr(ccxt, exchange_id)
        
        self.exchange = exchange_class({
            'apiKey': api_key,
            'secret': secret,
            'sandbox': sandbox,
            'enableRateLimit': True,  # IMPORTANTE!
            'options': {
                'defaultType': 'spot',  # o 'future' per derivati
            }
        })
        
        self.exchange_id = exchange_id
        self._markets_loaded = False
        logger.info(f"Exchange {exchange_id} initialized (sandbox={sandbox})")
    
    def load_markets(self) -> Dict:
        """Carica informazioni sui mercati."""
        if not self._markets_loaded:
            self.exchange.load_markets()
            self._markets_loaded = True
            logger.info(f"Loaded {len(self.exchange.markets)} markets")
        return self.exchange.markets
    
    # ═══════════════════════════════════════════════════════════════════════
    # PUBLIC DATA (no auth required)
    # ═══════════════════════════════════════════════════════════════════════
    
    def get_ticker(self, symbol: str) -> Dict:
        """
        Ottieni prezzo corrente.
        
        Returns:
            {
                'symbol': 'BTC/USDT',
                'bid': 50000.0,
                'ask': 50001.0,
                'last': 50000.5,
                'volume': 1234.5,
                ...
            }
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            logger.debug(f"Ticker {symbol}: bid={ticker['bid']}, ask={ticker['ask']}")
            return ticker
        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching ticker: {e}")
            raise
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching ticker: {e}")
            raise
    
    def get_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> List:
        """
        Ottieni candlestick.
        
        Args:
            symbol: 'BTC/USDT'
            timeframe: '1m', '5m', '15m', '1h', '4h', '1d'
            limit: numero di candele
        
        Returns:
            [[timestamp, open, high, low, close, volume], ...]
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            logger.debug(f"Fetched {len(ohlcv)} candles for {symbol} {timeframe}")
            return ohlcv
        except Exception as e:
            logger.error(f"Error fetching OHLCV: {e}")
            raise
    
    def get_order_book(self, symbol: str, limit: int = 20) -> Dict:
        """
        Ottieni order book.
        
        Returns:
            {
                'bids': [[price, amount], ...],
                'asks': [[price, amount], ...],
            }
        """
        return self.exchange.fetch_order_book(symbol, limit)
    
    # ═══════════════════════════════════════════════════════════════════════
    # PRIVATE DATA (auth required)
    # ═══════════════════════════════════════════════════════════════════════
    
    def get_balance(self) -> Dict:
        """
        Ottieni saldo account.
        
        Returns:
            {
                'BTC': {'free': 1.0, 'used': 0.5, 'total': 1.5},
                'USDT': {'free': 10000.0, 'used': 0.0, 'total': 10000.0},
            }
        """
        balance = self.exchange.fetch_balance()
        # Filtra solo valute con saldo > 0
        return {k: v for k, v in balance.items() 
                if isinstance(v, dict) and v.get('total', 0) > 0}
    
    # ═══════════════════════════════════════════════════════════════════════
    # ORDER MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════
    
    def create_market_order(self, symbol: str, side: str, amount: float) -> Dict:
        """
        Crea ordine market.
        
        Args:
            symbol: 'BTC/USDT'
            side: 'buy' o 'sell'
            amount: quantità in base currency
        """
        logger.info(f"Creating MARKET {side} order: {amount} {symbol}")
        
        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=amount
            )
            logger.info(f"Order created: {order['id']}")
            return order
        except ccxt.InsufficientFunds as e:
            logger.error(f"Insufficient funds: {e}")
            raise
        except ccxt.InvalidOrder as e:
            logger.error(f"Invalid order: {e}")
            raise
    
    def create_limit_order(self, symbol: str, side: str, amount: float, price: float) -> Dict:
        """
        Crea ordine limit.
        """
        logger.info(f"Creating LIMIT {side} order: {amount} {symbol} @ {price}")
        
        order = self.exchange.create_order(
            symbol=symbol,
            type='limit',
            side=side,
            amount=amount,
            price=price
        )
        return order
    
    def cancel_order(self, order_id: str, symbol: str) -> Dict:
        """Cancella ordine."""
        logger.info(f"Cancelling order {order_id}")
        return self.exchange.cancel_order(order_id, symbol)
    
    def get_order(self, order_id: str, symbol: str) -> Dict:
        """Ottieni stato ordine."""
        return self.exchange.fetch_order(order_id, symbol)
    
    def get_open_orders(self, symbol: str = None) -> List:
        """Ottieni ordini aperti."""
        return self.exchange.fetch_open_orders(symbol)


# Esempio di uso:
if __name__ == "__main__":
    # Sempre iniziare con TESTNET!
    client = ExchangeClient('binance', sandbox=True)
    client.load_markets()
    
    ticker = client.get_ticker('BTC/USDT')
    print(f"BTC price: {ticker['last']}")
    
    ohlcv = client.get_ohlcv('BTC/USDT', '1h', 10)
    print(f"Last 10 candles: {len(ohlcv)}")
'''


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 1.3: RATE LIMITING
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    1.3 TEORIA: RATE LIMITING                                 │
└──────────────────────────────────────────────────────────────────────────────┘

RATE LIMITING:
──────────────
Gli exchange limitano le richieste per evitare abusi.
Superare i limiti = BAN temporaneo o permanente!

Binance: ~1200 requests/min (pesi diversi per endpoint)
Bybit: ~120 requests/min
Kraken: ~15-20 requests/sec

STRATEGIE:
1. enableRateLimit: True (ccxt gestisce automaticamente)
2. Caching dei dati
3. WebSocket per dati real-time (invece di polling)
"""

import functools
from datetime import datetime, timedelta

class RateLimiter:
    """Rate limiter per API calls."""
    
    def __init__(self, max_calls: int, period: float):
        """
        Args:
            max_calls: Numero massimo di chiamate
            period: Periodo in secondi
        """
        self.max_calls = max_calls
        self.period = period
        self.calls = []
    
    def wait_if_needed(self):
        """Attende se necessario per rispettare il rate limit."""
        now = datetime.now()
        
        # Rimuovi chiamate vecchie
        self.calls = [c for c in self.calls 
                      if now - c < timedelta(seconds=self.period)]
        
        if len(self.calls) >= self.max_calls:
            oldest = min(self.calls)
            sleep_time = (oldest + timedelta(seconds=self.period) - now).total_seconds()
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self.calls.append(now)


def rate_limited(limiter: RateLimiter):
    """Decorator per rate limiting."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            limiter.wait_if_needed()
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 1.4: PRIMO BOT BASE
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    1.4 PRIMO BOT: STRUTTURA BASE                             │
└──────────────────────────────────────────────────────────────────────────────┘
"""

SIMPLE_BOT_STRUCTURE = '''
"""
Struttura base di un trading bot.
Questo è lo scheletro su cui costruirai il tuo bot di scalping.
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TradingBot')


class TradingBot(ABC):
    """
    Base class per trading bot.
    Implementa il loop principale e gestione errori.
    """
    
    def __init__(self, exchange_client, symbol: str, interval: int = 60):
        """
        Args:
            exchange_client: Client exchange (il tuo ExchangeClient)
            symbol: Trading pair (es: 'BTC/USDT')
            interval: Secondi tra ogni ciclo
        """
        self.exchange = exchange_client
        self.symbol = symbol
        self.interval = interval
        self.running = False
        self.position = None
    
    @abstractmethod
    def analyze(self) -> Optional[str]:
        """
        Analizza il mercato e restituisce segnale.
        
        Returns:
            'buy', 'sell', o None
        """
        pass
    
    @abstractmethod
    def execute_signal(self, signal: str):
        """Esegue il segnale di trading."""
        pass
    
    def run(self):
        """Main loop del bot."""
        self.running = True
        logger.info(f"Bot started for {self.symbol}")
        
        while self.running:
            try:
                # 1. Analizza mercato
                signal = self.analyze()
                
                # 2. Esegui se c'è segnale
                if signal:
                    logger.info(f"Signal detected: {signal}")
                    self.execute_signal(signal)
                
                # 3. Attendi prossimo ciclo
                time.sleep(self.interval)
                
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                self.running = False
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(self.interval)
    
    def stop(self):
        """Ferma il bot."""
        self.running = False
        logger.info("Bot stopping...")


class SimpleMovingAverageBot(TradingBot):
    """
    Esempio: Bot con strategia SMA crossover.
    NON usare in produzione - solo esempio didattico!
    """
    
    def __init__(self, exchange_client, symbol: str, 
                 fast_period: int = 10, slow_period: int = 20):
        super().__init__(exchange_client, symbol)
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def analyze(self) -> Optional[str]:
        """Analizza con SMA crossover."""
        # Fetch candles
        candles = self.exchange.get_ohlcv(
            self.symbol, '1h', 
            limit=self.slow_period + 5
        )
        
        closes = [c[4] for c in candles]  # Close prices
        
        # Calcola SMA
        fast_sma = sum(closes[-self.fast_period:]) / self.fast_period
        slow_sma = sum(closes[-self.slow_period:]) / self.slow_period
        
        # Segnali
        if fast_sma > slow_sma and not self.position:
            return 'buy'
        elif fast_sma < slow_sma and self.position:
            return 'sell'
        
        return None
    
    def execute_signal(self, signal: str):
        """Esegue ordine."""
        if signal == 'buy':
            # In un bot reale, calcola size basato su risk management
            amount = 0.001  # Esempio: quantità fissa
            order = self.exchange.create_market_order(
                self.symbol, 'buy', amount
            )
            self.position = order
            logger.info(f"Opened position: {order}")
        
        elif signal == 'sell' and self.position:
            order = self.exchange.create_market_order(
                self.symbol, 'sell', self.position['amount']
            )
            self.position = None
            logger.info(f"Closed position: {order}")


# Uso:
# client = ExchangeClient('binance', sandbox=True)
# bot = SimpleMovingAverageBot(client, 'BTC/USDT')
# bot.run()
'''


# ══════════════════════════════════════════════════════════════════════════════
#                    ESERCIZI PRATICI
# ══════════════════════════════════════════════════════════════════════════════

EXERCISES = """
══════════════════════════════════════════════════════════════════════════════
                    ESERCIZI PRATICI
══════════════════════════════════════════════════════════════════════════════

ESERCIZIO 1: Data Classes
─────────────────────────
Crea una dataclass `Trade` con:
- id, symbol, side, price, amount, timestamp, fee
- Proprietà `total_cost` che calcola price * amount + fee
- Proprietà `is_profitable(current_price)` per trade chiuso

ESERCIZIO 2: OHLCV Analysis
───────────────────────────
Data una lista di OHLCV:
- Calcola SMA(10) e SMA(20)
- Identifica candele "doji" (body < 10% del range)
- Trova il massimo e minimo degli ultimi N periodi

ESERCIZIO 3: Order Book
───────────────────────
Dato un order book:
{'bids': [[100, 1.5], [99, 2.0], [98, 3.0]],
 'asks': [[101, 1.0], [102, 1.5], [103, 2.0]]}

Calcola:
- Spread
- Midprice
- Total bid/ask volume
- Slippage per un ordine di X amount

ESERCIZIO 4: Risk Management
────────────────────────────
Crea una classe `RiskManager` che:
- Calcola position size basato su % rischio del capitale
- Verifica che il trade rispetti max drawdown
- Calcola stop loss e take profit prices

ESERCIZIO 5: Exchange Wrapper
─────────────────────────────
Estendi ExchangeClient con:
- Metodo per calcolare prezzo medio di acquisto
- Cache per ticker con TTL (time-to-live)
- Logging di tutte le operazioni su file


══════════════════════════════════════════════════════════════════════════════
"""


if __name__ == "__main__":
    print("=" * 70)
    print("SPECIALIZZAZIONE TRADING - Module 1: Foundations")
    print("=" * 70)
    print("""
    Contenuti:
    print(CCXT_WRAPPER_CODE)      # Codice wrapper exchange
    print(SIMPLE_BOT_STRUCTURE)   # Struttura bot base
    print(EXERCISES)              # Esercizi pratici
    """)
