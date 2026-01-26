"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    CCXT INTEGRATION - TRADING BOT REALE                      ║
║                                                                              ║
║                 Connessione a Exchange Crypto (Binance, Bybit)               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

PREREQUISITI:
pip install ccxt python-dotenv pandas

SICUREZZA:
- MAI committare API keys su git
- Usa SEMPRE .env per le credenziali
- Inizia SEMPRE con testnet/paper trading

═══════════════════════════════════════════════════════════════════════════════
"""

import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ccxt_trading')


# ══════════════════════════════════════════════════════════════════════════════
#                    CONFIGURAZIONE ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════

ENV_TEMPLATE = """
# .env file - NON COMMITTARE MAI!
# Copia questo in .env e inserisci le tue chiavi

# Binance Testnet (per testing)
BINANCE_TESTNET_API_KEY=your_testnet_api_key
BINANCE_TESTNET_SECRET=your_testnet_secret

# Binance Production (quando sei pronto)
BINANCE_API_KEY=your_api_key
BINANCE_SECRET=your_secret

# Bybit Testnet
BYBIT_TESTNET_API_KEY=your_testnet_api_key
BYBIT_TESTNET_SECRET=your_testnet_secret

# Trading Parameters
DEFAULT_SYMBOL=BTC/USDT
DEFAULT_TIMEFRAME=1m
MAX_POSITION_SIZE=0.05
RISK_PER_TRADE=0.01
"""


def load_config():
    """Carica configurazione da .env"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        logger.warning("python-dotenv non installato. Usa variabili d'ambiente.")
    
    return {
        'binance_testnet': {
            'api_key': os.getenv('BINANCE_TESTNET_API_KEY'),
            'secret': os.getenv('BINANCE_TESTNET_SECRET')
        },
        'binance_prod': {
            'api_key': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_SECRET')
        },
        'bybit_testnet': {
            'api_key': os.getenv('BYBIT_TESTNET_API_KEY'),
            'secret': os.getenv('BYBIT_TESTNET_SECRET')
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
#                    DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

class Side(Enum):
    BUY = 'buy'
    SELL = 'sell'


class OrderType(Enum):
    MARKET = 'market'
    LIMIT = 'limit'


class OrderStatus(Enum):
    OPEN = 'open'
    CLOSED = 'closed'
    CANCELED = 'canceled'


@dataclass
class OHLCV:
    """Candlestick data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    @classmethod
    def from_ccxt(cls, data: list) -> 'OHLCV':
        """Crea da formato ccxt [timestamp, o, h, l, c, v]"""
        return cls(
            timestamp=datetime.fromtimestamp(data[0] / 1000),
            open=data[1],
            high=data[2],
            low=data[3],
            close=data[4],
            volume=data[5]
        )


@dataclass
class Ticker:
    """Market ticker."""
    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    timestamp: datetime


@dataclass
class Balance:
    """Account balance per asset."""
    asset: str
    free: float
    used: float
    total: float


@dataclass
class Order:
    """Order data."""
    id: str
    symbol: str
    side: Side
    order_type: OrderType
    amount: float
    price: Optional[float]
    filled: float
    remaining: float
    status: OrderStatus
    timestamp: datetime


# ══════════════════════════════════════════════════════════════════════════════
#                    EXCHANGE WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

class ExchangeWrapper:
    """
    Wrapper ccxt per trading.
    Supporta Binance e Bybit (estendibile).
    """
    
    SUPPORTED_EXCHANGES = ['binance', 'bybit']
    
    def __init__(
        self,
        exchange_id: str,
        api_key: str,
        secret: str,
        testnet: bool = True
    ):
        """
        Inizializza connessione exchange.
        
        Args:
            exchange_id: 'binance' o 'bybit'
            api_key: API key
            secret: API secret
            testnet: True per testnet (RACCOMANDATO per iniziare!)
        """
        try:
            import ccxt
        except ImportError:
            raise ImportError("Installa ccxt: pip install ccxt")
        
        if exchange_id not in self.SUPPORTED_EXCHANGES:
            raise ValueError(f"Exchange {exchange_id} non supportato")
        
        self.exchange_id = exchange_id
        self.testnet = testnet
        
        # Crea istanza exchange
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'  # o 'future' per futures
            }
        })
        
        # Configura testnet
        if testnet:
            self._setup_testnet()
        
        logger.info(f"Exchange {exchange_id} inizializzato (testnet={testnet})")
    
    def _setup_testnet(self):
        """Configura URL testnet."""
        if self.exchange_id == 'binance':
            self.exchange.set_sandbox_mode(True)
        elif self.exchange_id == 'bybit':
            self.exchange.set_sandbox_mode(True)
    
    # ═══════════════════════════════════════════════════════════════════════
    # MARKET DATA
    # ═══════════════════════════════════════════════════════════════════════
    
    def get_ticker(self, symbol: str) -> Ticker:
        """Ottieni ticker corrente."""
        data = self.exchange.fetch_ticker(symbol)
        return Ticker(
            symbol=data['symbol'],
            bid=data['bid'],
            ask=data['ask'],
            last=data['last'],
            volume=data['baseVolume'],
            timestamp=datetime.fromtimestamp(data['timestamp'] / 1000)
        )
    
    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1m',
        limit: int = 100
    ) -> List[OHLCV]:
        """
        Ottieni candlestick data.
        
        Args:
            symbol: es. 'BTC/USDT'
            timeframe: '1m', '5m', '15m', '1h', '4h', '1d'
            limit: numero di candele (max 1000 per la maggior parte degli exchange)
        """
        data = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        return [OHLCV.from_ccxt(candle) for candle in data]
    
    def get_orderbook(self, symbol: str, limit: int = 10) -> Dict:
        """Ottieni order book."""
        return self.exchange.fetch_order_book(symbol, limit=limit)
    
    # ═══════════════════════════════════════════════════════════════════════
    # ACCOUNT
    # ═══════════════════════════════════════════════════════════════════════
    
    def get_balance(self, asset: str = None) -> Dict[str, Balance]:
        """
        Ottieni balance account.
        
        Args:
            asset: Se specificato, restituisce solo quell'asset
        """
        data = self.exchange.fetch_balance()
        
        balances = {}
        for symbol, balance in data['total'].items():
            if balance > 0 or symbol in ['USDT', 'BTC', 'ETH']:
                balances[symbol] = Balance(
                    asset=symbol,
                    free=data['free'].get(symbol, 0),
                    used=data['used'].get(symbol, 0),
                    total=balance
                )
        
        if asset:
            return balances.get(asset)
        return balances
    
    # ═══════════════════════════════════════════════════════════════════════
    # ORDERS
    # ═══════════════════════════════════════════════════════════════════════
    
    def create_market_order(
        self,
        symbol: str,
        side: Side,
        amount: float
    ) -> Order:
        """
        Crea ordine market.
        
        Args:
            symbol: es. 'BTC/USDT'
            side: Side.BUY o Side.SELL
            amount: quantità in base currency (es. BTC)
        """
        logger.info(f"Creating MARKET {side.value} order: {amount} {symbol}")
        
        data = self.exchange.create_order(
            symbol=symbol,
            type='market',
            side=side.value,
            amount=amount
        )
        
        return self._parse_order(data)
    
    def create_limit_order(
        self,
        symbol: str,
        side: Side,
        amount: float,
        price: float
    ) -> Order:
        """
        Crea ordine limit.
        
        Args:
            symbol: es. 'BTC/USDT'
            side: Side.BUY o Side.SELL
            amount: quantità
            price: prezzo limite
        """
        logger.info(f"Creating LIMIT {side.value} order: {amount} {symbol} @ {price}")
        
        data = self.exchange.create_order(
            symbol=symbol,
            type='limit',
            side=side.value,
            amount=amount,
            price=price
        )
        
        return self._parse_order(data)
    
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancella ordine."""
        try:
            self.exchange.cancel_order(order_id, symbol)
            logger.info(f"Order {order_id} cancelled")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def get_order(self, order_id: str, symbol: str) -> Order:
        """Ottieni dettagli ordine."""
        data = self.exchange.fetch_order(order_id, symbol)
        return self._parse_order(data)
    
    def get_open_orders(self, symbol: str = None) -> List[Order]:
        """Ottieni ordini aperti."""
        data = self.exchange.fetch_open_orders(symbol)
        return [self._parse_order(o) for o in data]
    
    def _parse_order(self, data: Dict) -> Order:
        """Converte risposta ccxt in Order."""
        return Order(
            id=data['id'],
            symbol=data['symbol'],
            side=Side(data['side']),
            order_type=OrderType(data['type']),
            amount=data['amount'],
            price=data.get('price'),
            filled=data['filled'],
            remaining=data['remaining'],
            status=OrderStatus(data['status']),
            timestamp=datetime.fromtimestamp(data['timestamp'] / 1000)
        )
    
    # ═══════════════════════════════════════════════════════════════════════
    # UTILITIES
    # ═══════════════════════════════════════════════════════════════════════
    
    def get_min_order_amount(self, symbol: str) -> float:
        """Ottieni quantità minima per ordine."""
        markets = self.exchange.load_markets()
        market = markets.get(symbol)
        if market:
            return market['limits']['amount']['min']
        return 0.0
    
    def get_price_precision(self, symbol: str) -> int:
        """Ottieni precisione prezzo (decimali)."""
        markets = self.exchange.load_markets()
        market = markets.get(symbol)
        if market:
            return market['precision']['price']
        return 8
    
    def round_price(self, symbol: str, price: float) -> float:
        """Arrotonda prezzo alla precisione corretta."""
        precision = self.get_price_precision(symbol)
        return round(price, precision)


# ══════════════════════════════════════════════════════════════════════════════
#                    TRADING BOT CON CCXT
# ══════════════════════════════════════════════════════════════════════════════

class ScalpingBot:
    """
    Bot di scalping completo con ccxt.
    """
    
    def __init__(
        self,
        exchange: ExchangeWrapper,
        symbol: str = 'BTC/USDT',
        timeframe: str = '1m',
        risk_per_trade: float = 0.01,
        max_position_pct: float = 0.1
    ):
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self.risk_per_trade = risk_per_trade
        self.max_position_pct = max_position_pct
        
        self.running = False
        self.current_position = None
        self.trades_today = []
        
        logger.info(f"Bot initialized: {symbol} @ {timeframe}")
    
    def calculate_position_size(self) -> float:
        """Calcola size posizione basata su risk management."""
        # Ottieni balance
        balance = self.exchange.get_balance('USDT')
        if not balance:
            return 0.0
        
        usdt_available = balance.free
        
        # Max position size
        max_position = usdt_available * self.max_position_pct
        
        # Risk-based size
        risk_amount = usdt_available * self.risk_per_trade
        
        # Usa il minore
        position_usdt = min(max_position, risk_amount * 10)  # 10:1 ratio
        
        # Converti in BTC
        ticker = self.exchange.get_ticker(self.symbol)
        position_size = position_usdt / ticker.last
        
        # Verifica min order
        min_amount = self.exchange.get_min_order_amount(self.symbol)
        if position_size < min_amount:
            logger.warning(f"Position size {position_size} < min {min_amount}")
            return 0.0
        
        return round(position_size, 6)
    
    def check_signal(self) -> Optional[Side]:
        """
        Analizza mercato e genera segnale.
        Implementa qui la tua strategia da Pine Script!
        """
        # Ottieni candele
        candles = self.exchange.get_ohlcv(self.symbol, self.timeframe, limit=50)
        
        if len(candles) < 20:
            return None
        
        # Esempio semplice: EMA crossover
        closes = [c.close for c in candles]
        
        ema_fast = self._ema(closes, 9)
        ema_slow = self._ema(closes, 21)
        
        # Segnale
        if ema_fast[-1] > ema_slow[-1] and ema_fast[-2] <= ema_slow[-2]:
            return Side.BUY
        elif ema_fast[-1] < ema_slow[-1] and ema_fast[-2] >= ema_slow[-2]:
            return Side.SELL
        
        return None
    
    def _ema(self, data: List[float], period: int) -> List[float]:
        """Calcola EMA."""
        if len(data) < period:
            return data
        
        multiplier = 2 / (period + 1)
        ema = [sum(data[:period]) / period]
        
        for price in data[period:]:
            ema.append((price - ema[-1]) * multiplier + ema[-1])
        
        return ema
    
    def execute_trade(self, side: Side):
        """Esegui trade."""
        position_size = self.calculate_position_size()
        
        if position_size <= 0:
            logger.warning("Cannot execute: position size is 0")
            return
        
        try:
            order = self.exchange.create_market_order(
                self.symbol,
                side,
                position_size
            )
            
            logger.info(f"Trade executed: {order}")
            self.trades_today.append(order)
            
            if side == Side.BUY:
                self.current_position = {
                    'side': side,
                    'entry_price': order.price or self.exchange.get_ticker(self.symbol).last,
                    'amount': order.filled
                }
            else:
                self.current_position = None
                
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
    
    def run(self, interval_seconds: int = 60):
        """
        Avvia bot loop.
        
        Args:
            interval_seconds: secondi tra ogni check
        """
        self.running = True
        logger.info(f"Bot started. Checking every {interval_seconds}s")
        
        while self.running:
            try:
                signal = self.check_signal()
                
                if signal:
                    logger.info(f"Signal detected: {signal}")
                    
                    # Se non abbiamo posizione e signal è BUY
                    if self.current_position is None and signal == Side.BUY:
                        self.execute_trade(signal)
                    
                    # Se abbiamo posizione long e signal è SELL
                    elif self.current_position and signal == Side.SELL:
                        self.execute_trade(signal)
                
                time.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                self.stop()
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(interval_seconds)
    
    def stop(self):
        """Ferma il bot."""
        self.running = False
        logger.info("Bot stopped")
    
    def get_stats(self) -> Dict:
        """Statistiche trading."""
        if not self.trades_today:
            return {'trades': 0, 'pnl': 0}
        
        return {
            'trades': len(self.trades_today),
            'pnl': 'calculate based on trades'
        }


# ══════════════════════════════════════════════════════════════════════════════
#                    ESEMPIO DI USO
# ══════════════════════════════════════════════════════════════════════════════

def example_usage():
    """
    Esempio completo di utilizzo.
    NOTA: Usa SEMPRE testnet per testare!
    """
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                         ESEMPIO CCXT BOT                                 ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    
    STEP 1: Crea file .env con le tue API keys (testnet!)
    
    STEP 2: Esegui questo codice:
    
    ```python
    from ccxt_trading import ExchangeWrapper, ScalpingBot, Side, load_config
    
    # Carica config
    config = load_config()
    
    # Crea exchange wrapper (TESTNET!)
    exchange = ExchangeWrapper(
        exchange_id='binance',
        api_key=config['binance_testnet']['api_key'],
        secret=config['binance_testnet']['secret'],
        testnet=True  # IMPORTANTE!
    )
    
    # Test connessione
    ticker = exchange.get_ticker('BTC/USDT')
    print(f"BTC/USDT: {ticker.last}")
    
    # Test balance
    balance = exchange.get_balance()
    print(f"Balance: {balance}")
    
    # Crea bot
    bot = ScalpingBot(
        exchange=exchange,
        symbol='BTC/USDT',
        timeframe='1m',
        risk_per_trade=0.01
    )
    
    # Avvia (Ctrl+C per fermare)
    # bot.run(interval_seconds=60)
    ```
    
    STEP 3: Quando sei sicuro, passa a produzione (CON CAUTELA!)
    """)


def demo_without_api():
    """
    Demo senza API keys - solo per vedere la struttura.
    """
    print("=" * 70)
    print("DEMO CCXT TRADING BOT (senza connessione)")
    print("=" * 70)
    
    print("""
    Struttura del bot:
    
    1. ExchangeWrapper
       - Connessione a Binance/Bybit
       - get_ticker(), get_ohlcv(), get_balance()
       - create_market_order(), create_limit_order()
    
    2. ScalpingBot
       - calculate_position_size() - Risk management
       - check_signal() - Strategia (PERSONALIZZA QUESTA!)
       - execute_trade() - Esecuzione ordini
       - run() - Main loop
    
    3. Data Structures
       - OHLCV, Ticker, Balance, Order
       - Side, OrderType, OrderStatus
    
    Per usare con vero exchange:
    1. pip install ccxt python-dotenv
    2. Crea account testnet Binance
    3. Genera API keys testnet
    4. Crea .env con le chiavi
    5. Testa su testnet per ALMENO 1 settimana
    6. Solo dopo considera denaro reale
    """)


if __name__ == "__main__":
    demo_without_api()
    example_usage()
