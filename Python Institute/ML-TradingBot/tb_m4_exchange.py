#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    TRADING BOT - MODULE 4                                    ║
║                    EXCHANGE CONNECTION (CCXT)                                 ║
║                    TB-M4: 20% del percorso Trading                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

SYLLABUS TB MODULE 4:
├── TB 4.1 - CCXT Library Overview
├── TB 4.2 - Market Data Fetching
├── TB 4.3 - Order Management
├── TB 4.4 - Account & Position Management
├── TB 4.5 - Error Handling & Rate Limits
└── TB 4.6 - Paper Trading vs Live Trading

PREREQUISITI: TB-M1 to M3, PCPP1 (network)
⚠️ IMPORTANTE: Inizia SEMPRE con TESTNET!
"""

import os

print("=" * 70)
print("TB 4.1 - CCXT LIBRARY OVERVIEW")
print("=" * 70)

print("""
📋 CCXT: CryptoCurrency eXchange Trading Library

pip install ccxt python-dotenv

ESEMPIO BASE:
import ccxt

exchange = ccxt.binance({
    'apiKey': 'YOUR_API_KEY',
    'secret': 'YOUR_SECRET',
    'sandbox': True,  # TESTNET!
})
""")

print("\n" + "=" * 70)
print("TB 4.2 - MARKET DATA FETCHING")
print("=" * 70)

print("""
🔹 TICKER:
ticker = exchange.fetch_ticker('BTC/USDT')

🔹 OHLCV:
ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=100)
df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])

🔹 ORDER BOOK:
orderbook = exchange.fetch_order_book('BTC/USDT', limit=10)
""")

print("\n" + "=" * 70)
print("TB 4.3 - ORDER MANAGEMENT")
print("=" * 70)

print("""
🔹 MARKET ORDER:
order = exchange.create_market_buy_order('BTC/USDT', 0.001)

🔹 LIMIT ORDER:
order = exchange.create_limit_buy_order('BTC/USDT', 0.001, 40000)

🔹 CANCEL ORDER:
exchange.cancel_order(order_id, 'BTC/USDT')
""")

print("\n" + "=" * 70)
print("TB 4.4 - ACCOUNT MANAGEMENT")
print("=" * 70)

print("""
balance = exchange.fetch_balance()
print(balance['USDT']['free'])  # Disponibile
print(balance['BTC']['total'])  # Totale
""")

print("\n" + "=" * 70)
print("TB 4.5 - ERROR HANDLING")
print("=" * 70)

print("""
try:
    order = exchange.create_market_buy_order('BTC/USDT', 0.001)
except ccxt.InsufficientFunds as e:
    print(f"Fondi insufficienti: {e}")
except ccxt.NetworkError as e:
    print(f"Errore rete: {e}")
""")

print("\n" + "=" * 70)
print("TB 4.6 - PAPER VS LIVE")
print("=" * 70)

print("""
⚠️ WORKFLOW:
1. BACKTEST → Valida logica
2. PAPER TRADING (testnet) → Valida esecuzione
3. LIVE con piccole size → Monitora attentamente
""")

print("\n" + "=" * 70)
print("QUIZ - TB MODULE 4")
print("=" * 70)

print("""
Q1. sandbox=True significa? → TESTNET
Q2. fetch_ohlcv restituisce? → Candele storiche
Q3. API keys dove? → In .env (mai committare)
Q4. Prima di live? → Backtest + paper trading
Q5. Stop loss? → Sempre attivo
""")

print("\n" + "=" * 70)
print("TB MODULE 4 COMPLETATO!")
print("=" * 70)
