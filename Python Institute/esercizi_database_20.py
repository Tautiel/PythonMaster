"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ESERCIZI DATABASE PYTHON - 20 ESERCIZI                     ║
║                  SQLite, SQL, PostgreSQL, SQLAlchemy ORM                      ║
║                                                                               ║
║  Livello 1 (1-5):   SQL Base - CRUD, Query semplici                          ║
║  Livello 2 (6-10):  SQL Intermedio - JOIN, GROUP BY, Aggregazioni            ║
║  Livello 3 (11-15): SQLAlchemy ORM - Modelli, Relazioni                      ║
║  Livello 4 (16-20): Progetti Trading - Sistemi completi                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sqlite3
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Optional, Tuple
import os

print("=" * 70)
print("ESERCIZI DATABASE PYTHON")
print("20 Esercizi Progressivi")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def create_test_db():
    """Crea database di test con dati di esempio."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.executescript("""
        -- Tabella portfolio
        CREATE TABLE portfolios (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            initial_balance REAL DEFAULT 10000,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Tabella trades
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY,
            portfolio_id INTEGER,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL CHECK(side IN ('BUY', 'SELL')),
            quantity REAL NOT NULL,
            price REAL NOT NULL,
            fee REAL DEFAULT 0,
            strategy TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
        );
        
        -- Tabella positions
        CREATE TABLE positions (
            id INTEGER PRIMARY KEY,
            portfolio_id INTEGER,
            symbol TEXT NOT NULL,
            quantity REAL DEFAULT 0,
            avg_price REAL DEFAULT 0,
            realized_pnl REAL DEFAULT 0,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (portfolio_id) REFERENCES portfolios(id),
            UNIQUE(portfolio_id, symbol)
        );
        
        -- Tabella candele
        CREATE TABLE candles (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            UNIQUE(symbol, timeframe, timestamp)
        );
        
        -- Dati di esempio
        INSERT INTO portfolios (id, name, initial_balance) VALUES 
            (1, 'Trading Bot', 10000),
            (2, 'Long Term', 50000);
        
        INSERT INTO trades (portfolio_id, symbol, side, quantity, price, fee, strategy, created_at) VALUES 
            (1, 'BTC', 'BUY', 0.1, 42000, 4.2, 'RSI', '2024-01-05 10:00:00'),
            (1, 'BTC', 'BUY', 0.05, 44000, 2.2, 'SMA', '2024-01-10 14:00:00'),
            (1, 'BTC', 'SELL', 0.08, 46000, 3.68, 'RSI', '2024-01-15 09:00:00'),
            (1, 'ETH', 'BUY', 1.0, 2500, 2.5, 'RSI', '2024-01-07 11:00:00'),
            (1, 'ETH', 'BUY', 0.5, 2600, 1.3, 'Breakout', '2024-01-12 16:00:00'),
            (1, 'ETH', 'SELL', 0.8, 2800, 2.24, 'RSI', '2024-01-18 10:00:00'),
            (1, 'SOL', 'BUY', 50, 90, 4.5, 'SMA', '2024-01-08 13:00:00'),
            (1, 'SOL', 'SELL', 30, 110, 3.3, 'SMA', '2024-01-20 15:00:00'),
            (2, 'BTC', 'BUY', 0.5, 43000, 21.5, 'DCA', '2024-01-02 08:00:00'),
            (2, 'ETH', 'BUY', 5.0, 2400, 12.0, 'DCA', '2024-01-03 09:00:00');
        
        INSERT INTO positions (portfolio_id, symbol, quantity, avg_price, realized_pnl) VALUES 
            (1, 'BTC', 0.07, 42666.67, 320),
            (1, 'ETH', 0.7, 2533.33, 213.36),
            (1, 'SOL', 20, 90, 600),
            (2, 'BTC', 0.5, 43000, 0),
            (2, 'ETH', 5.0, 2400, 0);
    """)
    
    # Aggiungi candele di esempio
    base_time = int(datetime(2024, 1, 1).timestamp())
    candles = []
    import random
    random.seed(42)
    
    for symbol, base_price in [('BTC', 42000), ('ETH', 2500), ('SOL', 90)]:
        price = base_price
        for i in range(100):
            change = random.uniform(-0.02, 0.02) * price
            open_p = price
            close_p = price + change
            high_p = max(open_p, close_p) + abs(change) * 0.5
            low_p = min(open_p, close_p) - abs(change) * 0.5
            volume = random.uniform(1000000, 5000000)
            
            candles.append((symbol, '1h', base_time + i * 3600, 
                           open_p, high_p, low_p, close_p, volume))
            price = close_p
    
    cursor.executemany("""
        INSERT INTO candles (symbol, timeframe, timestamp, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, candles)
    conn.commit()
    
    return conn


# ══════════════════════════════════════════════════════════════════════════════
# LIVELLO 1: SQL BASE (Esercizi 1-5)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("📚 LIVELLO 1: SQL BASE")
print("═" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# ESERCIZIO 1: Connessione e SELECT Base
# ─────────────────────────────────────────────────────────────────────────────

print("\n📝 ESERCIZIO 1: Connessione e SELECT Base")
print("-" * 50)
print("""
OBIETTIVO: Impara a connetterti a SQLite e fare query SELECT base.

TASKS:
1. Crea una connessione a un database SQLite in memoria
2. Crea una tabella 'crypto' con colonne: id, symbol, name, price
3. Inserisci 3 record (BTC, ETH, SOL con prezzi a tua scelta)
4. Fai una SELECT per recuperare tutti i record
5. Fai una SELECT solo per symbol e price
6. Usa row_factory per accedere ai risultati per nome colonna

HINT: 
- sqlite3.connect(":memory:") per database in memoria
- conn.row_factory = sqlite3.Row per accesso per nome
""")

def exercise_1_solution():
    """Soluzione Esercizio 1"""
    # 1. Connessione
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    
    # 2. Crea tabella
    cursor.execute("""
        CREATE TABLE crypto (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            name TEXT NOT NULL,
            price REAL NOT NULL
        )
    """)
    
    # 3. Inserisci dati
    cryptos = [
        ('BTC', 'Bitcoin', 45000.00),
        ('ETH', 'Ethereum', 3000.00),
        ('SOL', 'Solana', 100.00)
    ]
    cursor.executemany(
        "INSERT INTO crypto (symbol, name, price) VALUES (?, ?, ?)",
        cryptos
    )
    conn.commit()
    
    # 4. Select tutti
    cursor.execute("SELECT * FROM crypto")
    print("Tutti i record:")
    for row in cursor.fetchall():
        print(f"   {row}")
    
    # 5. Select colonne specifiche
    cursor.execute("SELECT symbol, price FROM crypto")
    print("\nSolo symbol e price:")
    for row in cursor.fetchall():
        print(f"   {row[0]}: ${row[1]}")
    
    # 6. Con row_factory
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM crypto")
    print("\nCon row_factory:")
    for row in cursor.fetchall():
        print(f"   {row['symbol']} ({row['name']}): ${row['price']}")
    
    conn.close()
    return True

# exercise_1_solution()


# ─────────────────────────────────────────────────────────────────────────────
# ESERCIZIO 2: INSERT con Parametri
# ─────────────────────────────────────────────────────────────────────────────

print("\n📝 ESERCIZIO 2: INSERT con Parametri")
print("-" * 50)
print("""
OBIETTIVO: Impara a inserire dati in modo sicuro usando parametri.

TASKS:
1. Crea funzione `insert_trade(conn, symbol, side, quantity, price)`
2. La funzione deve usare parametri (?) per evitare SQL injection
3. Deve ritornare l'ID del trade inserito
4. Crea funzione `insert_many_trades(conn, trades_list)`
5. Usa executemany per inserimento batch

HINT:
- cursor.lastrowid per ottenere l'ID
- executemany per inserimenti multipli
- SEMPRE usare (?, ?, ?) MAI f-strings!
""")

def exercise_2_solution():
    """Soluzione Esercizio 2"""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    
    # Setup tabella
    cursor.execute("""
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity REAL NOT NULL,
            price REAL NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    def insert_trade(conn, symbol: str, side: str, quantity: float, price: float) -> int:
        """Inserisce un trade e ritorna l'ID."""
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO trades (symbol, side, quantity, price) VALUES (?, ?, ?, ?)",
            (symbol, side, quantity, price)
        )
        conn.commit()
        return cursor.lastrowid
    
    def insert_many_trades(conn, trades: list) -> int:
        """Inserisce più trade e ritorna il numero di righe inserite."""
        cursor = conn.cursor()
        cursor.executemany(
            "INSERT INTO trades (symbol, side, quantity, price) VALUES (?, ?, ?, ?)",
            trades
        )
        conn.commit()
        return cursor.rowcount
    
    # Test singolo insert
    trade_id = insert_trade(conn, 'BTC', 'BUY', 0.1, 45000)
    print(f"Inserito trade ID: {trade_id}")
    
    # Test batch insert
    many_trades = [
        ('ETH', 'BUY', 1.0, 3000),
        ('SOL', 'BUY', 10, 100),
        ('BTC', 'SELL', 0.05, 46000),
    ]
    count = insert_many_trades(conn, many_trades)
    print(f"Inseriti {count} trade in batch")
    
    # Verifica
    cursor.execute("SELECT COUNT(*) FROM trades")
    total = cursor.fetchone()[0]
    print(f"Totale trade nel database: {total}")
    
    conn.close()
    return True

# exercise_2_solution()


# ─────────────────────────────────────────────────────────────────────────────
# ESERCIZIO 3: SELECT con WHERE e Filtri
# ─────────────────────────────────────────────────────────────────────────────

print("\n📝 ESERCIZIO 3: SELECT con WHERE e Filtri")
print("-" * 50)
print("""
OBIETTIVO: Impara a filtrare dati con WHERE.

TASKS:
1. Crea funzione `get_trades_by_symbol(conn, symbol)` → List[dict]
2. Crea funzione `get_trades_by_side(conn, side)` → List[dict]
3. Crea funzione `get_trades_in_price_range(conn, min_price, max_price)` → List[dict]
4. Crea funzione `get_recent_trades(conn, hours)` → List[dict]

USA IL DATABASE DI TEST: conn = create_test_db()

HINT:
- WHERE symbol = ?
- WHERE price BETWEEN ? AND ?
- datetime('now', '-N hours')
""")

def exercise_3_solution():
    """Soluzione Esercizio 3"""
    conn = create_test_db()
    
    def get_trades_by_symbol(conn, symbol: str) -> List[dict]:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM trades WHERE symbol = ?", (symbol,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_trades_by_side(conn, side: str) -> List[dict]:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM trades WHERE side = ?", (side,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_trades_in_price_range(conn, min_price: float, max_price: float) -> List[dict]:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM trades WHERE price BETWEEN ? AND ?",
            (min_price, max_price)
        )
        return [dict(row) for row in cursor.fetchall()]
    
    def get_recent_trades(conn, hours: int) -> List[dict]:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM trades WHERE created_at > datetime('now', ?)",
            (f'-{hours} hours',)
        )
        return [dict(row) for row in cursor.fetchall()]
    
    # Test
    print("Trade BTC:", len(get_trades_by_symbol(conn, 'BTC')))
    print("Trade BUY:", len(get_trades_by_side(conn, 'BUY')))
    print("Trade $40k-50k:", len(get_trades_in_price_range(conn, 40000, 50000)))
    
    conn.close()
    return True

# exercise_3_solution()


# ─────────────────────────────────────────────────────────────────────────────
# ESERCIZIO 4: UPDATE e DELETE
# ─────────────────────────────────────────────────────────────────────────────

print("\n📝 ESERCIZIO 4: UPDATE e DELETE")
print("-" * 50)
print("""
OBIETTIVO: Impara a modificare e eliminare dati.

TASKS:
1. Crea funzione `update_trade_price(conn, trade_id, new_price)` → bool
2. Crea funzione `update_trade_notes(conn, trade_id, notes)` → bool
3. Crea funzione `delete_trade(conn, trade_id)` → bool
4. Crea funzione `delete_trades_by_symbol(conn, symbol)` → int (num deleted)

ATTENZIONE: Ritorna True/False se la modifica ha avuto effetto (rowcount > 0)

HINT:
- cursor.rowcount per sapere quante righe sono state modificate
- UPDATE ... SET ... WHERE ...
- DELETE FROM ... WHERE ...
""")

def exercise_4_solution():
    """Soluzione Esercizio 4"""
    conn = create_test_db()
    
    # Aggiungi colonna notes se non esiste
    try:
        conn.execute("ALTER TABLE trades ADD COLUMN notes TEXT")
    except:
        pass
    
    def update_trade_price(conn, trade_id: int, new_price: float) -> bool:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE trades SET price = ? WHERE id = ?",
            (new_price, trade_id)
        )
        conn.commit()
        return cursor.rowcount > 0
    
    def update_trade_notes(conn, trade_id: int, notes: str) -> bool:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE trades SET notes = ? WHERE id = ?",
            (notes, trade_id)
        )
        conn.commit()
        return cursor.rowcount > 0
    
    def delete_trade(conn, trade_id: int) -> bool:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM trades WHERE id = ?", (trade_id,))
        conn.commit()
        return cursor.rowcount > 0
    
    def delete_trades_by_symbol(conn, symbol: str) -> int:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM trades WHERE symbol = ?", (symbol,))
        conn.commit()
        return cursor.rowcount
    
    # Test
    print(f"Update price trade 1: {update_trade_price(conn, 1, 43000)}")
    print(f"Update notes trade 1: {update_trade_notes(conn, 1, 'Modified')}")
    print(f"Delete trade 999 (non esiste): {delete_trade(conn, 999)}")
    print(f"Delete trades SOL: {delete_trades_by_symbol(conn, 'SOL')} eliminati")
    
    conn.close()
    return True

# exercise_4_solution()


# ─────────────────────────────────────────────────────────────────────────────
# ESERCIZIO 5: ORDER BY e LIMIT
# ─────────────────────────────────────────────────────────────────────────────

print("\n📝 ESERCIZIO 5: ORDER BY e LIMIT")
print("-" * 50)
print("""
OBIETTIVO: Impara a ordinare e limitare risultati.

TASKS:
1. Crea funzione `get_top_trades_by_value(conn, n)` → top N trade per valore
2. Crea funzione `get_latest_trades(conn, n)` → ultimi N trade per data
3. Crea funzione `get_trades_sorted(conn, sort_by, ascending=True)` → trade ordinati
4. Crea funzione `get_page(conn, page, page_size)` → paginazione

HINT:
- ORDER BY column ASC/DESC
- LIMIT n OFFSET m
- quantity * price AS value
""")

def exercise_5_solution():
    """Soluzione Esercizio 5"""
    conn = create_test_db()
    
    def get_top_trades_by_value(conn, n: int) -> List[dict]:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT *, (quantity * price) AS value 
            FROM trades 
            ORDER BY value DESC 
            LIMIT ?
        """, (n,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_latest_trades(conn, n: int) -> List[dict]:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM trades 
            ORDER BY created_at DESC 
            LIMIT ?
        """, (n,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_trades_sorted(conn, sort_by: str, ascending: bool = True) -> List[dict]:
        direction = "ASC" if ascending else "DESC"
        # Whitelist delle colonne per sicurezza
        valid_columns = ['id', 'symbol', 'side', 'quantity', 'price', 'created_at']
        if sort_by not in valid_columns:
            raise ValueError(f"Invalid column: {sort_by}")
        
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM trades ORDER BY {sort_by} {direction}")
        return [dict(row) for row in cursor.fetchall()]
    
    def get_page(conn, page: int, page_size: int) -> List[dict]:
        offset = (page - 1) * page_size
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM trades ORDER BY id LIMIT ? OFFSET ?",
            (page_size, offset)
        )
        return [dict(row) for row in cursor.fetchall()]
    
    # Test
    print("Top 3 trade per valore:")
    for t in get_top_trades_by_value(conn, 3):
        print(f"   ${t['value']:,.2f}")
    
    print("\nUltimi 3 trade:")
    for t in get_latest_trades(conn, 3):
        print(f"   {t['created_at']}: {t['symbol']}")
    
    print("\nPagina 1 (3 per pagina):")
    for t in get_page(conn, 1, 3):
        print(f"   ID {t['id']}: {t['symbol']}")
    
    conn.close()
    return True

# exercise_5_solution()


# ══════════════════════════════════════════════════════════════════════════════
# LIVELLO 2: SQL INTERMEDIO (Esercizi 6-10)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("📚 LIVELLO 2: SQL INTERMEDIO")
print("═" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# ESERCIZIO 6: Aggregazioni (COUNT, SUM, AVG)
# ─────────────────────────────────────────────────────────────────────────────

print("\n📝 ESERCIZIO 6: Aggregazioni")
print("-" * 50)
print("""
OBIETTIVO: Impara a usare funzioni di aggregazione.

TASKS:
1. Conta totale trade
2. Somma totale volume (quantity * price)
3. Media prezzo per simbolo
4. Min/Max prezzo per simbolo
5. Crea funzione `get_symbol_stats(conn, symbol)` → dict con tutte le stats

HINT:
- COUNT(*), SUM(), AVG(), MIN(), MAX()
- GROUP BY per raggruppare
""")

def exercise_6_solution():
    """Soluzione Esercizio 6"""
    conn = create_test_db()
    cursor = conn.cursor()
    
    # 1. Conta totale
    cursor.execute("SELECT COUNT(*) FROM trades")
    total = cursor.fetchone()[0]
    print(f"Totale trade: {total}")
    
    # 2. Somma volume
    cursor.execute("SELECT SUM(quantity * price) FROM trades")
    volume = cursor.fetchone()[0]
    print(f"Volume totale: ${volume:,.2f}")
    
    # 3. Media prezzo per simbolo
    cursor.execute("""
        SELECT symbol, AVG(price) as avg_price 
        FROM trades 
        GROUP BY symbol
    """)
    print("\nMedia prezzo per simbolo:")
    for row in cursor.fetchall():
        print(f"   {row['symbol']}: ${row['avg_price']:,.2f}")
    
    # 4. Min/Max
    cursor.execute("""
        SELECT symbol, MIN(price) as min_price, MAX(price) as max_price
        FROM trades 
        GROUP BY symbol
    """)
    print("\nMin/Max per simbolo:")
    for row in cursor.fetchall():
        print(f"   {row['symbol']}: ${row['min_price']:,.2f} - ${row['max_price']:,.2f}")
    
    # 5. Funzione stats
    def get_symbol_stats(conn, symbol: str) -> dict:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                COUNT(*) as trade_count,
                SUM(quantity) as total_qty,
                SUM(quantity * price) as total_volume,
                AVG(price) as avg_price,
                MIN(price) as min_price,
                MAX(price) as max_price
            FROM trades
            WHERE symbol = ?
        """, (symbol,))
        row = cursor.fetchone()
        return dict(row) if row else {}
    
    print("\nStats BTC:")
    stats = get_symbol_stats(conn, 'BTC')
    for k, v in stats.items():
        print(f"   {k}: {v}")
    
    conn.close()
    return True

# exercise_6_solution()


# ─────────────────────────────────────────────────────────────────────────────
# ESERCIZIO 7: GROUP BY e HAVING
# ─────────────────────────────────────────────────────────────────────────────

print("\n📝 ESERCIZIO 7: GROUP BY e HAVING")
print("-" * 50)
print("""
OBIETTIVO: Impara a raggruppare e filtrare gruppi.

TASKS:
1. Conta trade per simbolo
2. Conta trade per strategia
3. Trade per simbolo con almeno 2 trade (HAVING)
4. Volume totale per simbolo, solo se > 10000
5. Trade per giorno della settimana

HINT:
- GROUP BY column
- HAVING COUNT(*) > N (filtro DOPO il raggruppamento)
- strftime('%w', date) per giorno settimana
""")

def exercise_7_solution():
    """Soluzione Esercizio 7"""
    conn = create_test_db()
    cursor = conn.cursor()
    
    # 1. Trade per simbolo
    cursor.execute("""
        SELECT symbol, COUNT(*) as count 
        FROM trades 
        GROUP BY symbol
        ORDER BY count DESC
    """)
    print("Trade per simbolo:")
    for row in cursor.fetchall():
        print(f"   {row['symbol']}: {row['count']}")
    
    # 2. Trade per strategia
    cursor.execute("""
        SELECT strategy, COUNT(*) as count 
        FROM trades 
        WHERE strategy IS NOT NULL
        GROUP BY strategy
    """)
    print("\nTrade per strategia:")
    for row in cursor.fetchall():
        print(f"   {row['strategy']}: {row['count']}")
    
    # 3. Simboli con almeno 2 trade
    cursor.execute("""
        SELECT symbol, COUNT(*) as count 
        FROM trades 
        GROUP BY symbol
        HAVING count >= 2
    """)
    print("\nSimboli con almeno 2 trade:")
    for row in cursor.fetchall():
        print(f"   {row['symbol']}: {row['count']}")
    
    # 4. Volume > 10000
    cursor.execute("""
        SELECT symbol, SUM(quantity * price) as volume
        FROM trades
        GROUP BY symbol
        HAVING volume > 10000
    """)
    print("\nSimboli con volume > 10000:")
    for row in cursor.fetchall():
        print(f"   {row['symbol']}: ${row['volume']:,.2f}")
    
    # 5. Per giorno settimana (0=domenica, 6=sabato)
    cursor.execute("""
        SELECT 
            CASE strftime('%w', created_at)
                WHEN '0' THEN 'Domenica'
                WHEN '1' THEN 'Lunedì'
                WHEN '2' THEN 'Martedì'
                WHEN '3' THEN 'Mercoledì'
                WHEN '4' THEN 'Giovedì'
                WHEN '5' THEN 'Venerdì'
                WHEN '6' THEN 'Sabato'
            END as day_name,
            COUNT(*) as count
        FROM trades
        GROUP BY strftime('%w', created_at)
        ORDER BY strftime('%w', created_at)
    """)
    print("\nTrade per giorno:")
    for row in cursor.fetchall():
        print(f"   {row['day_name']}: {row['count']}")
    
    conn.close()
    return True

# exercise_7_solution()


# ─────────────────────────────────────────────────────────────────────────────
# ESERCIZIO 8: JOIN
# ─────────────────────────────────────────────────────────────────────────────

print("\n📝 ESERCIZIO 8: JOIN")
print("-" * 50)
print("""
OBIETTIVO: Impara a combinare dati da più tabelle.

TASKS:
1. Lista trade con nome portfolio (JOIN trades + portfolios)
2. Posizioni con nome portfolio
3. Portfolio con conteggio trade
4. LEFT JOIN per trovare portfolio senza trade

HINT:
- INNER JOIN: solo match
- LEFT JOIN: tutti a sinistra + match
- ON t.portfolio_id = p.id
""")

def exercise_8_solution():
    """Soluzione Esercizio 8"""
    conn = create_test_db()
    cursor = conn.cursor()
    
    # 1. Trade con nome portfolio
    cursor.execute("""
        SELECT t.*, p.name as portfolio_name
        FROM trades t
        INNER JOIN portfolios p ON t.portfolio_id = p.id
        ORDER BY t.created_at DESC
        LIMIT 5
    """)
    print("Trade con portfolio:")
    for row in cursor.fetchall():
        print(f"   [{row['portfolio_name']}] {row['symbol']} {row['side']}")
    
    # 2. Posizioni con nome portfolio
    cursor.execute("""
        SELECT pos.*, p.name as portfolio_name
        FROM positions pos
        INNER JOIN portfolios p ON pos.portfolio_id = p.id
        WHERE pos.quantity > 0
    """)
    print("\nPosizioni attive:")
    for row in cursor.fetchall():
        print(f"   [{row['portfolio_name']}] {row['symbol']}: {row['quantity']}")
    
    # 3. Portfolio con conteggio trade
    cursor.execute("""
        SELECT p.name, COUNT(t.id) as trade_count
        FROM portfolios p
        LEFT JOIN trades t ON p.id = t.portfolio_id
        GROUP BY p.id
    """)
    print("\nTrade per portfolio:")
    for row in cursor.fetchall():
        print(f"   {row['name']}: {row['trade_count']} trade")
    
    # 4. Aggiungiamo un portfolio senza trade per testare LEFT JOIN
    cursor.execute("INSERT INTO portfolios (name) VALUES ('Empty Portfolio')")
    
    cursor.execute("""
        SELECT p.name
        FROM portfolios p
        LEFT JOIN trades t ON p.id = t.portfolio_id
        WHERE t.id IS NULL
    """)
    print("\nPortfolio senza trade:")
    for row in cursor.fetchall():
        print(f"   {row['name']}")
    
    conn.close()
    return True

# exercise_8_solution()


# ─────────────────────────────────────────────────────────────────────────────
# ESERCIZIO 9: Subquery
# ─────────────────────────────────────────────────────────────────────────────

print("\n📝 ESERCIZIO 9: Subquery")
print("-" * 50)
print("""
OBIETTIVO: Impara a usare query annidate.

TASKS:
1. Trade con prezzo sopra la media
2. Trade nel portfolio con più volume
3. Simboli che hanno avuto sia BUY che SELL
4. Ultimo trade per ogni simbolo

HINT:
- WHERE price > (SELECT AVG(price) FROM trades)
- Subquery come filtro
- Subquery correlate
""")

def exercise_9_solution():
    """Soluzione Esercizio 9"""
    conn = create_test_db()
    cursor = conn.cursor()
    
    # 1. Trade sopra la media
    cursor.execute("""
        SELECT symbol, price
        FROM trades
        WHERE price > (SELECT AVG(price) FROM trades)
    """)
    print("Trade sopra la media:")
    for row in cursor.fetchall():
        print(f"   {row['symbol']}: ${row['price']:,.2f}")
    
    # 2. Trade nel portfolio con più volume
    cursor.execute("""
        SELECT * FROM trades
        WHERE portfolio_id = (
            SELECT portfolio_id
            FROM trades
            GROUP BY portfolio_id
            ORDER BY SUM(quantity * price) DESC
            LIMIT 1
        )
    """)
    print(f"\nTrade nel portfolio top: {len(cursor.fetchall())}")
    
    # 3. Simboli con sia BUY che SELL
    cursor.execute("""
        SELECT DISTINCT symbol FROM trades t1
        WHERE EXISTS (
            SELECT 1 FROM trades t2 
            WHERE t2.symbol = t1.symbol AND t2.side = 'BUY'
        )
        AND EXISTS (
            SELECT 1 FROM trades t3 
            WHERE t3.symbol = t1.symbol AND t3.side = 'SELL'
        )
    """)
    print("\nSimboli con BUY e SELL:")
    for row in cursor.fetchall():
        print(f"   {row['symbol']}")
    
    # 4. Ultimo trade per simbolo
    cursor.execute("""
        SELECT * FROM trades t1
        WHERE created_at = (
            SELECT MAX(created_at) FROM trades t2
            WHERE t2.symbol = t1.symbol
        )
    """)
    print("\nUltimo trade per simbolo:")
    for row in cursor.fetchall():
        print(f"   {row['symbol']}: {row['created_at']}")
    
    conn.close()
    return True

# exercise_9_solution()


# ─────────────────────────────────────────────────────────────────────────────
# ESERCIZIO 10: Transazioni
# ─────────────────────────────────────────────────────────────────────────────

print("\n📝 ESERCIZIO 10: Transazioni")
print("-" * 50)
print("""
OBIETTIVO: Impara a gestire transazioni atomiche.

TASKS:
1. Crea funzione `transfer_funds(conn, from_portfolio, to_portfolio, amount)`
   che trasferisce fondi tra portfolio in modo atomico
2. Implementa rollback in caso di errore
3. Crea funzione `execute_trade_with_position_update(conn, trade_data)`
   che crea trade E aggiorna position in una transazione

HINT:
- try/except con rollback
- conn.commit() solo se tutto ok
- Usa context manager se possibile
""")

def exercise_10_solution():
    """Soluzione Esercizio 10"""
    conn = create_test_db()
    
    # Aggiungi colonna balance se non esiste
    try:
        conn.execute("ALTER TABLE portfolios ADD COLUMN balance REAL DEFAULT 10000")
    except:
        pass
    conn.execute("UPDATE portfolios SET balance = initial_balance")
    conn.commit()
    
    def transfer_funds(conn, from_id: int, to_id: int, amount: float) -> bool:
        """Trasferisce fondi tra portfolio atomicamente."""
        cursor = conn.cursor()
        try:
            # Verifica saldo sufficiente
            cursor.execute("SELECT balance FROM portfolios WHERE id = ?", (from_id,))
            from_balance = cursor.fetchone()
            if not from_balance or from_balance['balance'] < amount:
                print("   ❌ Saldo insufficiente")
                return False
            
            # Sottrai dal mittente
            cursor.execute(
                "UPDATE portfolios SET balance = balance - ? WHERE id = ?",
                (amount, from_id)
            )
            
            # Aggiungi al destinatario
            cursor.execute(
                "UPDATE portfolios SET balance = balance + ? WHERE id = ?",
                (amount, to_id)
            )
            
            conn.commit()
            print(f"   ✅ Trasferiti ${amount}")
            return True
            
        except Exception as e:
            conn.rollback()
            print(f"   ❌ Errore: {e}")
            return False
    
    def execute_trade_with_position(conn, portfolio_id: int, symbol: str,
                                   side: str, qty: float, price: float) -> bool:
        """Esegue trade e aggiorna position atomicamente."""
        cursor = conn.cursor()
        try:
            # 1. Crea trade
            cursor.execute("""
                INSERT INTO trades (portfolio_id, symbol, side, quantity, price)
                VALUES (?, ?, ?, ?, ?)
            """, (portfolio_id, symbol, side, qty, price))
            
            # 2. Aggiorna position
            cursor.execute("""
                SELECT * FROM positions 
                WHERE portfolio_id = ? AND symbol = ?
            """, (portfolio_id, symbol))
            pos = cursor.fetchone()
            
            if pos:
                if side == 'BUY':
                    new_qty = pos['quantity'] + qty
                    new_avg = ((pos['avg_price'] * pos['quantity']) + (price * qty)) / new_qty
                    cursor.execute("""
                        UPDATE positions SET quantity = ?, avg_price = ?
                        WHERE portfolio_id = ? AND symbol = ?
                    """, (new_qty, new_avg, portfolio_id, symbol))
                else:
                    new_qty = pos['quantity'] - qty
                    cursor.execute("""
                        UPDATE positions SET quantity = ?
                        WHERE portfolio_id = ? AND symbol = ?
                    """, (new_qty, portfolio_id, symbol))
            else:
                cursor.execute("""
                    INSERT INTO positions (portfolio_id, symbol, quantity, avg_price)
                    VALUES (?, ?, ?, ?)
                """, (portfolio_id, symbol, qty if side == 'BUY' else -qty, price))
            
            conn.commit()
            print(f"   ✅ Trade + position aggiornati")
            return True
            
        except Exception as e:
            conn.rollback()
            print(f"   ❌ Rollback: {e}")
            return False
    
    # Test
    print("Test trasferimento:")
    transfer_funds(conn, 1, 2, 1000)
    transfer_funds(conn, 1, 2, 999999)  # Dovrebbe fallire
    
    print("\nTest trade + position:")
    execute_trade_with_position(conn, 1, 'AVAX', 'BUY', 10, 35)
    
    conn.close()
    return True

# exercise_10_solution()


# ══════════════════════════════════════════════════════════════════════════════
# LIVELLO 3: SQLALCHEMY ORM (Esercizi 11-15)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("📚 LIVELLO 3: SQLALCHEMY ORM")
print("═" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# ESERCIZIO 11: Definizione Modelli
# ─────────────────────────────────────────────────────────────────────────────

print("\n📝 ESERCIZIO 11: Definizione Modelli ORM")
print("-" * 50)
print("""
OBIETTIVO: Impara a definire modelli SQLAlchemy.

TASKS:
Crea i seguenti modelli (puoi usare pseudo-codice o codice reale):

1. Modello Cryptocurrency:
   - id (primary key)
   - symbol (string, unique)
   - name (string)
   - current_price (float)
   - market_cap (float, optional)
   - updated_at (datetime)

2. Modello WatchlistItem:
   - id (primary key)
   - user_id (foreign key)
   - crypto_id (foreign key to Cryptocurrency)
   - alert_price (float, optional)
   - created_at (datetime)

3. Aggiungi relazioni:
   - Cryptocurrency.watchlist_items → List[WatchlistItem]
   - WatchlistItem.crypto → Cryptocurrency

HINT:
- Mapped[type] = mapped_column(...)
- relationship("Model", back_populates="...")
""")

# Soluzione come stringa dato che richiede SQLAlchemy installato
exercise_11_code = '''
from sqlalchemy import String, Float, ForeignKey, DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from datetime import datetime
from typing import List, Optional

class Base(DeclarativeBase):
    pass

class Cryptocurrency(Base):
    __tablename__ = 'cryptocurrencies'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), unique=True)
    name: Mapped[str] = mapped_column(String(100))
    current_price: Mapped[float] = mapped_column(Float)
    market_cap: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    
    # Relazione
    watchlist_items: Mapped[List["WatchlistItem"]] = relationship(
        back_populates="crypto"
    )
    
    def __repr__(self):
        return f"Cryptocurrency({self.symbol}: ${self.current_price})"

class WatchlistItem(Base):
    __tablename__ = 'watchlist_items'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))
    crypto_id: Mapped[int] = mapped_column(ForeignKey('cryptocurrencies.id'))
    alert_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relazione
    crypto: Mapped["Cryptocurrency"] = relationship(back_populates="watchlist_items")
    
    def __repr__(self):
        return f"WatchlistItem(crypto={self.crypto_id}, alert=${self.alert_price})"
'''

print("\nSoluzione (codice SQLAlchemy):")
print(exercise_11_code)


# ─────────────────────────────────────────────────────────────────────────────
# ESERCIZIO 12: CRUD con ORM
# ─────────────────────────────────────────────────────────────────────────────

print("\n📝 ESERCIZIO 12: CRUD con ORM")
print("-" * 50)
print("""
OBIETTIVO: Impara le operazioni CRUD con SQLAlchemy ORM.

TASKS (pseudo-codice o reale):

1. CREATE:
   - Funzione add_crypto(session, symbol, name, price)
   - Funzione add_watchlist_item(session, user_id, crypto_id, alert_price)

2. READ:
   - Funzione get_crypto_by_symbol(session, symbol)
   - Funzione get_all_cryptos(session)
   - Funzione get_watchlist_for_user(session, user_id)

3. UPDATE:
   - Funzione update_crypto_price(session, symbol, new_price)
   - Funzione update_alert(session, item_id, new_alert_price)

4. DELETE:
   - Funzione remove_from_watchlist(session, item_id)
   - Funzione remove_crypto(session, symbol) - con cascade

HINT:
- session.add(obj) per insert
- session.get(Model, id) per get by PK
- session.scalars(select(Model).where(...))
""")

exercise_12_code = '''
from sqlalchemy import select
from sqlalchemy.orm import Session

def add_crypto(session: Session, symbol: str, name: str, price: float):
    crypto = Cryptocurrency(symbol=symbol, name=name, current_price=price)
    session.add(crypto)
    session.flush()  # Per ottenere ID prima del commit
    return crypto

def get_crypto_by_symbol(session: Session, symbol: str):
    stmt = select(Cryptocurrency).where(Cryptocurrency.symbol == symbol)
    return session.scalar(stmt)

def get_all_cryptos(session: Session):
    stmt = select(Cryptocurrency).order_by(Cryptocurrency.market_cap.desc())
    return list(session.scalars(stmt))

def update_crypto_price(session: Session, symbol: str, new_price: float):
    crypto = get_crypto_by_symbol(session, symbol)
    if crypto:
        crypto.current_price = new_price
        return True
    return False

def remove_from_watchlist(session: Session, item_id: int):
    item = session.get(WatchlistItem, item_id)
    if item:
        session.delete(item)
        return True
    return False

# Esempio uso
with Session(engine) as session:
    # Create
    btc = add_crypto(session, "BTC", "Bitcoin", 45000)
    
    # Read
    crypto = get_crypto_by_symbol(session, "BTC")
    all_cryptos = get_all_cryptos(session)
    
    # Update  
    update_crypto_price(session, "BTC", 46000)
    
    # Delete
    remove_from_watchlist(session, 1)
    
    session.commit()
'''

print("\nSoluzione (codice SQLAlchemy):")
print(exercise_12_code)


# ─────────────────────────────────────────────────────────────────────────────
# ESERCIZIO 13-15: Repository Pattern e Progetti
# ─────────────────────────────────────────────────────────────────────────────

print("\n📝 ESERCIZI 13-15: Pattern Avanzati")
print("-" * 50)
print("""
ESERCIZIO 13: Repository Pattern
- Crea CryptoRepository con metodi specializzati
- Crea WatchlistRepository 
- Implementa filtri e paginazione

ESERCIZIO 14: Relazioni Complesse
- Aggiungi modello User con relazione Many-to-Many con Crypto via WatchlistItem
- Implementa eager loading con joinedload/selectinload

ESERCIZIO 15: Migrazioni
- Scrivi una migrazione Alembic che aggiunge colonna 'volume_24h' a Cryptocurrency
- Scrivi downgrade che la rimuove

Questi esercizi richiedono SQLAlchemy installato.
Vedi module_database_part2_postgresql_orm.py per esempi completi.
""")


# ══════════════════════════════════════════════════════════════════════════════
# LIVELLO 4: PROGETTI TRADING (Esercizi 16-20)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("📚 LIVELLO 4: PROGETTI TRADING")
print("═" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# ESERCIZIO 16: Trade Logger
# ─────────────────────────────────────────────────────────────────────────────

print("\n📝 ESERCIZIO 16: Trade Logger")
print("-" * 50)
print("""
OBIETTIVO: Sistema completo di logging trade.

TASKS:
1. Crea funzione `log_trade(conn, trade_data)` che salva trade
2. Crea funzione `get_trades_by_symbol(conn, symbol)` 
3. Crea funzione `get_trades_by_date_range(conn, start, end)`
4. Crea funzione `calculate_statistics(conn, portfolio_id)` → PnL, win rate, etc.
5. Crea funzione `export_to_csv(trades)` → stringa CSV
""")

def exercise_16_solution():
    """Soluzione Esercizio 16"""
    conn = create_test_db()
    
    def log_trade(conn, portfolio_id: int, symbol: str, side: str,
                 quantity: float, price: float, strategy: str = None) -> int:
        """Logga un trade e ritorna l'ID."""
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO trades (portfolio_id, symbol, side, quantity, price, strategy)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (portfolio_id, symbol, side, quantity, price, strategy))
        conn.commit()
        return cursor.lastrowid
    
    def get_trades_by_symbol(conn, symbol: str) -> List[dict]:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM trades WHERE symbol = ?", (symbol,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_trades_by_date_range(conn, start: str, end: str) -> List[dict]:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM trades 
            WHERE date(created_at) BETWEEN date(?) AND date(?)
        """, (start, end))
        return [dict(row) for row in cursor.fetchall()]
    
    def calculate_statistics(conn, portfolio_id: int) -> dict:
        cursor = conn.cursor()
        
        # Statistiche base
        cursor.execute("""
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN side='BUY' THEN 1 ELSE 0 END) as buys,
                SUM(CASE WHEN side='SELL' THEN 1 ELSE 0 END) as sells,
                SUM(quantity * price) as total_volume,
                SUM(fee) as total_fees
            FROM trades
            WHERE portfolio_id = ?
        """, (portfolio_id,))
        row = cursor.fetchone()
        
        return {
            'total_trades': row['total_trades'],
            'buys': row['buys'],
            'sells': row['sells'],
            'total_volume': row['total_volume'],
            'total_fees': row['total_fees']
        }
    
    def export_to_csv(trades: List[dict]) -> str:
        if not trades:
            return ""
        headers = list(trades[0].keys())
        lines = [','.join(headers)]
        for t in trades:
            lines.append(','.join(str(t[h]) for h in headers))
        return '\n'.join(lines)
    
    # Test
    print("1. Log nuovo trade:")
    new_id = log_trade(conn, 1, 'AVAX', 'BUY', 25, 38.5, 'RSI')
    print(f"   Nuovo trade ID: {new_id}")
    
    print("\n2. Trade per simbolo (BTC):")
    btc_trades = get_trades_by_symbol(conn, 'BTC')
    for t in btc_trades:
        print(f"   {t['side']} {t['quantity']} @ ${t['price']}")
    
    print("\n3. Trade nel range date:")
    trades = get_trades_by_date_range(conn, '2024-01-01', '2024-01-10')
    print(f"   Trovati {len(trades)} trade")
    
    print("\n4. Statistiche:")
    stats = calculate_statistics(conn, 1)
    for k, v in stats.items():
        print(f"   {k}: {v}")
    
    print("\n5. Export CSV (prime 3 righe):")
    csv = export_to_csv(btc_trades)
    for line in csv.split('\n')[:3]:
        print(f"   {line}")
    
    conn.close()
    return True

# exercise_16_solution()


# ─────────────────────────────────────────────────────────────────────────────
# ESERCIZIO 17: Portfolio Manager
# ─────────────────────────────────────────────────────────────────────────────

print("\n📝 ESERCIZIO 17: Portfolio Manager")
print("-" * 50)
print("""
OBIETTIVO: Gestisci posizioni e calcola performance.

TASKS:
1. Implementa update_position dopo ogni trade
2. Calcola valore portfolio dato prezzi correnti
3. Calcola unrealized P&L per ogni posizione
4. Calcola allocazione % per asset
5. Genera report portfolio completo
""")

def exercise_17_solution():
    """Soluzione Esercizio 17"""
    conn = create_test_db()
    cursor = conn.cursor()
    
    def update_position(conn, portfolio_id: int, symbol: str, 
                       side: str, quantity: float, price: float) -> float:
        """Aggiorna posizione dopo trade. Ritorna PnL se SELL."""
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT quantity, avg_price, realized_pnl FROM positions
            WHERE portfolio_id = ? AND symbol = ?
        """, (portfolio_id, symbol))
        pos = cursor.fetchone()
        
        pnl = 0
        
        if side == 'BUY':
            if pos:
                new_qty = pos['quantity'] + quantity
                new_avg = ((pos['avg_price'] * pos['quantity']) + (price * quantity)) / new_qty
                cursor.execute("""
                    UPDATE positions SET quantity = ?, avg_price = ?, updated_at = datetime('now')
                    WHERE portfolio_id = ? AND symbol = ?
                """, (new_qty, new_avg, portfolio_id, symbol))
            else:
                cursor.execute("""
                    INSERT INTO positions (portfolio_id, symbol, quantity, avg_price)
                    VALUES (?, ?, ?, ?)
                """, (portfolio_id, symbol, quantity, price))
        else:  # SELL
            if pos and pos['quantity'] >= quantity:
                pnl = (price - pos['avg_price']) * quantity
                new_qty = pos['quantity'] - quantity
                new_pnl = pos['realized_pnl'] + pnl
                
                if new_qty > 0:
                    cursor.execute("""
                        UPDATE positions SET quantity = ?, realized_pnl = ?, updated_at = datetime('now')
                        WHERE portfolio_id = ? AND symbol = ?
                    """, (new_qty, new_pnl, portfolio_id, symbol))
                else:
                    cursor.execute("""
                        DELETE FROM positions WHERE portfolio_id = ? AND symbol = ?
                    """, (portfolio_id, symbol))
        
        conn.commit()
        return pnl
    
    def get_portfolio_value(conn, portfolio_id: int, current_prices: dict) -> float:
        """Calcola valore totale portfolio."""
        cursor = conn.cursor()
        cursor.execute("""
            SELECT symbol, quantity FROM positions
            WHERE portfolio_id = ? AND quantity > 0
        """, (portfolio_id,))
        
        total = 0
        for row in cursor.fetchall():
            price = current_prices.get(row['symbol'], 0)
            total += row['quantity'] * price
        return total
    
    def get_unrealized_pnl(conn, portfolio_id: int, current_prices: dict) -> List[dict]:
        """Calcola unrealized PnL per posizione."""
        cursor = conn.cursor()
        cursor.execute("""
            SELECT symbol, quantity, avg_price FROM positions
            WHERE portfolio_id = ? AND quantity > 0
        """, (portfolio_id,))
        
        results = []
        for row in cursor.fetchall():
            current = current_prices.get(row['symbol'], row['avg_price'])
            pnl = (current - row['avg_price']) * row['quantity']
            pnl_pct = ((current / row['avg_price']) - 1) * 100 if row['avg_price'] else 0
            results.append({
                'symbol': row['symbol'],
                'quantity': row['quantity'],
                'avg_price': row['avg_price'],
                'current_price': current,
                'unrealized_pnl': pnl,
                'unrealized_pnl_pct': pnl_pct
            })
        return results
    
    def get_allocation(conn, portfolio_id: int, current_prices: dict) -> List[dict]:
        """Calcola allocazione % per asset."""
        total = get_portfolio_value(conn, portfolio_id, current_prices)
        if total == 0:
            return []
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT symbol, quantity FROM positions
            WHERE portfolio_id = ? AND quantity > 0
        """, (portfolio_id,))
        
        results = []
        for row in cursor.fetchall():
            value = row['quantity'] * current_prices.get(row['symbol'], 0)
            results.append({
                'symbol': row['symbol'],
                'value': value,
                'allocation_pct': (value / total) * 100
            })
        return sorted(results, key=lambda x: x['allocation_pct'], reverse=True)
    
    # Test con prezzi correnti simulati
    current_prices = {'BTC': 48000, 'ETH': 3200, 'SOL': 120}
    
    print("1. Unrealized PnL:")
    for p in get_unrealized_pnl(conn, 1, current_prices):
        print(f"   {p['symbol']}: ${p['unrealized_pnl']:+,.2f} ({p['unrealized_pnl_pct']:+.1f}%)")
    
    print("\n2. Portfolio Value:")
    value = get_portfolio_value(conn, 1, current_prices)
    print(f"   ${value:,.2f}")
    
    print("\n3. Allocation:")
    for a in get_allocation(conn, 1, current_prices):
        print(f"   {a['symbol']}: {a['allocation_pct']:.1f}%")
    
    conn.close()
    return True

# exercise_17_solution()


# ─────────────────────────────────────────────────────────────────────────────
# ESERCIZIO 18: OHLCV Manager
# ─────────────────────────────────────────────────────────────────────────────

print("\n📝 ESERCIZIO 18: OHLCV Manager")
print("-" * 50)
print("""
OBIETTIVO: Gestisci storico candele per backtesting.

TASKS:
1. Funzione insert_candle (con upsert per evitare duplicati)
2. Funzione get_candles(symbol, timeframe, start, end)
3. Funzione resample_candles(candles, new_timeframe) - es: 1h → 4h
4. Funzione calculate_indicators - SMA, EMA sul close
5. Funzione find_patterns - candele con volume sopra media
""")

def exercise_18_solution():
    """Soluzione Esercizio 18"""
    conn = create_test_db()
    
    def insert_candle(conn, symbol: str, timeframe: str, timestamp: int,
                     o: float, h: float, l: float, c: float, v: float) -> bool:
        """Insert o update candela (upsert)."""
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO candles (symbol, timeframe, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, timeframe, timestamp) 
            DO UPDATE SET open=?, high=?, low=?, close=?, volume=?
        """, (symbol, timeframe, timestamp, o, h, l, c, v, o, h, l, c, v))
        conn.commit()
        return True
    
    def get_candles(conn, symbol: str, timeframe: str, 
                   start_ts: int = None, end_ts: int = None) -> List[dict]:
        """Recupera candele per range temporale."""
        cursor = conn.cursor()
        query = "SELECT * FROM candles WHERE symbol = ? AND timeframe = ?"
        params = [symbol, timeframe]
        
        if start_ts:
            query += " AND timestamp >= ?"
            params.append(start_ts)
        if end_ts:
            query += " AND timestamp <= ?"
            params.append(end_ts)
        
        query += " ORDER BY timestamp"
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def calculate_sma(candles: List[dict], period: int) -> List[float]:
        """Calcola Simple Moving Average."""
        closes = [c['close'] for c in candles]
        sma = []
        for i in range(len(closes)):
            if i < period - 1:
                sma.append(None)
            else:
                window = closes[i - period + 1:i + 1]
                sma.append(sum(window) / period)
        return sma
    
    def calculate_ema(candles: List[dict], period: int) -> List[float]:
        """Calcola Exponential Moving Average."""
        closes = [c['close'] for c in candles]
        ema = []
        multiplier = 2 / (period + 1)
        
        for i, close in enumerate(closes):
            if i == 0:
                ema.append(close)
            elif i < period - 1:
                # Usa SMA per il periodo iniziale
                ema.append(sum(closes[:i+1]) / (i + 1))
            else:
                ema.append((close - ema[-1]) * multiplier + ema[-1])
        return ema
    
    def find_high_volume_candles(candles: List[dict], threshold: float = 1.5) -> List[dict]:
        """Trova candele con volume sopra la media * threshold."""
        if not candles:
            return []
        avg_volume = sum(c['volume'] for c in candles) / len(candles)
        return [c for c in candles if c['volume'] > avg_volume * threshold]
    
    # Test
    print("1. Get candles BTC 1h:")
    candles = get_candles(conn, 'BTC', '1h')
    print(f"   {len(candles)} candele caricate")
    
    print("\n2. SMA(20) ultime 5 candele:")
    sma = calculate_sma(candles, 20)
    for i, v in enumerate(sma[-5:]):
        if v:
            print(f"   SMA: ${v:,.2f}")
    
    print("\n3. EMA(12) ultime 5 candele:")
    ema = calculate_ema(candles, 12)
    for v in ema[-5:]:
        print(f"   EMA: ${v:,.2f}")
    
    print("\n4. Candele alto volume (>1.5x media):")
    high_vol = find_high_volume_candles(candles)
    print(f"   Trovate {len(high_vol)} candele")
    
    conn.close()
    return True

# exercise_18_solution()


# ─────────────────────────────────────────────────────────────────────────────
# ESERCIZIO 19: Performance Analyzer
# ─────────────────────────────────────────────────────────────────────────────

print("\n📝 ESERCIZIO 19: Performance Analyzer")
print("-" * 50)
print("""
OBIETTIVO: Analizza performance trading.

TASKS:
1. Calcola Win Rate (% trade in profitto)
2. Calcola Profit Factor (gross profit / gross loss)
3. Calcola Max Drawdown
4. Calcola Sharpe Ratio (semplificato)
5. Genera report completo per periodo
""")

def exercise_19_solution():
    """Soluzione Esercizio 19"""
    conn = create_test_db()
    
    def get_closed_trades_pnl(conn, portfolio_id: int) -> List[float]:
        """Calcola PnL per ogni "round trip" (buy + sell)."""
        cursor = conn.cursor()
        
        # Per ogni simbolo, abbina BUY e SELL
        cursor.execute("""
            SELECT symbol FROM trades 
            WHERE portfolio_id = ?
            GROUP BY symbol
            HAVING SUM(CASE WHEN side='BUY' THEN 1 ELSE 0 END) > 0
               AND SUM(CASE WHEN side='SELL' THEN 1 ELSE 0 END) > 0
        """, (portfolio_id,))
        symbols = [row['symbol'] for row in cursor.fetchall()]
        
        pnls = []
        for symbol in symbols:
            cursor.execute("""
                SELECT 
                    (SELECT SUM(quantity * price) FROM trades 
                     WHERE portfolio_id = ? AND symbol = ? AND side = 'SELL') -
                    (SELECT SUM(quantity * price) FROM trades 
                     WHERE portfolio_id = ? AND symbol = ? AND side = 'BUY') as pnl
            """, (portfolio_id, symbol, portfolio_id, symbol))
            row = cursor.fetchone()
            if row and row['pnl'] is not None:
                pnls.append(row['pnl'])
        
        return pnls
    
    def calculate_win_rate(pnls: List[float]) -> float:
        """Percentuale di trade in profitto."""
        if not pnls:
            return 0
        wins = sum(1 for p in pnls if p > 0)
        return (wins / len(pnls)) * 100
    
    def calculate_profit_factor(pnls: List[float]) -> float:
        """Rapporto gross profit / gross loss."""
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def calculate_max_drawdown(equity_curve: List[float]) -> float:
        """Massimo drawdown percentuale."""
        if not equity_curve:
            return 0
        
        max_dd = 0
        peak = equity_curve[0]
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def generate_performance_report(conn, portfolio_id: int) -> dict:
        """Genera report completo."""
        cursor = conn.cursor()
        
        # Trade totali
        cursor.execute("""
            SELECT COUNT(*) as total,
                   SUM(quantity * price) as volume,
                   SUM(fee) as fees
            FROM trades WHERE portfolio_id = ?
        """, (portfolio_id,))
        row = cursor.fetchone()
        
        pnls = get_closed_trades_pnl(conn, portfolio_id)
        total_pnl = sum(pnls) if pnls else 0
        
        # Posizioni attuali
        cursor.execute("""
            SELECT SUM(realized_pnl) as realized
            FROM positions WHERE portfolio_id = ?
        """, (portfolio_id,))
        realized = cursor.fetchone()['realized'] or 0
        
        return {
            'total_trades': row['total'],
            'total_volume': row['volume'],
            'total_fees': row['fees'],
            'closed_positions': len(pnls),
            'total_pnl': total_pnl,
            'realized_pnl': realized,
            'win_rate': calculate_win_rate(pnls),
            'profit_factor': calculate_profit_factor(pnls)
        }
    
    # Test
    print("Performance Report Portfolio 1:")
    report = generate_performance_report(conn, 1)
    for k, v in report.items():
        if isinstance(v, float):
            print(f"   {k}: {v:,.2f}")
        else:
            print(f"   {k}: {v}")
    
    conn.close()
    return True

# exercise_19_solution()


# ─────────────────────────────────────────────────────────────────────────────
# ESERCIZIO 20: Trading Bot Database Layer
# ─────────────────────────────────────────────────────────────────────────────

print("\n📝 ESERCIZIO 20: Trading Bot Database Layer (PROGETTO FINALE)")
print("-" * 50)
print("""
OBIETTIVO: Crea un layer database completo per il tuo trading bot.

TASKS:
Crea una classe TradingDatabase che gestisca:

1. Setup:
   - __init__(self, db_path) - crea connessione
   - create_tables() - crea schema se non esiste
   - close() - chiude connessione

2. Trade Operations:
   - log_trade(trade_data) → trade_id
   - get_open_orders()
   - update_order_status(order_id, status)

3. Position Operations:
   - update_position(portfolio_id, symbol, side, qty, price)
   - get_position(portfolio_id, symbol)
   - get_all_positions(portfolio_id)

4. Market Data:
   - save_candles(candles_list)
   - get_candles(symbol, timeframe, limit)
   - get_latest_price(symbol)

5. Analytics:
   - get_portfolio_summary(portfolio_id)
   - get_trade_history(portfolio_id, days)

Implementa come classe Python completa!
""")

def exercise_20_solution():
    """Soluzione Esercizio 20 - Trading Bot Database Layer"""
    
    class TradingDatabase:
        """Database layer completo per trading bot."""
        
        def __init__(self, db_path: str = ":memory:"):
            self.db_path = db_path
            self.conn = sqlite3.connect(db_path)
            self.conn.row_factory = sqlite3.Row
            self.create_tables()
        
        def create_tables(self):
            """Crea schema database se non esiste."""
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS portfolios (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    initial_balance REAL DEFAULT 10000,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY,
                    portfolio_id INTEGER,
                    order_id TEXT UNIQUE,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT DEFAULT 'MARKET',
                    quantity REAL NOT NULL,
                    price REAL,
                    filled_price REAL,
                    fee REAL DEFAULT 0,
                    status TEXT DEFAULT 'PENDING',
                    strategy TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    filled_at TEXT,
                    FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
                );
                
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY,
                    portfolio_id INTEGER,
                    symbol TEXT NOT NULL,
                    quantity REAL DEFAULT 0,
                    avg_price REAL DEFAULT 0,
                    realized_pnl REAL DEFAULT 0,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (portfolio_id) REFERENCES portfolios(id),
                    UNIQUE(portfolio_id, symbol)
                );
                
                CREATE TABLE IF NOT EXISTS candles (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    UNIQUE(symbol, timeframe, timestamp)
                );
                
                CREATE INDEX IF NOT EXISTS idx_trades_portfolio ON trades(portfolio_id);
                CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
                CREATE INDEX IF NOT EXISTS idx_candles_lookup ON candles(symbol, timeframe, timestamp);
            """)
            self.conn.commit()
        
        def close(self):
            """Chiude connessione."""
            self.conn.close()
        
        # Trade Operations
        def log_trade(self, portfolio_id: int, symbol: str, side: str,
                     quantity: float, price: float = None, **kwargs) -> int:
            """Logga nuovo trade."""
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO trades (portfolio_id, symbol, side, quantity, price, 
                                   order_type, strategy, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                portfolio_id, symbol, side, quantity, price,
                kwargs.get('order_type', 'MARKET'),
                kwargs.get('strategy'),
                kwargs.get('status', 'FILLED')
            ))
            self.conn.commit()
            return cursor.lastrowid
        
        def get_open_orders(self) -> List[dict]:
            """Recupera ordini aperti."""
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM trades WHERE status = 'PENDING'")
            return [dict(row) for row in cursor.fetchall()]
        
        def update_order_status(self, trade_id: int, status: str, 
                               filled_price: float = None) -> bool:
            """Aggiorna stato ordine."""
            cursor = self.conn.cursor()
            if filled_price:
                cursor.execute("""
                    UPDATE trades SET status = ?, filled_price = ?, filled_at = datetime('now')
                    WHERE id = ?
                """, (status, filled_price, trade_id))
            else:
                cursor.execute("UPDATE trades SET status = ? WHERE id = ?", (status, trade_id))
            self.conn.commit()
            return cursor.rowcount > 0
        
        # Position Operations
        def update_position(self, portfolio_id: int, symbol: str, 
                          side: str, quantity: float, price: float) -> float:
            """Aggiorna posizione dopo trade. Ritorna PnL per SELL."""
            cursor = self.conn.cursor()
            
            cursor.execute("""
                SELECT quantity, avg_price, realized_pnl FROM positions
                WHERE portfolio_id = ? AND symbol = ?
            """, (portfolio_id, symbol))
            pos = cursor.fetchone()
            
            pnl = 0
            
            if side == 'BUY':
                if pos:
                    new_qty = pos['quantity'] + quantity
                    new_avg = ((pos['avg_price'] * pos['quantity']) + (price * quantity)) / new_qty
                    cursor.execute("""
                        UPDATE positions SET quantity = ?, avg_price = ?, updated_at = datetime('now')
                        WHERE portfolio_id = ? AND symbol = ?
                    """, (new_qty, new_avg, portfolio_id, symbol))
                else:
                    cursor.execute("""
                        INSERT INTO positions (portfolio_id, symbol, quantity, avg_price)
                        VALUES (?, ?, ?, ?)
                    """, (portfolio_id, symbol, quantity, price))
            
            elif side == 'SELL' and pos and pos['quantity'] >= quantity:
                pnl = (price - pos['avg_price']) * quantity
                new_qty = pos['quantity'] - quantity
                
                if new_qty > 0:
                    cursor.execute("""
                        UPDATE positions SET quantity = ?, realized_pnl = realized_pnl + ?,
                                           updated_at = datetime('now')
                        WHERE portfolio_id = ? AND symbol = ?
                    """, (new_qty, pnl, portfolio_id, symbol))
                else:
                    cursor.execute("""
                        UPDATE positions SET quantity = 0, realized_pnl = realized_pnl + ?,
                                           updated_at = datetime('now')
                        WHERE portfolio_id = ? AND symbol = ?
                    """, (pnl, portfolio_id, symbol))
            
            self.conn.commit()
            return pnl
        
        def get_position(self, portfolio_id: int, symbol: str) -> Optional[dict]:
            """Recupera posizione specifica."""
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT * FROM positions WHERE portfolio_id = ? AND symbol = ?
            """, (portfolio_id, symbol))
            row = cursor.fetchone()
            return dict(row) if row else None
        
        def get_all_positions(self, portfolio_id: int) -> List[dict]:
            """Recupera tutte le posizioni attive."""
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT * FROM positions WHERE portfolio_id = ? AND quantity > 0
            """, (portfolio_id,))
            return [dict(row) for row in cursor.fetchall()]
        
        # Market Data
        def save_candles(self, candles: List[tuple]):
            """Salva candele in batch."""
            cursor = self.conn.cursor()
            cursor.executemany("""
                INSERT OR REPLACE INTO candles 
                (symbol, timeframe, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, candles)
            self.conn.commit()
        
        def get_candles(self, symbol: str, timeframe: str, limit: int = 100) -> List[dict]:
            """Recupera ultime N candele."""
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT * FROM candles 
                WHERE symbol = ? AND timeframe = ?
                ORDER BY timestamp DESC LIMIT ?
            """, (symbol, timeframe, limit))
            return [dict(row) for row in cursor.fetchall()][::-1]  # Ordine cronologico
        
        def get_latest_price(self, symbol: str) -> Optional[float]:
            """Recupera ultimo prezzo."""
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT close FROM candles 
                WHERE symbol = ?
                ORDER BY timestamp DESC LIMIT 1
            """, (symbol,))
            row = cursor.fetchone()
            return row['close'] if row else None
        
        # Analytics
        def get_portfolio_summary(self, portfolio_id: int) -> dict:
            """Genera sommario portfolio."""
            cursor = self.conn.cursor()
            
            # Trade stats
            cursor.execute("""
                SELECT COUNT(*) as total_trades,
                       SUM(quantity * COALESCE(filled_price, price)) as total_volume,
                       SUM(fee) as total_fees
                FROM trades WHERE portfolio_id = ?
            """, (portfolio_id,))
            trades = cursor.fetchone()
            
            # Positions
            positions = self.get_all_positions(portfolio_id)
            
            # Realized PnL
            cursor.execute("""
                SELECT SUM(realized_pnl) as realized FROM positions WHERE portfolio_id = ?
            """, (portfolio_id,))
            realized = cursor.fetchone()['realized'] or 0
            
            return {
                'total_trades': trades['total_trades'] or 0,
                'total_volume': trades['total_volume'] or 0,
                'total_fees': trades['total_fees'] or 0,
                'open_positions': len(positions),
                'realized_pnl': realized,
                'positions': positions
            }
        
        def get_trade_history(self, portfolio_id: int, days: int = 30) -> List[dict]:
            """Recupera storico trade."""
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT * FROM trades 
                WHERE portfolio_id = ?
                  AND created_at > datetime('now', ?)
                ORDER BY created_at DESC
            """, (portfolio_id, f'-{days} days'))
            return [dict(row) for row in cursor.fetchall()]
    
    # Test della classe
    print("Inizializzazione TradingDatabase...")
    db = TradingDatabase()
    
    # Crea portfolio
    db.conn.execute("INSERT INTO portfolios (name) VALUES ('Test Bot')")
    db.conn.commit()
    
    print("\n1. Log trades:")
    t1 = db.log_trade(1, 'BTC', 'BUY', 0.1, 45000, strategy='RSI')
    t2 = db.log_trade(1, 'BTC', 'SELL', 0.05, 47000, strategy='RSI')
    print(f"   Trade IDs: {t1}, {t2}")
    
    print("\n2. Update positions:")
    db.update_position(1, 'BTC', 'BUY', 0.1, 45000)
    pnl = db.update_position(1, 'BTC', 'SELL', 0.05, 47000)
    print(f"   PnL dalla vendita: ${pnl:,.2f}")
    
    print("\n3. Get position:")
    pos = db.get_position(1, 'BTC')
    print(f"   BTC: {pos['quantity']} @ ${pos['avg_price']}")
    
    print("\n4. Save candles:")
    import time
    ts = int(time.time())
    candles = [
        ('BTC', '1h', ts - 3600, 44000, 44500, 43800, 44200, 1000000),
        ('BTC', '1h', ts, 44200, 45000, 44000, 44800, 1200000),
    ]
    db.save_candles(candles)
    print(f"   Salvate {len(candles)} candele")
    
    print("\n5. Portfolio summary:")
    summary = db.get_portfolio_summary(1)
    for k, v in summary.items():
        if k != 'positions':
            print(f"   {k}: {v}")
    
    db.close()
    print("\n✅ TradingDatabase completo!")
    return True

# exercise_20_solution()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Esegue tutti gli esercizi."""
    print("\n" + "═" * 70)
    print("ESECUZIONE ESERCIZI DATABASE")
    print("═" * 70)
    
    exercises = [
        ("Esercizio 1: Connessione e SELECT", exercise_1_solution),
        ("Esercizio 2: INSERT con Parametri", exercise_2_solution),
        ("Esercizio 3: SELECT con WHERE", exercise_3_solution),
        ("Esercizio 4: UPDATE e DELETE", exercise_4_solution),
        ("Esercizio 5: ORDER BY e LIMIT", exercise_5_solution),
        ("Esercizio 6: Aggregazioni", exercise_6_solution),
        ("Esercizio 7: GROUP BY e HAVING", exercise_7_solution),
        ("Esercizio 8: JOIN", exercise_8_solution),
        ("Esercizio 9: Subquery", exercise_9_solution),
        ("Esercizio 10: Transazioni", exercise_10_solution),
        ("Esercizio 16: Trade Logger", exercise_16_solution),
        ("Esercizio 17: Portfolio Manager", exercise_17_solution),
        ("Esercizio 18: OHLCV Manager", exercise_18_solution),
        ("Esercizio 19: Performance Analyzer", exercise_19_solution),
        ("Esercizio 20: Trading Database", exercise_20_solution),
    ]
    
    for name, func in exercises:
        print(f"\n{'─' * 60}")
        print(f"▶ {name}")
        print('─' * 60)
        try:
            func()
            print(f"✅ {name} completato")
        except Exception as e:
            print(f"❌ {name} errore: {e}")
    
    print("\n" + "═" * 70)
    print("🎉 TUTTI GLI ESERCIZI COMPLETATI!")
    print("═" * 70)


if __name__ == "__main__":
    main()
