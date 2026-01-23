"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MODULE: DATABASE PER PYTHON - PARTE 1                      ║
║                        SQLite & SQL Fondamentale                              ║
║                                                                               ║
║  Versione: 2025.1                                                             ║
║  Prerequisiti: Funzioni, Dizionari, Context Managers (base)                   ║
║  Posizione nel corso: Dopo Giorno 45 (dopo OOP basics)                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

INDICE:
═══════
1. Introduzione ai Database
2. SQLite: Il Database Integrato in Python
3. SQL Fondamentale: CRUD Operations
4. Query Avanzate: JOIN, GROUP BY, Subquery
5. Indici e Ottimizzazione
6. Transazioni e ACID
7. Pattern per Trading Bot
8. Best Practices e Errori Comuni

"""

import sqlite3
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from contextlib import contextmanager
import os

print("=" * 70)
print("MODULE: DATABASE PER PYTHON - PARTE 1")
print("SQLite & SQL Fondamentale")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 1: INTRODUZIONE AI DATABASE
# ══════════════════════════════════════════════════════════════════════════════

def section1_intro():
    """
    Cos'è un Database e Perché Ti Serve
    ────────────────────────────────────
    
    Un database è un sistema organizzato per memorizzare, gestire e recuperare
    dati in modo efficiente. A differenza dei file (JSON, CSV), un database:
    
    ✅ Gestisce milioni di record efficientemente
    ✅ Permette query complesse (cerca, filtra, aggrega)
    ✅ Garantisce integrità dei dati (transazioni ACID)
    ✅ Supporta accesso concorrente (più processi insieme)
    ✅ Persiste i dati in modo sicuro
    
    PER IL TRADING BOT:
    ───────────────────
    - Storico candele OHLCV (migliaia/milioni di record)
    - Log di tutti i trade eseguiti
    - Stato del portfolio nel tempo
    - Backtesting su dati storici
    - Audit trail per compliance
    
    TIPI DI DATABASE:
    ─────────────────
    
    ┌─────────────────┬────────────────────┬─────────────────────────────┐
    │ Tipo            │ Esempi             │ Uso                         │
    ├─────────────────┼────────────────────┼─────────────────────────────┤
    │ Relazionale     │ SQLite, PostgreSQL │ Dati strutturati, relazioni │
    │ Document        │ MongoDB            │ Dati flessibili, JSON-like  │
    │ Time-Series     │ TimescaleDB        │ Dati temporali, metriche    │
    │ Key-Value       │ Redis              │ Cache, sessioni             │
    │ Graph           │ Neo4j              │ Relazioni complesse         │
    └─────────────────┴────────────────────┴─────────────────────────────┘
    
    PER QUESTO CORSO:
    ─────────────────
    SQLite → Sviluppo locale, bot singolo, backtesting
    PostgreSQL → Produzione, multi-utente, alta disponibilità
    
    """
    print("\n" + "═" * 60)
    print("📚 SEZIONE 1: INTRODUZIONE AI DATABASE")
    print("═" * 60)
    
    # Esempio: Perché non basta un dizionario?
    print("\n>>> Esempio: Dizionario vs Database")
    
    # Con dizionario - problemi evidenti
    trades_dict = {
        "trade_1": {"symbol": "BTC", "price": 45000, "qty": 0.1},
        "trade_2": {"symbol": "ETH", "price": 3000, "qty": 1.5},
        # Se il programma crasha, questi dati SPARISCONO!
    }
    
    # Per filtrare devi iterare TUTTO
    btc_trades = [t for t in trades_dict.values() if t["symbol"] == "BTC"]
    print(f"   Dizionario: {len(btc_trades)} trade BTC (scan completo!)")
    
    # Con database - vantaggi
    print("\n   Database vantaggi:")
    print("   ✅ SELECT * FROM trades WHERE symbol='BTC' -- Usa indice!")
    print("   ✅ Dati persistenti su disco")
    print("   ✅ Query complesse in millisecondi")
    print("   ✅ Transazioni atomiche")
    
    return True


# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 2: SQLITE BASICS
# ══════════════════════════════════════════════════════════════════════════════

def section2_sqlite_basics():
    """
    SQLite: Il Database Embedded di Python
    ───────────────────────────────────────
    
    SQLite è INCLUSO in Python (modulo sqlite3), non serve installare nulla!
    È un database "embedded" - il database è un singolo file sul disco.
    
    VANTAGGI SQLITE:
    ✅ Zero configurazione
    ✅ Database = un file .db
    ✅ Perfetto per sviluppo e test
    ✅ Fino a 281 TB di dati (teorico)
    ✅ Letture molto veloci
    
    LIMITAZIONI:
    ❌ Una sola connessione write alla volta
    ❌ Niente connessioni di rete (locale only)
    ❌ Non ideale per alta concorrenza
    
    """
    print("\n" + "═" * 60)
    print("📚 SEZIONE 2: SQLITE BASICS")
    print("═" * 60)
    
    # 2.1 Connessione Base
    print("\n>>> 2.1 Connessione Base")
    
    # Connessione a database in memoria (per test)
    conn_memory = sqlite3.connect(":memory:")
    print(f"   Connesso a database in memoria: {conn_memory}")
    
    # Connessione a file (persistente)
    db_path = "trading_example.db"
    conn_file = sqlite3.connect(db_path)
    print(f"   Connesso a database file: {db_path}")
    
    # SEMPRE chiudere la connessione!
    conn_memory.close()
    conn_file.close()
    
    # 2.2 Pattern Context Manager (RACCOMANDATO)
    print("\n>>> 2.2 Context Manager Pattern")
    
    # Questo pattern garantisce che la connessione sia chiusa
    with sqlite3.connect(":memory:") as conn:
        print("   Dentro il context manager - connessione aperta")
        # Esegui operazioni qui
    print("   Fuori dal context manager - connessione chiusa automaticamente")
    
    # 2.3 Cursor Object
    print("\n>>> 2.3 Cursor Object")
    
    with sqlite3.connect(":memory:") as conn:
        # Il cursor esegue le query
        cursor = conn.cursor()
        print(f"   Cursor creato: {cursor}")
        
        # Ogni query passa attraverso il cursor
        cursor.execute("SELECT sqlite_version()")
        version = cursor.fetchone()[0]
        print(f"   SQLite versione: {version}")
    
    # 2.4 PRAGMA - Configurazione SQLite
    print("\n>>> 2.4 PRAGMA Configurazioni")
    
    with sqlite3.connect(":memory:") as conn:
        cursor = conn.cursor()
        
        # PRAGMA più importanti per trading
        pragmas = [
            ("journal_mode", "WAL"),      # Write-Ahead Logging: letture + scritture concurrent
            ("synchronous", "NORMAL"),     # Bilancio sicurezza/velocità
            ("cache_size", "-64000"),      # 64MB di cache
            ("temp_store", "MEMORY"),      # Temp tables in RAM
        ]
        
        for pragma, value in pragmas:
            cursor.execute(f"PRAGMA {pragma} = {value}")
            cursor.execute(f"PRAGMA {pragma}")
            actual = cursor.fetchone()[0]
            print(f"   {pragma}: {actual}")
    
    # 2.5 Row Factory
    print("\n>>> 2.5 Row Factory per accesso ai dati")
    
    with sqlite3.connect(":memory:") as conn:
        # Default: tuple
        cursor = conn.cursor()
        cursor.execute("SELECT 1 as id, 'BTC' as symbol")
        row = cursor.fetchone()
        print(f"   Tuple default: {row} - accesso: row[0]={row[0]}")
        
        # Con Row factory: accesso per nome!
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT 1 as id, 'BTC' as symbol")
        row = cursor.fetchone()
        print(f"   Row factory: {dict(row)} - accesso: row['symbol']={row['symbol']}")
    
    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)
    
    return True


# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 3: CRUD OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

def section3_crud_operations():
    """
    CRUD: Create, Read, Update, Delete
    ───────────────────────────────────
    
    Le 4 operazioni fondamentali su qualsiasi database:
    
    CREATE → INSERT INTO
    READ   → SELECT
    UPDATE → UPDATE ... SET
    DELETE → DELETE FROM
    
    """
    print("\n" + "═" * 60)
    print("📚 SEZIONE 3: CRUD OPERATIONS")
    print("═" * 60)
    
    # Setup database di test
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # 3.1 CREATE TABLE
    print("\n>>> 3.1 CREATE TABLE")
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL CHECK(side IN ('BUY', 'SELL')),
            quantity REAL NOT NULL,
            price REAL NOT NULL,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            notes TEXT
        )
    """)
    print("   ✅ Tabella 'trades' creata")
    
    # Spiegazione colonne:
    # - PRIMARY KEY: identificatore unico
    # - AUTOINCREMENT: SQLite genera ID automaticamente
    # - NOT NULL: campo obbligatorio
    # - CHECK: constraint di validazione
    # - DEFAULT: valore di default
    
    # 3.2 INSERT - Inserimento dati
    print("\n>>> 3.2 INSERT - Inserimento dati")
    
    # Singolo insert
    cursor.execute("""
        INSERT INTO trades (symbol, side, quantity, price, notes)
        VALUES ('BTC', 'BUY', 0.1, 45000, 'Primo acquisto')
    """)
    print(f"   ✅ Inserito trade, ID: {cursor.lastrowid}")
    
    # Insert con parametri (SICURO contro SQL injection!)
    trade_data = ('ETH', 'BUY', 1.5, 3000, 'Accumulo')
    cursor.execute("""
        INSERT INTO trades (symbol, side, quantity, price, notes)
        VALUES (?, ?, ?, ?, ?)
    """, trade_data)
    print(f"   ✅ Inserito trade ETH, ID: {cursor.lastrowid}")
    
    # Insert multiplo con executemany
    many_trades = [
        ('BTC', 'SELL', 0.05, 47000, 'Presa profitto'),
        ('SOL', 'BUY', 10, 100, 'Nuova posizione'),
        ('ETH', 'SELL', 0.5, 3200, 'Presa profitto parziale'),
    ]
    cursor.executemany("""
        INSERT INTO trades (symbol, side, quantity, price, notes)
        VALUES (?, ?, ?, ?, ?)
    """, many_trades)
    conn.commit()  # IMPORTANTE: salva le modifiche!
    print(f"   ✅ Inseriti {len(many_trades)} trade con executemany")
    
    # 3.3 SELECT - Lettura dati
    print("\n>>> 3.3 SELECT - Lettura dati")
    
    # Select tutti i record
    cursor.execute("SELECT * FROM trades")
    all_trades = cursor.fetchall()
    print(f"   Tutti i trade ({len(all_trades)} totali):")
    for t in all_trades:
        print(f"      #{t['id']}: {t['side']} {t['quantity']} {t['symbol']} @ ${t['price']}")
    
    # Select con WHERE
    cursor.execute("SELECT * FROM trades WHERE symbol = ?", ('BTC',))
    btc_trades = cursor.fetchall()
    print(f"\n   Trade BTC: {len(btc_trades)}")
    
    # Select colonne specifiche
    cursor.execute("SELECT symbol, SUM(quantity) as total_qty FROM trades GROUP BY symbol")
    for row in cursor.fetchall():
        print(f"   {row['symbol']}: {row['total_qty']} totali")
    
    # 3.4 UPDATE - Modifica dati
    print("\n>>> 3.4 UPDATE - Modifica dati")
    
    cursor.execute("""
        UPDATE trades 
        SET notes = 'AGGIORNATO: ' || notes 
        WHERE id = 1
    """)
    conn.commit()
    print(f"   ✅ Modificate {cursor.rowcount} righe")
    
    # Verifica
    cursor.execute("SELECT notes FROM trades WHERE id = 1")
    print(f"   Nuove note: {cursor.fetchone()['notes']}")
    
    # 3.5 DELETE - Eliminazione dati
    print("\n>>> 3.5 DELETE - Eliminazione dati")
    
    # Prima conta
    cursor.execute("SELECT COUNT(*) FROM trades")
    before = cursor.fetchone()[0]
    
    # Elimina
    cursor.execute("DELETE FROM trades WHERE symbol = 'SOL'")
    conn.commit()
    
    # Dopo conta
    cursor.execute("SELECT COUNT(*) FROM trades")
    after = cursor.fetchone()[0]
    print(f"   Eliminati {before - after} trade SOL ({before} → {after})")
    
    conn.close()
    return True


# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 4: QUERY AVANZATE
# ══════════════════════════════════════════════════════════════════════════════

def section4_advanced_queries():
    """
    Query Avanzate: JOIN, GROUP BY, Subquery
    ─────────────────────────────────────────
    
    Le query avanzate permettono di:
    - Combinare dati da più tabelle (JOIN)
    - Aggregare dati (GROUP BY, SUM, AVG, COUNT)
    - Filtrare aggregazioni (HAVING)
    - Query annidate (Subquery)
    
    """
    print("\n" + "═" * 60)
    print("📚 SEZIONE 4: QUERY AVANZATE")
    print("═" * 60)
    
    # Setup con dati più complessi
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Crea schema multi-tabella
    cursor.executescript("""
        -- Tabella asset
        CREATE TABLE assets (
            id INTEGER PRIMARY KEY,
            symbol TEXT UNIQUE NOT NULL,
            name TEXT,
            category TEXT
        );
        
        -- Tabella portfolio
        CREATE TABLE portfolios (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Tabella trades
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY,
            portfolio_id INTEGER,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity REAL NOT NULL,
            price REAL NOT NULL,
            fee REAL DEFAULT 0,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
        );
        
        -- Dati di esempio
        INSERT INTO assets VALUES 
            (1, 'BTC', 'Bitcoin', 'crypto'),
            (2, 'ETH', 'Ethereum', 'crypto'),
            (3, 'SOL', 'Solana', 'crypto'),
            (4, 'AAPL', 'Apple Inc', 'stock');
        
        INSERT INTO portfolios VALUES 
            (1, 'Trading Bot', '2024-01-01'),
            (2, 'Long Term', '2024-01-01');
        
        INSERT INTO trades (portfolio_id, symbol, side, quantity, price, fee, timestamp) VALUES 
            (1, 'BTC', 'BUY', 0.1, 42000, 4.2, '2024-01-05 10:00'),
            (1, 'BTC', 'BUY', 0.05, 44000, 2.2, '2024-01-10 14:00'),
            (1, 'BTC', 'SELL', 0.08, 46000, 3.68, '2024-01-15 09:00'),
            (1, 'ETH', 'BUY', 1, 2500, 2.5, '2024-01-07 11:00'),
            (1, 'ETH', 'BUY', 0.5, 2600, 1.3, '2024-01-12 16:00'),
            (2, 'BTC', 'BUY', 0.5, 43000, 21.5, '2024-01-02 08:00'),
            (2, 'SOL', 'BUY', 100, 90, 9, '2024-01-08 12:00');
    """)
    conn.commit()
    
    # 4.1 JOIN - Combinare tabelle
    print("\n>>> 4.1 JOIN - Combinare tabelle")
    
    # INNER JOIN: solo record che matchano in entrambe
    cursor.execute("""
        SELECT t.*, a.name as asset_name, a.category
        FROM trades t
        INNER JOIN assets a ON t.symbol = a.symbol
        WHERE t.portfolio_id = 1
        ORDER BY t.timestamp
    """)
    print("   INNER JOIN (trades + asset info):")
    for row in cursor.fetchall():
        print(f"      {row['timestamp'][:10]}: {row['side']} {row['quantity']} "
              f"{row['asset_name']} ({row['category']})")
    
    # LEFT JOIN: tutti i record a sinistra, anche senza match
    cursor.execute("""
        SELECT a.symbol, a.name, COUNT(t.id) as trade_count
        FROM assets a
        LEFT JOIN trades t ON a.symbol = t.symbol
        GROUP BY a.symbol
    """)
    print("\n   LEFT JOIN (tutti gli asset con count trade):")
    for row in cursor.fetchall():
        print(f"      {row['symbol']}: {row['trade_count']} trade")
    
    # 4.2 GROUP BY - Aggregazioni
    print("\n>>> 4.2 GROUP BY - Aggregazioni")
    
    cursor.execute("""
        SELECT 
            symbol,
            COUNT(*) as num_trades,
            SUM(CASE WHEN side='BUY' THEN quantity ELSE -quantity END) as net_qty,
            ROUND(AVG(price), 2) as avg_price,
            ROUND(SUM(fee), 2) as total_fees
        FROM trades
        WHERE portfolio_id = 1
        GROUP BY symbol
    """)
    print("   Statistiche per asset (Portfolio 1):")
    for row in cursor.fetchall():
        print(f"      {row['symbol']}: {row['num_trades']} trades, "
              f"net: {row['net_qty']}, avg: ${row['avg_price']}, fees: ${row['total_fees']}")
    
    # 4.3 HAVING - Filtrare aggregazioni
    print("\n>>> 4.3 HAVING - Filtrare aggregazioni")
    
    cursor.execute("""
        SELECT symbol, COUNT(*) as trade_count
        FROM trades
        GROUP BY symbol
        HAVING trade_count >= 2
    """)
    print("   Asset con almeno 2 trade:")
    for row in cursor.fetchall():
        print(f"      {row['symbol']}: {row['trade_count']} trade")
    
    # 4.4 Subquery
    print("\n>>> 4.4 Subquery")
    
    # Subquery nel WHERE
    cursor.execute("""
        SELECT * FROM trades
        WHERE price > (SELECT AVG(price) FROM trades WHERE symbol = 'BTC')
        AND symbol = 'BTC'
    """)
    print("   Trade BTC sopra il prezzo medio:")
    for row in cursor.fetchall():
        print(f"      {row['side']} @ ${row['price']}")
    
    # Subquery come tabella derivata
    cursor.execute("""
        SELECT p.name, stats.total_volume
        FROM portfolios p
        INNER JOIN (
            SELECT portfolio_id, SUM(quantity * price) as total_volume
            FROM trades
            GROUP BY portfolio_id
        ) stats ON p.id = stats.portfolio_id
    """)
    print("\n   Volume totale per portfolio:")
    for row in cursor.fetchall():
        print(f"      {row['name']}: ${row['total_volume']:,.2f}")
    
    # 4.5 Window Functions (SQLite 3.25+)
    print("\n>>> 4.5 Window Functions")
    
    cursor.execute("""
        SELECT 
            timestamp,
            symbol,
            side,
            price,
            AVG(price) OVER (PARTITION BY symbol ORDER BY timestamp 
                            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) as moving_avg
        FROM trades
        WHERE portfolio_id = 1
        ORDER BY timestamp
    """)
    print("   Moving Average (3 periodi) per trade:")
    for row in cursor.fetchall():
        ma = row['moving_avg']
        print(f"      {row['timestamp'][:10]} {row['symbol']}: ${row['price']} "
              f"(MA3: ${ma:.2f})" if ma else "")
    
    conn.close()
    return True


# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 5: INDICI E OTTIMIZZAZIONE
# ══════════════════════════════════════════════════════════════════════════════

def section5_indexes():
    """
    Indici: Velocizzare le Query
    ────────────────────────────
    
    Un indice è una struttura dati che velocizza le ricerche.
    Pensa a un indice di un libro: invece di leggere ogni pagina,
    vai direttamente alla pagina giusta.
    
    REGOLA: Crea indici sulle colonne usate in WHERE, JOIN, ORDER BY
    
    """
    print("\n" + "═" * 60)
    print("📚 SEZIONE 5: INDICI E OTTIMIZZAZIONE")
    print("═" * 60)
    
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    
    # Crea tabella con molti dati
    cursor.execute("""
        CREATE TABLE candles (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            open REAL, high REAL, low REAL, close REAL,
            volume REAL
        )
    """)
    
    # Inserisci dati di test
    import random
    data = []
    symbols = ['BTC', 'ETH', 'SOL']
    base_ts = 1704067200  # 2024-01-01
    for symbol in symbols:
        for i in range(10000):
            data.append((
                symbol, '1h', base_ts + i * 3600,
                random.uniform(40000, 50000),
                random.uniform(40000, 50000),
                random.uniform(40000, 50000),
                random.uniform(40000, 50000),
                random.uniform(1000000, 5000000)
            ))
    
    cursor.executemany("""
        INSERT INTO candles (symbol, timeframe, timestamp, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, data)
    conn.commit()
    
    print(f"\n>>> Inseriti {len(data)} record")
    
    # 5.1 Query senza indice
    print("\n>>> 5.1 Query SENZA indice")
    
    # EXPLAIN QUERY PLAN mostra come SQLite esegue la query
    cursor.execute("""
        EXPLAIN QUERY PLAN
        SELECT * FROM candles 
        WHERE symbol = 'BTC' AND timestamp > 1704200000
        ORDER BY timestamp
    """)
    print("   Plan: " + cursor.fetchone()[3])  # "SCAN" = legge tutto!
    
    # 5.2 Creazione indice
    print("\n>>> 5.2 Creazione INDICE")
    
    cursor.execute("""
        CREATE INDEX idx_candles_lookup 
        ON candles(symbol, timeframe, timestamp)
    """)
    print("   ✅ Creato indice composito su (symbol, timeframe, timestamp)")
    
    # 5.3 Query con indice
    print("\n>>> 5.3 Query CON indice")
    
    cursor.execute("""
        EXPLAIN QUERY PLAN
        SELECT * FROM candles 
        WHERE symbol = 'BTC' AND timestamp > 1704200000
        ORDER BY timestamp
    """)
    plan = cursor.fetchone()[3]
    print("   Plan: " + plan)  # Ora usa SEARCH e INDEX!
    
    # 5.4 Regole per indici efficaci
    print("\n>>> 5.4 Regole per indici efficaci")
    
    rules = """
    ✅ CREA indici su colonne nel WHERE frequente
    ✅ CREA indici su colonne JOIN
    ✅ CREA indici su colonne ORDER BY
    ✅ INDICE COMPOSITO: ordine colonne = ordine query
    
    ❌ NON creare troppi indici (rallentano INSERT/UPDATE)
    ❌ NON creare indici su colonne con pochi valori unici
    ❌ NON indicizzare piccole tabelle (<1000 righe)
    """
    print(rules)
    
    # 5.5 ANALYZE per statistiche
    print(">>> 5.5 ANALYZE")
    cursor.execute("ANALYZE")
    print("   ✅ ANALYZE completato - SQLite ottimizzerà le query")
    
    conn.close()
    return True


# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 6: TRANSAZIONI E ACID
# ══════════════════════════════════════════════════════════════════════════════

def section6_transactions():
    """
    Transazioni: Operazioni Atomiche
    ─────────────────────────────────
    
    Una transazione è un gruppo di operazioni che devono:
    - Completare TUTTE o NESSUNA (Atomic)
    - Lasciare il database in stato valido (Consistent)
    - Essere isolate da altre transazioni (Isolated)
    - Persistere anche dopo crash (Durable)
    
    = ACID
    
    ESEMPIO TRADING:
    Se trasferisci crypto tra 2 wallet:
    1. Sottrai da wallet A
    2. Aggiungi a wallet B
    
    Se step 2 fallisce dopo step 1, hai PERSO crypto!
    Con transazione: o entrambi o nessuno.
    
    """
    print("\n" + "═" * 60)
    print("📚 SEZIONE 6: TRANSAZIONI E ACID")
    print("═" * 60)
    
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Setup
    cursor.executescript("""
        CREATE TABLE wallets (
            id INTEGER PRIMARY KEY,
            name TEXT,
            balance REAL DEFAULT 0 CHECK(balance >= 0)
        );
        INSERT INTO wallets (name, balance) VALUES ('Hot Wallet', 10000);
        INSERT INTO wallets (name, balance) VALUES ('Cold Wallet', 50000);
    """)
    conn.commit()
    
    # 6.1 Transazione con commit
    print("\n>>> 6.1 Transazione con COMMIT")
    
    def show_balances():
        cursor.execute("SELECT name, balance FROM wallets")
        for w in cursor.fetchall():
            print(f"      {w['name']}: ${w['balance']:,.2f}")
    
    print("   Prima:")
    show_balances()
    
    # Trasferimento atomico
    amount = 5000
    try:
        cursor.execute("UPDATE wallets SET balance = balance - ? WHERE name = 'Hot Wallet'", (amount,))
        cursor.execute("UPDATE wallets SET balance = balance + ? WHERE name = 'Cold Wallet'", (amount,))
        conn.commit()  # COMMIT: rende permanenti le modifiche
        print(f"\n   ✅ Trasferiti ${amount}")
    except Exception as e:
        conn.rollback()  # ROLLBACK: annulla tutto
        print(f"   ❌ Errore: {e}")
    
    print("\n   Dopo:")
    show_balances()
    
    # 6.2 Transazione con rollback
    print("\n>>> 6.2 Transazione con ROLLBACK")
    
    print("   Prima:")
    show_balances()
    
    # Tentativo di trasferimento che viola il CHECK constraint
    amount = 100000  # Più del saldo!
    try:
        cursor.execute("BEGIN TRANSACTION")
        cursor.execute("UPDATE wallets SET balance = balance - ? WHERE name = 'Hot Wallet'", (amount,))
        cursor.execute("UPDATE wallets SET balance = balance + ? WHERE name = 'Cold Wallet'", (amount,))
        conn.commit()
    except sqlite3.IntegrityError as e:
        conn.rollback()
        print(f"   ❌ ROLLBACK: {e}")
    
    print("\n   Dopo (invariato!):")
    show_balances()
    
    # 6.3 Context manager per transazioni
    print("\n>>> 6.3 Context Manager per transazioni")
    
    @contextmanager
    def transaction(connection):
        """Context manager che gestisce commit/rollback automaticamente."""
        try:
            yield connection.cursor()
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    
    # Uso pulito
    try:
        with transaction(conn) as cur:
            cur.execute("UPDATE wallets SET balance = balance + 1000 WHERE name = 'Hot Wallet'")
            print("   ✅ Transazione completata con context manager")
    except Exception as e:
        print(f"   ❌ Errore: {e}")
    
    show_balances()
    
    # 6.4 Isolation levels
    print("\n>>> 6.4 Isolation Levels")
    print("""
    SQLite supporta diversi livelli di isolamento:
    
    DEFERRED (default): Lock acquisito alla prima write
    IMMEDIATE: Lock acquisito subito (altre conn possono leggere)
    EXCLUSIVE: Lock esclusivo (nessun'altra operazione)
    
    Per trading bot: IMMEDIATE è spesso il migliore
    """)
    
    conn.close()
    return True


# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 7: PATTERN PER TRADING BOT
# ══════════════════════════════════════════════════════════════════════════════

def section7_trading_patterns():
    """
    Pattern Database per Trading Bot
    ─────────────────────────────────
    
    Pattern specifici che userai nel tuo bot:
    1. Schema OHLCV ottimizzato
    2. Trade logging con audit trail
    3. Position tracking
    4. Performance metrics
    
    """
    print("\n" + "═" * 60)
    print("📚 SEZIONE 7: PATTERN PER TRADING BOT")
    print("═" * 60)
    
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # 7.1 Schema OHLCV Ottimizzato
    print("\n>>> 7.1 Schema OHLCV Ottimizzato")
    
    cursor.executescript("""
        -- Prezzi come INTEGER per evitare floating point errors
        -- Moltiplica per 10^8 (satoshi style)
        
        CREATE TABLE candles (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            timestamp INTEGER NOT NULL,  -- Unix timestamp, non TEXT!
            open INTEGER NOT NULL,       -- price * 100_000_000
            high INTEGER NOT NULL,
            low INTEGER NOT NULL,
            close INTEGER NOT NULL,
            volume INTEGER NOT NULL,
            is_complete INTEGER DEFAULT 1,
            
            UNIQUE(symbol, timeframe, timestamp)
        );
        
        -- Indice per lookup rapido
        CREATE INDEX idx_candles_lookup ON candles(symbol, timeframe, timestamp);
    """)
    print("   ✅ Tabella candles creata con prezzi INTEGER")
    
    PRICE_SCALE = 100_000_000  # 8 decimali
    
    def store_price(price: float) -> int:
        """Converti prezzo float a integer scalato."""
        return int(price * PRICE_SCALE)
    
    def read_price(scaled: int) -> float:
        """Converti integer scalato a prezzo float."""
        return scaled / PRICE_SCALE
    
    # Inserisci candela
    cursor.execute("""
        INSERT INTO candles (symbol, timeframe, timestamp, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, ('BTC', '1h', 1704067200, 
          store_price(42150.50), store_price(42300.00),
          store_price(42100.00), store_price(42250.75),
          store_price(1500000)))
    
    # Leggi candela
    cursor.execute("SELECT * FROM candles WHERE symbol = 'BTC'")
    row = cursor.fetchone()
    print(f"   BTC: O={read_price(row['open'])}, H={read_price(row['high'])}, "
          f"L={read_price(row['low'])}, C={read_price(row['close'])}")
    
    # 7.2 Trade Logging con Audit Trail
    print("\n>>> 7.2 Trade Logging con Audit Trail")
    
    cursor.executescript("""
        CREATE TABLE trade_log (
            id INTEGER PRIMARY KEY,
            order_id TEXT UNIQUE,        -- ID dall'exchange
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            order_type TEXT NOT NULL,    -- MARKET, LIMIT
            quantity REAL NOT NULL,
            price REAL,                  -- NULL per MARKET
            filled_price REAL,           -- Prezzo effettivo
            filled_qty REAL,
            fee REAL,
            fee_currency TEXT,
            status TEXT DEFAULT 'PENDING',
            strategy TEXT,               -- Quale strategia ha generato
            signal_data TEXT,            -- JSON con dati del segnale
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT,
            executed_at TEXT
        );
        
        CREATE INDEX idx_trades_symbol ON trade_log(symbol);
        CREATE INDEX idx_trades_status ON trade_log(status);
        CREATE INDEX idx_trades_strategy ON trade_log(strategy);
    """)
    print("   ✅ Tabella trade_log con audit trail")
    
    # 7.3 Position Tracking
    print("\n>>> 7.3 Position Tracking")
    
    cursor.executescript("""
        CREATE TABLE positions (
            id INTEGER PRIMARY KEY,
            symbol TEXT UNIQUE NOT NULL,
            quantity REAL NOT NULL DEFAULT 0,
            avg_entry_price REAL,
            unrealized_pnl REAL,
            realized_pnl REAL DEFAULT 0,
            last_updated TEXT DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Vista per posizioni attive
        CREATE VIEW active_positions AS
        SELECT * FROM positions WHERE quantity != 0;
    """)
    print("   ✅ Tabella positions con vista active_positions")
    
    # 7.4 Funzione per aggiornare posizione dopo trade
    print("\n>>> 7.4 Funzione Update Position")
    
    def update_position_after_trade(cursor, symbol: str, side: str, 
                                   qty: float, price: float):
        """
        Aggiorna posizione dopo un trade.
        Calcola nuovo avg entry price e PnL realizzato.
        """
        cursor.execute("SELECT * FROM positions WHERE symbol = ?", (symbol,))
        pos = cursor.fetchone()
        
        if pos is None:
            # Nuova posizione
            if side == 'BUY':
                cursor.execute("""
                    INSERT INTO positions (symbol, quantity, avg_entry_price)
                    VALUES (?, ?, ?)
                """, (symbol, qty, price))
        else:
            current_qty = pos['quantity']
            current_avg = pos['avg_entry_price'] or 0
            realized_pnl = pos['realized_pnl'] or 0
            
            if side == 'BUY':
                # Aumenta posizione: calcola nuovo avg
                new_qty = current_qty + qty
                if new_qty != 0:
                    new_avg = ((current_avg * current_qty) + (price * qty)) / new_qty
                else:
                    new_avg = 0
                cursor.execute("""
                    UPDATE positions 
                    SET quantity = ?, avg_entry_price = ?, last_updated = CURRENT_TIMESTAMP
                    WHERE symbol = ?
                """, (new_qty, new_avg, symbol))
            else:  # SELL
                # Riduce posizione: calcola PnL realizzato
                sell_qty = min(qty, current_qty)  # Non vendere più di quanto hai
                pnl = (price - current_avg) * sell_qty
                new_qty = current_qty - sell_qty
                
                cursor.execute("""
                    UPDATE positions 
                    SET quantity = ?, realized_pnl = ?, last_updated = CURRENT_TIMESTAMP
                    WHERE symbol = ?
                """, (new_qty, realized_pnl + pnl, symbol))
                
                return pnl
        return 0
    
    # Simula alcuni trade
    trades = [
        ('BTC', 'BUY', 0.1, 42000),
        ('BTC', 'BUY', 0.05, 44000),
        ('BTC', 'SELL', 0.08, 46000),
    ]
    
    for symbol, side, qty, price in trades:
        pnl = update_position_after_trade(cursor, symbol, side, qty, price)
        if pnl:
            print(f"   {side} {qty} {symbol} @ ${price} → PnL: ${pnl:.2f}")
        else:
            print(f"   {side} {qty} {symbol} @ ${price}")
    conn.commit()
    
    # Mostra posizione finale
    cursor.execute("SELECT * FROM positions WHERE symbol = 'BTC'")
    pos = cursor.fetchone()
    print(f"\n   Posizione finale BTC:")
    print(f"      Quantity: {pos['quantity']}")
    print(f"      Avg Entry: ${pos['avg_entry_price']:.2f}")
    print(f"      Realized PnL: ${pos['realized_pnl']:.2f}")
    
    conn.close()
    return True


# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 8: BEST PRACTICES E ERRORI COMUNI
# ══════════════════════════════════════════════════════════════════════════════

def section8_best_practices():
    """
    Best Practices e Errori da Evitare
    ───────────────────────────────────
    
    """
    print("\n" + "═" * 60)
    print("📚 SEZIONE 8: BEST PRACTICES")
    print("═" * 60)
    
    print("""
    ✅ BEST PRACTICES:
    ──────────────────
    
    1. USA SEMPRE PARAMETRI (?) PER I VALORI
       ❌ f"SELECT * FROM users WHERE name = '{name}'"  # SQL INJECTION!
       ✅ cursor.execute("SELECT * FROM users WHERE name = ?", (name,))
    
    2. USA CONTEXT MANAGERS
       ❌ conn = sqlite3.connect(db); ... ; conn.close()
       ✅ with sqlite3.connect(db) as conn: ...
    
    3. USA row_factory = sqlite3.Row
       ❌ row[0], row[1], row[2]  # Cosa sono?
       ✅ row['symbol'], row['price'], row['quantity']
    
    4. COMMIT ESPLICITI PER WRITE
       cursor.execute("INSERT ...")
       conn.commit()  # Non dimenticare!
    
    5. INDICI SULLE COLONNE CERCATE
       CREATE INDEX idx_symbol ON trades(symbol)
    
    6. PRAGMA WAL PER CONCORRENZA
       PRAGMA journal_mode = WAL
    
    ❌ ERRORI COMUNI:
    ─────────────────
    
    1. SQL INJECTION
       Mai concatenare stringhe nelle query!
    
    2. DIMENTICARE COMMIT
       Le modifiche non persistono senza commit()
    
    3. CONNESSIONI NON CHIUSE
       Memory leak e file lock
    
    4. FLOAT PER PREZZI
       0.1 + 0.2 != 0.3 in floating point!
       Usa INTEGER scalati o Decimal
    
    5. TIMESTAMP COME STRINGA
       ❌ "2024-01-15 10:30:00" → difficile da ordinare/filtrare
       ✅ Unix timestamp INTEGER → veloce, comparabile
    
    6. DELETE/UPDATE SENZA WHERE
       DELETE FROM trades;  # ⚠️ ELIMINA TUTTO!
       Sempre: DELETE FROM trades WHERE id = ?
    """)
    
    return True


# ══════════════════════════════════════════════════════════════════════════════
# MAIN - Esecuzione
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Esegue tutte le sezioni del modulo."""
    
    sections = [
        ("Sezione 1: Introduzione", section1_intro),
        ("Sezione 2: SQLite Basics", section2_sqlite_basics),
        ("Sezione 3: CRUD Operations", section3_crud_operations),
        ("Sezione 4: Query Avanzate", section4_advanced_queries),
        ("Sezione 5: Indici", section5_indexes),
        ("Sezione 6: Transazioni", section6_transactions),
        ("Sezione 7: Pattern Trading", section7_trading_patterns),
        ("Sezione 8: Best Practices", section8_best_practices),
    ]
    
    print("\n" + "═" * 70)
    print("ESECUZIONE MODULO DATABASE PARTE 1")
    print("═" * 70)
    
    for name, func in sections:
        try:
            func()
            print(f"\n✅ {name} completata")
        except Exception as e:
            print(f"\n❌ {name} errore: {e}")
    
    print("\n" + "═" * 70)
    print("🎉 MODULO DATABASE PARTE 1 COMPLETATO!")
    print("═" * 70)
    print("""
    HAI IMPARATO:
    ─────────────
    ✅ Cos'è un database e perché usarlo
    ✅ SQLite: connessione, PRAGMA, ottimizzazioni
    ✅ SQL CRUD: INSERT, SELECT, UPDATE, DELETE
    ✅ Query avanzate: JOIN, GROUP BY, subquery
    ✅ Indici per performance
    ✅ Transazioni ACID
    ✅ Pattern reali per trading bot
    ✅ Best practices e errori da evitare
    
    PROSSIMO: Parte 2 - PostgreSQL e SQLAlchemy ORM
    """)


if __name__ == "__main__":
    main()
