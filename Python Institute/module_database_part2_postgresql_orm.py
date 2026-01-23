"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MODULE: DATABASE PER PYTHON - PARTE 2                      ║
║                        PostgreSQL & SQLAlchemy ORM                            ║
║                                                                               ║
║  Versione: 2025.1                                                             ║
║  Prerequisiti: Parte 1 (SQLite), OOP, Context Managers                        ║
║  Posizione nel corso: Dopo Giorno 60 (o dopo Parte 1)                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

INDICE:
═══════
1. PostgreSQL: Introduzione e Differenze da SQLite
2. Connessione con psycopg (psycopg3)
3. SQLAlchemy Core: Query Programmatiche
4. SQLAlchemy ORM: Modelli e Relazioni
5. Pattern Avanzati: Repository, Unit of Work
6. Migrazioni con Alembic
7. Async Support (asyncpg + SQLAlchemy async)
8. TimescaleDB per Time-Series
9. Progetto Completo: Trading Database System

NOTA: Questa parte richiede PostgreSQL installato per eseguire gli esempi.
      Gli esempi con SQLAlchemy funzionano anche con SQLite per testing.

"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Optional, Any, Type, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import os

print("=" * 70)
print("MODULE: DATABASE PER PYTHON - PARTE 2")
print("PostgreSQL & SQLAlchemy ORM")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 1: POSTGRESQL - INTRODUZIONE
# ══════════════════════════════════════════════════════════════════════════════

def section1_postgresql_intro():
    """
    PostgreSQL vs SQLite: Quando Usare Cosa
    ────────────────────────────────────────
    
    PostgreSQL è un database relazionale enterprise-grade:
    ✅ Client-Server (può servire molti client contemporaneamente)
    ✅ Supporto ACID completo con MVCC
    ✅ Estensioni potenti (TimescaleDB, PostGIS, etc.)
    ✅ Tipi di dato avanzati (JSONB, Array, UUID, etc.)
    ✅ Full-text search integrato
    ✅ Replication e high availability
    
    QUANDO SCEGLIERE COSA:
    ──────────────────────
    
    ┌────────────────────────┬──────────────────────┬────────────────────────┐
    │ Criterio               │ SQLite               │ PostgreSQL             │
    ├────────────────────────┼──────────────────────┼────────────────────────┤
    │ Setup                  │ Zero (incluso)       │ Richiede installazione │
    │ Concorrenza            │ 1 writer             │ Molti writer           │
    │ Network access         │ No                   │ Sì                     │
    │ Dataset size           │ < 10GB ideale        │ TB scale               │
    │ Backup/Replica         │ Manuale              │ Built-in               │
    │ Estensioni             │ Limitate             │ Ricche                 │
    │ Produzione             │ Embedded apps        │ Web apps, API          │
    └────────────────────────┴──────────────────────┴────────────────────────┘
    
    PER IL TUO TRADING BOT:
    ───────────────────────
    - Sviluppo/Backtest → SQLite (semplice, veloce)
    - Produzione singolo bot → SQLite con WAL
    - Multi-bot/Dashboard → PostgreSQL
    - Time-series massive → PostgreSQL + TimescaleDB
    
    """
    print("\n" + "═" * 60)
    print("📚 SEZIONE 1: POSTGRESQL INTRODUZIONE")
    print("═" * 60)
    
    print("""
    DIFFERENZE SINTASSI SQL:
    ────────────────────────
    
    SQLite                          PostgreSQL
    ───────────────────────────────────────────────────────
    INTEGER PRIMARY KEY             SERIAL / BIGSERIAL
    AUTOINCREMENT                   (automatico con SERIAL)
    TEXT                            TEXT / VARCHAR(n)
    REAL                            REAL / DOUBLE PRECISION
    BLOB                            BYTEA
    datetime('now')                 NOW() / CURRENT_TIMESTAMP
    
    TIPI ESCLUSIVI POSTGRESQL:
    ──────────────────────────
    - UUID: identificatori unici universali
    - JSONB: JSON binario (query-able!)
    - ARRAY: colonne array
    - INET/CIDR: indirizzi IP
    - INTERVAL: durate temporali
    - NUMERIC(p,s): precisione arbitraria
    """)
    
    # Esempio: Differenza pratica
    print("\n>>> Esempio: Schema in PostgreSQL")
    
    postgresql_schema = """
    -- PostgreSQL schema per trading
    CREATE TABLE IF NOT EXISTS trades (
        id BIGSERIAL PRIMARY KEY,           -- Auto-incrementing bigint
        order_id UUID DEFAULT gen_random_uuid(),  -- UUID automatico
        symbol VARCHAR(20) NOT NULL,
        side VARCHAR(4) CHECK(side IN ('BUY', 'SELL')),
        quantity NUMERIC(18, 8) NOT NULL,   -- Precisione per crypto
        price NUMERIC(18, 8) NOT NULL,
        metadata JSONB,                      -- Dati extra flessibili
        tags TEXT[],                         -- Array di tags
        created_at TIMESTAMPTZ DEFAULT NOW(),
        
        CONSTRAINT positive_qty CHECK(quantity > 0)
    );
    
    -- Indice su JSONB
    CREATE INDEX idx_trades_metadata ON trades USING GIN(metadata);
    
    -- Indice parziale (solo BUY)
    CREATE INDEX idx_trades_buy ON trades(created_at) WHERE side = 'BUY';
    """
    print(postgresql_schema)
    
    return True


# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 2: PSYCOPG3 - DRIVER MODERNO
# ══════════════════════════════════════════════════════════════════════════════

def section2_psycopg():
    """
    psycopg3: Il Driver PostgreSQL Moderno
    ───────────────────────────────────────
    
    psycopg3 (pacchetto 'psycopg') è il driver raccomandato per nuovi progetti.
    Vantaggi rispetto a psycopg2:
    
    ✅ Async nativo (asyncio built-in)
    ✅ Type hints completi
    ✅ Connection pooling avanzato
    ✅ Prepared statements automatici
    ✅ COPY protocol per bulk insert
    ✅ Performance ~3x rispetto a psycopg2
    
    INSTALLAZIONE:
    pip install psycopg[binary]
    pip install psycopg_pool
    
    """
    print("\n" + "═" * 60)
    print("📚 SEZIONE 2: PSYCOPG3")
    print("═" * 60)
    
    print("""
    >>> 2.1 Connessione Base
    
    ```python
    import psycopg
    from psycopg.rows import dict_row
    
    # Connessione semplice
    conn = psycopg.connect(
        host="localhost",
        port=5432,
        dbname="trading_db",
        user="trader",
        password="secret"
    )
    
    # Oppure con connection string
    conn = psycopg.connect("postgresql://trader:secret@localhost:5432/trading_db")
    
    # Con context manager (raccomandato)
    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM trades")
            rows = cur.fetchall()
    ```
    
    >>> 2.2 Row Factory per dizionari
    
    ```python
    from psycopg.rows import dict_row, namedtuple_row
    
    with psycopg.connect(conninfo) as conn:
        # Cursor con risultati come dizionari
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM trades WHERE symbol = %s", ('BTC',))
            for row in cur:
                print(row['symbol'], row['price'])  # Accesso per nome!
    ```
    
    >>> 2.3 Parametri e Sicurezza
    
    ```python
    # SEMPRE usare parametri, MAI concatenare stringhe!
    
    # ✅ CORRETTO - Parametri posizionali
    cur.execute("SELECT * FROM trades WHERE symbol = %s", (symbol,))
    
    # ✅ CORRETTO - Parametri nominati
    cur.execute(
        "SELECT * FROM trades WHERE symbol = %(sym)s AND price > %(min_price)s",
        {'sym': 'BTC', 'min_price': 40000}
    )
    
    # ❌ SBAGLIATO - SQL Injection!
    cur.execute(f"SELECT * FROM trades WHERE symbol = '{symbol}'")
    ```
    
    >>> 2.4 Batch Insert con executemany
    
    ```python
    trades = [
        ('BTC', 'BUY', 0.1, 45000),
        ('ETH', 'BUY', 1.5, 3000),
        ('SOL', 'SELL', 10, 100),
    ]
    
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO trades (symbol, side, quantity, price) VALUES (%s, %s, %s, %s)",
            trades
        )
    conn.commit()
    ```
    
    >>> 2.5 COPY Protocol (Super Veloce per Bulk)
    
    ```python
    # 10-100x più veloce di INSERT per grandi volumi
    
    with conn.cursor() as cur:
        with cur.copy("COPY trades (symbol, side, quantity, price) FROM STDIN") as copy:
            for trade in large_trade_list:
                copy.write_row(trade)
    conn.commit()
    ```
    
    >>> 2.6 Connection Pool
    
    ```python
    from psycopg_pool import ConnectionPool
    
    # Crea pool (fallo una volta all'avvio)
    pool = ConnectionPool(
        conninfo="postgresql://trader:secret@localhost/trading_db",
        min_size=4,
        max_size=20,
        max_idle=300,      # Chiudi connessioni idle dopo 5 min
        max_lifetime=1800  # Ricicla connessioni ogni 30 min
    )
    
    # Usa connessioni dal pool
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM trades")
            # ... elabora risultati
    # Connessione ritorna al pool automaticamente
    
    # Chiudi pool a fine programma
    pool.close()
    ```
    """)
    
    return True


# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 3: SQLALCHEMY CORE
# ══════════════════════════════════════════════════════════════════════════════

def section3_sqlalchemy_core():
    """
    SQLAlchemy Core: Query Programmatiche
    ─────────────────────────────────────
    
    SQLAlchemy ha 2 modalità:
    1. CORE: Query builder programmatico (SQL expression language)
    2. ORM: Object-Relational Mapping (modelli Python)
    
    Core è utile quando:
    - Vuoi controllo totale sulle query
    - Performance critica
    - Query complesse dinamiche
    
    """
    print("\n" + "═" * 60)
    print("📚 SEZIONE 3: SQLALCHEMY CORE")
    print("═" * 60)
    
    print("""
    >>> 3.1 Setup Engine e Metadata
    
    ```python
    from sqlalchemy import create_engine, MetaData, Table, Column
    from sqlalchemy import Integer, String, Float, DateTime, text
    from sqlalchemy.sql import select, insert, update, delete
    
    # Engine = connessione al database
    # Per SQLite (test):
    engine = create_engine("sqlite:///trading.db", echo=True)
    
    # Per PostgreSQL (produzione):
    engine = create_engine(
        "postgresql+psycopg://user:pass@localhost:5432/trading_db",
        pool_size=5,
        max_overflow=10
    )
    
    # Metadata = registro delle tabelle
    metadata = MetaData()
    ```
    
    >>> 3.2 Definizione Tabelle
    
    ```python
    from sqlalchemy import Table, Column, Integer, String, Float, DateTime, ForeignKey
    from sqlalchemy import CheckConstraint, UniqueConstraint
    from datetime import datetime
    
    # Definisci tabella trades
    trades = Table(
        'trades',
        metadata,
        Column('id', Integer, primary_key=True),
        Column('symbol', String(20), nullable=False),
        Column('side', String(4), nullable=False),
        Column('quantity', Float, nullable=False),
        Column('price', Float, nullable=False),
        Column('created_at', DateTime, default=datetime.utcnow),
        
        CheckConstraint("side IN ('BUY', 'SELL')", name='valid_side'),
        CheckConstraint("quantity > 0", name='positive_qty')
    )
    
    # Crea tabelle nel database
    metadata.create_all(engine)
    ```
    
    >>> 3.3 Insert con Core
    
    ```python
    from sqlalchemy import insert
    
    with engine.connect() as conn:
        # Singolo insert
        stmt = insert(trades).values(
            symbol='BTC', side='BUY', quantity=0.1, price=45000
        )
        result = conn.execute(stmt)
        print(f"Inserito ID: {result.inserted_primary_key}")
        
        # Multi insert
        stmt = insert(trades)
        conn.execute(stmt, [
            {'symbol': 'ETH', 'side': 'BUY', 'quantity': 1.0, 'price': 3000},
            {'symbol': 'SOL', 'side': 'BUY', 'quantity': 10, 'price': 100},
        ])
        
        conn.commit()
    ```
    
    >>> 3.4 Select con Core
    
    ```python
    from sqlalchemy import select, and_, or_, func
    
    with engine.connect() as conn:
        # Select base
        stmt = select(trades)
        result = conn.execute(stmt)
        for row in result:
            print(row.symbol, row.price)
        
        # Select con filtri
        stmt = select(trades).where(
            and_(
                trades.c.symbol == 'BTC',
                trades.c.price > 40000
            )
        )
        
        # Select con aggregazioni
        stmt = select(
            trades.c.symbol,
            func.count().label('num_trades'),
            func.sum(trades.c.quantity).label('total_qty'),
            func.avg(trades.c.price).label('avg_price')
        ).group_by(trades.c.symbol)
        
        # Order e Limit
        stmt = select(trades).order_by(
            trades.c.created_at.desc()
        ).limit(10)
    ```
    
    >>> 3.5 Update e Delete
    
    ```python
    from sqlalchemy import update, delete
    
    with engine.connect() as conn:
        # Update
        stmt = update(trades).where(
            trades.c.id == 1
        ).values(price=46000)
        conn.execute(stmt)
        
        # Delete
        stmt = delete(trades).where(
            trades.c.symbol == 'SOL'
        )
        result = conn.execute(stmt)
        print(f"Eliminati: {result.rowcount}")
        
        conn.commit()
    ```
    
    >>> 3.6 Raw SQL quando serve
    
    ```python
    from sqlalchemy import text
    
    with engine.connect() as conn:
        # Query raw con parametri
        result = conn.execute(
            text("SELECT * FROM trades WHERE symbol = :sym"),
            {'sym': 'BTC'}
        )
        
        # Per query complesse/ottimizzate
        result = conn.execute(text('''
            WITH recent_trades AS (
                SELECT * FROM trades 
                WHERE created_at > NOW() - INTERVAL '1 hour'
            )
            SELECT symbol, AVG(price) as avg_price
            FROM recent_trades
            GROUP BY symbol
        '''))
    ```
    """)
    
    return True


# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 4: SQLALCHEMY ORM
# ══════════════════════════════════════════════════════════════════════════════

def section4_sqlalchemy_orm():
    """
    SQLAlchemy ORM: Modelli Python
    ──────────────────────────────
    
    ORM = Object-Relational Mapping
    Ogni tabella diventa una classe Python.
    Ogni riga diventa un oggetto Python.
    
    Vantaggi:
    ✅ Codice più leggibile e pythonic
    ✅ Validazione a livello di oggetto
    ✅ Relazioni automatiche
    ✅ Lazy loading
    
    """
    print("\n" + "═" * 60)
    print("📚 SEZIONE 4: SQLALCHEMY ORM")
    print("═" * 60)
    
    print("""
    >>> 4.1 Definizione Modelli (SQLAlchemy 2.x style)
    
    ```python
    from sqlalchemy import create_engine, ForeignKey, String, Float, DateTime
    from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
    from sqlalchemy.orm import Session
    from datetime import datetime
    from typing import Optional, List
    
    # Base class per tutti i modelli
    class Base(DeclarativeBase):
        pass
    
    # Modello Trade
    class Trade(Base):
        __tablename__ = 'trades'
        
        id: Mapped[int] = mapped_column(primary_key=True)
        symbol: Mapped[str] = mapped_column(String(20))
        side: Mapped[str] = mapped_column(String(4))
        quantity: Mapped[float] = mapped_column(Float)
        price: Mapped[float] = mapped_column(Float)
        portfolio_id: Mapped[int] = mapped_column(ForeignKey('portfolios.id'))
        created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
        
        # Relazione con Portfolio
        portfolio: Mapped['Portfolio'] = relationship(back_populates='trades')
        
        def __repr__(self):
            return f"Trade({self.side} {self.quantity} {self.symbol} @ {self.price})"
        
        @property
        def total_value(self) -> float:
            return self.quantity * self.price
    
    # Modello Portfolio
    class Portfolio(Base):
        __tablename__ = 'portfolios'
        
        id: Mapped[int] = mapped_column(primary_key=True)
        name: Mapped[str] = mapped_column(String(100))
        created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
        
        # Relazione one-to-many con Trade
        trades: Mapped[List['Trade']] = relationship(back_populates='portfolio')
        
        @property
        def trade_count(self) -> int:
            return len(self.trades)
    ```
    
    >>> 4.2 Session e CRUD
    
    ```python
    from sqlalchemy.orm import Session
    
    engine = create_engine("sqlite:///trading.db")
    Base.metadata.create_all(engine)
    
    # CREATE
    with Session(engine) as session:
        # Crea portfolio
        portfolio = Portfolio(name="Trading Bot")
        session.add(portfolio)
        
        # Crea trade collegato
        trade = Trade(
            symbol='BTC',
            side='BUY',
            quantity=0.1,
            price=45000,
            portfolio=portfolio  # Relazione automatica!
        )
        session.add(trade)
        session.commit()
    
    # READ
    with Session(engine) as session:
        # Query base
        trades = session.query(Trade).all()
        
        # Nuovo stile SQLAlchemy 2.x
        from sqlalchemy import select
        stmt = select(Trade).where(Trade.symbol == 'BTC')
        trades = session.scalars(stmt).all()
        
        # Con relazioni (eager loading)
        from sqlalchemy.orm import joinedload
        stmt = select(Portfolio).options(joinedload(Portfolio.trades))
        portfolios = session.scalars(stmt).unique().all()
        
        for p in portfolios:
            print(f"{p.name}: {p.trade_count} trades")
    
    # UPDATE
    with Session(engine) as session:
        trade = session.get(Trade, 1)  # Get by primary key
        trade.price = 46000
        session.commit()
    
    # DELETE
    with Session(engine) as session:
        trade = session.get(Trade, 1)
        session.delete(trade)
        session.commit()
    ```
    
    >>> 4.3 Query Avanzate ORM
    
    ```python
    from sqlalchemy import select, func, and_, or_
    from sqlalchemy.orm import Session
    
    with Session(engine) as session:
        # Filtri multipli
        stmt = select(Trade).where(
            and_(
                Trade.symbol == 'BTC',
                Trade.side == 'BUY',
                Trade.price.between(40000, 50000)
            )
        )
        
        # Aggregazioni
        stmt = select(
            Trade.symbol,
            func.count(Trade.id).label('count'),
            func.sum(Trade.quantity).label('total_qty')
        ).group_by(Trade.symbol)
        
        result = session.execute(stmt)
        for row in result:
            print(f"{row.symbol}: {row.count} trades, {row.total_qty} total")
        
        # Subquery
        subq = select(func.avg(Trade.price)).where(
            Trade.symbol == 'BTC'
        ).scalar_subquery()
        
        stmt = select(Trade).where(Trade.price > subq)
    ```
    
    >>> 4.4 Relazioni Avanzate
    
    ```python
    from sqlalchemy.orm import Mapped, mapped_column, relationship
    from typing import List, Optional
    
    class User(Base):
        __tablename__ = 'users'
        
        id: Mapped[int] = mapped_column(primary_key=True)
        name: Mapped[str] = mapped_column(String(100))
        
        # One-to-many: un utente ha molti portfolio
        portfolios: Mapped[List['Portfolio']] = relationship(
            back_populates='owner',
            cascade='all, delete-orphan'  # Delete portfolios se user eliminato
        )
    
    class Portfolio(Base):
        __tablename__ = 'portfolios'
        
        id: Mapped[int] = mapped_column(primary_key=True)
        name: Mapped[str] = mapped_column(String(100))
        user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))
        
        # Many-to-one: un portfolio appartiene a un utente
        owner: Mapped['User'] = relationship(back_populates='portfolios')
        
        # One-to-many: un portfolio ha molti trade
        trades: Mapped[List['Trade']] = relationship(
            back_populates='portfolio',
            lazy='selectin'  # Eager loading per default
        )
    ```
    """)
    
    return True


# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 5: PATTERN REPOSITORY
# ══════════════════════════════════════════════════════════════════════════════

def section5_repository_pattern():
    """
    Pattern Repository: Separazione della Logica
    ─────────────────────────────────────────────
    
    Il pattern Repository separa la logica di business
    dall'accesso ai dati. Vantaggi:
    
    ✅ Codice più testabile (mock del repository)
    ✅ Cambio database trasparente
    ✅ Logica di query centralizzata
    ✅ Codice più leggibile
    
    """
    print("\n" + "═" * 60)
    print("📚 SEZIONE 5: PATTERN REPOSITORY")
    print("═" * 60)
    
    print("""
    >>> 5.1 Repository Base
    
    ```python
    from abc import ABC, abstractmethod
    from typing import Generic, TypeVar, List, Optional
    from sqlalchemy.orm import Session
    from sqlalchemy import select
    
    T = TypeVar('T')
    
    class BaseRepository(Generic[T], ABC):
        '''Repository generico per operazioni CRUD.'''
        
        def __init__(self, session: Session, model: type[T]):
            self.session = session
            self.model = model
        
        def get(self, id: int) -> Optional[T]:
            return self.session.get(self.model, id)
        
        def get_all(self) -> List[T]:
            stmt = select(self.model)
            return list(self.session.scalars(stmt))
        
        def add(self, entity: T) -> T:
            self.session.add(entity)
            return entity
        
        def delete(self, entity: T) -> None:
            self.session.delete(entity)
        
        def commit(self) -> None:
            self.session.commit()
    ```
    
    >>> 5.2 Repository Specifico per Trade
    
    ```python
    from datetime import datetime, timedelta
    from sqlalchemy import select, func, and_
    
    class TradeRepository(BaseRepository[Trade]):
        '''Repository con metodi specifici per Trade.'''
        
        def __init__(self, session: Session):
            super().__init__(session, Trade)
        
        def get_by_symbol(self, symbol: str) -> List[Trade]:
            stmt = select(Trade).where(Trade.symbol == symbol)
            return list(self.session.scalars(stmt))
        
        def get_recent(self, hours: int = 24) -> List[Trade]:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            stmt = select(Trade).where(
                Trade.created_at > cutoff
            ).order_by(Trade.created_at.desc())
            return list(self.session.scalars(stmt))
        
        def get_by_portfolio(self, portfolio_id: int) -> List[Trade]:
            stmt = select(Trade).where(Trade.portfolio_id == portfolio_id)
            return list(self.session.scalars(stmt))
        
        def get_statistics(self, symbol: str) -> dict:
            stmt = select(
                func.count(Trade.id).label('count'),
                func.sum(Trade.quantity).label('total_qty'),
                func.avg(Trade.price).label('avg_price'),
                func.min(Trade.price).label('min_price'),
                func.max(Trade.price).label('max_price')
            ).where(Trade.symbol == symbol)
            
            row = self.session.execute(stmt).one()
            return {
                'count': row.count,
                'total_qty': row.total_qty,
                'avg_price': row.avg_price,
                'min_price': row.min_price,
                'max_price': row.max_price
            }
        
        def get_pnl(self, portfolio_id: int) -> float:
            '''Calcola P&L realizzato per un portfolio.'''
            buys = select(
                Trade.symbol,
                func.sum(Trade.quantity).label('qty'),
                func.sum(Trade.quantity * Trade.price).label('cost')
            ).where(
                and_(Trade.portfolio_id == portfolio_id, Trade.side == 'BUY')
            ).group_by(Trade.symbol).subquery()
            
            sells = select(
                Trade.symbol,
                func.sum(Trade.quantity).label('qty'),
                func.sum(Trade.quantity * Trade.price).label('revenue')
            ).where(
                and_(Trade.portfolio_id == portfolio_id, Trade.side == 'SELL')
            ).group_by(Trade.symbol).subquery()
            
            # Join e calcola P&L...
            # (query complessa semplificata qui)
            return 0.0
    ```
    
    >>> 5.3 Unit of Work Pattern
    
    ```python
    class UnitOfWork:
        '''Gestisce la transazione e i repository.'''
        
        def __init__(self, session_factory):
            self.session_factory = session_factory
        
        def __enter__(self):
            self.session = self.session_factory()
            self.trades = TradeRepository(self.session)
            self.portfolios = PortfolioRepository(self.session)
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type:
                self.session.rollback()
            self.session.close()
        
        def commit(self):
            self.session.commit()
        
        def rollback(self):
            self.session.rollback()
    
    # Uso
    from sqlalchemy.orm import sessionmaker
    
    Session = sessionmaker(bind=engine)
    
    with UnitOfWork(Session) as uow:
        # Tutte le operazioni nella stessa transazione
        trade = Trade(symbol='BTC', side='BUY', quantity=0.1, price=45000)
        uow.trades.add(trade)
        
        stats = uow.trades.get_statistics('BTC')
        print(stats)
        
        uow.commit()  # Tutto o niente
    ```
    """)
    
    return True


# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 6: ALEMBIC - MIGRAZIONI
# ══════════════════════════════════════════════════════════════════════════════

def section6_alembic():
    """
    Alembic: Migrazioni Database
    ────────────────────────────
    
    Quando cambi lo schema (aggiungi colonne, tabelle, etc.),
    Alembic gestisce le migrazioni in modo tracciabile.
    
    PERCHÉ SERVE:
    - Traccia la storia dello schema
    - Sincronizza sviluppo/staging/produzione
    - Rollback se qualcosa va storto
    - Lavoro in team su stesso database
    
    """
    print("\n" + "═" * 60)
    print("📚 SEZIONE 6: ALEMBIC MIGRAZIONI")
    print("═" * 60)
    
    print("""
    >>> 6.1 Setup Alembic
    
    ```bash
    # Installa
    pip install alembic
    
    # Inizializza nella directory del progetto
    alembic init alembic
    
    # Struttura creata:
    # alembic/
    #   env.py          # Configurazione
    #   versions/       # File di migrazione
    # alembic.ini       # Config file
    ```
    
    >>> 6.2 Configurazione env.py
    
    ```python
    # alembic/env.py
    from logging.config import fileConfig
    from sqlalchemy import engine_from_config, pool
    from alembic import context
    
    # Importa i tuoi modelli
    from myapp.models import Base
    
    config = context.config
    target_metadata = Base.metadata
    
    def run_migrations_online():
        connectable = engine_from_config(
            config.get_section(config.config_ini_section),
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )
        
        with connectable.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata
            )
            with context.begin_transaction():
                context.run_migrations()
    ```
    
    >>> 6.3 Creare Migrazioni
    
    ```bash
    # Auto-genera migrazione da differenze nei modelli
    alembic revision --autogenerate -m "add fee column to trades"
    
    # Crea migrazione vuota (manuale)
    alembic revision -m "custom migration"
    ```
    
    >>> 6.4 File di Migrazione
    
    ```python
    # alembic/versions/abc123_add_fee_column.py
    
    from alembic import op
    import sqlalchemy as sa
    
    revision = 'abc123'
    down_revision = 'xyz789'  # Migrazione precedente
    
    def upgrade():
        '''Applica la migrazione.'''
        op.add_column('trades', 
            sa.Column('fee', sa.Float(), nullable=True, default=0)
        )
        op.create_index('idx_trades_fee', 'trades', ['fee'])
    
    def downgrade():
        '''Rollback della migrazione.'''
        op.drop_index('idx_trades_fee')
        op.drop_column('trades', 'fee')
    ```
    
    >>> 6.5 Comandi Alembic
    
    ```bash
    # Applica tutte le migrazioni pending
    alembic upgrade head
    
    # Applica una specifica migrazione
    alembic upgrade abc123
    
    # Rollback ultima migrazione
    alembic downgrade -1
    
    # Rollback a versione specifica
    alembic downgrade xyz789
    
    # Mostra storia migrazioni
    alembic history
    
    # Mostra versione corrente
    alembic current
    ```
    
    >>> 6.6 Best Practices Migrazioni
    
    ```
    ✅ Testa SEMPRE le migrazioni in staging prima di produzione
    ✅ Fai backup prima di migrare in produzione
    ✅ Scrivi SEMPRE il downgrade
    ✅ Migrazioni piccole e frequenti > grandi e rare
    ✅ Non modificare migrazioni già applicate in produzione
    ✅ Usa transazioni (op.batch_alter_table per SQLite)
    ```
    """)
    
    return True


# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 7: ASYNC SUPPORT
# ══════════════════════════════════════════════════════════════════════════════

def section7_async():
    """
    Async Database: Per Alta Performance
    ─────────────────────────────────────
    
    Async è utile quando:
    - Molte richieste concurrent (web API)
    - I/O bound operations (attesa database)
    - Real-time applications
    
    Per trading bot singolo: sync è spesso sufficiente.
    Per sistema multi-bot/API: async scala meglio.
    
    """
    print("\n" + "═" * 60)
    print("📚 SEZIONE 7: ASYNC SUPPORT")
    print("═" * 60)
    
    print("""
    >>> 7.1 psycopg3 Async
    
    ```python
    import asyncio
    import psycopg
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool
    
    # Pool async
    pool = AsyncConnectionPool(
        conninfo="postgresql://user:pass@localhost/db",
        min_size=4,
        max_size=20
    )
    
    async def get_trades(symbol: str):
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    "SELECT * FROM trades WHERE symbol = %s",
                    (symbol,)
                )
                return await cur.fetchall()
    
    async def main():
        # Concurrent queries
        results = await asyncio.gather(
            get_trades('BTC'),
            get_trades('ETH'),
            get_trades('SOL')
        )
        for trades in results:
            print(f"Found {len(trades)} trades")
    
    asyncio.run(main())
    ```
    
    >>> 7.2 SQLAlchemy Async
    
    ```python
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from sqlalchemy import select
    
    # Engine async
    engine = create_async_engine(
        "postgresql+asyncpg://user:pass@localhost/db",
        echo=True
    )
    
    # Session factory async
    async_session = async_sessionmaker(engine, expire_on_commit=False)
    
    async def get_trades_orm(symbol: str):
        async with async_session() as session:
            stmt = select(Trade).where(Trade.symbol == symbol)
            result = await session.execute(stmt)
            return result.scalars().all()
    
    async def create_trade(trade_data: dict):
        async with async_session() as session:
            trade = Trade(**trade_data)
            session.add(trade)
            await session.commit()
            return trade
    ```
    
    >>> 7.3 Pattern Async Repository
    
    ```python
    from typing import List, Optional
    
    class AsyncTradeRepository:
        def __init__(self, session: AsyncSession):
            self.session = session
        
        async def get(self, id: int) -> Optional[Trade]:
            return await self.session.get(Trade, id)
        
        async def get_by_symbol(self, symbol: str) -> List[Trade]:
            stmt = select(Trade).where(Trade.symbol == symbol)
            result = await self.session.execute(stmt)
            return list(result.scalars())
        
        async def add(self, trade: Trade) -> Trade:
            self.session.add(trade)
            await self.session.flush()
            return trade
        
        async def save(self) -> None:
            await self.session.commit()
    
    # Uso
    async def process_trades():
        async with async_session() as session:
            repo = AsyncTradeRepository(session)
            
            btc_trades = await repo.get_by_symbol('BTC')
            
            new_trade = Trade(symbol='BTC', side='BUY', quantity=0.1, price=45000)
            await repo.add(new_trade)
            await repo.save()
    ```
    """)
    
    return True


# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 8: TIMESCALEDB
# ══════════════════════════════════════════════════════════════════════════════

def section8_timescaledb():
    """
    TimescaleDB: Time-Series Powerhouse
    ────────────────────────────────────
    
    TimescaleDB è un'estensione PostgreSQL ottimizzata per dati temporali.
    Perfetto per:
    - Storico candele OHLCV
    - Metriche di sistema
    - Log trade
    
    VANTAGGI:
    ✅ Insert 10-100x più veloci per time-series
    ✅ Compressione automatica (90%+)
    ✅ Query temporali super ottimizzate
    ✅ Continuous aggregates (pre-calcolo)
    ✅ Data retention policies
    
    """
    print("\n" + "═" * 60)
    print("📚 SEZIONE 8: TIMESCALEDB")
    print("═" * 60)
    
    print("""
    >>> 8.1 Setup TimescaleDB
    
    ```sql
    -- Abilita estensione
    CREATE EXTENSION IF NOT EXISTS timescaledb;
    
    -- Crea tabella normale
    CREATE TABLE candles (
        time TIMESTAMPTZ NOT NULL,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        open NUMERIC(18,8) NOT NULL,
        high NUMERIC(18,8) NOT NULL,
        low NUMERIC(18,8) NOT NULL,
        close NUMERIC(18,8) NOT NULL,
        volume NUMERIC(18,8) NOT NULL
    );
    
    -- Converti in hypertable (la magia!)
    SELECT create_hypertable('candles', 'time');
    
    -- Indice per lookup
    CREATE INDEX ON candles (symbol, timeframe, time DESC);
    ```
    
    >>> 8.2 Continuous Aggregates (Pre-calcolo)
    
    ```sql
    -- Crea aggregato che si aggiorna automaticamente
    CREATE MATERIALIZED VIEW candles_1h
    WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('1 hour', time) AS bucket,
        symbol,
        first(open, time) AS open,
        max(high) AS high,
        min(low) AS low,
        last(close, time) AS close,
        sum(volume) AS volume
    FROM candles
    WHERE timeframe = '1m'
    GROUP BY bucket, symbol;
    
    -- Refresh policy automatico
    SELECT add_continuous_aggregate_policy('candles_1h',
        start_offset => INTERVAL '3 hours',
        end_offset => INTERVAL '1 hour',
        schedule_interval => INTERVAL '1 hour'
    );
    ```
    
    >>> 8.3 Compressione
    
    ```sql
    -- Abilita compressione
    ALTER TABLE candles SET (
        timescaledb.compress,
        timescaledb.compress_segmentby = 'symbol,timeframe'
    );
    
    -- Policy: comprimi dati > 7 giorni
    SELECT add_compression_policy('candles', INTERVAL '7 days');
    
    -- Compressione manuale
    SELECT compress_chunk(chunk) 
    FROM show_chunks('candles', older_than => INTERVAL '7 days') chunk;
    ```
    
    >>> 8.4 Data Retention
    
    ```sql
    -- Elimina automaticamente dati > 1 anno
    SELECT add_retention_policy('candles', INTERVAL '1 year');
    ```
    
    >>> 8.5 Query Ottimizzate
    
    ```sql
    -- Ultima candela per ogni symbol
    SELECT DISTINCT ON (symbol)
        symbol, time, close
    FROM candles
    ORDER BY symbol, time DESC;
    
    -- OHLCV per periodo
    SELECT
        time_bucket('1 day', time) AS day,
        symbol,
        first(open, time) AS open,
        max(high) AS high,
        min(low) AS low,
        last(close, time) AS close,
        sum(volume) AS volume
    FROM candles
    WHERE time > NOW() - INTERVAL '30 days'
      AND symbol = 'BTC'
    GROUP BY day, symbol
    ORDER BY day;
    
    -- Moving average
    SELECT
        time,
        close,
        avg(close) OVER (ORDER BY time ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS sma_20
    FROM candles
    WHERE symbol = 'BTC' AND timeframe = '1d';
    ```
    
    >>> 8.6 Python + TimescaleDB
    
    ```python
    import psycopg
    from datetime import datetime, timezone
    
    async def store_candles(conn, candles: list):
        '''Bulk insert candele in TimescaleDB.'''
        async with conn.cursor() as cur:
            async with cur.copy(
                "COPY candles (time, symbol, timeframe, open, high, low, close, volume) FROM STDIN"
            ) as copy:
                for c in candles:
                    await copy.write_row((
                        datetime.fromtimestamp(c['time'], tz=timezone.utc),
                        c['symbol'],
                        c['timeframe'],
                        c['open'],
                        c['high'],
                        c['low'],
                        c['close'],
                        c['volume']
                    ))
    
    async def get_ohlcv(conn, symbol: str, start: datetime, end: datetime):
        '''Query OHLCV da TimescaleDB.'''
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute('''
                SELECT time, open, high, low, close, volume
                FROM candles
                WHERE symbol = %s
                  AND time BETWEEN %s AND %s
                ORDER BY time
            ''', (symbol, start, end))
            return await cur.fetchall()
    ```
    """)
    
    return True


# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 9: PROGETTO COMPLETO
# ══════════════════════════════════════════════════════════════════════════════

def section9_complete_project():
    """
    Progetto Completo: Trading Database System
    ──────────────────────────────────────────
    
    Struttura production-ready per il tuo trading bot.
    
    """
    print("\n" + "═" * 60)
    print("📚 SEZIONE 9: PROGETTO COMPLETO")
    print("═" * 60)
    
    print("""
    >>> 9.1 Struttura Directory
    
    ```
    trading_bot/
    ├── database/
    │   ├── __init__.py
    │   ├── models.py          # Definizione modelli ORM
    │   ├── repositories.py    # Pattern repository
    │   ├── connection.py      # Engine e session factory
    │   └── migrations/        # Alembic migrations
    │       └── versions/
    ├── services/
    │   ├── trade_service.py   # Business logic
    │   └── market_service.py
    └── main.py
    ```
    
    >>> 9.2 database/models.py
    
    ```python
    from datetime import datetime
    from typing import List, Optional
    from sqlalchemy import String, Float, ForeignKey, Index
    from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
    
    class Base(DeclarativeBase):
        pass
    
    class Portfolio(Base):
        __tablename__ = 'portfolios'
        
        id: Mapped[int] = mapped_column(primary_key=True)
        name: Mapped[str] = mapped_column(String(100))
        initial_balance: Mapped[float] = mapped_column(Float, default=10000)
        created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
        
        trades: Mapped[List['Trade']] = relationship(back_populates='portfolio')
        positions: Mapped[List['Position']] = relationship(back_populates='portfolio')
    
    class Trade(Base):
        __tablename__ = 'trades'
        __table_args__ = (
            Index('idx_trades_lookup', 'portfolio_id', 'symbol', 'created_at'),
        )
        
        id: Mapped[int] = mapped_column(primary_key=True)
        portfolio_id: Mapped[int] = mapped_column(ForeignKey('portfolios.id'))
        symbol: Mapped[str] = mapped_column(String(20))
        side: Mapped[str] = mapped_column(String(4))
        quantity: Mapped[float] = mapped_column(Float)
        price: Mapped[float] = mapped_column(Float)
        fee: Mapped[float] = mapped_column(Float, default=0)
        strategy: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
        created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
        
        portfolio: Mapped['Portfolio'] = relationship(back_populates='trades')
        
        @property
        def total_value(self) -> float:
            return self.quantity * self.price
        
        @property
        def net_value(self) -> float:
            return self.total_value - self.fee
    
    class Position(Base):
        __tablename__ = 'positions'
        __table_args__ = (
            Index('idx_positions_unique', 'portfolio_id', 'symbol', unique=True),
        )
        
        id: Mapped[int] = mapped_column(primary_key=True)
        portfolio_id: Mapped[int] = mapped_column(ForeignKey('portfolios.id'))
        symbol: Mapped[str] = mapped_column(String(20))
        quantity: Mapped[float] = mapped_column(Float, default=0)
        avg_entry_price: Mapped[float] = mapped_column(Float, default=0)
        realized_pnl: Mapped[float] = mapped_column(Float, default=0)
        updated_at: Mapped[datetime] = mapped_column(
            default=datetime.utcnow, onupdate=datetime.utcnow
        )
        
        portfolio: Mapped['Portfolio'] = relationship(back_populates='positions')
        
        def update_after_buy(self, qty: float, price: float):
            new_total = self.quantity + qty
            self.avg_entry_price = (
                (self.avg_entry_price * self.quantity + price * qty) / new_total
            )
            self.quantity = new_total
        
        def update_after_sell(self, qty: float, price: float) -> float:
            pnl = (price - self.avg_entry_price) * qty
            self.quantity -= qty
            self.realized_pnl += pnl
            return pnl
    ```
    
    >>> 9.3 database/connection.py
    
    ```python
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker, Session
    from contextlib import contextmanager
    from .models import Base
    
    class Database:
        def __init__(self, url: str = "sqlite:///trading.db"):
            self.engine = create_engine(url, echo=False)
            self.SessionLocal = sessionmaker(bind=self.engine)
        
        def create_tables(self):
            Base.metadata.create_all(self.engine)
        
        @contextmanager
        def session(self) -> Session:
            session = self.SessionLocal()
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()
    
    # Singleton
    db = Database()
    ```
    
    >>> 9.4 database/repositories.py
    
    ```python
    from typing import List, Optional
    from sqlalchemy.orm import Session
    from sqlalchemy import select
    from .models import Trade, Portfolio, Position
    
    class TradeRepository:
        def __init__(self, session: Session):
            self.session = session
        
        def add(self, trade: Trade) -> Trade:
            self.session.add(trade)
            self.session.flush()
            return trade
        
        def get_by_portfolio(self, portfolio_id: int) -> List[Trade]:
            stmt = select(Trade).where(Trade.portfolio_id == portfolio_id)
            return list(self.session.scalars(stmt))
    
    class PositionRepository:
        def __init__(self, session: Session):
            self.session = session
        
        def get_or_create(self, portfolio_id: int, symbol: str) -> Position:
            stmt = select(Position).where(
                Position.portfolio_id == portfolio_id,
                Position.symbol == symbol
            )
            pos = self.session.scalar(stmt)
            if not pos:
                pos = Position(portfolio_id=portfolio_id, symbol=symbol)
                self.session.add(pos)
            return pos
    ```
    
    >>> 9.5 services/trade_service.py
    
    ```python
    from database.connection import db
    from database.models import Trade, Position
    from database.repositories import TradeRepository, PositionRepository
    
    class TradeService:
        def execute_trade(
            self, portfolio_id: int, symbol: str, side: str, 
            qty: float, price: float, strategy: str = None
        ) -> dict:
            with db.session() as session:
                trade_repo = TradeRepository(session)
                pos_repo = PositionRepository(session)
                
                # Crea trade
                trade = Trade(
                    portfolio_id=portfolio_id,
                    symbol=symbol,
                    side=side,
                    quantity=qty,
                    price=price,
                    strategy=strategy
                )
                trade_repo.add(trade)
                
                # Aggiorna posizione
                position = pos_repo.get_or_create(portfolio_id, symbol)
                if side == 'BUY':
                    position.update_after_buy(qty, price)
                    pnl = 0
                else:
                    pnl = position.update_after_sell(qty, price)
                
                return {
                    'trade_id': trade.id,
                    'realized_pnl': pnl,
                    'position_qty': position.quantity
                }
    ```
    """)
    
    return True


# ══════════════════════════════════════════════════════════════════════════════
# MAIN - Esecuzione
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Esegue tutte le sezioni del modulo."""
    
    sections = [
        ("Sezione 1: PostgreSQL Intro", section1_postgresql_intro),
        ("Sezione 2: psycopg3", section2_psycopg),
        ("Sezione 3: SQLAlchemy Core", section3_sqlalchemy_core),
        ("Sezione 4: SQLAlchemy ORM", section4_sqlalchemy_orm),
        ("Sezione 5: Repository Pattern", section5_repository_pattern),
        ("Sezione 6: Alembic Migrazioni", section6_alembic),
        ("Sezione 7: Async Support", section7_async),
        ("Sezione 8: TimescaleDB", section8_timescaledb),
        ("Sezione 9: Progetto Completo", section9_complete_project),
    ]
    
    print("\n" + "═" * 70)
    print("ESECUZIONE MODULO DATABASE PARTE 2")
    print("═" * 70)
    
    for name, func in sections:
        try:
            func()
            print(f"\n✅ {name} completata")
        except Exception as e:
            print(f"\n❌ {name} errore: {e}")
    
    print("\n" + "═" * 70)
    print("🎉 MODULO DATABASE PARTE 2 COMPLETATO!")
    print("═" * 70)
    print("""
    HAI IMPARATO:
    ─────────────
    ✅ PostgreSQL vs SQLite: quando usare cosa
    ✅ psycopg3: driver moderno, pooling, COPY protocol
    ✅ SQLAlchemy Core: query programmatiche
    ✅ SQLAlchemy ORM 2.x: modelli, relazioni, type hints
    ✅ Pattern Repository e Unit of Work
    ✅ Alembic: migrazioni database
    ✅ Async: psycopg async, SQLAlchemy async
    ✅ TimescaleDB: time-series, continuous aggregates
    ✅ Progetto completo: struttura production-ready
    
    PROSSIMO PASSO: Esercizi pratici!
    """)


if __name__ == "__main__":
    main()
