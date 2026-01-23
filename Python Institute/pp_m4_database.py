"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║            PYTHON PROFESSIONAL (PP) - MODULE 4                               ║
║                   Database Programming & ORM                                 ║
║                                                                              ║
║                     Allineato al Syllabus PCPP2-32-20x                       ║
║                         OpenEDG Python Institute                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

PCPP2 Exam Block 4: Database Programming (25%)

STRUTTURA MODULO:
├── Section 4.1: Database Basics & SQL Review
├── Section 4.2: SQLite with Python
├── Section 4.3: SQLAlchemy Core
├── Section 4.4: SQLAlchemy ORM
├── Section 4.5: Relationships
├── Section 4.6: Transactions & Sessions
├── Section 4.7: Alembic Migrations ⭐ NEW
├── Section 4.8: Connection Pooling
├── Section 4.9: Repository Pattern
├── Labs (15 esercizi)
└── Module 4 Test (40 domande)

TEMPO STIMATO: 10-12 ore

═══════════════════════════════════════════════════════════════════════════════
"""

import sqlite3
from typing import List, Dict, Optional, Any
from contextlib import contextmanager
from datetime import datetime


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 4.1: DATABASE BASICS
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    4.1 TEORIA: FONDAMENTI DATABASE                           │
└──────────────────────────────────────────────────────────────────────────────┘

TIPI DI DATABASE:
─────────────────
Relational (SQL): PostgreSQL, MySQL, SQLite
- Dati strutturati in tabelle
- Schema fisso
- ACID compliance
- Query con SQL

NoSQL: MongoDB, Redis, Cassandra
- Schema flessibile
- Scalabilità orizzontale
- Vari modelli (document, key-value, graph)


SQL ESSENTIALS:
───────────────
DDL (Data Definition Language): CREATE, ALTER, DROP
DML (Data Manipulation Language): SELECT, INSERT, UPDATE, DELETE
DCL (Data Control Language): GRANT, REVOKE
TCL (Transaction Control): COMMIT, ROLLBACK


ACID PROPERTIES:
────────────────
A - Atomicity: Transazione tutto-o-niente
C - Consistency: Dati sempre validi
I - Isolation: Transazioni isolate
D - Durability: Dati persistenti dopo commit
"""


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 4.1                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_4_1 = """
Q1. ACID sta per:
    A) Atomic, Consistent, Isolated, Durable
    B) Advanced, Complete, Integrated, Dynamic
    C) Async, Cached, Indexed, Distributed
    D) Automatic, Controlled, Independent, Direct

Q2. DDL include:
    A) SELECT, INSERT    B) CREATE, DROP    C) COMMIT, ROLLBACK    D) GRANT, REVOKE

Q3. DML include:
    A) CREATE, ALTER    B) SELECT, INSERT, UPDATE    C) GRANT, REVOKE    D) COMMIT

Q4. Una transazione atomica:
    A) È veloce    B) È tutto-o-niente    C) È asincrona    D) È distribuita

Q5. Isolation garantisce:
    A) Velocità    B) Transazioni non si interferiscono    C) Backup    D) Compressione
"""

ANSWERS_4_1 = """
RISPOSTE QUIZ 4.1:
Q1: A - Atomic, Consistent, Isolated, Durable
Q2: B - CREATE, DROP, ALTER (Data Definition)
Q3: B - SELECT, INSERT, UPDATE, DELETE (Data Manipulation)
Q4: B - Tutto-o-niente (completa o rollback)
Q5: B - Transazioni non si interferiscono
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 4.2: SQLITE WITH PYTHON
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    4.2 TEORIA: SQLITE                                        │
└──────────────────────────────────────────────────────────────────────────────┘
"""

# CONNESSIONE E OPERAZIONI BASE
def sqlite_basics():
    conn = sqlite3.connect('example.db')
    cursor = conn.cursor()
    
    # CREATE TABLE
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # INSERT con parametri (SICURO!)
    cursor.execute(
        "INSERT OR IGNORE INTO users (username, email) VALUES (?, ?)",
        ('marco', 'marco@example.com')
    )
    
    # SELECT
    cursor.execute("SELECT * FROM users WHERE username = ?", ('marco',))
    user = cursor.fetchone()
    
    conn.commit()
    conn.close()
    return user


# CONTEXT MANAGER
@contextmanager
def get_db_connection(db_path: str):
    """Context manager per connessioni database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Risultati come dizionari
    try:
        yield conn
    finally:
        conn.close()


# USO
# with get_db_connection('example.db') as conn:
#     cursor = conn.cursor()
#     cursor.execute("SELECT * FROM users")
#     for row in cursor:
#         print(dict(row))


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 4.2                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_4_2 = """
Q1. sqlite3.connect(':memory:') crea:
    A) File temporaneo    B) Database in RAM    C) Connessione remota    D) Errore

Q2. cursor.fetchone() restituisce:
    A) Lista    B) Tupla o None    C) Dict    D) Int

Q3. ? in execute() previene:
    A) Errori sintassi    B) SQL injection    C) Duplicati    D) Timeout

Q4. row_factory = sqlite3.Row permette:
    A) Righe veloci    B) Accesso per nome colonna    C) Compressione    D) Encryption

Q5. conn.commit() è necessario per:
    A) SELECT    B) INSERT/UPDATE/DELETE    C) Entrambi    D) Nessuno
"""

ANSWERS_4_2 = """
RISPOSTE QUIZ 4.2:
Q1: B - Database in RAM
Q2: B - Tupla o None
Q3: B - SQL injection
Q4: B - Accesso per nome colonna
Q5: B - INSERT/UPDATE/DELETE (modifiche)
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 4.3: SQLALCHEMY CORE
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    4.3 TEORIA: SQLALCHEMY CORE                               │
└──────────────────────────────────────────────────────────────────────────────┘

SQLAlchemy ha 2 livelli:
1. CORE - SQL Expression Language (più basso livello)
2. ORM - Object Relational Mapper (alto livello)
"""

# pip install sqlalchemy

from sqlalchemy import create_engine, MetaData, Table, Column
from sqlalchemy import Integer, String, DateTime, ForeignKey
from sqlalchemy import select, insert, update, delete

# ENGINE - Connessione al database
# engine = create_engine('sqlite:///example.db', echo=True)
# engine = create_engine('postgresql://user:pass@localhost/dbname')

"""
CONNECTION STRINGS:
───────────────────
SQLite:     sqlite:///path/to/file.db
            sqlite:///:memory:
PostgreSQL: postgresql://user:password@host:port/database
MySQL:      mysql+pymysql://user:password@host:port/database
"""


# METADATA E TABELLE (Core)
metadata = MetaData()

users_table = Table(
    'users', metadata,
    Column('id', Integer, primary_key=True),
    Column('username', String(50), nullable=False, unique=True),
    Column('email', String(100), nullable=False),
    Column('created_at', DateTime, default=datetime.utcnow)
)

# CREAZIONE TABELLE
# metadata.create_all(engine)


# OPERAZIONI CON CORE
def sqlalchemy_core_example(engine):
    with engine.connect() as conn:
        # INSERT
        stmt = insert(users_table).values(
            username='marco', 
            email='marco@example.com'
        )
        conn.execute(stmt)
        conn.commit()
        
        # SELECT
        stmt = select(users_table).where(users_table.c.username == 'marco')
        result = conn.execute(stmt)
        for row in result:
            print(row)
        
        # UPDATE
        stmt = update(users_table).where(
            users_table.c.username == 'marco'
        ).values(email='new@example.com')
        conn.execute(stmt)
        conn.commit()
        
        # DELETE
        stmt = delete(users_table).where(users_table.c.id == 1)
        conn.execute(stmt)
        conn.commit()


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 4.3                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_4_3 = """
Q1. create_engine() crea:
    A) Tabella    B) Connessione/pool    C) Query    D) Schema

Q2. MetaData() contiene:
    A) Dati    B) Schema delle tabelle    C) Connessioni    D) Query

Q3. metadata.create_all(engine) fa:
    A) Crea engine    B) Crea tabelle definite    C) Elimina tutto    D) Backup

Q4. users_table.c.username accede a:
    A) Dati    B) Colonna    C) Riga    D) Tabella

Q5. SQLAlchemy Core è:
    A) ORM    B) SQL Expression Language    C) Driver    D) GUI
"""

ANSWERS_4_3 = """
RISPOSTE QUIZ 4.3:
Q1: B - Connessione/pool al database
Q2: B - Schema delle tabelle (definizioni)
Q3: B - Crea le tabelle definite in metadata
Q4: B - Colonna (c = columns)
Q5: B - SQL Expression Language (più basso livello dell'ORM)
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 4.4: SQLALCHEMY ORM
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    4.4 TEORIA: SQLALCHEMY ORM                                │
└──────────────────────────────────────────────────────────────────────────────┘

ORM = Object Relational Mapper
Mappa classi Python ↔ tabelle database
"""

from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from datetime import datetime

# BASE CLASS
Base = declarative_base()


# MODELLO (mapped class)
class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), nullable=False, unique=True)
    email = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship (vedremo dopo)
    posts = relationship("Post", back_populates="author")
    
    def __repr__(self):
        return f"<User(username='{self.username}')>"


class Post(Base):
    __tablename__ = 'posts'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    content = Column(String)
    user_id = Column(Integer, ForeignKey('users.id'))
    
    author = relationship("User", back_populates="posts")
    
    def __repr__(self):
        return f"<Post(title='{self.title}')>"


# SETUP
# engine = create_engine('sqlite:///example.db')
# Base.metadata.create_all(engine)
# Session = sessionmaker(bind=engine)


# OPERAZIONI ORM
def orm_example(Session):
    session = Session()
    
    try:
        # CREATE
        user = User(username='marco', email='marco@example.com')
        session.add(user)
        session.commit()
        
        # READ
        user = session.query(User).filter_by(username='marco').first()
        users = session.query(User).all()
        
        # Python 2.0 style
        # from sqlalchemy import select
        # stmt = select(User).where(User.username == 'marco')
        # user = session.execute(stmt).scalar_one_or_none()
        
        # UPDATE
        user.email = 'new@example.com'
        session.commit()
        
        # DELETE
        session.delete(user)
        session.commit()
        
    except Exception as e:
        session.rollback()
        raise
    finally:
        session.close()


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 4.4                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_4_4 = """
Q1. declarative_base() crea:
    A) Engine    B) Classe base per modelli ORM    C) Session    D) Query

Q2. __tablename__ specifica:
    A) Nome classe    B) Nome tabella nel DB    C) Nome colonna    D) Schema

Q3. session.add(obj) fa:
    A) Salva subito    B) Aggiunge alla session (pending)    C) Query    D) Delete

Q4. session.commit() fa:
    A) Chiude session    B) Salva modifiche su DB    C) Rollback    D) Query

Q5. session.rollback() fa:
    A) Salva    B) Annulla modifiche non committate    C) Chiude    D) Query

Q6. filter_by(username='x') è equivalente a:
    A) filter(username='x')    B) filter(User.username == 'x')    C) where(username='x')    D) get('x')
"""

ANSWERS_4_4 = """
RISPOSTE QUIZ 4.4:
Q1: B - Classe base per modelli ORM
Q2: B - Nome tabella nel database
Q3: B - Aggiunge alla session (pending, non ancora salvato)
Q4: B - Salva modifiche su database
Q5: B - Annulla modifiche non committate
Q6: B - filter(User.username == 'x')
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 4.5: RELATIONSHIPS
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    4.5 TEORIA: RELATIONSHIPS                                 │
└──────────────────────────────────────────────────────────────────────────────┘
"""

# ONE-TO-MANY
class Author(Base):
    __tablename__ = 'authors'
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    
    # One author has many books
    books = relationship("Book", back_populates="author")


class Book(Base):
    __tablename__ = 'books'
    id = Column(Integer, primary_key=True)
    title = Column(String(200))
    author_id = Column(Integer, ForeignKey('authors.id'))
    
    # Many books have one author
    author = relationship("Author", back_populates="books")


# MANY-TO-MANY
from sqlalchemy import Table

# Association table
book_tags = Table(
    'book_tags', Base.metadata,
    Column('book_id', Integer, ForeignKey('books.id')),
    Column('tag_id', Integer, ForeignKey('tags.id'))
)


class Tag(Base):
    __tablename__ = 'tags'
    id = Column(Integer, primary_key=True)
    name = Column(String(50))
    
    books = relationship("Book", secondary=book_tags, back_populates="tags")


# Aggiungi a Book:
# tags = relationship("Tag", secondary=book_tags, back_populates="books")


# ONE-TO-ONE
class Profile(Base):
    __tablename__ = 'profiles'
    id = Column(Integer, primary_key=True)
    bio = Column(String)
    user_id = Column(Integer, ForeignKey('users.id'))
    
    user = relationship("User", back_populates="profile", uselist=False)


# N+1 PROBLEM E EAGER LOADING
"""
N+1 PROBLEM:
────────────
Quando carichi N oggetti e poi accedi alle relazioni, 
fai 1 + N query (inefficiente!).

SOLUZIONE: Eager Loading
"""

from sqlalchemy.orm import joinedload, selectinload

def eager_loading_example(session):
    # LAZY LOADING (default) - N+1 problem
    authors = session.query(Author).all()
    for author in authors:
        print(author.books)  # Una query per ogni author!
    
    # EAGER LOADING - joined (una query con JOIN)
    authors = session.query(Author).options(
        joinedload(Author.books)
    ).all()
    
    # EAGER LOADING - subquery (due query separate)
    authors = session.query(Author).options(
        selectinload(Author.books)
    ).all()


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 4.5                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_4_5 = """
Q1. ForeignKey definisce:
    A) Primary key    B) Riferimento a altra tabella    C) Index    D) Constraint

Q2. relationship() con back_populates crea:
    A) Tabella    B) Relazione bidirezionale    C) Index    D) Constraint

Q3. secondary= in relationship serve per:
    A) One-to-one    B) One-to-many    C) Many-to-many    D) Backup

Q4. uselist=False indica:
    A) One-to-one    B) One-to-many    C) Many-to-many    D) Nessuna relazione

Q5. Il problema N+1 si risolve con:
    A) Più query    B) Eager loading    C) Più tabelle    D) Index

Q6. joinedload fa:
    A) Query separate    B) Una query con JOIN    C) Lazy loading    D) Delete cascade
"""

ANSWERS_4_5 = """
RISPOSTE QUIZ 4.5:
Q1: B - Riferimento a altra tabella
Q2: B - Relazione bidirezionale (entrambi i lati)
Q3: C - Many-to-many (association table)
Q4: A - One-to-one (singolo oggetto, non lista)
Q5: B - Eager loading (joinedload/selectinload)
Q6: B - Una query con JOIN
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 4.6: TRANSACTIONS & SESSIONS
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    4.6 TEORIA: TRANSACTIONS                                  │
└──────────────────────────────────────────────────────────────────────────────┘
"""

from sqlalchemy.orm import Session

def transaction_example(engine):
    """Gestione transazioni con context manager."""
    with Session(engine) as session:
        try:
            # Operazioni
            user = User(username='test', email='test@example.com')
            session.add(user)
            
            # Più operazioni nella stessa transazione
            post = Post(title='Test', author=user)
            session.add(post)
            
            # Commit se tutto ok
            session.commit()
            
        except Exception:
            # Rollback se errore
            session.rollback()
            raise


# SESSION SCOPED (per web app)
from sqlalchemy.orm import scoped_session

# Session = scoped_session(sessionmaker(bind=engine))
# Ogni thread ottiene la propria session


# NESTED TRANSACTIONS (SAVEPOINT)
def nested_transaction_example(session):
    """Savepoint per transazioni parziali."""
    session.begin()  # Transazione principale
    
    try:
        session.add(User(username='user1', email='u1@example.com'))
        
        # Nested transaction (savepoint)
        session.begin_nested()
        try:
            session.add(User(username='user2', email='u2@example.com'))
            session.commit()  # Commit savepoint
        except:
            session.rollback()  # Rollback solo savepoint
        
        session.commit()  # Commit transazione principale
    except:
        session.rollback()


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 4.6                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_4_6 = """
Q1. Session(engine) con context manager:
    A) Richiede commit manuale    B) Auto-commit    C) Auto-close    D) Niente

Q2. session.begin_nested() crea:
    A) Nuova sessione    B) Savepoint    C) Nuova connessione    D) Backup

Q3. scoped_session serve per:
    A) Single thread    B) Thread-local sessions    C) Async    D) Testing

Q4. Dopo session.rollback():
    A) Dati salvati    B) Modifiche annullate    C) Session chiusa    D) Errore

Q5. ACID: Atomicity garantisce:
    A) Velocità    B) Tutto-o-niente    C) Isolamento    D) Durabilità
"""

ANSWERS_4_6 = """
RISPOSTE QUIZ 4.6:
Q1: C - Auto-close (ma commit manuale)
Q2: B - Savepoint (transazione nested)
Q3: B - Thread-local sessions (una per thread)
Q4: B - Modifiche annullate (non committate)
Q5: B - Tutto-o-niente
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 4.7: ALEMBIC MIGRATIONS ⭐ IMPORTANTE PER PCPP2
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    4.7 TEORIA: ALEMBIC MIGRATIONS                            │
└──────────────────────────────────────────────────────────────────────────────┘

ALEMBIC = Tool per database migrations in SQLAlchemy.

PERCHÉ MIGRATIONS?
──────────────────
- Schema database evolve nel tempo
- Serve tracciare cambiamenti
- Applicare/revertire modifiche in modo controllato
- Sincronizzare schema tra ambienti (dev/staging/prod)


INSTALLAZIONE E SETUP:
──────────────────────
pip install alembic
alembic init alembic    # Crea struttura directory
"""

ALEMBIC_STRUCTURE = """
STRUTTURA DIRECTORY ALEMBIC:
────────────────────────────

myproject/
├── alembic/
│   ├── versions/           # Migration files
│   │   ├── 001_initial.py
│   │   ├── 002_add_email.py
│   │   └── ...
│   ├── env.py              # Environment configuration
│   ├── script.py.mako      # Template per nuove migrations
│   └── README
├── alembic.ini             # Alembic configuration
└── models.py               # SQLAlchemy models
"""


ALEMBIC_INI_EXAMPLE = """
# alembic.ini - File di configurazione principale

[alembic]
# Path to migration scripts
script_location = alembic

# Database URL (può essere override in env.py)
sqlalchemy.url = sqlite:///./app.db

# Template per file di migration
file_template = %%(year)d%%(month).2d%%(day).2d_%%(rev)s_%%(slug)s

# Timezone per i timestamp
# timezone = UTC

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic
"""


ENV_PY_EXAMPLE = """
# alembic/env.py - Configurazione ambiente

from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

# Import dei modelli per auto-detection
from myapp.models import Base

# Alembic Config object
config = context.config

# Logging setup
fileConfig(config.config_file_name)

# MetaData per autogenerate
target_metadata = Base.metadata


def run_migrations_offline():
    '''Run migrations in 'offline' mode (SQL output).'''
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    '''Run migrations in 'online' mode (direct DB connection).'''
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


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
"""


"""
═══════════════════════════════════════════════════════════════════════════════
                    COMANDI ALEMBIC PRINCIPALI
═══════════════════════════════════════════════════════════════════════════════

# Inizializzare Alembic in un progetto
alembic init alembic

# Creare nuova migration (manuale)
alembic revision -m "description"

# Creare migration auto-generata (confronta modelli con DB)
alembic revision --autogenerate -m "description"

# Applicare migrations (upgrade)
alembic upgrade head          # Applica tutte
alembic upgrade +1            # Applica la prossima
alembic upgrade abc123        # Applica fino a revision specifica

# Revertire migrations (downgrade)
alembic downgrade -1          # Revert l'ultima
alembic downgrade base        # Revert tutte
alembic downgrade abc123      # Revert fino a revision specifica

# Vedere stato corrente
alembic current               # Revision corrente
alembic history               # Storia migrations
alembic heads                 # Ultime revisions

# Generare SQL senza eseguire
alembic upgrade head --sql > migration.sql
"""


MIGRATION_FILE_EXAMPLE = '''
"""Add email column to users

Revision ID: 002_add_email
Revises: 001_initial
Create Date: 2025-01-23 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers
revision = '002_add_email'
down_revision = '001_initial'
branch_labels = None
depends_on = None


def upgrade():
    """
    Upgrade: Aggiunge colonna email alla tabella users.
    """
    op.add_column('users', 
        sa.Column('email', sa.String(100), nullable=True)
    )
    
    # Popola dati esistenti
    op.execute("UPDATE users SET email = username || '@example.com'")
    
    # Rendi NOT NULL
    op.alter_column('users', 'email', nullable=False)
    
    # Aggiungi index
    op.create_index('ix_users_email', 'users', ['email'], unique=True)


def downgrade():
    """
    Downgrade: Rimuove colonna email.
    """
    op.drop_index('ix_users_email', 'users')
    op.drop_column('users', 'email')
'''


"""
═══════════════════════════════════════════════════════════════════════════════
                    OPERAZIONI ALEMBIC (op.*)
═══════════════════════════════════════════════════════════════════════════════

TABELLE:
    op.create_table('name', Column(...), ...)
    op.drop_table('name')
    op.rename_table('old', 'new')

COLONNE:
    op.add_column('table', Column('name', Type))
    op.drop_column('table', 'column')
    op.alter_column('table', 'column', nullable=False, new_column_name='new')

INDEX:
    op.create_index('name', 'table', ['column'])
    op.drop_index('name', 'table')

FOREIGN KEY:
    op.create_foreign_key('name', 'source', 'target', ['col'], ['col'])
    op.drop_constraint('name', 'table')

SQL RAW:
    op.execute("UPDATE users SET active = true")

BATCH (per SQLite che non supporta ALTER):
    with op.batch_alter_table('users') as batch_op:
        batch_op.add_column(Column('age', Integer))
        batch_op.drop_column('old_column')
"""


# ESEMPIO PRATICO: SISTEMA DI MIGRATIONS

class MigrationManager:
    """
    Wrapper per gestire migrations programmaticamente.
    Utile per testing e deployment automatizzato.
    """
    
    def __init__(self, alembic_cfg_path: str = 'alembic.ini'):
        from alembic.config import Config
        from alembic import command
        
        self.config = Config(alembic_cfg_path)
        self.command = command
    
    def upgrade(self, revision: str = 'head'):
        """Applica migrations fino a revision specificata."""
        self.command.upgrade(self.config, revision)
    
    def downgrade(self, revision: str = '-1'):
        """Revert migrations."""
        self.command.downgrade(self.config, revision)
    
    def current(self) -> str:
        """Mostra revision corrente."""
        self.command.current(self.config)
    
    def history(self):
        """Mostra storia migrations."""
        self.command.history(self.config)
    
    def autogenerate(self, message: str):
        """Genera migration auto-detect da modelli."""
        self.command.revision(
            self.config, 
            message=message, 
            autogenerate=True
        )


# TESTING CON MIGRATIONS
def test_with_migrations(engine, session_factory):
    """
    Pattern per testing con database migrations.
    """
    from alembic.config import Config
    from alembic import command
    
    # Setup: applica tutte le migrations
    alembic_cfg = Config('alembic.ini')
    command.upgrade(alembic_cfg, 'head')
    
    # Test
    session = session_factory()
    try:
        # ... test code ...
        pass
    finally:
        session.close()
    
    # Teardown: revert tutto
    command.downgrade(alembic_cfg, 'base')


"""
BEST PRACTICES MIGRATIONS:
──────────────────────────
1. Una migration per ogni cambiamento logico
2. Scrivi sempre upgrade() E downgrade()
3. Testa migrations su copia del database prod
4. Non modificare migrations già applicate in prod
5. Usa autogenerate ma RIVEDI sempre il codice generato
6. Gestisci dati esistenti (populate, migrate data)
7. Usa batch operations per SQLite
8. Committa migrations in version control
"""


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 4.7 - ALEMBIC                                │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_4_7 = """
Q1. Alembic serve per:
    A) Query optimization    B) Database migrations    C) Connection pooling    D) Backup

Q2. alembic init crea:
    A) Database    B) Struttura directory per migrations    C) Tabelle    D) Modelli

Q3. alembic revision --autogenerate:
    A) Esegue migrations    B) Genera migration confrontando modelli con DB
    C) Revert migrations    D) Mostra history

Q4. alembic upgrade head applica:
    A) Prima migration    B) Tutte le migrations    C) Ultima migration    D) Nessuna

Q5. alembic downgrade -1:
    A) Applica una migration    B) Revert l'ultima migration    C) Revert tutte    D) Errore

Q6. op.add_column() è usato in:
    A) env.py    B) Migration file (upgrade)    C) alembic.ini    D) models.py

Q7. down_revision in una migration indica:
    A) Prossima migration    B) Migration precedente    C) Prima migration    D) Niente

Q8. alembic current mostra:
    A) Tutte le migrations    B) Revision attualmente applicata    C) Prossima migration    D) Errori

Q9. Per SQLite, alterazioni colonne richiedono:
    A) op.alter_column()    B) op.batch_alter_table()    C) DROP + CREATE    D) Non supportato

Q10. target_metadata in env.py contiene:
     A) Configurazione    B) MetaData dei modelli per autogenerate    C) Connessione    D) History

Q11. alembic upgrade head --sql genera:
     A) Errore    B) SQL senza eseguire    C) Migration file    D) Backup

Q12. Migrations devono essere:
     A) Solo upgrade    B) Sempre upgrade E downgrade    C) Solo downgrade    D) Opzionali
"""

ANSWERS_4_7 = """
RISPOSTE QUIZ 4.7 - ALEMBIC:
Q1: B - Database migrations (schema versioning)
Q2: B - Struttura directory per migrations
Q3: B - Genera migration confrontando modelli con DB attuale
Q4: B - Tutte le migrations (fino a head)
Q5: B - Revert l'ultima migration
Q6: B - Migration file nella funzione upgrade()
Q7: B - Migration precedente (dependency)
Q8: B - Revision attualmente applicata al DB
Q9: B - op.batch_alter_table() (SQLite non supporta ALTER)
Q10: B - MetaData dei modelli per autogenerate
Q11: B - SQL senza eseguire (per review o deploy manuale)
Q12: B - Sempre upgrade E downgrade (reversibilità)
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 4.8: CONNECTION POOLING
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    4.8 TEORIA: CONNECTION POOLING                            │
└──────────────────────────────────────────────────────────────────────────────┘

Connection Pool = riusa connessioni invece di crearne nuove.
Migliora performance in applicazioni con molte richieste.
"""

from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool, NullPool, StaticPool

# ENGINE CON POOL CONFIGURATO
engine = create_engine(
    'postgresql://user:pass@localhost/db',
    poolclass=QueuePool,
    pool_size=5,          # Connessioni nel pool
    max_overflow=10,      # Extra connessioni temporanee
    pool_timeout=30,      # Timeout per ottenere connessione
    pool_recycle=1800,    # Ricrea connessioni dopo N secondi
    pool_pre_ping=True    # Verifica connessione prima dell'uso
)


"""
TIPI DI POOL:
─────────────
QueuePool (default): Pool con coda, connessioni riutilizzate
NullPool: Nessun pool, nuova connessione ogni volta
StaticPool: Pool con una sola connessione (per SQLite in-memory)
SingletonThreadPool: Una connessione per thread
"""


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 4.8                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_4_8 = """
Q1. Connection pooling serve per:
    A) Sicurezza    B) Riusare connessioni    C) Backup    D) Logging

Q2. pool_size=5 significa:
    A) 5 tabelle    B) 5 connessioni nel pool    C) 5 query max    D) 5 MB

Q3. pool_pre_ping=True:
    A) Disabilita pool    B) Verifica connessione prima dell'uso    C) Crea backup    D) Log

Q4. NullPool significa:
    A) Pool infinito    B) Nessun pool    C) Pool vuoto    D) Pool statico

Q5. pool_recycle=1800 ricrea connessioni ogni:
    A) 1800 query    B) 30 minuti    C) 1800 righe    D) Mai
"""

ANSWERS_4_8 = """
RISPOSTE QUIZ 4.8:
Q1: B - Riusare connessioni (performance)
Q2: B - 5 connessioni mantenute nel pool
Q3: B - Verifica connessione prima dell'uso (evita stale connections)
Q4: B - Nessun pool (nuova connessione ogni volta)
Q5: B - 30 minuti (1800 secondi)
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 4.9: REPOSITORY PATTERN
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    4.9 TEORIA: REPOSITORY PATTERN                            │
└──────────────────────────────────────────────────────────────────────────────┘

Repository = astrae l'accesso ai dati.
Separa logica business da dettagli database.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Optional

T = TypeVar('T')


# REPOSITORY ASTRATTO
class Repository(ABC, Generic[T]):
    @abstractmethod
    def get(self, id: int) -> Optional[T]:
        pass
    
    @abstractmethod
    def get_all(self) -> List[T]:
        pass
    
    @abstractmethod
    def add(self, entity: T) -> T:
        pass
    
    @abstractmethod
    def update(self, entity: T) -> T:
        pass
    
    @abstractmethod
    def delete(self, id: int) -> bool:
        pass


# IMPLEMENTAZIONE CONCRETA
class UserRepository(Repository):
    def __init__(self, session):
        self.session = session
    
    def get(self, id: int) -> Optional[User]:
        return self.session.query(User).get(id)
    
    def get_all(self) -> List[User]:
        return self.session.query(User).all()
    
    def get_by_username(self, username: str) -> Optional[User]:
        return self.session.query(User).filter_by(
            username=username
        ).first()
    
    def add(self, user: User) -> User:
        self.session.add(user)
        self.session.flush()  # Ottiene ID senza commit
        return user
    
    def update(self, user: User) -> User:
        self.session.merge(user)
        return user
    
    def delete(self, id: int) -> bool:
        user = self.get(id)
        if user:
            self.session.delete(user)
            return True
        return False


# UNIT OF WORK PATTERN
class UnitOfWork:
    """
    Coordina repositories e gestisce transazioni.
    """
    def __init__(self, session_factory):
        self.session_factory = session_factory
    
    def __enter__(self):
        self.session = self.session_factory()
        self.users = UserRepository(self.session)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.rollback()
        self.session.close()
    
    def commit(self):
        self.session.commit()
    
    def rollback(self):
        self.session.rollback()


# USO
def create_user_example(uow_factory):
    with uow_factory() as uow:
        user = User(username='new_user', email='new@example.com')
        uow.users.add(user)
        uow.commit()


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 4.9                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_4_9 = """
Q1. Repository Pattern serve per:
    A) Velocità    B) Astrarre accesso dati    C) Sicurezza    D) Logging

Q2. Unit of Work coordina:
    A) Tabelle    B) Repositories e transazioni    C) Colonne    D) Index

Q3. session.flush() vs commit():
    A) Identici    B) flush non salva su disco    C) commit non salva    D) flush è più lento

Q4. Repository astrae:
    A) UI    B) Dettagli database    C) Network    D) Config

Q5. Generic[T] in Repository permette:
    A) Performance    B) Type hints generici    C) Async    D) Caching
"""

ANSWERS_4_9 = """
RISPOSTE QUIZ 4.9:
Q1: B - Astrarre accesso dati (separare business logic da DB)
Q2: B - Repositories e transazioni
Q3: B - flush sincronizza con DB ma non committa (rollbackable)
Q4: B - Dettagli database
Q5: B - Type hints generici (Repository[User], Repository[Post], etc.)
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    MODULE 4 FINAL TEST
# ══════════════════════════════════════════════════════════════════════════════

MODULE_4_FINAL_TEST = """
═══════════════════════════════════════════════════════════════════════════════
                    PP MODULE 4 - TEST FINALE (40 domande)
                             Tempo: 50 minuti
                             Pass: 70% (28/40)
═══════════════════════════════════════════════════════════════════════════════

SEZIONE A: BASICS & SQLITE (8 domande)
──────────────────────────────────────

Q1. ACID - A sta per:
    A) Advanced    B) Atomic    C) Async    D) Automatic

Q2. DDL include:
    A) SELECT    B) CREATE, DROP    C) COMMIT    D) INSERT

Q3. cursor.fetchone() restituisce:
    A) Lista    B) Tupla o None    C) Dict    D) Int

Q4. ? in execute() previene:
    A) Errori    B) SQL injection    C) Duplicati    D) Timeout


SEZIONE B: SQLALCHEMY (8 domande)
─────────────────────────────────

Q5. create_engine() crea:
    A) Tabella    B) Connessione/pool    C) Query    D) Modello

Q6. declarative_base() crea:
    A) Engine    B) Base per modelli ORM    C) Session    D) Query

Q7. session.add() fa:
    A) Salva subito    B) Aggiunge a pending    C) Query    D) Delete

Q8. session.commit() fa:
    A) Chiude    B) Salva su DB    C) Rollback    D) Query


SEZIONE C: RELATIONSHIPS (6 domande)
────────────────────────────────────

Q9. ForeignKey definisce:
    A) Primary key    B) Riferimento altra tabella    C) Index    D) Constraint

Q10. secondary= serve per:
     A) One-to-one    B) Many-to-many    C) One-to-many    D) Backup

Q11. uselist=False indica:
     A) Many-to-many    B) One-to-one    C) Lista vuota    D) Errore

Q12. joinedload risolve:
     A) N+1 problem    B) Duplicati    C) Deadlock    D) Timeout


SEZIONE D: TRANSACTIONS (6 domande)
───────────────────────────────────

Q13. session.rollback() fa:
     A) Salva    B) Annulla modifiche    C) Chiude    D) Query

Q14. begin_nested() crea:
     A) Nuova session    B) Savepoint    C) Nuova connessione    D) Backup

Q15. scoped_session serve per:
     A) Single thread    B) Thread-local sessions    C) Async    D) Test

Q16. ACID - I sta per:
     A) Indexed    B) Isolation    C) Integration    D) Instance


SEZIONE E: ALEMBIC (8 domande)
──────────────────────────────

Q17. Alembic serve per:
     A) Query    B) Migrations    C) Pooling    D) Backup

Q18. alembic upgrade head:
     A) Revert tutto    B) Applica tutte migrations    C) Mostra history    D) Crea migration

Q19. alembic downgrade -1:
     A) Applica una    B) Revert l'ultima    C) Revert tutte    D) Errore

Q20. --autogenerate confronta:
     A) File    B) Modelli con DB    C) Config    D) Logs

Q21. op.add_column() è in:
     A) env.py    B) upgrade() della migration    C) alembic.ini    D) models.py

Q22. down_revision indica:
     A) Prossima    B) Precedente    C) Prima    D) Ultima

Q23. Per SQLite ALTER serve:
     A) op.alter_column    B) batch_alter_table    C) DROP+CREATE    D) Non supportato

Q24. alembic current mostra:
     A) Tutte    B) Revision corrente    C) Prossima    D) Errori


SEZIONE F: POOLING & PATTERNS (4 domande)
─────────────────────────────────────────

Q25. pool_size=5 significa:
     A) 5 tabelle    B) 5 connessioni    C) 5 query    D) 5 MB

Q26. NullPool significa:
     A) Pool infinito    B) Nessun pool    C) Pool vuoto    D) Pool statico

Q27. Repository Pattern astrae:
     A) UI    B) Accesso dati    C) Network    D) Config

Q28. Unit of Work coordina:
     A) Tabelle    B) Repos e transazioni    C) Colonne    D) Index


═══════════════════════════════════════════════════════════════════════════════
                              FINE TEST
═══════════════════════════════════════════════════════════════════════════════
"""

MODULE_4_FINAL_ANSWERS = """
═══════════════════════════════════════════════════════════════════════════════
                    PP MODULE 4 - RISPOSTE TEST FINALE
═══════════════════════════════════════════════════════════════════════════════

Q1: B    Q2: B    Q3: B    Q4: B    Q5: B    Q6: B    Q7: B    Q8: B
Q9: B    Q10: B   Q11: B   Q12: A   Q13: B   Q14: B   Q15: B   Q16: B
Q17: B   Q18: B   Q19: B   Q20: B   Q21: B   Q22: B   Q23: B   Q24: B
Q25: B   Q26: B   Q27: B   Q28: B

PUNTEGGIO:
──────────
36-40: Eccellente! Pronto per PCPP2
32-35: Ottimo!
28-31: Buono (70% pass)
<28:   Rivedi sezioni deboli

SEZIONI DA RIVEDERE:
────────────────────
A (Q1-4):   Basics & SQLite
B (Q5-8):   SQLAlchemy
C (Q9-12):  Relationships
D (Q13-16): Transactions
E (Q17-24): Alembic ⭐
F (Q25-28): Pooling & Patterns

═══════════════════════════════════════════════════════════════════════════════
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    LABS - ESERCIZI PRATICI
# ══════════════════════════════════════════════════════════════════════════════

LABS = """
═══════════════════════════════════════════════════════════════════════════════
                    PP MODULE 4 - LABS
═══════════════════════════════════════════════════════════════════════════════

LAB 1: Crea un database SQLite con tabelle users e posts (one-to-many).

LAB 2: Implementa CRUD operations con SQLAlchemy ORM.

LAB 3: Aggiungi relationship many-to-many (posts ↔ tags).

LAB 4: Implementa eager loading per risolvere N+1 problem.

LAB 5: Crea un Repository generico riusabile.

LAB 6: Implementa Unit of Work pattern.

LAB 7: Configura connection pooling ottimale per web app.

LAB 8: Inizializza Alembic in un progetto esistente.

LAB 9: Crea migration per aggiungere colonna con dati default.

LAB 10: Crea migration per rinominare tabella.

LAB 11: Crea migration per aggiungere foreign key a tabella esistente.

LAB 12: Implementa downgrade per ogni migration creata.

LAB 13: Testa migrations con database di test.

LAB 14: Crea script che applica migrations in CI/CD.

LAB 15: Implementa sistema completo: models → repository → migrations.
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    ESECUZIONE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 78)
    print("PYTHON PROFESSIONAL - MODULE 4")
    print("Database Programming & ORM")
    print("=" * 78)
    print("""
    CONTENUTO:
    ──────────
    Section 4.1: Database Basics
    Section 4.2: SQLite
    Section 4.3: SQLAlchemy Core
    Section 4.4: SQLAlchemy ORM
    Section 4.5: Relationships
    Section 4.6: Transactions
    Section 4.7: Alembic Migrations ⭐
    Section 4.8: Connection Pooling
    Section 4.9: Repository Pattern
    
    COMANDI:
    ────────
    print(QUIZ_4_7)   → Quiz Alembic
    print(LABS)       → Esercizi pratici
    print(MODULE_4_FINAL_TEST)    → Test finale
    print(MODULE_4_FINAL_ANSWERS) → Risposte
    """)
