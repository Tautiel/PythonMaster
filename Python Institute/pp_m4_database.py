#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PYTHON PROFESSIONAL 2 - MODULE 4                          ║
║                    DATABASE PROGRAMMING                                       ║
║                    PCPP2 Prep - Database expected ~15% of exam               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Note: sqlite3 is covered in detail in module_database_part1_sqlite.py
      PostgreSQL/ORM in module_database_part2_postgresql_orm.py
This is a summary for PCPP2 exam prep.
"""

import sqlite3

# ══════════════════════════════════════════════════════════════════════════════
# 4.1 SQLITE3 BASICS
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("4.1 SQLITE3 BASICS")
print("=" * 70)

print("""
📋 SQLITE3 - Built-in Python database

# Connect (creates file if not exists)
conn = sqlite3.connect('database.db')
conn = sqlite3.connect(':memory:')  # In-memory database

# Get cursor
cursor = conn.cursor()

# Execute SQL
cursor.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
cursor.execute("INSERT INTO users (name) VALUES (?)", ("Marco",))

# Commit changes
conn.commit()

# Fetch results
cursor.execute("SELECT * FROM users")
row = cursor.fetchone()      # One row
rows = cursor.fetchall()     # All rows
rows = cursor.fetchmany(5)   # 5 rows

# Close
cursor.close()
conn.close()
""")

# Demo
conn = sqlite3.connect(':memory:')
cursor = conn.cursor()

# Create table
cursor.execute('''
    CREATE TABLE products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        price REAL
    )
''')

# Insert
cursor.execute("INSERT INTO products (name, price) VALUES (?, ?)", ("Laptop", 999.99))
cursor.execute("INSERT INTO products (name, price) VALUES (?, ?)", ("Mouse", 29.99))
conn.commit()

# Query
cursor.execute("SELECT * FROM products")
print(f"Products: {cursor.fetchall()}")

# ══════════════════════════════════════════════════════════════════════════════
# 4.2 PARAMETERIZED QUERIES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("4.2 PARAMETERIZED QUERIES (Prevent SQL Injection!)")
print("=" * 70)

print("""
📋 ALWAYS USE PARAMETERIZED QUERIES!

# BAD - SQL Injection vulnerable!
cursor.execute(f"SELECT * FROM users WHERE name = '{user_input}'")

# GOOD - Parameterized (? placeholder)
cursor.execute("SELECT * FROM users WHERE name = ?", (user_input,))

# GOOD - Named placeholders
cursor.execute("SELECT * FROM users WHERE name = :name", {"name": user_input})

# executemany for bulk inserts
data = [("A", 10), ("B", 20), ("C", 30)]
cursor.executemany("INSERT INTO products (name, price) VALUES (?, ?)", data)
""")

# ══════════════════════════════════════════════════════════════════════════════
# 4.3 TRANSACTIONS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("4.3 TRANSACTIONS")
print("=" * 70)

print("""
📋 TRANSACTION MANAGEMENT:

# Auto-commit mode
conn = sqlite3.connect('db.sqlite', isolation_level=None)

# Manual commit (default)
conn = sqlite3.connect('db.sqlite')
try:
    cursor.execute("INSERT ...")
    cursor.execute("UPDATE ...")
    conn.commit()  # Save all changes
except Exception:
    conn.rollback()  # Undo all changes

# Context manager (auto-commit on success, rollback on exception)
with conn:
    cursor.execute("INSERT ...")
    cursor.execute("UPDATE ...")
""")

# ══════════════════════════════════════════════════════════════════════════════
# 4.4 ROW FACTORY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("4.4 ROW FACTORY")
print("=" * 70)

# sqlite3.Row - access columns by name
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

cursor.execute("SELECT * FROM products WHERE id = 1")
row = cursor.fetchone()
print(f"Row as sqlite3.Row: name={row['name']}, price={row['price']}")

print("""
📋 ROW FACTORY:

# Default: tuples
cursor.fetchone()  # (1, 'Laptop', 999.99)

# sqlite3.Row: access by name
conn.row_factory = sqlite3.Row
row = cursor.fetchone()
row['name']  # 'Laptop'
row[0]       # 1 (still works)
row.keys()   # Column names

# Custom factory
def dict_factory(cursor, row):
    return {col[0]: row[i] for i, col in enumerate(cursor.description)}
conn.row_factory = dict_factory
""")

# ══════════════════════════════════════════════════════════════════════════════
# 4.5 ORM CONCEPTS (SQLAlchemy)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("4.5 ORM CONCEPTS (SQLAlchemy)")
print("=" * 70)

print("""
📋 ORM (Object-Relational Mapping):

from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker

# Create engine
engine = create_engine('sqlite:///database.db')

# Define model
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False)
    email = Column(String(100), unique=True)

# Create tables
Base.metadata.create_all(engine)

# Session
Session = sessionmaker(bind=engine)
session = Session()

# CRUD operations
# Create
user = User(name='Marco', email='marco@email.com')
session.add(user)
session.commit()

# Read
user = session.query(User).filter_by(name='Marco').first()
users = session.query(User).all()

# Update
user.email = 'new@email.com'
session.commit()

# Delete
session.delete(user)
session.commit()
""")

# ══════════════════════════════════════════════════════════════════════════════
# 4.6 CONNECTION POOLING
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("4.6 CONNECTION POOLING")
print("=" * 70)

print("""
📋 CONNECTION POOLING:

Why?
  - Database connections are expensive to create
  - Pool reuses existing connections
  - Improves performance

SQLAlchemy pooling:
engine = create_engine(
    'postgresql://user:pass@localhost/db',
    pool_size=5,           # Number of connections
    max_overflow=10,       # Extra connections when pool full
    pool_timeout=30,       # Wait time for connection
    pool_recycle=3600      # Recycle connections after 1 hour
)

# With raw psycopg2
from psycopg2 import pool

connection_pool = pool.SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    host='localhost',
    database='mydb'
)

conn = connection_pool.getconn()
# Use connection...
connection_pool.putconn(conn)  # Return to pool
""")

# ══════════════════════════════════════════════════════════════════════════════
# QUIZ
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("QUIZ")
print("=" * 70)

print("""
Q1. sqlite3.connect(':memory:') creates?
    → In-memory database

Q2. cursor.fetchone() returns?
    → Single row or None

Q3. Why use parameterized queries?
    → Prevent SQL injection

Q4. conn.commit() does what?
    → Saves all changes to database

Q5. conn.rollback() does what?
    → Undoes all uncommitted changes

Q6. sqlite3.Row allows?
    → Access columns by name

Q7. ORM stands for?
    → Object-Relational Mapping

Q8. Connection pooling benefit?
    → Reuses connections, improves performance
""")

conn.close()

print("\n" + "=" * 70)
print("DATABASE MODULE COMPLETE!")
print("=" * 70)
