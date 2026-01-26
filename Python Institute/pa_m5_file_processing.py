#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PYTHON PROFESSIONAL 1 - MODULE 5                          ║
║                    FILE PROCESSING                                            ║
║                    PCPP1-32-101 Section 5: 15% (6 domande)                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

SYLLABUS:
├── PCPP1 5.1 - sqlite3 (already covered in database module)
├── PCPP1 5.2 - XML processing
├── PCPP1 5.3 - CSV processing
├── PCPP1 5.4 - logging module
└── PCPP1 5.5 - ConfigParser
"""

import csv
import logging
import configparser
import xml.etree.ElementTree as ET
import io
import tempfile
import os

# ══════════════════════════════════════════════════════════════════════════════
# 5.1 CSV PROCESSING (ESAME!)
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("5.1 CSV PROCESSING (ESAME!)")
print("=" * 70)

print("""
📋 CSV MODULE:

csv.reader(file)      - Legge CSV come liste
csv.writer(file)      - Scrive CSV da liste
csv.DictReader(file)  - Legge CSV come dizionari
csv.DictWriter(file)  - Scrive CSV da dizionari
""")

# Writing CSV
csv_data = io.StringIO()
writer = csv.writer(csv_data)
writer.writerow(['name', 'age', 'city'])  # Header
writer.writerow(['Marco', 25, 'Milan'])
writer.writerow(['Anna', 30, 'Rome'])
writer.writerow(['Luca', 22, 'Naples'])

csv_content = csv_data.getvalue()
print("CSV generato:")
print(csv_content)

# Reading CSV with reader
print("📐 csv.reader() - Legge come liste:")
csv_data = io.StringIO(csv_content)
reader = csv.reader(csv_data)
header = next(reader)  # Prima riga (header)
print(f"Header: {header}")
for row in reader:
    print(f"  {row}")

# Reading CSV with DictReader
print("\n📐 csv.DictReader() - Legge come dizionari:")
csv_data = io.StringIO(csv_content)
reader = csv.DictReader(csv_data)
for row in reader:
    print(f"  {dict(row)}")

# Writing with DictWriter
print("\n📐 csv.DictWriter() - Scrive da dizionari:")
print("""
data = [
    {'name': 'Marco', 'age': 25},
    {'name': 'Anna', 'age': 30}
]

with open('output.csv', 'w', newline='') as f:
    fieldnames = ['name', 'age']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)
""")

# CSV Options
print("""
📋 CSV OPTIONS:

# Custom delimiter
reader = csv.reader(file, delimiter=';')

# Custom quote character
reader = csv.reader(file, quotechar='"')

# Tab-separated
reader = csv.reader(file, delimiter='\\t')

# Skip initial spaces
reader = csv.reader(file, skipinitialspace=True)
""")

# ══════════════════════════════════════════════════════════════════════════════
# 5.2 LOGGING MODULE (ESAME!)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("5.2 LOGGING MODULE (ESAME!)")
print("=" * 70)

print("""
📋 LOGGING LEVELS (dal meno al più grave):

DEBUG    (10) - Informazioni dettagliate, diagnostica
INFO     (20) - Conferma che le cose funzionano
WARNING  (30) - Qualcosa di inaspettato, ma programma funziona
ERROR    (40) - Errore serio, funzione non eseguita
CRITICAL (50) - Errore grave, programma potrebbe terminare
""")

# Basic logging
print("\n📐 Basic Logging:")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Create logger
logger = logging.getLogger(__name__)

# Log messages (solo printed in output reale)
print("logger.debug('Debug message')")
print("logger.info('Info message')")
print("logger.warning('Warning message')")
print("logger.error('Error message')")
print("logger.critical('Critical message')")

# Logging configuration
print("""
📋 LOGGING CONFIGURATION:

# Basic config
logging.basicConfig(
    level=logging.DEBUG,               # Minimum level to log
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='app.log',                # Log to file
    filemode='a'                       # 'a'=append, 'w'=overwrite
)

# Format placeholders:
%(asctime)s    - Timestamp
%(name)s       - Logger name
%(levelname)s  - Level (DEBUG, INFO, etc.)
%(message)s    - Log message
%(filename)s   - Source filename
%(lineno)d     - Line number
%(funcName)s   - Function name
""")

# Handlers
print("""
📋 LOGGING HANDLERS:

# File handler
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Stream handler (console)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

# Rotating file handler
from logging.handlers import RotatingFileHandler
handler = RotatingFileHandler(
    'app.log',
    maxBytes=1000000,    # 1MB
    backupCount=5        # Keep 5 backup files
)
""")

# Exception logging
print("""
📋 LOGGING EXCEPTIONS:

try:
    result = 1 / 0
except Exception as e:
    logger.exception("An error occurred")  # Includes traceback
    # OR
    logger.error("Error: %s", e, exc_info=True)
""")

# ══════════════════════════════════════════════════════════════════════════════
# 5.3 CONFIGPARSER
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("5.3 CONFIGPARSER")
print("=" * 70)

print("""
📋 INI FILE FORMAT:

[section1]
key1 = value1
key2 = value2

[section2]
key3 = value3
""")

# Create config
config = configparser.ConfigParser()

config['DEFAULT'] = {
    'ServerAliveInterval': '45',
    'Compression': 'yes'
}

config['bitbucket.org'] = {
    'User': 'hg'
}

config['topsecret.server.com'] = {
    'Host Port': '50022',
    'ForwardX11': 'no'
}

# Write to string (simulate file)
config_str = io.StringIO()
config.write(config_str)
print("Config generato:")
print(config_str.getvalue()[:200] + "...")

# Reading config
print("\n📐 Reading Configuration:")
config_data = """
[database]
host = localhost
port = 5432
name = mydb
user = admin
password = secret

[logging]
level = DEBUG
file = app.log
"""

config = configparser.ConfigParser()
config.read_string(config_data)

print(f"Sections: {config.sections()}")
print(f"database.host: {config['database']['host']}")
print(f"database.port: {config.getint('database', 'port')}")
print(f"logging.level: {config.get('logging', 'level')}")

# Type conversion methods
print("""
📋 TYPE CONVERSION METHODS:

config.get(section, option)          - String
config.getint(section, option)       - Integer
config.getfloat(section, option)     - Float
config.getboolean(section, option)   - Boolean (yes/no, true/false, 1/0)

# With fallback default
config.get(section, option, fallback='default')
""")

# ══════════════════════════════════════════════════════════════════════════════
# 5.4 XML PROCESSING (ADVANCED)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("5.4 XML PROCESSING (ADVANCED)")
print("=" * 70)

# Create XML
print("📐 Creating XML:")
root = ET.Element('catalog')

book1 = ET.SubElement(root, 'book', id='1')
title1 = ET.SubElement(book1, 'title')
title1.text = 'Python Basics'
price1 = ET.SubElement(book1, 'price')
price1.text = '29.99'

book2 = ET.SubElement(root, 'book', id='2')
title2 = ET.SubElement(book2, 'title')
title2.text = 'Advanced Python'
price2 = ET.SubElement(book2, 'price')
price2.text = '49.99'

# Pretty print (sort of)
xml_str = ET.tostring(root, encoding='unicode')
print(f"Generated XML:\n{xml_str}")

# Parsing and searching
print("\n📐 Parsing and Searching:")
print(f"root.tag: {root.tag}")
print(f"root.findall('book'): {len(root.findall('book'))} books")

for book in root.findall('book'):
    book_id = book.get('id')
    title = book.find('title').text
    price = book.find('price').text
    print(f"  Book {book_id}: {title} - ${price}")

# XPath-like queries
print("""
📋 FINDING ELEMENTS:

root.find('tag')          - First direct child
root.findall('tag')       - All direct children
root.find('.//tag')       - First anywhere (recursive)
root.findall('.//tag')    - All anywhere (recursive)
root.findall(".//*[@id]") - All with 'id' attribute
""")

# ══════════════════════════════════════════════════════════════════════════════
# 5.5 COMBINING CONCEPTS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("5.5 PRACTICAL EXAMPLE")
print("=" * 70)

print("""
📋 COMPLETE EXAMPLE: Config-driven CSV to XML converter

# config.ini
[input]
file = data.csv
delimiter = ,

[output]
file = output.xml
root_element = records

[logging]
level = INFO
file = converter.log

# converter.py
import csv
import xml.etree.ElementTree as ET
import logging
import configparser

def convert(config_file):
    # Load config
    config = configparser.ConfigParser()
    config.read(config_file)
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        filename=config['logging']['file']
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting conversion")
    
    # Read CSV
    with open(config['input']['file']) as f:
        reader = csv.DictReader(f, delimiter=config['input']['delimiter'])
        rows = list(reader)
    
    logger.info(f"Read {len(rows)} rows from CSV")
    
    # Create XML
    root = ET.Element(config['output']['root_element'])
    for row in rows:
        record = ET.SubElement(root, 'record')
        for key, value in row.items():
            elem = ET.SubElement(record, key)
            elem.text = value
    
    # Write XML
    tree = ET.ElementTree(root)
    tree.write(config['output']['file'])
    
    logger.info("Conversion complete")
""")

# ══════════════════════════════════════════════════════════════════════════════
# QUIZ
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("QUIZ DI VERIFICA")
print("=" * 70)

print("""
Q1. csv.DictReader legge righe come?
    A) Liste  B) Tuple  C) Dizionari  D) Set
    → RISPOSTA: C

Q2. Quale logging level è più grave?
    A) DEBUG  B) INFO  C) WARNING  D) CRITICAL
    → RISPOSTA: D

Q3. logging.DEBUG ha valore numerico?
    A) 0  B) 10  C) 20  D) 30
    → RISPOSTA: B

Q4. ConfigParser legge file in formato?
    A) JSON  B) XML  C) INI  D) YAML
    → RISPOSTA: C

Q5. config.getint() restituisce?
    A) String  B) Integer  C) Float  D) Boolean
    → RISPOSTA: B

Q6. ET.SubElement() crea?
    A) Root  B) Child element  C) Attribute  D) Text
    → RISPOSTA: B

Q7. element.find('.//tag') cerca?
    A) Solo figli diretti  B) Ricorsivamente  C) Attributi  D) Testo
    → RISPOSTA: B

Q8. csv.writer.writerow() accetta?
    A) String  B) Dict  C) List/tuple  D) Any
    → RISPOSTA: C
""")

print("\n" + "=" * 70)
print("FILE PROCESSING MODULE COMPLETATO!")
print("TUTTI I MODULI PCPP1 COMPLETATI!")
print("=" * 70)
