"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║              PYTHON ADVANCED (PA) - MODULE 4                                 ║
║         File Processing, Data Formats & Configuration                        ║
║                                                                              ║
║                     Allineato al Syllabus PCPP1-32-10x                       ║
║                         OpenEDG Python Institute                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

PCPP1 Exam Block 4: File Processing (25%)

STRUTTURA MODULO:
├── Section 4.1: Text Files & Encoding
├── Section 4.2: CSV Processing
├── Section 4.3: JSON Processing
├── Section 4.4: XML Processing
├── Section 4.5: SQLite Database
├── Section 4.6: Logging
├── Section 4.7: ConfigParser ⭐ NEW
├── Section 4.8: pathlib
├── Labs (15 esercizi)
└── Module 4 Test (40 domande)

TEMPO STIMATO: 8-10 ore

═══════════════════════════════════════════════════════════════════════════════
"""

import os
import csv
import json
import sqlite3
import logging
import configparser
from pathlib import Path
from xml.etree import ElementTree as ET


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 4.1: TEXT FILES & ENCODING
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    4.1 TEORIA: FILE DI TESTO                                 │
└──────────────────────────────────────────────────────────────────────────────┘

MODI DI APERTURA:
─────────────────
'r'  - Read (default) - file deve esistere
'w'  - Write - crea/sovrascrive
'a'  - Append - crea/aggiunge alla fine
'x'  - Exclusive create - errore se esiste
'r+' - Read and write
'w+' - Write and read (sovrascrive)
'a+' - Append and read

'b'  - Binary mode (es: 'rb', 'wb')
't'  - Text mode (default)


ENCODING:
─────────
UTF-8 è lo standard per testo internazionale.
Sempre specificare encoding per evitare problemi cross-platform.
"""

# SCRITTURA FILE
with open('test.txt', 'w', encoding='utf-8') as f:
    f.write("Prima riga\n")
    f.write("Seconda riga\n")
    f.writelines(["Terza\n", "Quarta\n"])

# LETTURA FILE
with open('test.txt', 'r', encoding='utf-8') as f:
    content = f.read()       # Tutto il file come stringa
    # content = f.readline()  # Una riga
    # content = f.readlines() # Lista di righe

# ITERAZIONE (memory efficient)
with open('test.txt', 'r', encoding='utf-8') as f:
    for line in f:
        print(line.strip())

# POSIZIONE NEL FILE
with open('test.txt', 'r+', encoding='utf-8') as f:
    pos = f.tell()        # Posizione corrente
    f.seek(0)             # Torna all'inizio
    f.seek(0, 2)          # Va alla fine (0 offset from end)


"""
PATHLIB (moderno):
──────────────────
"""

from pathlib import Path

path = Path('test.txt')
content = path.read_text(encoding='utf-8')
path.write_text("Nuovo contenuto", encoding='utf-8')

# Operazioni su path
path.exists()
path.is_file()
path.is_dir()
path.name       # 'test.txt'
path.stem       # 'test'
path.suffix     # '.txt'
path.parent     # directory parent


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 4.1                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_4_1 = """
Q1. Quale modo sovrascrive un file esistente?
    A) 'r'    B) 'w'    C) 'a'    D) 'r+'

Q2. Quale modo aggiunge alla fine del file?
    A) 'r'    B) 'w'    C) 'a'    D) 'x'

Q3. 'rb' significa:
    A) Read backup    B) Read binary    C) Read both    D) Read buffer

Q4. readlines() restituisce:
    A) Una stringa    B) Lista di stringhe    C) Un intero    D) Un dizionario

Q5. seek(0) fa:
    A) Chiude il file    B) Va all'inizio    C) Elimina contenuto    D) Conta righe

Q6. Path('dir/file.txt').stem restituisce:
    A) 'dir/file.txt'    B) 'file'    C) '.txt'    D) 'dir'

Q7. Path('dir/file.txt').suffix restituisce:
    A) 'file.txt'    B) 'file'    C) '.txt'    D) 'txt'

Q8. encoding='utf-8' è importante per:
    A) Velocità    B) Caratteri internazionali    C) Compressione    D) Sicurezza
"""

ANSWERS_4_1 = """
RISPOSTE QUIZ 4.1:
Q1: B - 'w' sovrascrive (write)
Q2: C - 'a' aggiunge (append)
Q3: B - Read binary
Q4: B - Lista di stringhe (con \\n)
Q5: B - Va all'inizio (posizione 0)
Q6: B - 'file' (nome senza estensione)
Q7: C - '.txt' (estensione con punto)
Q8: B - Caratteri internazionali (€, ü, 中文, etc.)
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 4.2: CSV PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    4.2 TEORIA: CSV                                           │
└──────────────────────────────────────────────────────────────────────────────┘

CSV = Comma Separated Values
Formato semplice per dati tabellari.
"""

import csv

# SCRITTURA CSV
data = [
    ['Nome', 'Età', 'Città'],
    ['Marco', 30, 'Milano'],
    ['Anna', 25, 'Roma']
]

with open('data.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(data)
    # oppure: writer.writerow(['Singola', 'Riga'])


# LETTURA CSV
with open('data.csv', 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)  # Lista di stringhe


# DICTREADER/DICTWRITER (più comodo!)
with open('data.csv', 'r', newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row['Nome'], row['Età'])  # Accesso per nome colonna


data_dict = [
    {'Nome': 'Marco', 'Età': 30, 'Città': 'Milano'},
    {'Nome': 'Anna', 'Età': 25, 'Città': 'Roma'}
]

with open('data.csv', 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['Nome', 'Età', 'Città']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()  # Scrive intestazione
    writer.writerows(data_dict)


# DELIMITATORI CUSTOM
with open('data.tsv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter='\t')  # Tab separated
    writer.writerows(data)


"""
newline='' È IMPORTANTE!
────────────────────────
Su Windows, senza newline='', csv aggiunge righe vuote extra.
È una best practice sempre includerlo.
"""


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 4.2                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_4_2 = """
Q1. csv.reader restituisce:
    A) Dizionari    B) Liste    C) Tuple    D) Stringhe

Q2. csv.DictReader restituisce:
    A) Dizionari    B) Liste    C) Tuple    D) Stringhe

Q3. newline='' serve per:
    A) Velocità    B) Gestire newlines cross-platform    C) Compressione    D) Encoding

Q4. csv.writer.writerow() scrive:
    A) Tutto il file    B) Una riga    C) Un carattere    D) Un header

Q5. DictWriter.writeheader() scrive:
    A) I dati    B) L'intestazione    C) Il footer    D) I metadati

Q6. delimiter='\\t' crea un file:
    A) Comma separated    B) Tab separated    C) Pipe separated    D) Space separated

Q7. Per leggere un CSV con ; come separatore:
    A) csv.reader(f, delimiter=';')    B) csv.reader(f, sep=';')
    C) csv.reader(f, separator=';')    D) csv.semicolon_reader(f)
"""

ANSWERS_4_2 = """
RISPOSTE QUIZ 4.2:
Q1: B - Liste (righe come liste di stringhe)
Q2: A - Dizionari (chiavi = header)
Q3: B - Gestire newlines cross-platform
Q4: B - Una riga
Q5: B - L'intestazione (fieldnames)
Q6: B - Tab separated (TSV)
Q7: A - csv.reader(f, delimiter=';')
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 4.3: JSON PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    4.3 TEORIA: JSON                                          │
└──────────────────────────────────────────────────────────────────────────────┘

JSON = JavaScript Object Notation
Standard per API web e configurazioni.
"""

import json

# PYTHON → JSON (serialization)
data = {
    'nome': 'Marco',
    'età': 30,
    'lingue': ['Python', 'JavaScript'],
    'attivo': True,
    'saldo': None
}

# A stringa
json_str = json.dumps(data)
print(json_str)

# A file
with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)


# JSON → PYTHON (deserialization)
# Da stringa
data = json.loads(json_str)

# Da file
with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)


"""
MAPPING TIPI:
─────────────
Python          JSON
───────         ────
dict        →   object {}
list/tuple  →   array []
str         →   string ""
int/float   →   number
True/False  →   true/false
None        →   null
"""


# OPZIONI UTILI
json.dumps(data, 
    indent=2,           # Pretty print
    sort_keys=True,     # Ordina chiavi
    ensure_ascii=False  # Permette caratteri non-ASCII
)


# CUSTOM ENCODER (per tipi non supportati)
from datetime import datetime

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

data = {'timestamp': datetime.now()}
json.dumps(data, cls=CustomEncoder)


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 4.3                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_4_3 = """
Q1. json.dumps() converte Python in:
    A) File    B) Stringa JSON    C) Bytes    D) Dizionario

Q2. json.loads() converte JSON string in:
    A) File    B) Stringa    C) Oggetto Python    D) Bytes

Q3. json.dump() scrive su:
    A) Stringa    B) File    C) Console    D) Database

Q4. None in Python diventa in JSON:
    A) None    B) 'None'    C) null    D) undefined

Q5. True in Python diventa in JSON:
    A) True    B) 'True'    C) true    D) 1

Q6. indent=2 in json.dumps serve per:
    A) Compressione    B) Pretty print    C) Validazione    D) Encoding

Q7. ensure_ascii=False permette:
    A) Solo ASCII    B) Caratteri Unicode    C) Binary    D) Compressione
"""

ANSWERS_4_3 = """
RISPOSTE QUIZ 4.3:
Q1: B - Stringa JSON
Q2: C - Oggetto Python (dict, list, etc.)
Q3: B - File
Q4: C - null
Q5: C - true (lowercase in JSON)
Q6: B - Pretty print (formattazione leggibile)
Q7: B - Caratteri Unicode (€, ü, etc.)
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 4.4: XML PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    4.4 TEORIA: XML                                           │
└──────────────────────────────────────────────────────────────────────────────┘

XML = eXtensible Markup Language
Formato strutturato per dati gerarchici.
"""

from xml.etree import ElementTree as ET

# PARSING XML
xml_string = """
<users>
    <user id="1">
        <name>Marco</name>
        <age>30</age>
    </user>
    <user id="2">
        <name>Anna</name>
        <age>25</age>
    </user>
</users>
"""

# Da stringa
root = ET.fromstring(xml_string)

# Da file
# tree = ET.parse('data.xml')
# root = tree.getroot()


# NAVIGAZIONE
print(root.tag)  # 'users'

for user in root.findall('user'):
    user_id = user.get('id')  # Attributo
    name = user.find('name').text  # Testo elemento
    age = user.find('age').text
    print(f"ID: {user_id}, Nome: {name}, Età: {age}")


# METODI DI RICERCA
root.find('user')       # Primo match o None
root.findall('user')    # Lista di tutti i match
root.findall('.//name') # Ricerca ricorsiva


# CREAZIONE XML
root = ET.Element('users')

user1 = ET.SubElement(root, 'user')
user1.set('id', '1')

name = ET.SubElement(user1, 'name')
name.text = 'Marco'

# Salvataggio
tree = ET.ElementTree(root)
tree.write('output.xml', encoding='utf-8', xml_declaration=True)


# MODIFICA
for user in root.findall('user'):
    age = user.find('age')
    if age is not None:
        age.text = str(int(age.text) + 1)  # Incrementa età


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 4.4                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_4_4 = """
Q1. ET.fromstring() parsa XML da:
    A) File    B) Stringa    C) URL    D) Database

Q2. ET.parse() parsa XML da:
    A) File    B) Stringa    C) URL    D) Database

Q3. element.find() restituisce:
    A) Lista    B) Primo match o None    C) Tutti i match    D) Errore

Q4. element.findall() restituisce:
    A) Primo match    B) Lista (anche vuota)    C) None    D) Errore

Q5. element.get('attr') restituisce:
    A) Testo elemento    B) Valore attributo    C) Tag    D) Figli

Q6. element.text restituisce:
    A) Attributi    B) Contenuto testuale    C) Tag    D) Figli

Q7. ET.SubElement(parent, 'tag') crea:
    A) Root element    B) Child element    C) Attributo    D) Commento
"""

ANSWERS_4_4 = """
RISPOSTE QUIZ 4.4:
Q1: B - Stringa (ET.parse per file)
Q2: A - File
Q3: B - Primo match o None
Q4: B - Lista (anche vuota se nessun match)
Q5: B - Valore attributo
Q6: B - Contenuto testuale tra i tag
Q7: B - Child element (figlio)
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 4.5: SQLITE DATABASE
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    4.5 TEORIA: SQLITE                                        │
└──────────────────────────────────────────────────────────────────────────────┘

SQLite = Database relazionale embedded (file-based).
Incluso in Python, nessuna installazione richiesta.
"""

import sqlite3

# CONNESSIONE
conn = sqlite3.connect('database.db')  # Crea se non esiste
# conn = sqlite3.connect(':memory:')   # Database in memoria
cursor = conn.cursor()


# CREAZIONE TABELLA
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        age INTEGER,
        email TEXT UNIQUE
    )
''')


# INSERT (con parametri - SICURO!)
cursor.execute(
    "INSERT INTO users (name, age, email) VALUES (?, ?, ?)",
    ('Marco', 30, 'marco@example.com')
)

# Insert multipli
users = [
    ('Anna', 25, 'anna@example.com'),
    ('Luca', 35, 'luca@example.com')
]
cursor.executemany(
    "INSERT INTO users (name, age, email) VALUES (?, ?, ?)",
    users
)


# SELECT
cursor.execute("SELECT * FROM users")
all_rows = cursor.fetchall()    # Lista di tuple
one_row = cursor.fetchone()     # Una tupla o None

cursor.execute("SELECT * FROM users WHERE age > ?", (25,))
for row in cursor:  # Iterazione efficiente
    print(row)


# UPDATE
cursor.execute(
    "UPDATE users SET age = ? WHERE name = ?",
    (31, 'Marco')
)


# DELETE
cursor.execute("DELETE FROM users WHERE id = ?", (1,))


# COMMIT E CHIUSURA
conn.commit()  # Salva le modifiche!
conn.close()


# CONTEXT MANAGER (best practice)
with sqlite3.connect('database.db') as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    # commit automatico se nessuna eccezione


# ROW FACTORY (risultati come dizionari)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()
cursor.execute("SELECT * FROM users")
row = cursor.fetchone()
print(row['name'])  # Accesso per nome colonna


"""
⚠️ SQL INJECTION PREVENTION:
────────────────────────────
MAI concatenare stringhe! Usare sempre parametri (?)

SBAGLIATO:
cursor.execute(f"SELECT * FROM users WHERE name = '{name}'")

CORRETTO:
cursor.execute("SELECT * FROM users WHERE name = ?", (name,))
"""


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 4.5                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_4_5 = """
Q1. sqlite3.connect(':memory:') crea:
    A) File temporaneo    B) Database in RAM    C) Connessione remota    D) Errore

Q2. cursor.execute() con ? serve per:
    A) Velocità    B) Prevenire SQL injection    C) Formattazione    D) Debug

Q3. fetchall() restituisce:
    A) Una tupla    B) Lista di tuple    C) Un dizionario    D) Un intero

Q4. fetchone() restituisce:
    A) Lista    B) Tupla o None    C) Dizionario    D) Intero

Q5. conn.commit() serve per:
    A) Leggere dati    B) Salvare modifiche    C) Chiudere connessione    D) Creare tabelle

Q6. executemany() serve per:
    A) Una query    B) Query multiple con dati diversi    C) Transazioni    D) Backup

Q7. row_factory = sqlite3.Row permette:
    A) Righe più veloci    B) Accesso per nome colonna    C) Compressione    D) Encryption
"""

ANSWERS_4_5 = """
RISPOSTE QUIZ 4.5:
Q1: B - Database in RAM (memoria)
Q2: B - Prevenire SQL injection (parametrizzazione)
Q3: B - Lista di tuple
Q4: B - Tupla o None
Q5: B - Salvare modifiche (persist)
Q6: B - Query multiple con dati diversi
Q7: B - Accesso per nome colonna (come dizionario)
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 4.6: LOGGING
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    4.6 TEORIA: LOGGING                                       │
└──────────────────────────────────────────────────────────────────────────────┘

Logging = registrazione eventi durante l'esecuzione.
Meglio di print() per applicazioni in produzione.
"""

import logging

# CONFIGURAZIONE BASE
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app.log',  # Ometti per stdout
    filemode='a'         # 'w' per sovrascrivere
)


# LIVELLI (dal meno al più grave)
logging.debug("Informazione di debug")      # 10
logging.info("Informazione generale")       # 20
logging.warning("Attenzione")               # 30
logging.error("Errore")                     # 40
logging.critical("Errore critico")          # 50


# LOGGER PERSONALIZZATO
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Handler per file
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.ERROR)

# Handler per console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Aggiungi handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("Questo va solo in console")
logger.error("Questo va sia in console che in file")


# LOGGING ECCEZIONI
try:
    x = 1 / 0
except Exception:
    logger.exception("Errore durante divisione")  # Include traceback!


# ROTATING FILE HANDLER
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

# Rotazione per dimensione
handler = RotatingFileHandler(
    'app.log',
    maxBytes=1024*1024,  # 1MB
    backupCount=5
)

# Rotazione per tempo
handler = TimedRotatingFileHandler(
    'app.log',
    when='midnight',     # 'S', 'M', 'H', 'D', 'midnight'
    backupCount=7
)


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 4.6                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_4_6 = """
Q1. Quale livello è il PIÙ GRAVE?
    A) DEBUG    B) WARNING    C) ERROR    D) CRITICAL

Q2. Quale livello è il MENO GRAVE?
    A) DEBUG    B) INFO    C) WARNING    D) ERROR

Q3. logger.exception() include:
    A) Solo messaggio    B) Traceback completo    C) Solo errore    D) Niente

Q4. RotatingFileHandler ruota per:
    A) Tempo    B) Dimensione    C) Data    D) Errori

Q5. TimedRotatingFileHandler ruota per:
    A) Dimensione    B) Tempo    C) Errori    D) Livello

Q6. logging.getLogger(__name__) restituisce:
    A) Root logger    B) Logger col nome del modulo    C) Nuovo file    D) Handler

Q7. basicConfig() può essere chiamato:
    A) Infinite volte    B) Solo una volta (prima chiamata vince)    C) Mai    D) Solo con file

Q8. handler.setLevel(ERROR) significa:
    A) Solo ERROR    B) ERROR e superiori    C) Tutto sotto ERROR    D) Disabilita logging
"""

ANSWERS_4_6 = """
RISPOSTE QUIZ 4.6:
Q1: D - CRITICAL (50)
Q2: A - DEBUG (10)
Q3: B - Traceback completo
Q4: B - Dimensione (maxBytes)
Q5: B - Tempo (when='midnight', etc.)
Q6: B - Logger col nome del modulo
Q7: B - Solo una volta (successive chiamate ignorate)
Q8: B - ERROR e superiori (ERROR, CRITICAL)
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 4.7: CONFIGPARSER ⭐ IMPORTANTE PER PCPP1
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    4.7 TEORIA: CONFIGPARSER                                  │
└──────────────────────────────────────────────────────────────────────────────┘

ConfigParser = modulo per leggere/scrivere file di configurazione INI.
Formato standard per configurazioni applicative.

FORMATO FILE INI:
─────────────────
[section_name]
key = value
another_key = another value

[another_section]
setting = data
"""

import configparser

# ═══════════════════════════════════════════════════════════════════════════
# CREAZIONE E SCRITTURA CONFIG
# ═══════════════════════════════════════════════════════════════════════════

config = configparser.ConfigParser()

# Aggiungere sezioni e valori
config['DEFAULT'] = {
    'debug': 'false',
    'log_level': 'INFO'
}

config['database'] = {
    'host': 'localhost',
    'port': '5432',
    'name': 'myapp',
    'user': 'admin'
}

config['api'] = {
    'url': 'https://api.example.com',
    'timeout': '30',
    'retry': '3'
}

# Scrivere su file
with open('config.ini', 'w') as f:
    config.write(f)

"""
Il file config.ini sarà:

[DEFAULT]
debug = false
log_level = INFO

[database]
host = localhost
port = 5432
name = myapp
user = admin

[api]
url = https://api.example.com
timeout = 30
retry = 3
"""


# ═══════════════════════════════════════════════════════════════════════════
# LETTURA CONFIG
# ═══════════════════════════════════════════════════════════════════════════

config = configparser.ConfigParser()
config.read('config.ini')

# Accesso ai valori
host = config['database']['host']          # 'localhost'
port = config['database']['port']          # '5432' (stringa!)

# Conversione tipi
port_int = config.getint('database', 'port')     # 5432 (int)
timeout = config.getfloat('api', 'timeout')      # 30.0 (float)
debug = config.getboolean('DEFAULT', 'debug')    # False (bool)

# Valori con default (se chiave non esiste)
missing = config.get('database', 'password', fallback='secret')


# ═══════════════════════════════════════════════════════════════════════════
# SEZIONE DEFAULT
# ═══════════════════════════════════════════════════════════════════════════

"""
La sezione [DEFAULT] è SPECIALE:
- I suoi valori sono ereditati da TUTTE le altre sezioni
- È come un "template" per valori comuni
"""

config = configparser.ConfigParser()
config['DEFAULT'] = {'timeout': '30'}
config['api1'] = {'url': 'https://api1.com'}
config['api2'] = {'url': 'https://api2.com', 'timeout': '60'}

# api1 eredita timeout da DEFAULT
print(config['api1']['timeout'])  # '30'

# api2 sovrascrive
print(config['api2']['timeout'])  # '60'


# ═══════════════════════════════════════════════════════════════════════════
# ITERAZIONE E VERIFICA
# ═══════════════════════════════════════════════════════════════════════════

config = configparser.ConfigParser()
config.read('config.ini')

# Lista sezioni (esclude DEFAULT)
sections = config.sections()  # ['database', 'api']

# Verifica esistenza sezione
if config.has_section('database'):
    print("Database section exists")

# Verifica esistenza chiave
if config.has_option('database', 'host'):
    print("Host option exists")

# Iterare su sezione
for key, value in config['database'].items():
    print(f"{key} = {value}")

# Iterare su tutte le sezioni
for section in config.sections():
    print(f"[{section}]")
    for key, value in config[section].items():
        print(f"  {key} = {value}")


# ═══════════════════════════════════════════════════════════════════════════
# MODIFICA CONFIG
# ═══════════════════════════════════════════════════════════════════════════

config = configparser.ConfigParser()
config.read('config.ini')

# Modificare valore
config['database']['port'] = '3306'

# Aggiungere sezione
config.add_section('cache')
config.set('cache', 'enabled', 'true')
config.set('cache', 'ttl', '3600')

# Rimuovere
config.remove_option('api', 'retry')  # Rimuove chiave
config.remove_section('cache')        # Rimuove sezione

# Salvare modifiche
with open('config.ini', 'w') as f:
    config.write(f)


# ═══════════════════════════════════════════════════════════════════════════
# INTERPOLAZIONE (variabili)
# ═══════════════════════════════════════════════════════════════════════════

"""
ConfigParser supporta INTERPOLAZIONE = riferimenti a altre variabili.

BASIC INTERPOLATION (default):
Usa %(key)s per riferirsi ad altre chiavi nella stessa sezione o DEFAULT.
"""

config_string = """
[paths]
home = /home/user
data = %(home)s/data
logs = %(home)s/logs
"""

config = configparser.ConfigParser()
config.read_string(config_string)

print(config['paths']['data'])  # '/home/user/data'
print(config['paths']['logs'])  # '/home/user/logs'


"""
EXTENDED INTERPOLATION:
Usa ${section:key} per riferirsi a qualsiasi sezione.
"""

config = configparser.ConfigParser(
    interpolation=configparser.ExtendedInterpolation()
)

config_string = """
[DEFAULT]
base_url = https://api.example.com

[production]
url = ${DEFAULT:base_url}/v2

[development]
url = http://localhost:8000
"""

config.read_string(config_string)
print(config['production']['url'])  # 'https://api.example.com/v2'


# ═══════════════════════════════════════════════════════════════════════════
# OPZIONI AVANZATE
# ═══════════════════════════════════════════════════════════════════════════

# Case-insensitive keys (default)
config = configparser.ConfigParser()
config['section'] = {'Key': 'value'}
print(config['section']['key'])  # 'value' (lowercase funziona!)

# Case-sensitive keys
config = configparser.RawConfigParser()
config.optionxform = str  # Mantieni case originale

# Permettere chiavi senza valori
config = configparser.ConfigParser(allow_no_value=True)
config.read_string("[section]\nkey_without_value")

# Delimitatori custom
config = configparser.ConfigParser(delimiters=('=', ':'))

# Commenti custom
config = configparser.ConfigParser(comment_prefixes=('#', ';'))


# ═══════════════════════════════════════════════════════════════════════════
# ESEMPIO PRATICO: CONFIG TRADING BOT
# ═══════════════════════════════════════════════════════════════════════════

def create_trading_config():
    """Crea configurazione per trading bot."""
    config = configparser.ConfigParser()
    
    config['DEFAULT'] = {
        'log_level': 'INFO',
        'dry_run': 'true'
    }
    
    config['exchange'] = {
        'name': 'binance',
        'testnet': 'true',
        'rate_limit': '1200'
    }
    
    config['strategy'] = {
        'name': 'scalping_ema',
        'timeframe': '5m',
        'ema_fast': '9',
        'ema_slow': '21',
        'rsi_period': '14',
        'rsi_overbought': '70',
        'rsi_oversold': '30'
    }
    
    config['risk'] = {
        'max_position_size': '0.01',
        'stop_loss_pct': '2.0',
        'take_profit_pct': '4.0',
        'max_daily_trades': '10'
    }
    
    with open('trading_bot.ini', 'w') as f:
        config.write(f)
    
    return config


def load_trading_config(filename='trading_bot.ini'):
    """Carica configurazione trading bot con validazione."""
    config = configparser.ConfigParser()
    
    if not config.read(filename):
        raise FileNotFoundError(f"Config file {filename} not found")
    
    # Validazione
    required_sections = ['exchange', 'strategy', 'risk']
    for section in required_sections:
        if not config.has_section(section):
            raise ValueError(f"Missing required section: {section}")
    
    # Converti tipi
    settings = {
        'exchange': config['exchange']['name'],
        'testnet': config.getboolean('exchange', 'testnet'),
        'timeframe': config['strategy']['timeframe'],
        'ema_fast': config.getint('strategy', 'ema_fast'),
        'ema_slow': config.getint('strategy', 'ema_slow'),
        'stop_loss': config.getfloat('risk', 'stop_loss_pct'),
        'take_profit': config.getfloat('risk', 'take_profit_pct'),
        'dry_run': config.getboolean('DEFAULT', 'dry_run')
    }
    
    return settings


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 4.7 - CONFIGPARSER                           │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_4_7 = """
Q1. ConfigParser legge file in formato:
    A) JSON    B) INI    C) XML    D) YAML

Q2. config.read('file.ini') restituisce:
    A) Contenuto file    B) Lista di file letti    C) True/False    D) None

Q3. config.getint('section', 'key') restituisce:
    A) Stringa    B) Intero    C) Float    D) Boolean

Q4. config.getboolean() interpreta come True:
    A) Solo 'true'    B) 'true', 'yes', 'on', '1'    C) Solo '1'    D) Solo 'True'

Q5. La sezione [DEFAULT]:
    A) È obbligatoria    B) I suoi valori sono ereditati da tutte le sezioni
    C) Non può avere valori    D) È sempre vuota

Q6. config.sections() restituisce:
    A) Tutte le sezioni incluso DEFAULT    B) Tutte le sezioni escluso DEFAULT
    C) Solo DEFAULT    D) Lista vuota

Q7. %(key)s in un valore è:
    A) Un errore    B) Interpolazione (riferimento a altra chiave)
    C) Un commento    D) Escape character

Q8. config.has_section('name') restituisce:
    A) Il contenuto    B) True/False    C) La sezione    D) Errore

Q9. Per salvare modifiche al config:
    A) config.save()    B) config.write(file_object)    C) config.commit()    D) Automatico

Q10. ConfigParser è case-sensitive per le chiavi?
     A) Sì    B) No (default)    C) Dipende dal file    D) Solo su Windows

Q11. config.get('section', 'key', fallback='default') - fallback serve quando:
     A) Sempre    B) La chiave non esiste    C) Il valore è vuoto    D) Mai

Q12. Per usare ${section:key} serve:
     A) BasicInterpolation    B) ExtendedInterpolation    C) RawConfigParser    D) Non è possibile
"""

ANSWERS_4_7 = """
RISPOSTE QUIZ 4.7 - CONFIGPARSER:
Q1: B - INI format ([section] key=value)
Q2: B - Lista di file letti con successo
Q3: B - Intero (converte automaticamente)
Q4: B - 'true', 'yes', 'on', '1' (case insensitive)
Q5: B - I suoi valori sono ereditati da tutte le sezioni
Q6: B - Tutte le sezioni escluso DEFAULT
Q7: B - Interpolazione (riferimento a altra chiave)
Q8: B - True/False
Q9: B - config.write(file_object) con open()
Q10: B - No, le chiavi sono lowercase di default
Q11: B - La chiave non esiste
Q12: B - ExtendedInterpolation
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 4.8: PATHLIB
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    4.8 TEORIA: PATHLIB                                       │
└──────────────────────────────────────────────────────────────────────────────┘

pathlib = modulo moderno per manipolazione path (Python 3.4+).
Più leggibile e cross-platform rispetto a os.path.
"""

from pathlib import Path

# CREAZIONE PATH
p = Path('folder/file.txt')
p = Path.cwd()              # Current working directory
p = Path.home()             # Home directory
p = Path('/absolute/path')

# OPERATORE / PER JOIN
p = Path.home() / 'documents' / 'file.txt'
# Equivalente a: os.path.join(os.path.expanduser('~'), 'documents', 'file.txt')


# PROPRIETÀ PATH
p = Path('/home/user/docs/file.txt')

p.name          # 'file.txt'
p.stem          # 'file'
p.suffix        # '.txt'
p.suffixes      # ['.txt'] (per file.tar.gz → ['.tar', '.gz'])
p.parent        # Path('/home/user/docs')
p.parents       # Sequenza di tutti i parent
p.parts         # ('/', 'home', 'user', 'docs', 'file.txt')
p.anchor        # '/' (root su Unix, 'C:\\' su Windows)


# VERIFICA
p.exists()      # True se esiste
p.is_file()     # True se è file
p.is_dir()      # True se è directory
p.is_absolute() # True se path assoluto


# LETTURA/SCRITTURA
p = Path('test.txt')
content = p.read_text(encoding='utf-8')
p.write_text('nuovo contenuto', encoding='utf-8')

data = p.read_bytes()       # Lettura binaria
p.write_bytes(b'binary')    # Scrittura binaria


# OPERAZIONI DIRECTORY
p = Path('new_folder')
p.mkdir(exist_ok=True)              # Crea directory
p.mkdir(parents=True, exist_ok=True) # Crea anche parent

# Iterazione contenuto
for item in Path('.').iterdir():
    print(item)

# Glob pattern
for py_file in Path('.').glob('*.py'):      # Solo questa directory
    print(py_file)

for py_file in Path('.').rglob('*.py'):     # Ricorsivo
    print(py_file)


# MODIFICA PATH
p = Path('/home/user/file.txt')
p.with_name('newfile.txt')      # /home/user/newfile.txt
p.with_suffix('.md')            # /home/user/file.md
p.with_stem('different')        # /home/user/different.txt (3.9+)


# RISOLUZIONE
p = Path('relative/path')
p.resolve()                     # Path assoluto
p.absolute()                    # Path assoluto (alternativa)


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 4.8                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_4_8 = """
Q1. Path.cwd() restituisce:
    A) Home directory    B) Current working directory    C) Root    D) Temp

Q2. Path.home() / 'docs' usa / per:
    A) Divisione    B) Join path    C) Errore    D) Commento

Q3. path.stem per 'file.txt' restituisce:
    A) 'file.txt'    B) 'file'    C) '.txt'    D) 'txt'

Q4. path.suffix per 'file.txt' restituisce:
    A) 'file'    B) 'txt'    C) '.txt'    D) 'file.txt'

Q5. path.glob('*.py') trova:
    A) Tutti i file    B) Solo file .py nella directory    C) File ricorsivamente    D) Niente

Q6. path.rglob('*.py') trova:
    A) Solo questa directory    B) Ricorsivamente    C) Solo root    D) Niente

Q7. mkdir(parents=True) crea:
    A) Solo la directory    B) Anche le directory parent    C) Errore    D) Solo file

Q8. path.resolve() restituisce:
    A) Path relativo    B) Path assoluto    C) Nome file    D) Estensione
"""

ANSWERS_4_8 = """
RISPOSTE QUIZ 4.8:
Q1: B - Current working directory
Q2: B - Join path (operatore overloaded)
Q3: B - 'file' (nome senza estensione)
Q4: C - '.txt' (estensione con punto)
Q5: B - Solo file .py nella directory corrente
Q6: B - Ricorsivamente in tutte le subdirectory
Q7: B - Anche le directory parent mancanti
Q8: B - Path assoluto
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    MODULE 4 FINAL TEST
# ══════════════════════════════════════════════════════════════════════════════

MODULE_4_FINAL_TEST = """
═══════════════════════════════════════════════════════════════════════════════
                    PA MODULE 4 - TEST FINALE (40 domande)
                             Tempo: 45 minuti
                             Pass: 70% (28/40)
═══════════════════════════════════════════════════════════════════════════════

SEZIONE A: FILE I/O (8 domande)
───────────────────────────────

Q1. 'w' mode:
    A) Legge    B) Sovrascrive/crea    C) Append    D) Errore se esiste

Q2. 'a' mode:
    A) Legge    B) Sovrascrive    C) Append    D) Errore se non esiste

Q3. readlines() restituisce:
    A) Stringa    B) Lista    C) Tupla    D) Generator

Q4. Path.stem per 'data.csv':
    A) 'data.csv'    B) 'data'    C) '.csv'    D) 'csv'


SEZIONE B: CSV/JSON (8 domande)
───────────────────────────────

Q5. csv.DictReader restituisce:
    A) Liste    B) Dizionari    C) Tuple    D) Stringhe

Q6. newline='' in csv serve per:
    A) Velocità    B) Cross-platform newlines    C) Encoding    D) Header

Q7. json.loads() converte:
    A) File→Python    B) String→Python    C) Python→File    D) Python→String

Q8. json.dump() scrive su:
    A) Stringa    B) File    C) Console    D) Lista

Q9. None diventa in JSON:
    A) None    B) 'null'    C) null    D) undefined

Q10. True diventa in JSON:
     A) True    B) true    C) 1    D) 'True'


SEZIONE C: XML (6 domande)
──────────────────────────

Q11. ET.fromstring() parsa da:
     A) File    B) Stringa    C) URL    D) Bytes

Q12. element.find() restituisce:
     A) Lista    B) Primo match o None    C) Tutti    D) Errore

Q13. element.findall() restituisce:
     A) Primo    B) Lista    C) None    D) Errore

Q14. element.get('attr') restituisce:
     A) Testo    B) Attributo    C) Tag    D) Figli


SEZIONE D: SQLITE (6 domande)
─────────────────────────────

Q15. cursor.fetchone() restituisce:
     A) Lista    B) Tupla o None    C) Dict    D) Int

Q16. cursor.fetchall() restituisce:
     A) Tupla    B) Lista di tuple    C) Dict    D) Generator

Q17. ? in execute() previene:
     A) Errori    B) SQL injection    C) Duplicati    D) Null

Q18. conn.commit() serve per:
     A) Leggere    B) Salvare modifiche    C) Chiudere    D) Creare


SEZIONE E: LOGGING (6 domande)
──────────────────────────────

Q19. Livello PIÙ GRAVE:
     A) DEBUG    B) WARNING    C) ERROR    D) CRITICAL

Q20. Livello MENO GRAVE:
     A) DEBUG    B) INFO    C) WARNING    D) ERROR

Q21. logger.exception() include:
     A) Solo msg    B) Traceback    C) Solo errore    D) Niente

Q22. RotatingFileHandler ruota per:
     A) Tempo    B) Dimensione    C) Errori    D) Livello


SEZIONE F: CONFIGPARSER (6 domande)
───────────────────────────────────

Q23. ConfigParser legge formato:
     A) JSON    B) INI    C) XML    D) YAML

Q24. config.getboolean() interpreta 'yes' come:
     A) Errore    B) True    C) False    D) 'yes'

Q25. Sezione [DEFAULT]:
     A) Obbligatoria    B) Ereditata da tutte    C) Ignorata    D) Solo commenti

Q26. config.sections() include DEFAULT?
     A) Sì    B) No    C) Dipende    D) Errore

Q27. %(key)s è:
     A) Errore    B) Interpolazione    C) Commento    D) Escape

Q28. config.write() scrive su:
     A) Stringa    B) File object    C) Path    D) Console


═══════════════════════════════════════════════════════════════════════════════
                              FINE TEST
═══════════════════════════════════════════════════════════════════════════════
"""

MODULE_4_FINAL_ANSWERS = """
═══════════════════════════════════════════════════════════════════════════════
                    PA MODULE 4 - RISPOSTE TEST FINALE
═══════════════════════════════════════════════════════════════════════════════

Q1: B    Q2: C    Q3: B    Q4: B    Q5: B
Q6: B    Q7: B    Q8: B    Q9: C    Q10: B
Q11: B   Q12: B   Q13: B   Q14: B   Q15: B
Q16: B   Q17: B   Q18: B   Q19: D   Q20: A
Q21: B   Q22: B   Q23: B   Q24: B   Q25: B
Q26: B   Q27: B   Q28: B

PUNTEGGIO:
──────────
36-40: Eccellente!
32-35: Ottimo!
28-31: Buono (70% pass)
<28:   Rivedi le sezioni deboli

═══════════════════════════════════════════════════════════════════════════════
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    LABS - ESERCIZI PRATICI
# ══════════════════════════════════════════════════════════════════════════════

LABS = """
═══════════════════════════════════════════════════════════════════════════════
                    PA MODULE 4 - LABS
═══════════════════════════════════════════════════════════════════════════════

LAB 1: Crea una funzione che legge un CSV e lo converte in lista di dizionari.

LAB 2: Crea una funzione che salva un dizionario Python come JSON formattato.

LAB 3: Parsa un file XML e estrai tutti i valori di un determinato tag.

LAB 4: Crea un database SQLite con tabella users e funzioni CRUD.

LAB 5: Configura logging con output sia su file che su console.

LAB 6: Crea un file di configurazione INI per un'applicazione e leggi i valori.

LAB 7: Usa pathlib per trovare tutti i file .py in una directory ricorsivamente.

LAB 8: Crea una classe ConfigManager che wrappa ConfigParser con validazione.

LAB 9: Implementa un sistema di logging rotativo per dimensione.

LAB 10: Crea uno script che converte CSV → JSON → XML.

LAB 11: Implementa un file di configurazione con interpolazione per il trading bot.

LAB 12: Crea una funzione che merge multiple config files con priorità.

LAB 13: Implementa backup automatico di un database SQLite.

LAB 14: Crea un log analyzer che parsa file di log e genera statistiche.

LAB 15: Implementa un sistema di configurazione environment-aware
        (dev/staging/production) usando ConfigParser.
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    ESECUZIONE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 78)
    print("PYTHON ADVANCED - MODULE 4")
    print("File Processing, Data Formats & Configuration")
    print("=" * 78)
    print("""
    CONTENUTO:
    ──────────
    Section 4.1: Text Files & Encoding
    Section 4.2: CSV Processing
    Section 4.3: JSON Processing
    Section 4.4: XML Processing
    Section 4.5: SQLite Database
    Section 4.6: Logging
    Section 4.7: ConfigParser ⭐ 
    Section 4.8: pathlib
    
    COMANDI:
    ────────
    print(QUIZ_4_1)   → Quiz File I/O
    print(QUIZ_4_7)   → Quiz ConfigParser
    print(LABS)       → Esercizi pratici
    print(MODULE_4_FINAL_TEST)    → Test finale
    print(MODULE_4_FINAL_ANSWERS) → Risposte
    """)
