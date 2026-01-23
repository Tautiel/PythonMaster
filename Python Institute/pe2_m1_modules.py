"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    PYTHON ESSENTIALS 2 - MODULE 1                            ║
║                    Modules, Packages, and PIP                                ║
║                                                                              ║
║                     Allineato al Syllabus PCAP-31-03                         ║
║                         OpenEDG Python Institute                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

PCAP-31-03 Exam Block 1: Modules and Packages (12%)

STRUTTURA MODULO:
├── Section 1.1: Importing Modules
├── Section 1.2: Module Search Path
├── Section 1.3: Selected Modules (math)
├── Section 1.4: Selected Modules (random)
├── Section 1.5: Selected Modules (platform, os, sys)
├── Section 1.6: Creating Your Own Modules
├── Section 1.7: Packages and __init__.py
├── Section 1.8: PIP and PyPI
├── Section 1.9: Virtual Environments
├── Labs (15 esercizi)
└── Module 1 Quiz (40 domande)

TEMPO STIMATO: 6-8 ore

═══════════════════════════════════════════════════════════════════════════════
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 1.1: IMPORTING MODULES
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    1.1 TEORIA: IMPORT                                        │
└──────────────────────────────────────────────────────────────────────────────┘

Un MODULO è un file .py contenente definizioni e codice Python.
Permette di organizzare e riutilizzare il codice.

TIPI DI MODULI:
───────────────
1. Built-in: math, random, os, sys, etc. (inclusi con Python)
2. Standard Library: json, csv, datetime, etc. (inclusi con Python)
3. Esterni (third-party): numpy, pandas, requests (installati con pip)
4. User-defined: i tuoi file .py


MODI DI IMPORTARE:
──────────────────
"""

# 1. import module - Importa tutto il modulo
import math
print(math.pi)        # 3.141592653589793
print(math.sqrt(16))  # 4.0

# 2. import module as alias - Rinomina il modulo
import math as m
print(m.pi)           # 3.141592653589793

# 3. from module import name - Importa specifici elementi
from math import pi, sqrt
print(pi)             # 3.141592653589793
print(sqrt(16))       # 4.0 (senza prefisso math.)

# 4. from module import name as alias - Rinomina elementi
from math import sqrt as radice
print(radice(25))     # 5.0

# 5. from module import * - Importa TUTTO (SCONSIGLIATO!)
from math import *    # Inquina il namespace, evitare!


"""
PERCHÉ EVITARE from module import *?
────────────────────────────────────
1. Non sai cosa stai importando
2. Può sovrascrivere nomi esistenti
3. Rende il codice meno leggibile
4. Difficile debug

ECCEZIONE: OK in shell interattiva per test rapidi.
"""


"""
COSA SUCCEDE QUANDO IMPORTI:
────────────────────────────
1. Python cerca il modulo
2. Compila in bytecode (.pyc) se necessario
3. Esegue il codice del modulo (una sola volta!)
4. Crea un oggetto modulo nel namespace
"""

# Verifica se modulo già importato
import sys
print('math' in sys.modules)  # True se già importato


"""
ATTRIBUTI SPECIALI DEI MODULI:
──────────────────────────────
"""
import math

print(math.__name__)    # 'math'
print(math.__file__)    # Path del file (se disponibile)
print(math.__doc__)     # Documentazione
print(dir(math))        # Lista tutti gli attributi


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 1.1                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_1_1 = """
Q1. Quale import richiede di usare il prefisso modulo.funzione?
    A) from math import sqrt
    B) import math
    C) from math import *
    D) from math import sqrt as s

Q2. "from math import *" è sconsigliato perché:
    A) È più lento
    B) Inquina il namespace
    C) Non funziona
    D) Richiede più memoria

Q3. import math as m - "m" è:
    A) Una copia di math
    B) Un alias per math
    C) Una funzione
    D) Un errore

Q4. Quante volte viene eseguito il codice di un modulo importato?
    A) Ogni volta che si usa
    B) Una sola volta (prima importazione)
    C) Mai
    D) Dipende dal modulo

Q5. dir(module) restituisce:
    A) Il path del modulo
    B) Lista degli attributi del modulo
    C) La documentazione
    D) Il codice sorgente

Q6. module.__name__ contiene:
    A) Il path
    B) Il nome del modulo come stringa
    C) La versione
    D) L'autore

Q7. sys.modules è:
    A) Lista dei moduli disponibili
    B) Dizionario dei moduli già importati
    C) Funzione per importare
    D) Path di ricerca
"""

ANSWERS_1_1 = """
RISPOSTE QUIZ 1.1:
Q1: B - import math richiede math.sqrt()
Q2: B - Inquina il namespace con nomi sconosciuti
Q3: B - Un alias (riferimento alternativo)
Q4: B - Una sola volta, poi cached in sys.modules
Q5: B - Lista degli attributi del modulo
Q6: B - Il nome del modulo come stringa
Q7: B - Dizionario dei moduli già importati/cached
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 1.2: MODULE SEARCH PATH
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    1.2 TEORIA: MODULE SEARCH PATH                            │
└──────────────────────────────────────────────────────────────────────────────┘

Quando fai "import modulo", Python cerca in quest'ordine:

1. Directory corrente (dove si trova lo script)
2. PYTHONPATH (variabile d'ambiente)
3. Directory di installazione Python (site-packages, etc.)

Il path completo è in sys.path
"""

import sys

# Visualizza il path di ricerca
print(sys.path)
# ['', '/usr/lib/python3.10', '/usr/lib/python3.10/lib-dynload', ...]

# Primo elemento '' = directory corrente


"""
MODIFICARE sys.path:
────────────────────
Puoi aggiungere directory a runtime (NON consigliato in produzione)
"""

import sys
sys.path.append('/path/to/my/modules')
# Ora Python cerca anche lì


"""
PYTHONPATH (variabile d'ambiente):
──────────────────────────────────
Meglio di modificare sys.path nel codice.

Linux/Mac:
    export PYTHONPATH="/path/to/modules:$PYTHONPATH"

Windows:
    set PYTHONPATH=C:\\path\\to\\modules;%PYTHONPATH%
"""


"""
TROVARE DOVE È UN MODULO:
─────────────────────────
"""

import math
print(math.__file__)  # /usr/lib/python3.10/lib-dynload/math.cpython-310-x86_64-linux-gnu.so

import json
print(json.__file__)  # /usr/lib/python3.10/json/__init__.py


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 1.2                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_1_2 = """
Q1. sys.path è:
    A) Una funzione
    B) Una lista di directory dove Python cerca moduli
    C) Un dizionario
    D) Una stringa

Q2. Il primo posto dove Python cerca un modulo è:
    A) site-packages
    B) PYTHONPATH
    C) Directory corrente
    D) /usr/lib/python

Q3. Per aggiungere una directory al path di ricerca:
    A) sys.path.add()
    B) sys.path.append()
    C) sys.path.insert()
    D) B e C sono corrette

Q4. PYTHONPATH è:
    A) Una funzione Python
    B) Una variabile d'ambiente
    C) Un file di configurazione
    D) Un modulo

Q5. module.__file__ contiene:
    A) Il codice
    B) Il path del file del modulo
    C) Il nome
    D) La documentazione
"""

ANSWERS_1_2 = """
RISPOSTE QUIZ 1.2:
Q1: B - Lista di directory dove Python cerca moduli
Q2: C - Directory corrente (primo elemento di sys.path)
Q3: D - Sia append() che insert() funzionano
Q4: B - Variabile d'ambiente del sistema operativo
Q5: B - Il path del file del modulo
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 1.3: MATH MODULE
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    1.3 TEORIA: MODULO MATH                                   │
└──────────────────────────────────────────────────────────────────────────────┘

Il modulo math fornisce funzioni matematiche per numeri reali.
Per numeri complessi, usa cmath.
"""

import math

# ═══════════════════════════════════════════════════════════════════════════
# COSTANTI
# ═══════════════════════════════════════════════════════════════════════════

print(math.pi)        # 3.141592653589793
print(math.e)         # 2.718281828459045 (numero di Eulero)
print(math.tau)       # 6.283185307179586 (2 * pi)
print(math.inf)       # Infinito positivo
print(math.nan)       # Not a Number


# ═══════════════════════════════════════════════════════════════════════════
# FUNZIONI DI ARROTONDAMENTO
# ═══════════════════════════════════════════════════════════════════════════

print(math.ceil(4.2))     # 5 (arrotonda verso l'alto)
print(math.floor(4.8))    # 4 (arrotonda verso il basso)
print(math.trunc(4.8))    # 4 (tronca decimali, come int())
print(math.trunc(-4.8))   # -4 (nota: floor(-4.8) = -5!)


# ═══════════════════════════════════════════════════════════════════════════
# POTENZE E RADICI
# ═══════════════════════════════════════════════════════════════════════════

print(math.sqrt(16))      # 4.0 (radice quadrata)
print(math.pow(2, 3))     # 8.0 (potenza, sempre float)
print(2 ** 3)             # 8 (alternativa, può essere int)
print(math.exp(1))        # 2.718... (e^1)
print(math.exp(2))        # 7.389... (e^2)


# ═══════════════════════════════════════════════════════════════════════════
# LOGARITMI
# ═══════════════════════════════════════════════════════════════════════════

print(math.log(math.e))   # 1.0 (logaritmo naturale, base e)
print(math.log(100, 10))  # 2.0 (logaritmo base 10)
print(math.log10(100))    # 2.0 (logaritmo base 10, più veloce)
print(math.log2(8))       # 3.0 (logaritmo base 2)


# ═══════════════════════════════════════════════════════════════════════════
# TRIGONOMETRIA (angoli in RADIANTI!)
# ═══════════════════════════════════════════════════════════════════════════

print(math.sin(math.pi / 2))  # 1.0 (seno di 90°)
print(math.cos(0))            # 1.0 (coseno di 0°)
print(math.tan(math.pi / 4))  # 1.0 (tangente di 45°)

# Inverse
print(math.asin(1))           # 1.5707... (π/2, arcoseno)
print(math.acos(1))           # 0.0 (arcocoseno)
print(math.atan(1))           # 0.7853... (π/4, arcotangente)

# Conversione gradi <-> radianti
print(math.radians(180))      # 3.14159... (π)
print(math.degrees(math.pi))  # 180.0


# ═══════════════════════════════════════════════════════════════════════════
# ALTRE FUNZIONI UTILI
# ═══════════════════════════════════════════════════════════════════════════

print(math.fabs(-5.5))        # 5.5 (valore assoluto, sempre float)
print(abs(-5))                # 5 (built-in, mantiene tipo)

print(math.factorial(5))      # 120 (5! = 5*4*3*2*1)

print(math.gcd(48, 18))       # 6 (massimo comun divisore)
print(math.lcm(4, 6))         # 12 (minimo comune multiplo, Python 3.9+)

print(math.hypot(3, 4))       # 5.0 (ipotenusa: sqrt(3² + 4²))

print(math.isnan(math.nan))   # True
print(math.isinf(math.inf))   # True
print(math.isfinite(1.5))     # True


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 1.3                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_1_3 = """
Q1. math.ceil(4.2) restituisce:
    A) 4    B) 5    C) 4.0    D) 4.2

Q2. math.floor(-4.8) restituisce:
    A) -4    B) -5    C) 4    D) 5

Q3. math.trunc(-4.8) restituisce:
    A) -4    B) -5    C) 4    D) 5

Q4. Le funzioni trigonometriche usano angoli in:
    A) Gradi    B) Radianti    C) Entrambi    D) Dipende

Q5. math.log(100, 10) restituisce:
    A) 10    B) 100    C) 2.0    D) 1.0

Q6. math.factorial(4) restituisce:
    A) 4    B) 16    C) 24    D) 256

Q7. math.hypot(3, 4) calcola:
    A) 3 + 4    B) 3 * 4    C) sqrt(3² + 4²)    D) 3² + 4²

Q8. math.pow(2, 3) restituisce:
    A) 8    B) 8.0    C) 6    D) 6.0

Q9. Per convertire gradi in radianti:
    A) math.radians()    B) math.degrees()    C) math.rad()    D) math.deg()

Q10. math.e è:
     A) 3.14159...    B) 2.71828...    C) 1.41421...    D) 1.61803...
"""

ANSWERS_1_3 = """
RISPOSTE QUIZ 1.3:
Q1: B - 5 (arrotonda verso l'alto)
Q2: B - -5 (arrotonda verso il basso, -5 < -4.8)
Q3: A - -4 (tronca verso zero)
Q4: B - Radianti
Q5: C - 2.0 (10² = 100)
Q6: C - 24 (4! = 4*3*2*1)
Q7: C - sqrt(3² + 4²) = 5.0 (teorema di Pitagora)
Q8: B - 8.0 (math.pow restituisce sempre float)
Q9: A - math.radians()
Q10: B - 2.71828... (numero di Eulero)
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 1.4: RANDOM MODULE
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    1.4 TEORIA: MODULO RANDOM                                 │
└──────────────────────────────────────────────────────────────────────────────┘

Il modulo random genera numeri pseudo-casuali.
NON usare per crittografia! Usa secrets per quello.
"""

import random

# ═══════════════════════════════════════════════════════════════════════════
# NUMERI CASUALI
# ═══════════════════════════════════════════════════════════════════════════

# Float casuale [0.0, 1.0)
print(random.random())        # 0.7234...

# Float casuale in range [a, b]
print(random.uniform(1, 10))  # 5.234...

# Intero casuale [a, b] (inclusi!)
print(random.randint(1, 6))   # 1, 2, 3, 4, 5 o 6

# Intero casuale [start, stop) con step
print(random.randrange(0, 10, 2))  # 0, 2, 4, 6 o 8


# ═══════════════════════════════════════════════════════════════════════════
# OPERAZIONI SU SEQUENZE
# ═══════════════════════════════════════════════════════════════════════════

lista = [1, 2, 3, 4, 5]

# Elemento casuale
print(random.choice(lista))   # Un elemento a caso

# Più elementi casuali (con ripetizione)
print(random.choices(lista, k=3))  # [2, 2, 5] possibile

# Più elementi casuali (senza ripetizione)
print(random.sample(lista, k=3))   # [3, 1, 5] tutti diversi

# Mescola lista IN PLACE
random.shuffle(lista)
print(lista)  # [3, 1, 5, 2, 4] ordine casuale


# ═══════════════════════════════════════════════════════════════════════════
# SEED (per riproducibilità)
# ═══════════════════════════════════════════════════════════════════════════

random.seed(42)  # Imposta seed
print(random.random())  # 0.6394267984578837 (sempre uguale con seed 42!)

random.seed(42)  # Reset seed
print(random.random())  # 0.6394267984578837 (identico!)


# ═══════════════════════════════════════════════════════════════════════════
# DISTRIBUZIONI
# ═══════════════════════════════════════════════════════════════════════════

# Distribuzione normale (gaussiana)
print(random.gauss(mu=0, sigma=1))  # Media 0, deviazione standard 1

# Distribuzione triangolare
print(random.triangular(low=0, high=10, mode=5))


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 1.4                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_1_4 = """
Q1. random.random() restituisce un float in:
    A) [0, 1]    B) [0.0, 1.0)    C) (0, 1)    D) [0, 100]

Q2. random.randint(1, 6) può restituire:
    A) Solo 1-5    B) Solo 2-5    C) 1, 2, 3, 4, 5 o 6    D) Float tra 1 e 6

Q3. random.choice(lista) restituisce:
    A) Indice casuale    B) Elemento casuale    C) Lista mescolata    D) Copia

Q4. random.shuffle(lista) modifica:
    A) Una copia    B) La lista originale    C) Niente    D) Restituisce nuova lista

Q5. random.sample(lista, k=3) restituisce:
    A) 3 elementi con ripetizione
    B) 3 elementi senza ripetizione
    C) La lista mescolata
    D) Un singolo elemento

Q6. random.seed(42) serve per:
    A) Generare 42
    B) Rendere i risultati riproducibili
    C) Velocizzare
    D) Crittografia

Q7. random.uniform(1, 10) restituisce:
    A) Intero 1-10    B) Float 1.0-10.0    C) Lista    D) Booleano

Q8. Per crittografia, invece di random usa:
    A) math    B) secrets    C) os    D) sys
"""

ANSWERS_1_4 = """
RISPOSTE QUIZ 1.4:
Q1: B - [0.0, 1.0) - include 0, esclude 1
Q2: C - 1, 2, 3, 4, 5 o 6 (entrambi gli estremi inclusi!)
Q3: B - Elemento casuale dalla lista
Q4: B - Modifica la lista originale (in place)
Q5: B - 3 elementi senza ripetizione
Q6: B - Rendere i risultati riproducibili
Q7: B - Float tra 1.0 e 10.0
Q8: B - secrets (crittograficamente sicuro)
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 1.5: PLATFORM, OS, SYS MODULES
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    1.5 TEORIA: PLATFORM, OS, SYS                             │
└──────────────────────────────────────────────────────────────────────────────┘
"""

# ═══════════════════════════════════════════════════════════════════════════
# PLATFORM - Informazioni sul sistema
# ═══════════════════════════════════════════════════════════════════════════

import platform

print(platform.system())           # 'Linux', 'Windows', 'Darwin' (macOS)
print(platform.machine())          # 'x86_64', 'AMD64', etc.
print(platform.processor())        # Info processore
print(platform.python_version())   # '3.10.0'
print(platform.python_implementation())  # 'CPython', 'PyPy', etc.
print(platform.platform())         # Info completa del sistema
print(platform.node())             # Nome host


# ═══════════════════════════════════════════════════════════════════════════
# OS - Interazione con sistema operativo
# ═══════════════════════════════════════════════════════════════════════════

import os

# Directory corrente
print(os.getcwd())                 # '/home/user/project'

# Cambia directory
# os.chdir('/path/to/dir')

# Lista file in directory
print(os.listdir('.'))             # ['file1.py', 'file2.py', ...]

# Crea directory
# os.mkdir('new_dir')              # Crea singola directory
# os.makedirs('path/to/new_dir')   # Crea path completo

# Rimuovi
# os.remove('file.txt')            # Rimuovi file
# os.rmdir('empty_dir')            # Rimuovi directory vuota

# Path operations
print(os.path.join('dir', 'file.txt'))  # 'dir/file.txt' (cross-platform!)
print(os.path.exists('file.txt'))       # True/False
print(os.path.isfile('file.txt'))       # True se è file
print(os.path.isdir('directory'))       # True se è directory
print(os.path.basename('/path/to/file.txt'))  # 'file.txt'
print(os.path.dirname('/path/to/file.txt'))   # '/path/to'
print(os.path.split('/path/to/file.txt'))     # ('/path/to', 'file.txt')
print(os.path.splitext('file.txt'))           # ('file', '.txt')

# Environment variables
print(os.environ.get('HOME'))      # '/home/user'
print(os.getenv('PATH'))           # Path di sistema

# Nome del sistema operativo
print(os.name)                     # 'posix' (Linux/Mac), 'nt' (Windows)


# ═══════════════════════════════════════════════════════════════════════════
# SYS - Sistema Python
# ═══════════════════════════════════════════════════════════════════════════

import sys

print(sys.version)                 # Versione Python completa
print(sys.version_info)            # (3, 10, 0, 'final', 0)
print(sys.platform)                # 'linux', 'win32', 'darwin'
print(sys.path)                    # Lista path di ricerca moduli
print(sys.modules)                 # Dict moduli importati
print(sys.executable)              # Path dell'interprete Python

# Argomenti command line
print(sys.argv)                    # ['script.py', 'arg1', 'arg2']

# Standard I/O
# sys.stdin, sys.stdout, sys.stderr

# Exit
# sys.exit(0)                      # Termina con codice 0 (successo)
# sys.exit(1)                      # Termina con codice 1 (errore)

# Dimensione oggetti
print(sys.getsizeof([1, 2, 3]))    # Byte usati dalla lista


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 1.5                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_1_5 = """
Q1. platform.system() restituisce:
    A) Versione Python    B) Nome SO (Linux/Windows/Darwin)    C) Architettura    D) Host

Q2. os.getcwd() restituisce:
    A) Home directory    B) Directory corrente    C) Root    D) Temp

Q3. os.path.join('a', 'b', 'c') è preferibile perché:
    A) È più veloce    B) È cross-platform    C) È più corto    D) È più sicuro

Q4. os.name su Windows restituisce:
    A) 'windows'    B) 'win32'    C) 'nt'    D) 'Windows'

Q5. sys.argv[0] contiene:
    A) Primo argomento    B) Nome dello script    C) Versione Python    D) Path Python

Q6. sys.exit(1) indica:
    A) Successo    B) Errore    C) Warning    D) Niente

Q7. os.path.splitext('file.txt') restituisce:
    A) ('file.txt', '')    B) ('file', '.txt')    C) ['file', 'txt']    D) 'file'

Q8. platform.python_version() restituisce:
    A) Solo major version    B) Stringa come '3.10.0'    C) Intero    D) Tupla

Q9. os.environ è:
    A) Una funzione    B) Un dizionario di variabili d'ambiente    C) Una lista    D) Una stringa

Q10. sys.getsizeof(obj) restituisce:
     A) Lunghezza    B) Dimensione in byte    C) Tipo    D) Hash
"""

ANSWERS_1_5 = """
RISPOSTE QUIZ 1.5:
Q1: B - Nome del sistema operativo
Q2: B - Directory corrente (current working directory)
Q3: B - È cross-platform (usa separatore corretto)
Q4: C - 'nt' (New Technology)
Q5: B - Nome dello script
Q6: B - Errore (0 = successo, non-zero = errore)
Q7: B - ('file', '.txt')
Q8: B - Stringa come '3.10.0'
Q9: B - Un dizionario di variabili d'ambiente
Q10: B - Dimensione in byte dell'oggetto
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 1.6: CREATING YOUR OWN MODULES
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    1.6 TEORIA: CREARE MODULI                                 │
└──────────────────────────────────────────────────────────────────────────────┘

Un modulo è semplicemente un file .py!
"""

# ═══════════════════════════════════════════════════════════════════════════
# ESEMPIO: mymodule.py
# ═══════════════════════════════════════════════════════════════════════════

MYMODULE_EXAMPLE = '''
"""
mymodule.py - Il mio modulo di esempio

Questo modulo fornisce funzioni utili per calcoli.
"""

# Variabile del modulo
VERSION = "1.0.0"
PI = 3.14159

# Funzione pubblica
def somma(a, b):
    """Somma due numeri."""
    return a + b

def moltiplica(a, b):
    """Moltiplica due numeri."""
    return a * b

# Funzione "privata" (convenzione _)
def _helper():
    """Funzione interna, non parte dell'API pubblica."""
    pass

# Classe
class Calculator:
    """Semplice calcolatore."""
    
    def add(self, a, b):
        return a + b

# Codice eseguito SOLO se il file è eseguito direttamente
if __name__ == "__main__":
    print("Eseguito direttamente!")
    print(f"2 + 3 = {somma(2, 3)}")
else:
    print("Importato come modulo")
'''


"""
__name__ E if __name__ == "__main__":
──────────────────────────────────────
- Se il file è eseguito direttamente: __name__ == "__main__"
- Se il file è importato: __name__ == nome del modulo

Questo pattern permette di:
1. Avere codice di test nel modulo
2. Importare il modulo senza eseguire il test
"""


# ═══════════════════════════════════════════════════════════════════════════
# USARE IL MODULO
# ═══════════════════════════════════════════════════════════════════════════

USAGE_EXAMPLE = '''
# main.py
import mymodule

print(mymodule.VERSION)           # "1.0.0"
print(mymodule.somma(2, 3))       # 5

calc = mymodule.Calculator()
print(calc.add(5, 3))             # 8

# Oppure
from mymodule import somma, Calculator
print(somma(10, 20))              # 30
'''


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 1.6                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_1_6 = """
Q1. Un modulo Python è:
    A) Una cartella    B) Un file .py    C) Una classe    D) Una funzione

Q2. __name__ quando il modulo è importato contiene:
    A) "__main__"    B) Il nome del modulo    C) None    D) ""

Q3. __name__ quando il file è eseguito direttamente contiene:
    A) Il nome del file    B) "__main__"    C) None    D) ""

Q4. if __name__ == "__main__": serve per:
    A) Importare moduli
    B) Eseguire codice solo quando il file è eseguito direttamente
    C) Definire variabili
    D) Creare classi

Q5. Una funzione che inizia con _ è:
    A) Errore    B) Privata per convenzione    C) Più veloce    D) Deprecata

Q6. Per rendere un modulo importabile, deve:
    A) Essere in site-packages
    B) Essere nel path di ricerca o directory corrente
    C) Avere permessi speciali
    D) Essere compilato
"""

ANSWERS_1_6 = """
RISPOSTE QUIZ 1.6:
Q1: B - Un file .py
Q2: B - Il nome del modulo (es. "mymodule")
Q3: B - "__main__"
Q4: B - Eseguire codice solo quando il file è eseguito direttamente
Q5: B - Privata per convenzione (non parte dell'API pubblica)
Q6: B - Essere nel path di ricerca o directory corrente
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 1.7: PACKAGES AND __init__.py
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    1.7 TEORIA: PACKAGES                                      │
└──────────────────────────────────────────────────────────────────────────────┘

Un PACKAGE è una directory contenente moduli e un file __init__.py

STRUTTURA:
──────────
mypackage/
├── __init__.py      # Rende la directory un package
├── module1.py
├── module2.py
└── subpackage/
    ├── __init__.py
    └── module3.py
"""

# ═══════════════════════════════════════════════════════════════════════════
# __init__.py
# ═══════════════════════════════════════════════════════════════════════════

"""
__init__.py può essere:
1. Vuoto (minimo necessario)
2. Contenere codice di inizializzazione
3. Definire __all__ per controllare "from package import *"
4. Importare submoduli per accesso più facile
"""

INIT_EXAMPLE = '''
# mypackage/__init__.py

# Versione del package
__version__ = "1.0.0"

# Controlla cosa viene esportato con "from mypackage import *"
__all__ = ['module1', 'module2', 'useful_function']

# Importa submoduli per accesso facile
from .module1 import ClassA
from .module2 import useful_function

# Codice di inizializzazione
print("Package mypackage inizializzato!")
'''


# ═══════════════════════════════════════════════════════════════════════════
# IMPORT DA PACKAGES
# ═══════════════════════════════════════════════════════════════════════════

IMPORT_EXAMPLES = '''
# Import del package
import mypackage
print(mypackage.__version__)

# Import di un modulo dal package
import mypackage.module1
mypackage.module1.function()

# Import con alias
import mypackage.module1 as m1
m1.function()

# Import diretto
from mypackage import module1
module1.function()

# Import di elemento specifico
from mypackage.module1 import MyClass
obj = MyClass()

# Import da subpackage
from mypackage.subpackage import module3
from mypackage.subpackage.module3 import something
'''


# ═══════════════════════════════════════════════════════════════════════════
# IMPORT RELATIVI (dentro un package)
# ═══════════════════════════════════════════════════════════════════════════

RELATIVE_IMPORT_EXAMPLE = '''
# mypackage/module2.py

# Import relativo - stesso livello
from . import module1
from .module1 import MyClass

# Import relativo - livello superiore
from .. import other_package
from ..other_package import something

# Import relativo - subpackage
from .subpackage import module3
'''


"""
NOTA: Import relativi funzionano SOLO dentro un package,
non in script eseguiti direttamente!
"""


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 1.7                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_1_7 = """
Q1. Un package Python è:
    A) Un file .py
    B) Una directory con __init__.py
    C) Una funzione
    D) Una variabile

Q2. __init__.py può essere:
    A) Solo vuoto
    B) Vuoto o con codice
    C) Solo con codice
    D) Non deve esistere

Q3. __all__ in __init__.py controlla:
    A) Cosa viene importato con import *
    B) Le versioni Python supportate
    C) I permessi
    D) La documentazione

Q4. "from . import module" è:
    A) Import assoluto
    B) Import relativo (stesso package)
    C) Errore
    D) Import di tutto

Q5. "from .. import module" significa:
    A) Import dallo stesso livello
    B) Import dal package parent
    C) Import da root
    D) Errore

Q6. Import relativi funzionano:
    A) Ovunque
    B) Solo dentro un package
    C) Solo in __init__.py
    D) Solo in Python 2

Q7. mypackage/__init__.py viene eseguito:
    A) Mai
    B) Quando il package è importato
    C) Solo con import *
    D) Solo manualmente
"""

ANSWERS_1_7 = """
RISPOSTE QUIZ 1.7:
Q1: B - Una directory con __init__.py
Q2: B - Può essere vuoto o contenere codice
Q3: A - Cosa viene importato con "from package import *"
Q4: B - Import relativo (stesso package)
Q5: B - Import dal package parent
Q6: B - Solo dentro un package (non script diretti)
Q7: B - Quando il package è importato
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 1.8: PIP AND PYPI
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    1.8 TEORIA: PIP AND PYPI                                  │
└──────────────────────────────────────────────────────────────────────────────┘

PIP = Package Installer for Python
PyPI = Python Package Index (repository pubblico)
"""

PIP_COMMANDS = """
═══════════════════════════════════════════════════════════════════════════════
                        COMANDI PIP ESSENZIALI
═══════════════════════════════════════════════════════════════════════════════

# Installare un package
pip install package_name
pip install requests

# Installare versione specifica
pip install requests==2.28.0
pip install requests>=2.20.0
pip install "requests>=2.20.0,<3.0.0"

# Aggiornare un package
pip install --upgrade requests
pip install -U requests

# Disinstallare
pip uninstall requests

# Lista packages installati
pip list
pip list --outdated        # Solo quelli con aggiornamenti

# Informazioni su un package
pip show requests

# Cercare packages (deprecato su PyPI)
pip search keyword         # Non funziona più, usa pypi.org

# Salvare dipendenze
pip freeze > requirements.txt

# Installare da requirements.txt
pip install -r requirements.txt

# Installare in modalità sviluppo (editable)
pip install -e .

# Installare da git
pip install git+https://github.com/user/repo.git

# Cache
pip cache purge            # Pulisce la cache
"""


# ═══════════════════════════════════════════════════════════════════════════
# REQUIREMENTS.TXT
# ═══════════════════════════════════════════════════════════════════════════

REQUIREMENTS_EXAMPLE = """
# requirements.txt

# Versione esatta
requests==2.28.0

# Versione minima
pandas>=1.5.0

# Range di versioni
numpy>=1.20.0,<2.0.0

# Qualsiasi versione
flask

# Da git
git+https://github.com/user/repo.git

# Con extras
requests[security]

# Commenti iniziano con #
"""


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 1.8                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_1_8 = """
Q1. PyPI è:
    A) Un comando    B) Repository di packages Python    C) Un IDE    D) Un modulo

Q2. pip install requests==2.28.0 installa:
    A) L'ultima versione    B) Versione esatta 2.28.0    C) Versione >= 2.28.0    D) Errore

Q3. pip freeze serve per:
    A) Bloccare pip    B) Generare lista dipendenze    C) Aggiornare    D) Disinstallare

Q4. pip install -r requirements.txt:
    A) Crea il file    B) Installa da file    C) Legge il file    D) Elimina il file

Q5. pip install --upgrade:
    A) Installa    B) Aggiorna    C) Rimuove    D) Lista

Q6. pip list mostra:
    A) Packages disponibili    B) Packages installati    C) Errori    D) Documentazione

Q7. pip show package mostra:
    A) Codice sorgente    B) Informazioni sul package    C) Dipendenze    D) B e C

Q8. >= in requirements.txt significa:
    A) Esattamente    B) Maggiore o uguale    C) Minore    D) Diverso
"""

ANSWERS_1_8 = """
RISPOSTE QUIZ 1.8:
Q1: B - Repository pubblico di packages Python
Q2: B - Versione esatta 2.28.0
Q3: B - Generare lista dipendenze installate
Q4: B - Installa tutti i packages elencati nel file
Q5: B - Aggiorna il package all'ultima versione
Q6: B - Packages installati nel sistema/ambiente
Q7: D - Sia informazioni che dipendenze del package
Q8: B - Maggiore o uguale (versione minima)
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 1.9: VIRTUAL ENVIRONMENTS
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    1.9 TEORIA: VIRTUAL ENVIRONMENTS                          │
└──────────────────────────────────────────────────────────────────────────────┘

Un VIRTUAL ENVIRONMENT è un ambiente Python isolato.
Permette di avere dipendenze diverse per progetti diversi.
"""

VENV_COMMANDS = """
═══════════════════════════════════════════════════════════════════════════════
                        VIRTUAL ENVIRONMENTS
═══════════════════════════════════════════════════════════════════════════════

# CREARE un virtual environment
python -m venv myenv
python -m venv .venv          # Convenzione: .venv nella root del progetto

# ATTIVARE
# Linux/Mac:
source myenv/bin/activate

# Windows:
myenv\\Scripts\\activate

# VERIFICARE (prompt cambia)
(myenv) $ which python        # /path/to/myenv/bin/python
(myenv) $ pip list            # Solo packages del venv

# INSTALLARE packages (isolati nel venv)
(myenv) $ pip install requests

# DISATTIVARE
(myenv) $ deactivate
$                             # Prompt torna normale

# RIMUOVERE (semplicemente elimina la directory)
rm -rf myenv                  # Linux/Mac
rmdir /s myenv                # Windows
"""


"""
PERCHÉ USARE VIRTUAL ENVIRONMENTS?
──────────────────────────────────
1. Isolamento: progetti diversi, dipendenze diverse
2. Riproducibilità: requirements.txt esatti
3. Pulizia: non inquini il Python di sistema
4. Testing: testare con versioni diverse

BEST PRACTICE:
──────────────
1. Un venv per progetto
2. Nomina .venv e aggiungilo a .gitignore
3. Salva sempre requirements.txt
4. Non committare il venv in git!
"""


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 1.9                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_1_9 = """
Q1. Un virtual environment serve per:
    A) Velocizzare Python    B) Isolare dipendenze    C) Debugging    D) Testing

Q2. python -m venv myenv crea:
    A) Un file    B) Una directory con ambiente isolato    C) Un package    D) Un modulo

Q3. Per attivare un venv su Linux:
    A) venv activate    B) source venv/bin/activate    C) activate venv    D) python venv

Q4. deactivate serve per:
    A) Eliminare il venv    B) Uscire dal venv    C) Disinstallare packages    D) Creare backup

Q5. Il venv dovrebbe essere in .gitignore?
    A) No    B) Sì    C) Dipende    D) Non importa

Q6. packages installati in un venv sono:
    A) Globali    B) Solo per quel venv    C) Condivisi    D) Temporanei

Q7. Come rimuovere un venv?
    A) pip remove    B) Eliminare la directory    C) venv delete    D) python -m venv --remove
"""

ANSWERS_1_9 = """
RISPOSTE QUIZ 1.9:
Q1: B - Isolare dipendenze tra progetti
Q2: B - Una directory con ambiente Python isolato
Q3: B - source venv/bin/activate
Q4: B - Uscire dal virtual environment
Q5: B - Sì (non committare venv in git)
Q6: B - Solo per quel venv (isolati)
Q7: B - Eliminare la directory (è solo una cartella)
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    MODULE 1 FINAL TEST
# ══════════════════════════════════════════════════════════════════════════════

MODULE_1_FINAL_TEST = """
═══════════════════════════════════════════════════════════════════════════════
                    PE2 MODULE 1 - TEST FINALE (40 domande)
                             Tempo: 45 minuti
                             Pass: 70% (28/40)
═══════════════════════════════════════════════════════════════════════════════

Q1. import math as m - per usare pi greco:
    A) pi    B) math.pi    C) m.pi    D) import pi

Q2. from math import * è sconsigliato perché:
    A) Lento    B) Inquina namespace    C) Non funziona    D) Deprecato

Q3. sys.path è:
    A) Stringa    B) Lista    C) Dizionario    D) Tupla

Q4. math.ceil(-4.2) restituisce:
    A) -5    B) -4    C) 4    D) 5

Q5. math.floor(-4.2) restituisce:
    A) -5    B) -4    C) 4    D) 5

Q6. random.randint(1, 10) può restituire 10?
    A) Sì    B) No    C) Dipende    D) Errore

Q7. random.random() restituisce valori in:
    A) [0, 1]    B) [0.0, 1.0)    C) (0, 1)    D) [0, 100)

Q8. random.shuffle(lista):
    A) Restituisce nuova lista    B) Modifica in place    C) Restituisce None    D) B e C

Q9. platform.system() su macOS restituisce:
    A) 'macOS'    B) 'Darwin'    C) 'Mac'    D) 'Apple'

Q10. os.getcwd() restituisce:
     A) Home    B) Current directory    C) Root    D) Temp

Q11. os.path.join('a', 'b') è preferibile perché:
     A) Più corto    B) Cross-platform    C) Più veloce    D) Più sicuro

Q12. sys.argv[0] contiene:
     A) Primo argomento    B) Nome script    C) Python path    D) Versione

Q13. __name__ quando importato:
     A) "__main__"    B) Nome modulo    C) None    D) ""

Q14. __name__ quando eseguito:
     A) Nome file    B) "__main__"    C) None    D) ""

Q15. Un package richiede:
     A) Solo directory    B) __init__.py    C) setup.py    D) main.py

Q16. __all__ in __init__.py controlla:
     A) Versioni    B) import *    C) Permessi    D) Docs

Q17. from . import x è:
     A) Import assoluto    B) Import relativo    C) Errore    D) Deprecato

Q18. pip install requests==2.0:
     A) Ultima versione    B) Versione 2.0    C) >= 2.0    D) Errore

Q19. pip freeze serve per:
     A) Bloccare    B) Generare requirements    C) Aggiornare    D) Cercare

Q20. pip install -r file.txt:
     A) Crea file    B) Installa da file    C) Legge    D) Rimuove

Q21. python -m venv env crea:
     A) File    B) Virtual environment    C) Package    D) Script

Q22. Per attivare venv su Linux:
     A) activate    B) source env/bin/activate    C) env start    D) python env

Q23. math.sqrt(16) restituisce:
     A) 4    B) 4.0    C) 256    D) 16

Q24. math.pow(2, 3) restituisce:
     A) 8    B) 8.0    C) 6    D) 6.0

Q25. math.log(e) restituisce:
     A) 0    B) 1.0    C) e    D) 2.718

Q26. random.choice([1,2,3]) restituisce:
     A) Lista    B) Un elemento    C) Indice    D) Tupla

Q27. random.sample(list, k=2):
     A) Con ripetizione    B) Senza ripetizione    C) Ordinato    D) Shuffle

Q28. random.seed(42) serve per:
     A) Generare 42    B) Riproducibilità    C) Sicurezza    D) Reset

Q29. os.name su Windows:
     A) 'windows'    B) 'win32'    C) 'nt'    D) 'Windows'

Q30. sys.exit(0) indica:
     A) Errore    B) Successo    C) Warning    D) Niente

Q31. module.__file__ contiene:
     A) Codice    B) Path    C) Nome    D) Versione

Q32. dir(module) restituisce:
     A) Path    B) Lista attributi    C) Docs    D) Codice

Q33. from .. import x significa:
     A) Stesso livello    B) Parent package    C) Root    D) Errore

Q34. >= in requirements.txt:
     A) Esatto    B) Minimo    C) Massimo    D) Diverso

Q35. Venv in .gitignore?
     A) No    B) Sì    C) Dipende    D) Mai

Q36. math.factorial(5):
     A) 5    B) 25    C) 120    D) 125

Q37. math.gcd(12, 8):
     A) 2    B) 4    C) 12    D) 24

Q38. os.path.exists() restituisce:
     A) Path    B) Boolean    C) File    D) Directory

Q39. sys.version_info è:
     A) Stringa    B) Tupla    C) Lista    D) Dict

Q40. pip show package mostra:
     A) Codice    B) Info e dipendenze    C) Solo versione    D) Errori


═══════════════════════════════════════════════════════════════════════════════
"""

MODULE_1_FINAL_ANSWERS = """
═══════════════════════════════════════════════════════════════════════════════
                    PE2 MODULE 1 - RISPOSTE TEST FINALE
═══════════════════════════════════════════════════════════════════════════════

Q1: C    Q2: B    Q3: B    Q4: B    Q5: A    Q6: A    Q7: B    Q8: D
Q9: B    Q10: B   Q11: B   Q12: B   Q13: B   Q14: B   Q15: B   Q16: B
Q17: B   Q18: B   Q19: B   Q20: B   Q21: B   Q22: B   Q23: B   Q24: B
Q25: B   Q26: B   Q27: B   Q28: B   Q29: C   Q30: B   Q31: B   Q32: B
Q33: B   Q34: B   Q35: B   Q36: C   Q37: B   Q38: B   Q39: B   Q40: B

PUNTEGGIO:
──────────
36-40: Eccellente!
32-35: Ottimo!
28-31: Buono (70% pass)
<28:   Rivedi le sezioni deboli

═══════════════════════════════════════════════════════════════════════════════
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    LABS
# ══════════════════════════════════════════════════════════════════════════════

LABS = """
═══════════════════════════════════════════════════════════════════════════════
                    PE2 MODULE 1 - LABS
═══════════════════════════════════════════════════════════════════════════════

LAB 1: Crea un modulo calculator.py con funzioni add, sub, mul, div.

LAB 2: Usa il modulo math per calcolare area e circonferenza di un cerchio.

LAB 3: Usa random per simulare il lancio di 2 dadi 1000 volte e conta le somme.

LAB 4: Usa random.shuffle per mescolare un mazzo di carte.

LAB 5: Usa os e platform per stampare info complete sul sistema.

LAB 6: Crea un package "geometry" con moduli circle.py e rectangle.py.

LAB 7: Scrivi __init__.py che espone le funzioni principali del package.

LAB 8: Crea un requirements.txt per un progetto con requests, pandas, numpy.

LAB 9: Crea e attiva un virtual environment, installa packages.

LAB 10: Usa sys.argv per creare uno script che accetta argomenti.

LAB 11: Implementa il pattern if __name__ == "__main__" nel tuo modulo.

LAB 12: Usa random.seed per generare sequenze riproducibili.

LAB 13: Usa math.log per calcolare il tempo di raddoppio di un investimento.

LAB 14: Crea un modulo con __all__ per controllare gli export.

LAB 15: Implementa un semplice generatore di password con random.
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    ESECUZIONE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 78)
    print("PYTHON ESSENTIALS 2 - MODULE 1")
    print("Modules, Packages, and PIP")
    print("=" * 78)
    print("""
    CONTENUTO:
    ──────────
    9 Sezioni di teoria con quiz
    15 Labs pratici
    Test finale (40 domande)
    
    COMANDI:
    ────────
    print(QUIZ_1_1)   → Quiz Import
    print(QUIZ_1_3)   → Quiz Math
    print(QUIZ_1_4)   → Quiz Random
    print(LABS)       → Esercizi pratici
    print(MODULE_1_FINAL_TEST)    → Test finale
    print(MODULE_1_FINAL_ANSWERS) → Risposte
    """)
