"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    PYTHON ESSENTIALS 2 - MODULE 4                            ║
║           Generators, Iterators, File I/O, os, datetime                      ║
║                                                                              ║
║                     Allineato al Syllabus PCAP-31-03                         ║
║                         OpenEDG Python Institute                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

PCAP-31-03 Exam Block 4: Miscellaneous (36%)

STRUTTURA MODULO:
├── Section 4.1: Generators and Iterators
├── Section 4.2: List/Dict/Set Comprehensions
├── Section 4.3: Lambdas and Closures
├── Section 4.4: File Handling
├── Section 4.5: os and os.path
├── Section 4.6: datetime Module
├── Labs (10 esercizi)
└── Module 4 Quiz (30 domande)

TEMPO STIMATO: 8-10 ore

═══════════════════════════════════════════════════════════════════════════════
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 4.1: GENERATORS AND ITERATORS
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    4.1 TEORIA: GENERATORI E ITERATORI                        │
└──────────────────────────────────────────────────────────────────────────────┘

ITERATOR: Oggetto che implementa __iter__() e __next__()
GENERATOR: Funzione che usa yield invece di return
"""

# ITERATOR MANUALE
class Counter:
    def __init__(self, max):
        self.max = max
        self.n = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.n >= self.max:
            raise StopIteration
        self.n += 1
        return self.n

counter = Counter(3)
print(next(counter))  # 1
print(next(counter))  # 2
print(next(counter))  # 3
# print(next(counter))  # StopIteration!


"""
GENERATOR FUNCTION (con yield):
───────────────────────────────
Più semplice di un iterator!
"""

def count_up_to(max):
    n = 1
    while n <= max:
        yield n  # "Restituisce" e SOSPENDE
        n += 1

gen = count_up_to(3)
print(next(gen))  # 1
print(next(gen))  # 2
print(next(gen))  # 3
# print(next(gen))  # StopIteration

# Usare in for loop
for n in count_up_to(3):
    print(n)  # 1, 2, 3


"""
GENERATOR EXPRESSION:
─────────────────────
Come list comprehension ma con parentesi tonde ()
"""

# List comprehension - crea TUTTA la lista in memoria
squares_list = [x**2 for x in range(1000)]

# Generator expression - crea valori ON DEMAND (lazy)
squares_gen = (x**2 for x in range(1000))

print(type(squares_gen))  # <class 'generator'>
print(next(squares_gen))  # 0
print(next(squares_gen))  # 1


"""
⚠️ TRAPPOLA: I generatori si ESAURISCONO!
────────────────────────────────────────
"""
gen = (x for x in [1, 2, 3])
print(list(gen))  # [1, 2, 3]
print(list(gen))  # [] - VUOTO! Il generatore è esaurito!


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 4.2: COMPREHENSIONS
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    4.2 TEORIA: COMPREHENSIONS                                │
└──────────────────────────────────────────────────────────────────────────────┘

LIST COMPREHENSION:
───────────────────
"""
# Sintassi base
squares = [x**2 for x in range(5)]
print(squares)  # [0, 1, 4, 9, 16]

# Con condizione (filter)
evens = [x for x in range(10) if x % 2 == 0]
print(evens)  # [0, 2, 4, 6, 8]

# Con if-else (transform)
labels = ["even" if x % 2 == 0 else "odd" for x in range(5)]
print(labels)  # ['even', 'odd', 'even', 'odd', 'even']

# Nested
matrix = [[j for j in range(3)] for i in range(3)]
print(matrix)  # [[0, 1, 2], [0, 1, 2], [0, 1, 2]]

# Flatten
flat = [x for row in matrix for x in row]
print(flat)  # [0, 1, 2, 0, 1, 2, 0, 1, 2]


"""
DICT COMPREHENSION:
───────────────────
"""
squares_dict = {x: x**2 for x in range(5)}
print(squares_dict)  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# Invertire un dizionario
original = {'a': 1, 'b': 2}
inverted = {v: k for k, v in original.items()}
print(inverted)  # {1: 'a', 2: 'b'}


"""
SET COMPREHENSION:
──────────────────
"""
unique_lengths = {len(word) for word in ["hello", "world", "hi"]}
print(unique_lengths)  # {2, 5}


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 4.3: LAMBDAS AND CLOSURES
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    4.3 TEORIA: LAMBDA E CLOSURE                              │
└──────────────────────────────────────────────────────────────────────────────┘

LAMBDA: Funzione anonima in una riga
Sintassi: lambda args: expression
"""

# Lambda semplice
square = lambda x: x**2
print(square(5))  # 25

# Equivalente a:
def square_func(x):
    return x**2

# Lambda con più argomenti
add = lambda x, y: x + y
print(add(3, 4))  # 7


"""
LAMBDA con sorted(), map(), filter():
──────────────────────────────────────
"""

# sorted con key
words = ["banana", "apple", "cherry"]
print(sorted(words, key=lambda w: len(w)))  # ['apple', 'banana', 'cherry']
print(sorted(words, key=lambda w: w[-1]))   # Per ultima lettera

# map: applica funzione a ogni elemento
nums = [1, 2, 3, 4]
squared = list(map(lambda x: x**2, nums))
print(squared)  # [1, 4, 9, 16]

# filter: filtra elementi
evens = list(filter(lambda x: x % 2 == 0, nums))
print(evens)  # [2, 4]


"""
CLOSURE:
────────
Una funzione che "ricorda" le variabili del suo scope esterno
"""

def make_multiplier(n):
    def multiplier(x):
        return x * n  # n è "catturato" dalla closure
    return multiplier

double = make_multiplier(2)
triple = make_multiplier(3)

print(double(5))  # 10
print(triple(5))  # 15


# ══════════════════════════════════════════════════════════════════════════════
#                     SECTION 4.1-4.3 - QUIZ
# ══════════════════════════════════════════════════════════════════════════════

QUIZ_4_1_4_3 = """
══════════════════════════════════════════════════════════════════════════════
QUIZ 4.1.1
══════════════════════════════════════════════════════════════════════════════
def gen():
    yield 1
    yield 2

g = gen()
print(next(g), next(g))

Stampa:
A) 1 1
B) 1 2
C) 2 2
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.1.2 - TRAPPOLA!
══════════════════════════════════════════════════════════════════════════════
g = (x for x in [1, 2, 3])
print(sum(g), sum(g))

Stampa:
A) 6 6
B) 6 0
C) 0 6
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.2.1
══════════════════════════════════════════════════════════════════════════════
print([x*2 for x in range(3)])

Stampa:
A) [0, 2, 4]
B) [2, 4, 6]
C) [0, 1, 2]
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.2.2
══════════════════════════════════════════════════════════════════════════════
print([x for x in range(10) if x % 3 == 0])

Stampa:
A) [3, 6, 9]
B) [0, 3, 6, 9]
C) [0, 3, 6, 9, 12]
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.2.3
══════════════════════════════════════════════════════════════════════════════
print({x: x**2 for x in [1, 2, 2, 3]})

Stampa:
A) {1: 1, 2: 4, 2: 4, 3: 9}
B) {1: 1, 2: 4, 3: 9}
C) Error
D) [(1,1), (2,4), (3,9)]

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.3.1
══════════════════════════════════════════════════════════════════════════════
f = lambda x, y: x + y
print(f(3, 4))

Stampa:
A) 7
B) 34
C) Error
D) lambda

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.3.2
══════════════════════════════════════════════════════════════════════════════
print(list(map(lambda x: x*2, [1, 2, 3])))

Stampa:
A) [1, 2, 3]
B) [2, 4, 6]
C) [1, 4, 9]
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.3.3
══════════════════════════════════════════════════════════════════════════════
print(list(filter(lambda x: x > 2, [1, 2, 3, 4])))

Stampa:
A) [1, 2]
B) [3, 4]
C) [True, True]
D) Error

Tua risposta: ___
"""


RISPOSTE_4_1_4_3 = """
══════════════════════════════════════════════════════════════════════════════
RISPOSTE QUIZ 4.1-4.3
══════════════════════════════════════════════════════════════════════════════

4.1.1: B) 1 2
       yield sospende e riprende, restituendo valori in sequenza.

4.1.2: B) 6 0
       I generatori si ESAURISCONO! Dopo sum(g), g è vuoto.

4.2.1: A) [0, 2, 4]
       range(3) = 0, 1, 2 → *2 = 0, 2, 4

4.2.2: B) [0, 3, 6, 9]
       0 è divisibile per 3! (0 % 3 == 0)

4.2.3: B) {1: 1, 2: 4, 3: 9}
       Dict non ha duplicati, il secondo 2 sovrascrive il primo.

4.3.1: A) 7
       Lambda con due parametri, restituisce la somma.

4.3.2: B) [2, 4, 6]
       map applica la funzione a ogni elemento.

4.3.3: B) [3, 4]
       filter mantiene solo elementi dove la lambda è True.
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 4.4: FILE HANDLING
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    4.4 TEORIA: FILE I/O                                      │
└──────────────────────────────────────────────────────────────────────────────┘

APRIRE UN FILE:
───────────────
open(filename, mode)

MODI:
- 'r'  : Read (default)
- 'w'  : Write (sovrascrive!)
- 'a'  : Append
- 'r+' : Read and Write
- 'b'  : Binary mode (rb, wb)
- 't'  : Text mode (default)
"""

# MODO CONSIGLIATO: with statement (chiude automaticamente)
# with open('file.txt', 'r') as f:
#     content = f.read()

# Metodi di lettura:
# f.read()      - Legge TUTTO il file come stringa
# f.read(n)     - Legge n caratteri
# f.readline()  - Legge UNA riga
# f.readlines() - Legge TUTTE le righe come lista

# Metodi di scrittura:
# f.write(str)      - Scrive stringa
# f.writelines(lst) - Scrive lista di stringhe (no newline!)


"""
ESEMPIO COMPLETO:
─────────────────
"""

# Scrittura
# with open('test.txt', 'w') as f:
#     f.write('Line 1\n')
#     f.write('Line 2\n')

# Lettura completa
# with open('test.txt', 'r') as f:
#     content = f.read()
#     print(content)

# Lettura riga per riga (memory efficient)
# with open('test.txt', 'r') as f:
#     for line in f:
#         print(line.strip())


"""
⚠️ ERRORI COMUNI:
────────────────
"""
# FileNotFoundError: file non esiste (in mode 'r')
# PermissionError: permessi insufficienti
# IsADirectoryError: path è una directory


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 4.5: os AND os.path
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    4.5 TEORIA: os MODULE                                     │
└──────────────────────────────────────────────────────────────────────────────┘
"""
import os
import os.path

# Info sistema
print(os.name)        # 'posix' (Linux/Mac) o 'nt' (Windows)
print(os.getcwd())    # Current working directory

# Operazioni directory
# os.mkdir('dir')      # Crea directory
# os.makedirs('a/b/c') # Crea directory ricorsivamente
# os.rmdir('dir')      # Rimuove directory (vuota)
# os.chdir('path')     # Cambia directory

# Elencare contenuti
# os.listdir('.')      # Lista file/cartelle

# Operazioni file
# os.remove('file')    # Elimina file
# os.rename('old', 'new')


"""
os.path - Manipolazione percorsi:
─────────────────────────────────
"""
path = '/home/user/file.txt'

print(os.path.exists(path))     # True/False
print(os.path.isfile(path))     # True se è un file
print(os.path.isdir(path))      # True se è una directory

print(os.path.basename(path))   # 'file.txt'
print(os.path.dirname(path))    # '/home/user'
print(os.path.split(path))      # ('/home/user', 'file.txt')
print(os.path.splitext(path))   # ('/home/user/file', '.txt')

# Costruire path (cross-platform!)
print(os.path.join('home', 'user', 'file.txt'))
# Linux: 'home/user/file.txt'
# Windows: 'home\\user\\file.txt'


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 4.6: datetime MODULE
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    4.6 TEORIA: datetime                                      │
└──────────────────────────────────────────────────────────────────────────────┘
"""
from datetime import date, time, datetime, timedelta

# DATE
today = date.today()
print(today)           # 2024-06-15
print(today.year)      # 2024
print(today.month)     # 6
print(today.day)       # 15
print(today.weekday()) # 0=Monday, 6=Sunday

specific_date = date(2024, 12, 25)

# TIME
t = time(14, 30, 45)  # 14:30:45
print(t.hour)         # 14
print(t.minute)       # 30
print(t.second)       # 45

# DATETIME (date + time)
now = datetime.now()
print(now)  # 2024-06-15 14:30:45.123456

# Creare datetime specifico
dt = datetime(2024, 12, 25, 10, 30, 0)

# Formattazione: strftime
print(now.strftime("%Y-%m-%d"))        # '2024-06-15'
print(now.strftime("%d/%m/%Y %H:%M"))  # '15/06/2024 14:30'

# Parsing: strptime
dt = datetime.strptime("2024-06-15", "%Y-%m-%d")


"""
TIMEDELTA - Differenze di tempo:
────────────────────────────────
"""
delta = timedelta(days=7, hours=5)
future = datetime.now() + delta
past = datetime.now() - timedelta(days=30)

# Differenza tra date
d1 = date(2024, 6, 15)
d2 = date(2024, 6, 1)
diff = d1 - d2
print(diff.days)  # 14


# ══════════════════════════════════════════════════════════════════════════════
#                     SECTION 4.4-4.6 - QUIZ
# ══════════════════════════════════════════════════════════════════════════════

QUIZ_4_4_4_6 = """
══════════════════════════════════════════════════════════════════════════════
QUIZ 4.4.1
══════════════════════════════════════════════════════════════════════════════
Quale mode SOVRASCRIVE il contenuto del file?

A) 'r'
B) 'w'
C) 'a'
D) 'r+'

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.4.2
══════════════════════════════════════════════════════════════════════════════
with open('file.txt', 'r') as f:
    data = f.read()
# Qui, f è...?

A) Ancora aperto
B) Chiuso automaticamente
C) None
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.4.3
══════════════════════════════════════════════════════════════════════════════
Quale metodo legge TUTTE le righe come lista?

A) read()
B) readline()
C) readlines()
D) readall()

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.5.1
══════════════════════════════════════════════════════════════════════════════
import os.path
print(os.path.basename('/home/user/file.txt'))

Stampa:
A) '/home/user'
B) 'file.txt'
C) '/home/user/file.txt'
D) 'file'

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.5.2
══════════════════════════════════════════════════════════════════════════════
import os.path
print(os.path.splitext('file.txt'))

Stampa:
A) ('file', 'txt')
B) ('file', '.txt')
C) ['file', '.txt']
D) ('file.', 'txt')

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.6.1
══════════════════════════════════════════════════════════════════════════════
from datetime import date
d = date(2024, 6, 15)
print(d.weekday())  # Sabato

Stampa:
A) 6 (Sunday)
B) 5 (Saturday)
C) 7
D) 'Saturday'

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.6.2
══════════════════════════════════════════════════════════════════════════════
from datetime import date
d1 = date(2024, 6, 15)
d2 = date(2024, 6, 10)
print(type(d1 - d2))

Stampa:
A) <class 'int'>
B) <class 'date'>
C) <class 'datetime.timedelta'>
D) Error

Tua risposta: ___
"""


RISPOSTE_4_4_4_6 = """
══════════════════════════════════════════════════════════════════════════════
RISPOSTE QUIZ 4.4-4.6
══════════════════════════════════════════════════════════════════════════════

4.4.1: B) 'w'
       'w' crea/sovrascrive. 'a' aggiunge alla fine.

4.4.2: B) Chiuso automaticamente
       with statement chiude automaticamente all'uscita dal blocco.

4.4.3: C) readlines()
       read() = stringa, readline() = una riga, readlines() = lista di righe.

4.5.1: B) 'file.txt'
       basename restituisce solo il nome del file.

4.5.2: B) ('file', '.txt')
       splitext separa nome ed ESTENSIONE (con il punto).

4.6.1: B) 5 (Saturday)
       weekday() restituisce 0=Monday, ..., 5=Saturday, 6=Sunday.

4.6.2: C) <class 'datetime.timedelta'>
       La differenza tra date restituisce un timedelta.
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    MODULE 4 - TEST FINALE
# ══════════════════════════════════════════════════════════════════════════════

MODULE_4_TEST = """
══════════════════════════════════════════════════════════════════════════════
                     PE2 MODULE 4 - TEST FINALE
                    30 domande - Target: 70% (21/30)
══════════════════════════════════════════════════════════════════════════════

Q1. def g(): yield 1; yield 2
    print(list(g())) → ?
    A) [1, 2]    B) [1]    C) Error    D) generator

Q2. g=(x for x in [1,2]); sum(g); sum(g) → ?
    A) 3, 3    B) 3, 0    C) 0, 3    D) Error

Q3. print(type((x for x in []))) → ?
    A) list    B) tuple    C) generator    D) Error

Q4. [x*2 for x in range(3)] → ?
    A) [0,2,4]    B) [2,4,6]    C) [0,1,2]    D) Error

Q5. [x for x in range(5) if x%2] → ?
    A) [0,2,4]    B) [1,3]    C) [0,1,2,3,4]    D) Error

Q6. {x:x**2 for x in [1,1,2]} → ?
    A) {1:1,1:1,2:4}    B) {1:1,2:4}    C) Error    D) [(1,1),(2,4)]

Q7. f=lambda x:x*2; f(3) → ?
    A) 6    B) 33    C) Error    D) lambda

Q8. list(map(str,[1,2,3])) → ?
    A) ['1','2','3']    B) [1,2,3]    C) '123'    D) Error

Q9. list(filter(None,[0,1,2,'',3])) → ?
    A) [0,1,2,'',3]    B) [1,2,3]    C) []    D) Error

Q10. Quale mode file aggiunge alla fine?
     A) 'r'    B) 'w'    C) 'a'    D) 'r+'

Q11. f.read() restituisce:
     A) Lista di righe    B) Una riga    C) Tutto come stringa    D) Bytes

Q12. f.readlines() restituisce:
     A) Lista di righe    B) Una riga    C) Tutto come stringa    D) Generator

Q13. with open(...) as f: ... dopo il blocco f è:
     A) Aperto    B) Chiuso    C) None    D) Error

Q14. os.path.basename('/a/b/c.txt') → ?
     A) '/a/b'    B) 'c.txt'    C) 'c'    D) '.txt'

Q15. os.path.dirname('/a/b/c.txt') → ?
     A) '/a/b'    B) 'c.txt'    C) '/a/b/'    D) 'c'

Q16. os.path.splitext('file.tar.gz') → ?
     A) ('file','tar.gz')    B) ('file.tar','.gz')    C) ('file','.tar.gz')    D) Error

Q17. os.name su Linux è:
     A) 'linux'    B) 'posix'    C) 'unix'    D) 'gnu'

Q18. date.today().weekday() per Monday → ?
     A) 0    B) 1    C) 7    D) 'Monday'

Q19. date(2024,6,15)-date(2024,6,10) → tipo?
     A) int    B) date    C) timedelta    D) Error

Q20. datetime.strftime("%Y") restituisce:
     A) Anno come int    B) Anno come stringa    C) Error    D) None

Q21. ["a" if x else "b" for x in [0,1,2]] → ?
     A) ['a','a','a']    B) ['b','a','a']    C) ['b','b','b']    D) Error

Q22. sorted([3,1,2], key=lambda x:-x) → ?
     A) [1,2,3]    B) [3,2,1]    C) [-3,-2,-1]    D) Error

Q23. def f(n):
         def g(x): return x*n
         return g
     f(3)(4) → ?
     A) 12    B) 7    C) 34    D) Error

Q24. next(iter([1,2,3])) → ?
     A) [1,2,3]    B) 1    C) iter    D) Error

Q25. timedelta(days=1,hours=2).total_seconds() → ?
     A) 90000    B) 93600    C) 86400    D) 3600

Q26. open('x.txt','w') se x.txt non esiste:
     A) FileNotFoundError    B) Crea il file    C) PermissionError    D) None

Q27. open('x.txt','r') se x.txt non esiste:
     A) FileNotFoundError    B) Crea il file    C) None    D) Returns empty

Q28. os.path.join('a','b','c') su Linux → ?
     A) 'a/b/c'    B) 'a\\b\\c'    C) 'abc'    D) ['a','b','c']

Q29. {len(w) for w in ['hi','hello','hi']} → ?
     A) {2,5,2}    B) {2,5}    C) [2,5]    D) Error

Q30. def g(): yield from [1,2]; list(g()) → ?
     A) [1,2]    B) [[1,2]]    C) Error    D) generator


══════════════════════════════════════════════════════════════════════════════
                              FINE TEST
══════════════════════════════════════════════════════════════════════════════
"""


MODULE_4_TEST_ANSWERS = """
══════════════════════════════════════════════════════════════════════════════
                    PE2 MODULE 4 - RISPOSTE
══════════════════════════════════════════════════════════════════════════════

Q1:  A) [1, 2]
Q2:  B) 3, 0 (generator esaurito!)
Q3:  C) generator
Q4:  A) [0,2,4]
Q5:  B) [1,3] (x%2 è truthy per dispari)
Q6:  B) {1:1,2:4} (no duplicati)
Q7:  A) 6
Q8:  A) ['1','2','3']
Q9:  B) [1,2,3] (filter(None,...) rimuove falsy)
Q10: C) 'a' (append)
Q11: C) Tutto come stringa
Q12: A) Lista di righe
Q13: B) Chiuso
Q14: B) 'c.txt'
Q15: A) '/a/b'
Q16: B) ('file.tar','.gz')
Q17: B) 'posix'
Q18: A) 0
Q19: C) timedelta
Q20: B) Anno come stringa
Q21: B) ['b','a','a'] (0 è falsy)
Q22: B) [3,2,1] (ordine decrescente)
Q23: A) 12 (closure!)
Q24: B) 1
Q25: B) 93600 (86400+7200)
Q26: B) Crea il file ('w' crea)
Q27: A) FileNotFoundError ('r' richiede esistenza)
Q28: A) 'a/b/c'
Q29: B) {2,5} (set no duplicati)
Q30: A) [1,2] (yield from)

PUNTEGGIO: ___/30
Target: 21/30 (70%)
"""


if __name__ == "__main__":
    print("=" * 78)
    print("PYTHON ESSENTIALS 2 - MODULE 4")
    print("Generators, File I/O, os, datetime")
    print("=" * 78)
    print("""
    CONTENUTO:
    - Generators e yield
    - List/Dict/Set Comprehensions  
    - Lambda, map, filter
    - File handling con with
    - os e os.path
    - datetime e timedelta
    
    TRAPPOLE:
    ⚠️  I generatori si ESAURISCONO dopo l'uso
    ⚠️  'w' sovrascrive, 'a' aggiunge
    ⚠️  splitext('a.tar.gz') → ('a.tar', '.gz')
    ⚠️  weekday(): 0=Monday, 6=Sunday
    ⚠️  filter(None, lst) rimuove tutti i falsy
    """)
