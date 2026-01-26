#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PYTHON ESSENTIALS 2 - MODULE 4                            ║
║                    MISCELLANEOUS: Lambdas, Closures, Generators, File I/O    ║
║                    PCAP-31-03 Section 5: 22% (9 domande)                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

SYLLABUS:
├── PCAP 5.1 - List comprehensions with conditions
├── PCAP 5.2 - Lambda functions, map(), filter()
├── PCAP 5.3 - Closures
├── PCAP 5.4 - I/O terminology (modes, streams, handles)
└── PCAP 5.5 - File I/O (open, read, write, readline, bytearray)
"""

# ══════════════════════════════════════════════════════════════════════════════
# 5.1 LIST COMPREHENSIONS (ADVANCED)
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("5.1 LIST COMPREHENSIONS (ADVANCED)")
print("=" * 70)

# Basic
squares = [x**2 for x in range(5)]
print(f"[x**2 for x in range(5)] = {squares}")

# With condition (filter)
evens = [x for x in range(10) if x % 2 == 0]
print(f"[x for x in range(10) if x % 2 == 0] = {evens}")

# With if-else (transform)
labels = ["even" if x % 2 == 0 else "odd" for x in range(5)]
print(f"['even' if x%2==0 else 'odd' for x in range(5)] = {labels}")

# Nested loops
matrix = [[i*j for j in range(1,4)] for i in range(1,4)]
print(f"Matrice 3x3: {matrix}")

# Flatten nested list
nested = [[1,2], [3,4], [5,6]]
flat = [item for sublist in nested for item in sublist]
print(f"Flatten {nested} = {flat}")

# Dict comprehension
squares_dict = {x: x**2 for x in range(5)}
print(f"Dict: {squares_dict}")

# Set comprehension
unique = {x % 3 for x in range(10)}
print(f"Set: {unique}")

# ══════════════════════════════════════════════════════════════════════════════
# 5.2 LAMBDA FUNCTIONS (ESAME!)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("5.2 LAMBDA FUNCTIONS (ESAME!)")
print("=" * 70)

print("""
📋 LAMBDA SYNTAX:
   lambda arguments: expression
   
   - Funzione anonima su una sola linea
   - Può avere più argomenti
   - Solo UNA espressione (no statements)
   - Restituisce automaticamente il risultato
""")

# Basic lambda
square = lambda x: x ** 2
print(f"square = lambda x: x**2")
print(f"square(5) = {square(5)}")

# Multiple arguments
add = lambda a, b: a + b
print(f"add = lambda a, b: a + b")
print(f"add(3, 4) = {add(3, 4)}")

# With default values
greet = lambda name="World": f"Hello, {name}!"
print(f"greet() = {greet()}")
print(f"greet('Marco') = {greet('Marco')}")

# ══════════════════════════════════════════════════════════════════════════════
# 5.3 map() AND filter() (ESAME!)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("5.3 map() AND filter() (ESAME!)")
print("=" * 70)

# MAP - Applica funzione a ogni elemento
print("📐 map(function, iterable)")
numbers = [1, 2, 3, 4, 5]

# Con lambda
squared = list(map(lambda x: x**2, numbers))
print(f"map(lambda x: x**2, {numbers}) = {squared}")

# Con funzione definita
def double(x):
    return x * 2

doubled = list(map(double, numbers))
print(f"map(double, {numbers}) = {doubled}")

# Map con multiple iterabili
a = [1, 2, 3]
b = [10, 20, 30]
sums = list(map(lambda x, y: x + y, a, b))
print(f"map(lambda x,y: x+y, {a}, {b}) = {sums}")

# FILTER - Filtra elementi che soddisfano condizione
print("\n📐 filter(function, iterable)")
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Numeri pari
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(f"filter(lambda x: x%2==0, {numbers}) = {evens}")

# Numeri > 5
greater = list(filter(lambda x: x > 5, numbers))
print(f"filter(lambda x: x>5, {numbers}) = {greater}")

# ⚠️ map/filter restituiscono ITERATORI, non liste!
print(f"\ntype(map(...)) = {type(map(lambda x: x, [1,2,3]))}")
print("Devi convertire in list() per vedere i risultati!")

# ══════════════════════════════════════════════════════════════════════════════
# 5.4 CLOSURES (ESAME!)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("5.4 CLOSURES (ESAME!)")
print("=" * 70)

print("""
📋 CLOSURE:
   Una funzione che "ricorda" le variabili dello scope in cui è stata definita,
   anche dopo che quello scope è terminato.
   
   Requisiti:
   1. Funzione annidata (inner function)
   2. Inner function riferisce variabile dello scope esterno
   3. La funzione esterna restituisce la inner function
""")

# Esempio classico
def outer(x):
    # x è nello scope di outer
    def inner(y):
        return x + y  # inner "ricorda" x
    return inner

add_5 = outer(5)  # x = 5 è "catturato"
add_10 = outer(10)  # x = 10 è "catturato"

print(f"add_5 = outer(5)")
print(f"add_5(3) = {add_5(3)}")   # 8
print(f"add_5(7) = {add_5(7)}")   # 12

print(f"\nadd_10 = outer(10)")
print(f"add_10(3) = {add_10(3)}")  # 13

# Esempio pratico: counter
def make_counter():
    count = 0
    def counter():
        nonlocal count  # Necessario per modificare
        count += 1
        return count
    return counter

counter1 = make_counter()
counter2 = make_counter()

print(f"\ncounter1(): {counter1()}, {counter1()}, {counter1()}")  # 1, 2, 3
print(f"counter2(): {counter2()}, {counter2()}")  # 1, 2 (indipendente!)

# ══════════════════════════════════════════════════════════════════════════════
# 5.5 GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("5.5 GENERATORS")
print("=" * 70)

# Generator function (usa yield)
def count_up_to(n):
    i = 1
    while i <= n:
        yield i
        i += 1

gen = count_up_to(3)
print(f"type(gen) = {type(gen)}")
print(f"next(gen) = {next(gen)}")  # 1
print(f"next(gen) = {next(gen)}")  # 2
print(f"next(gen) = {next(gen)}")  # 3
# next(gen)  # StopIteration!

# Generator expression
squares_gen = (x**2 for x in range(5))
print(f"\nGenerator expression: (x**2 for x in range(5))")
print(f"list(squares_gen) = {list(squares_gen)}")

# ══════════════════════════════════════════════════════════════════════════════
# 5.6 FILE I/O - TERMINOLOGY (ESAME!)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("5.6 FILE I/O - TERMINOLOGY (ESAME!)")
print("=" * 70)

print("""
📋 TERMINOLOGIA:

STREAM
   - Flusso di dati tra programma e file/device
   - Input stream: dati IN (lettura)
   - Output stream: dati OUT (scrittura)

HANDLE (File Object)
   - Oggetto Python che rappresenta il file aperto
   - Restituito da open()
   - Ha metodi: read(), write(), close()

MODES:
┌──────┬─────────────────────────────────────────────────────┐
│ Mode │ Descrizione                                         │
├──────┼─────────────────────────────────────────────────────┤
│ 'r'  │ Read (default) - file deve esistere                 │
│ 'w'  │ Write - crea/sovrascrive file                       │
│ 'a'  │ Append - aggiunge alla fine                         │
│ 'x'  │ Exclusive create - errore se esiste                 │
│ 'r+' │ Read+Write - file deve esistere                     │
│ 'w+' │ Write+Read - crea/sovrascrive                       │
│ 'a+' │ Append+Read                                         │
├──────┼─────────────────────────────────────────────────────┤
│ 'b'  │ Binary mode (es. 'rb', 'wb')                        │
│ 't'  │ Text mode (default)                                 │
└──────┴─────────────────────────────────────────────────────┘
""")

# ══════════════════════════════════════════════════════════════════════════════
# 5.7 FILE I/O - OPERATIONS (ESAME!)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("5.7 FILE I/O - OPERATIONS (ESAME!)")
print("=" * 70)

# WRITING
print("📐 WRITING:")
print("""
# Metodo 1: open/close manuale
f = open('file.txt', 'w')
f.write('Hello\\n')
f.write('World\\n')
f.close()  # IMPORTANTE!

# Metodo 2: with statement (RACCOMANDATO)
with open('file.txt', 'w') as f:
    f.write('Hello\\n')
    f.write('World\\n')
# File chiuso automaticamente!
""")

# READING
print("\n📐 READING:")
print("""
# read() - Legge TUTTO il file
with open('file.txt', 'r') as f:
    content = f.read()

# read(n) - Legge n caratteri
with open('file.txt', 'r') as f:
    first_10 = f.read(10)

# readline() - Legge UNA riga
with open('file.txt', 'r') as f:
    line1 = f.readline()  # Include '\\n'
    line2 = f.readline()

# readlines() - Lista di TUTTE le righe
with open('file.txt', 'r') as f:
    lines = f.readlines()  # ['Hello\\n', 'World\\n']

# Iterazione diretta (EFFICIENTE)
with open('file.txt', 'r') as f:
    for line in f:
        print(line.strip())
""")

# Esempio pratico
import tempfile
import os

# Creiamo un file temporaneo per demo
temp_file = os.path.join(tempfile.gettempdir(), 'demo.txt')

# Write
with open(temp_file, 'w') as f:
    f.write("Line 1\n")
    f.write("Line 2\n")
    f.write("Line 3\n")

# Read
print("\nEsempio pratico:")
with open(temp_file, 'r') as f:
    print(f"f.readline() = {repr(f.readline())}")
    print(f"f.readline() = {repr(f.readline())}")

with open(temp_file, 'r') as f:
    print(f"f.readlines() = {f.readlines()}")

# Cleanup
os.remove(temp_file)

# ══════════════════════════════════════════════════════════════════════════════
# 5.8 BINARY FILES AND bytearray (ESAME!)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("5.8 BINARY FILES AND bytearray (ESAME!)")
print("=" * 70)

print("""
📋 BINARY MODE:
   - Usa 'rb', 'wb' invece di 'r', 'w'
   - Lavora con bytes, non stringhe
   - Necessario per immagini, audio, etc.
""")

# bytearray - array mutabile di bytes
print("\n📐 bytearray:")
ba = bytearray(b'Hello')
print(f"bytearray(b'Hello') = {ba}")
print(f"ba[0] = {ba[0]}")  # 72 (code point di 'H')

# Modifica (bytearray è mutabile!)
ba[0] = 74  # 'J'
print(f"Dopo ba[0] = 74: {ba}")

# Append
ba.append(33)  # '!'
print(f"Dopo ba.append(33): {ba}")

# Conversione
print(f"ba.decode() = '{ba.decode()}'")

# Binary file I/O
print("""
# Scrittura binaria
with open('file.bin', 'wb') as f:
    f.write(b'Binary data')
    f.write(bytearray([0, 1, 2, 3]))

# Lettura binaria
with open('file.bin', 'rb') as f:
    data = f.read()  # bytes object
    ba = bytearray(data)  # Convertibile a bytearray
""")

# ══════════════════════════════════════════════════════════════════════════════
# 5.9 errno
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("5.9 errno MODULE")
print("=" * 70)

import errno

print("""
📋 errno - Codici di errore sistema

Usato quando si verificano errori I/O per identificare la causa.
""")

print(f"errno.ENOENT = {errno.ENOENT}")  # No such file
print(f"errno.EACCES = {errno.EACCES}")  # Permission denied
print(f"errno.EEXIST = {errno.EEXIST}")  # File exists

# ══════════════════════════════════════════════════════════════════════════════
# QUIZ
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("QUIZ DI VERIFICA")
print("=" * 70)

print("""
Q1. [x**2 for x in range(3)] = ?
    A) [0, 1, 4]  B) [1, 4, 9]  C) [0, 1, 2]  D) Error
    → RISPOSTA: A

Q2. list(filter(lambda x: x>2, [1,2,3,4])) = ?
    A) [1, 2]  B) [3, 4]  C) [2, 3, 4]  D) Error
    → RISPOSTA: B

Q3. list(map(lambda x: x*2, [1,2,3])) = ?
    A) [1, 2, 3]  B) [2, 4, 6]  C) [1, 4, 9]  D) Error
    → RISPOSTA: B

Q4. Cos'è una closure?
    A) Funzione che chiude file
    B) Funzione che ricorda variabili dello scope esterno
    C) Funzione ricorsiva
    D) Funzione senza parametri
    → RISPOSTA: B

Q5. Quale mode crea file e solleva errore se esiste?
    A) 'w'  B) 'a'  C) 'x'  D) 'r+'
    → RISPOSTA: C

Q6. readline() restituisce?
    A) Tutto il file  B) Una riga  C) Lista di righe  D) Un carattere
    → RISPOSTA: B

Q7. readlines() restituisce?
    A) String  B) Una riga  C) Lista di righe  D) Bytes
    → RISPOSTA: C

Q8. bytearray è?
    A) Immutabile  B) Mutabile  C) Solo lettura  D) Solo scrittura
    → RISPOSTA: B
""")

print("\n" + "=" * 70)
print("MODULO 4 COMPLETATO!")
print("PCAP MODULES COMPLETATI! → Ora: PCAP Exam Simulations")
print("=" * 70)
