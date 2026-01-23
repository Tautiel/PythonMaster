# 🎓 PYTHON MASTER PATH - Stile Python Institute

## La Logica OpenEDG

Il Python Institute usa un approccio preciso per ogni argomento:

```
┌─────────────────────────────────────────────────────────────┐
│  SEZIONE → TEORIA → ESEMPI → QUIZ → LAB → TEST MODULO      │
└─────────────────────────────────────────────────────────────┘
```

**Non è "impara a scrivere codice" ma "CAPISCI come Python funziona"**

Ogni sezione ha:
1. **Teoria**: Spiegazione concettuale
2. **Esempi**: Codice da ANALIZZARE (non copiare)
3. **Quiz**: "Cosa stampa questo codice?" (senza eseguire)
4. **Lab**: Esercizio pratico
5. **Test**: Verifica comprensione

---

# 📚 STRUTTURA CORSI PYTHON INSTITUTE

## Livello 1: Python Essentials 1 (PE1) → PCEP

| Modulo | Contenuto | Sezioni | Quiz | Lab |
|--------|-----------|---------|------|-----|
| **M1** | Intro to Python | 1.1-1.3 | 3 | 0 |
| **M2** | Data Types, Variables, Operators, I/O | 2.1-2.6 | 6 | 7 |
| **M3** | Boolean, Conditionals, Loops, Lists | 3.1-3.7 | 7 | 25 |
| **M4** | Functions, Tuples, Dicts, Exceptions | 4.1-4.7 | 7 | 30 |

**Totale PE1:** ~35 ore, 30+ Lab, Test finale

---

## Livello 2: Python Essentials 2 (PE2) → PCAP

| Modulo | Contenuto | Sezioni | Quiz | Lab |
|--------|-----------|---------|------|-----|
| **M1** | Modules, Packages, PIP | 1.1-1.4 | 4 | 5 |
| **M2** | Strings, Exceptions avanzate | 2.1-2.6 | 6 | 10 |
| **M3** | Object-Oriented Programming | 3.1-3.6 | 6 | 15 |
| **M4** | Generators, Files, datetime, os | 4.1-4.6 | 6 | 10 |

**Totale PE2:** ~40 ore, 40+ Lab, Test finale

---

## Livello 3: Python Advanced (PA) → PCPP1

| Modulo | Contenuto |
|--------|-----------|
| **M1** | Advanced OOP (decorators, metaclasses, ABC) |
| **M2** | Best Practices (PEP8, PEP257, type hints) |
| **M3** | GUI Programming (Tkinter) |
| **M4** | Network Programming (sockets, REST) |
| **M5** | File Processing (XML, CSV, logging, sqlite3) |

---

## Livello 4: Python Professional → PCPP2

| Modulo | Contenuto |
|--------|-----------|
| **M1** | Testing (unittest, pytest) |
| **M2** | Design Patterns (Singleton, Factory, Observer...) |
| **M3** | Concurrency (multiprocessing, threading) |
| **M4** | Advanced Networking |
| **M5** | SQL/NoSQL Databases |
| **M6** | Clean Code (SOLID, refactoring) |

---

# 🔀 INTEGRAZIONE: Python Institute + Trading Bot + AI

Il percorso Python Institute ti dà le **fondamenta logiche**.
Per trading e AI aggiungiamo **moduli specifici** dopo ogni livello.

```
PE1 (PCEP)
    │
    ├──→ [EXTRA] NumPy Basics (per calcoli)
    │
    ▼
PE2 (PCAP)
    │
    ├──→ [EXTRA] Pandas Basics (per dati OHLCV)
    ├──→ [EXTRA] ccxt intro (API exchange)
    │
    ▼
PA (PCPP1)
    │
    ├──→ [EXTRA] Async/WebSocket (real-time trading)
    ├──→ [EXTRA] Backtesting framework
    │
    ▼
PP (PCPP2)
    │
    ├──→ [EXTRA] ML/Scikit-learn
    ├──→ [EXTRA] TimescaleDB (già creato!)
    │
    ▼
🤖 TRADING BOT COMPLETO + AI
```

---

# 📖 PE1 - PYTHON ESSENTIALS 1 (Dettaglio)

## MODULE 1: Introduction to Python and Computer Programming

### Section 1.1 – Introduction to Programming
**Teoria:**
- Come funziona un programma
- Compilation vs Interpretation
- Lexis, Syntax, Semantics

**Quiz 1.1:**
```
Q: Quale affermazione è vera?
A) Python compila direttamente in codice macchina
B) Python è puramente interpretato
C) Python compila in bytecode, poi la PVM lo interpreta
D) Python non usa nessuna forma di compilazione

Risposta: ___
```

### Section 1.2 – Introduction to Python
**Teoria:**
- Storia di Python
- Python 2 vs Python 3
- Implementazioni (CPython, Jython, PyPy)

**Quiz 1.2:**
```
Q: Qual è l'implementazione standard di Python?
A) Jython
B) PyPy
C) CPython
D) IronPython

Risposta: ___
```

### Section 1.3 – Downloading and Installing Python
**Lab:** Installa Python e VS Code (opzionale - puoi usare edube.org)

---

## MODULE 2: Data Types, Variables, Operators, Basic I/O

### Section 2.1 – The print() function
**Teoria:**
- Invocazione funzioni
- Argomenti posizionali vs keyword
- sep, end parameters

**Quiz 2.1:**
```python
print("a", "b", "c", sep="-", end="!")
print("d")
```
Output: ___

**Lab 1-3:** print() exercises

### Section 2.2 – Python Literals
**Teoria:**
- Integers (decimale, binario 0b, ottale 0o, esadecimale 0x)
- Floats (notazione scientifica)
- Strings, Boolean, None

**Quiz 2.2:**
```python
print(0o17)
print(0xFF)
print(1e-2)
```
Output: ___

### Section 2.3 – Operators
**Teoria:**
- Aritmetici: +, -, *, /, //, %, **
- Precedenza e associatività
- ** associa a DESTRA!

**Quiz 2.3:**
```python
print(2 ** 3 ** 2)
print(-3 ** 2)
print((-3) ** 2)
print(17 // -4)
print(17 % -4)
```
Output: ___

### Section 2.4 – Variables
**Teoria:**
- Naming rules (PEP8)
- Assignment
- Shortcut operators (+=, -=, etc.)

**Quiz 2.4:**
```python
x = 1
x = x + x
x += x
x *= 2
print(x)
```
Output: ___

### Section 2.5 – Comments
**Teoria:** # per singola linea, niente multilinea nativo

### Section 2.6 – Input function
**Teoria:** input() restituisce SEMPRE stringa

**Quiz 2.6:**
```python
x = input()  # utente digita: 5
y = input()  # utente digita: 3
print(x + y)
```
Output: ___

---

## MODULE 3: Boolean, Conditionals, Loops, Lists

### Section 3.1 – Comparison and Boolean
**Teoria:**
- ==, !=, <, >, <=, >=
- Chained comparisons (1 < x < 10)
- and, or, not (precedenza: not > and > or)

**Quiz 3.1:**
```python
print(1 < 2 < 3)
print(1 < 2 > 0)
print(True or False and False)
print(not True or True and not False)
```
Output: ___

### Section 3.2 – Conditional Execution
**Teoria:**
- if, if-else, if-elif-else
- Nesting
- Ternary operator

**Quiz 3.2:**
```python
x = 5
y = 10 if x > 3 else 20
print(y)
```
Output: ___

### Section 3.3 – Loops
**Teoria:**
- while, for
- range(start, stop, step)
- break, continue, pass
- else clause (esegue se NO break)

**Quiz 3.3:**
```python
for i in range(3):
    print(i, end=" ")
else:
    print("done")
```
Output: ___

```python
for i in range(5):
    if i == 3:
        break
    print(i, end=" ")
else:
    print("done")
```
Output: ___

### Section 3.4 – Lists basics
**Teoria:**
- Creazione, indexing (positivo e negativo)
- Metodi: append, insert, remove, pop

**Quiz 3.4:**
```python
lst = [1, 2, 3]
lst.append(4)
lst.insert(0, 0)
lst.remove(2)
print(lst)
```
Output: ___

### Section 3.5 – Bubble Sort
**Lab:** Implementa bubble sort

### Section 3.6 – List Operations
**Teoria:**
- Slicing [start:stop:step]
- in, not in
- Shallow copy vs reference

**Quiz 3.6 (CRITICO!):**
```python
a = [1, 2, 3]
b = a
b.append(4)
print(a)
```
Output: ___

```python
a = [1, 2, 3]
b = a[:]
b.append(4)
print(a)
```
Output: ___

### Section 3.7 – Lists in Lists
**Teoria:** Liste annidate, matrici

**Quiz 3.7:**
```python
matrix = [[1, 2], [3, 4]]
print(matrix[1][0])
```
Output: ___

---

## MODULE 4: Functions, Tuples, Dictionaries, Exceptions

### Section 4.1 – Functions basics
**Teoria:**
- def, return
- Docstrings

### Section 4.2 – Parameters
**Teoria:**
- Positional vs keyword arguments
- Default values
- *args, **kwargs

**Quiz 4.2:**
```python
def f(a, b=2, c=3):
    return a + b + c

print(f(1))
print(f(1, c=10))
print(f(c=10, a=1))
```
Output: ___

### Section 4.3 – Return values
**Teoria:**
- return None implicito
- Multiple return values (tuple)

### Section 4.4 – Scopes
**Teoria:**
- Local vs Global
- global keyword
- LEGB rule

**Quiz 4.4 (TRAPPOLA CLASSICA!):**
```python
x = "global"

def f():
    print(x)
    x = "local"

f()
```
Output: ___

### Section 4.5 – Tuples
**Teoria:**
- Immutabili
- Packing/unpacking
- Come chiavi dizionario

**Quiz 4.5:**
```python
t = (1,)
print(type(t))

t2 = (1)
print(type(t2))
```
Output: ___

### Section 4.6 – Dictionaries
**Teoria:**
- Creazione, accesso
- Metodi: keys, values, items, get
- Chiavi duplicate

**Quiz 4.6:**
```python
d = {'a': 1, 'b': 2, 'a': 3}
print(d)
print(len(d))
```
Output: ___

### Section 4.7 – Exceptions
**Teoria:**
- try-except-else-finally
- raise
- Exception hierarchy

**Quiz 4.7:**
```python
try:
    print("A")
    x = 1/0
    print("B")
except ZeroDivisionError:
    print("C")
else:
    print("D")
finally:
    print("E")
```
Output: ___

---

# 📖 PE2 - PYTHON ESSENTIALS 2 (Dettaglio)

## MODULE 1: Modules, Packages, PIP

### Section 1.1 – Modules
**Teoria:**
- import, from...import, as
- __name__ == "__main__"
- dir()

**Quiz 1.1:**
```python
# file: mymod.py
print("Module loaded")

def greet():
    print("Hello")

# file: main.py
import mymod
import mymod
```
Quante volte stampa "Module loaded"? ___

### Section 1.2 – Standard Library
**Teoria:** math, random, platform, sys

### Section 1.3 – Packages
**Teoria:**
- __init__.py
- Import relativi

### Section 1.4 – PIP
**Teoria:** pip install, pip list, pip show

---

## MODULE 2: Strings, Exceptions avanzate

### Section 2.1 – String methods
**Teoria:** Tutti i metodi stringa (upper, lower, split, join, find, replace...)

**Quiz 2.1:**
```python
s = "a,b,,c"
print(s.split(","))

s2 = "a b  c"
print(s2.split())
print(s2.split(" "))
```
Output: ___

### Section 2.2 – String comparisons
**Teoria:** Ordinamento lessicografico, encoding

### Section 2.3 – Exceptions as objects
**Teoria:**
- Exception hierarchy
- Custom exceptions
- except as

**Quiz 2.3:**
```python
try:
    raise ValueError("test")
except Exception as e:
    print(type(e).__name__)
```
Output: ___

---

## MODULE 3: Object-Oriented Programming

### Section 3.1 – OOP Introduction
**Teoria:**
- Procedurale vs OOP
- Classi e oggetti

### Section 3.2 – Classes and Objects
**Teoria:**
- class, __init__, self
- Instance vs class attributes

**Quiz 3.2:**
```python
class Counter:
    count = 0
    
    def __init__(self):
        Counter.count += 1

a = Counter()
b = Counter()
c = Counter()
print(Counter.count)
print(a.count)
```
Output: ___

### Section 3.3 – Methods
**Teoria:**
- Instance methods
- @classmethod
- @staticmethod

### Section 3.4 – Inheritance
**Teoria:**
- Single inheritance
- super()
- MRO (Method Resolution Order)
- isinstance(), issubclass()

**Quiz 3.4:**
```python
class A:
    def __init__(self):
        print("A", end="")

class B(A):
    def __init__(self):
        print("B", end="")
        super().__init__()

b = B()
```
Output: ___

### Section 3.5 – Encapsulation
**Teoria:**
- Public, _protected, __private
- Name mangling

**Quiz 3.5:**
```python
class MyClass:
    def __init__(self):
        self.__secret = 42

obj = MyClass()
print(obj._MyClass__secret)
```
Output: ___

### Section 3.6 – Polymorphism
**Teoria:**
- Duck typing
- Method overriding

---

## MODULE 4: Generators, Files, Miscellaneous

### Section 4.1 – Generators
**Teoria:**
- yield
- Generator expressions
- Lazy evaluation

**Quiz 4.1:**
```python
g = (x**2 for x in range(3))
print(list(g))
print(list(g))
```
Output: ___

### Section 4.2 – Iterators
**Teoria:**
- __iter__, __next__
- StopIteration

### Section 4.3 – Closures
**Teoria:**
- Nested functions
- Captured variables

**Quiz 4.3:**
```python
def outer(x):
    def inner(y):
        return x + y
    return inner

f = outer(10)
print(f(5))
```
Output: ___

### Section 4.4 – File Processing
**Teoria:**
- open(), close(), with
- read(), readline(), readlines()
- write()
- Modes (r, w, a, b)

### Section 4.5 – os module
**Teoria:**
- os.path
- Directory operations

### Section 4.6 – datetime, time, calendar
**Teoria:**
- datetime objects
- Formatting
- Timedelta

---

# 🎯 COME USARE QUESTO DOCUMENTO

## Metodo di Studio (per ogni sezione)

1. **Leggi la teoria** (15-20 min)
2. **Studia gli esempi** - NON copiarli, ANALIZZALI
3. **Rispondi ai quiz** - SCRIVI la risposta PRIMA di verificare
4. **Se sbagli**: FERMATI e capisci perché
5. **Fai i lab** - Scrivi codice tu
6. **Test di modulo** - Verifica comprensione

## Regola d'Oro

> **"Se non sai prevedere l'output senza eseguire, non hai capito Python"**

---

# 📅 TIMELINE INTEGRATA

| Settimana | Focus | Certificazione | Extra Trading/AI |
|-----------|-------|----------------|------------------|
| 1-2 | PE1 Module 1-2 | PCEP prep | - |
| 3-4 | PE1 Module 3 | PCEP prep | - |
| 5-6 | PE1 Module 4 | PCEP prep | NumPy intro |
| 7 | **Review + PCEP** | 🎓 | - |
| 8-9 | PE2 Module 1-2 | PCAP prep | - |
| 10-12 | PE2 Module 3 (OOP) | PCAP prep | - |
| 13-14 | PE2 Module 4 | PCAP prep | Pandas intro |
| 15 | **Review + PCAP** | 🎓 | ccxt basics |
| 16-18 | PA Modules 1-2 | PCPP1 prep | - |
| 19-20 | PA Module 4 (Network) | PCPP1 prep | API exchange |
| 21-22 | PA Module 5 (Files) | PCPP1 prep | Async/WebSocket |
| 23-24 | **Review + PCPP1** | 🎓 | Backtesting |
| 25-28 | PP Modules 1-3 | PCPP2 prep | ML basics |
| 29-30 | PP Modules 4-5 | PCPP2 prep | TimescaleDB |
| 31-32 | **Review + PCPP2** | 🎓 | Trading Bot |
| 33-36 | **PROGETTO FINALE** | - | 🤖 Bot + AI |

**Totale: ~9 mesi** con 4 certificazioni + Trading Bot + AI basics

---

# ✅ CHECKLIST RISORSE

## Gratuite (Python Institute)
- [ ] edube.org - Python Essentials 1 (corso completo)
- [ ] edube.org - Python Essentials 2 (corso completo)
- [ ] pythoninstitute.org - Syllabus ufficiali
- [ ] Practice tests su edube

## I Tuoi File (già creati)
- [ ] python_logic_training.py - 80+ quiz logici
- [ ] esercizi_pcep_100.py - 100 esercizi PCEP
- [ ] module_database_part1_sqlite.py
- [ ] module_database_part2_postgresql_orm.py
- [ ] esercizi_database_20.py
- [ ] Tutti i file session1-4 esistenti

## Da Aggiungere
- [ ] Modulo NumPy/Pandas basics
- [ ] Modulo ccxt (API exchange)
- [ ] Modulo Async/WebSocket
- [ ] Modulo Backtesting

---

# 🔑 DIFFERENZA CHIAVE

| Approccio Normale | Approccio Python Institute |
|-------------------|---------------------------|
| "Scrivi un for loop" | "Cosa stampa questo for?" |
| Copia-incolla tutorial | Analizza e prevedi output |
| Tanti progetti subito | Prima CAPISCI, poi costruisci |
| "Funziona!" | "Capisco PERCHÉ funziona" |

**Questo approccio è più lento all'inizio, ma crea fondamenta SOLIDE.**

Quando arrivi a costruire il trading bot, non avrai dubbi su come funziona Python internamente. Saprai esattamente cosa fa ogni riga di codice.
