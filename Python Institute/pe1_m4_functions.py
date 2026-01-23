"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    PYTHON ESSENTIALS 1 - MODULE 4                            ║
║           Functions, Tuples, Dictionaries, Exceptions, Data Processing       ║
║                                                                              ║
║                     Allineato al Syllabus PCEP-30-02                         ║
║                         OpenEDG Python Institute                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

PCEP-30-02 Exam Blocks coperti:
- Block 3: Data Collections – Tuples, Dictionaries, Lists, Strings (25% restante)
- Block 4: Functions and Exceptions (28%)

STRUTTURA MODULO:
├── Section 4.1: Functions - Basics
├── Section 4.2: Functions - Parameters and Arguments
├── Section 4.3: Functions - Return and Scope
├── Section 4.4: Functions - Recursion
├── Section 4.5: Tuples
├── Section 4.6: Dictionaries
├── Section 4.7: Exceptions (try/except)
├── Labs (30 esercizi)
└── Module 4 Quiz (40 domande)

TEMPO STIMATO: 10-12 ore

═══════════════════════════════════════════════════════════════════════════════
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 4.1: FUNCTIONS - BASICS
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    4.1 TEORIA: FUNZIONI - BASE                               │
└──────────────────────────────────────────────────────────────────────────────┘

Una FUNZIONE è un blocco di codice riutilizzabile.

DEFINIZIONE E CHIAMATA:
───────────────────────
"""

def greet():
    """Questa è una docstring - documenta la funzione"""
    print("Hello!")

# Chiamata
greet()  # Output: Hello!


"""
FUNZIONI BUILT-IN vs USER-DEFINED:
──────────────────────────────────

Built-in (già disponibili):
    print(), len(), range(), type(), int(), str(), list(), etc.

User-defined (create da te):
    def my_function(): ...
"""


"""
ORDINE DI DEFINIZIONE:
──────────────────────
La funzione DEVE essere definita PRIMA di essere chiamata!
"""

# CORRETTO:
def say_hello():
    print("Hello")

say_hello()  # OK

# ERRORE:
# say_hi()  # NameError: name 'say_hi' is not defined
# def say_hi():
#     print("Hi")


"""
FUNZIONI CHE CHIAMANO ALTRE FUNZIONI:
─────────────────────────────────────
"""

def step1():
    print("Step 1")

def step2():
    print("Step 2")

def process():
    step1()
    step2()
    print("Done")

process()
# Output:
# Step 1
# Step 2
# Done


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 4.2: PARAMETERS AND ARGUMENTS
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    4.2 TEORIA: PARAMETRI E ARGOMENTI                         │
└──────────────────────────────────────────────────────────────────────────────┘

PARAMETRO: variabile nella DEFINIZIONE della funzione
ARGOMENTO: valore passato nella CHIAMATA della funzione
"""

def greet(name):       # 'name' è un PARAMETRO
    print(f"Hello, {name}!")

greet("Marco")         # "Marco" è un ARGOMENTO


"""
ARGOMENTI POSIZIONALI vs KEYWORD:
─────────────────────────────────
"""

def describe(name, age, city):
    print(f"{name}, {age} years, from {city}")

# Posizionali (ordine conta!)
describe("Marco", 30, "Milan")

# Keyword (ordine non conta!)
describe(city="Milan", age=30, name="Marco")

# Misti (posizionali PRIMA di keyword!)
describe("Marco", city="Milan", age=30)

# ERRORE: posizionale dopo keyword
# describe(name="Marco", 30, "Milan")  # SyntaxError


"""
VALORI DI DEFAULT:
──────────────────
"""

def greet(name, greeting="Hello"):
    print(f"{greeting}, {name}!")

greet("Marco")              # Hello, Marco!
greet("Marco", "Ciao")      # Ciao, Marco!

# REGOLA: parametri con default DOPO quelli senza default!
# def bad(x=1, y):  # SyntaxError!


"""
⚠️ TRAPPOLA ESAME: DEFAULT MUTABILE!
────────────────────────────────────
MAI usare liste/dict come default!
"""

def bad_append(item, lst=[]):  # SBAGLIATO!
    lst.append(item)
    return lst

print(bad_append(1))  # [1]
print(bad_append(2))  # [1, 2] - NON [2]! La lista è condivisa!
print(bad_append(3))  # [1, 2, 3]

# CORRETTO: usa None
def good_append(item, lst=None):
    if lst is None:
        lst = []
    lst.append(item)
    return lst


"""
*args - ARGOMENTI POSIZIONALI VARIABILI:
────────────────────────────────────────
"""

def sum_all(*args):
    print(type(args))  # <class 'tuple'>
    return sum(args)

print(sum_all(1, 2, 3))      # 6
print(sum_all(1, 2, 3, 4, 5)) # 15
print(sum_all())              # 0


"""
**kwargs - ARGOMENTI KEYWORD VARIABILI:
───────────────────────────────────────
"""

def print_info(**kwargs):
    print(type(kwargs))  # <class 'dict'>
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Marco", age=30, city="Milan")
# name: Marco
# age: 30
# city: Milan


"""
ORDINE DEI PARAMETRI:
─────────────────────
1. Posizionali normali
2. *args
3. Keyword con default
4. **kwargs

def func(a, b, *args, c=10, **kwargs):
    pass
"""


# ══════════════════════════════════════════════════════════════════════════════
#                     SECTION 4.2 - QUIZ
# ══════════════════════════════════════════════════════════════════════════════

QUIZ_4_2 = """
══════════════════════════════════════════════════════════════════════════════
QUIZ 4.2.1
══════════════════════════════════════════════════════════════════════════════
def func(a, b, c):
    print(a, b, c)

func(1, c=3, b=2)

Stampa:
A) 1 2 3
B) 1 3 2
C) Error
D) 3 2 1

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.2.2
══════════════════════════════════════════════════════════════════════════════
def func(a, b=10):
    print(a + b)

func(5)

Stampa:
A) 5
B) 15
C) Error
D) 10

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.2.3 - TRAPPOLA ESAME!
══════════════════════════════════════════════════════════════════════════════
def add(item, lst=[]):
    lst.append(item)
    return lst

print(add(1))
print(add(2))

Stampa:
A) [1]
   [2]
B) [1]
   [1, 2]
C) [1, 2]
   [1, 2]
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.2.4
══════════════════════════════════════════════════════════════════════════════
def func(*args):
    print(type(args))

func(1, 2, 3)

Stampa:
A) <class 'list'>
B) <class 'tuple'>
C) <class 'dict'>
D) <class 'args'>

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.2.5
══════════════════════════════════════════════════════════════════════════════
def func(**kwargs):
    print(type(kwargs))

func(a=1, b=2)

Stampa:
A) <class 'list'>
B) <class 'tuple'>
C) <class 'dict'>
D) <class 'kwargs'>

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.2.6
══════════════════════════════════════════════════════════════════════════════
def func(*args):
    return sum(args)

print(func())

Stampa:
A) 0
B) None
C) Error
D) ()

Tua risposta: ___
"""


RISPOSTE_4_2 = """
══════════════════════════════════════════════════════════════════════════════
RISPOSTE QUIZ 4.2
══════════════════════════════════════════════════════════════════════════════

4.2.1: A) 1 2 3
       a=1 (posizionale), b=2 (keyword), c=3 (keyword)
       L'ordine dei keyword non conta!

4.2.2: B) 15
       b ha default 10, quindi 5 + 10 = 15

4.2.3: B) [1]
          [1, 2]
       La lista default è CONDIVISA tra le chiamate!
       Questa è LA TRAPPOLA più comune nell'esame PCEP.

4.2.4: B) <class 'tuple'>
       *args raccoglie gli argomenti in una TUPLA

4.2.5: C) <class 'dict'>
       **kwargs raccoglie gli argomenti keyword in un DIZIONARIO

4.2.6: A) 0
       sum(()) = 0, tupla vuota ha somma 0
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 4.3: RETURN AND SCOPE
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    4.3 TEORIA: RETURN E SCOPE                                │
└──────────────────────────────────────────────────────────────────────────────┘

RETURN:
───────
"""

def add(a, b):
    return a + b

result = add(3, 5)
print(result)  # 8

# Senza return, la funzione restituisce None
def no_return():
    x = 5

print(no_return())  # None

# return multipli
def absolute(n):
    if n >= 0:
        return n
    else:
        return -n

# return termina la funzione IMMEDIATAMENTE
def test():
    return 1
    print("Mai eseguito")  # Codice irraggiungibile


"""
RETURN MULTIPLI (tuple unpacking):
──────────────────────────────────
"""

def min_max(lst):
    return min(lst), max(lst)  # Restituisce una tupla

minimum, maximum = min_max([3, 1, 4, 1, 5])
print(minimum, maximum)  # 1 5


"""
SCOPE - LEGB RULE:
──────────────────
L - Local (dentro la funzione)
E - Enclosing (funzione esterna, per funzioni annidate)
G - Global (livello modulo)
B - Built-in (funzioni predefinite Python)
"""

x = "global"  # Global

def outer():
    x = "enclosing"  # Enclosing
    
    def inner():
        x = "local"  # Local
        print(x)  # local
    
    inner()
    print(x)  # enclosing

outer()
print(x)  # global


"""
⚠️ TRAPPOLA ESAME: SHADOWING E UnboundLocalError
─────────────────────────────────────────────────
"""

x = 10

def func():
    print(x)  # Legge la x globale
    
func()  # 10

# MA...

x = 10

def func_error():
    print(x)  # UnboundLocalError!
    x = 20    # Questa riga rende x LOCALE per TUTTA la funzione

# func_error()  # UnboundLocalError: local variable 'x' referenced before assignment


"""
GLOBAL keyword:
───────────────
Permette di MODIFICARE una variabile globale da dentro una funzione
"""

counter = 0

def increment():
    global counter  # Dichiara che usiamo la variabile globale
    counter += 1

increment()
increment()
print(counter)  # 2


"""
NONLOCAL keyword:
─────────────────
Per modificare variabili della funzione ENCLOSING (non globale)
"""

def outer():
    x = 10
    
    def inner():
        nonlocal x  # Modifica la x di outer
        x = 20
    
    inner()
    print(x)  # 20

outer()


# ══════════════════════════════════════════════════════════════════════════════
#                     SECTION 4.3 - QUIZ
# ══════════════════════════════════════════════════════════════════════════════

QUIZ_4_3 = """
══════════════════════════════════════════════════════════════════════════════
QUIZ 4.3.1
══════════════════════════════════════════════════════════════════════════════
def func():
    pass

print(func())

Stampa:
A) pass
B) None
C) Error
D) (niente)

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.3.2
══════════════════════════════════════════════════════════════════════════════
def func():
    return 1, 2, 3

x = func()
print(type(x))

Stampa:
A) <class 'int'>
B) <class 'list'>
C) <class 'tuple'>
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.3.3 - CRITICO!
══════════════════════════════════════════════════════════════════════════════
x = 10

def func():
    x = 20
    print(x)

func()
print(x)

Stampa:
A) 20
   20
B) 20
   10
C) 10
   10
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.3.4 - CRITICO!
══════════════════════════════════════════════════════════════════════════════
x = 10

def func():
    global x
    x = 20

func()
print(x)

Stampa:
A) 10
B) 20
C) Error
D) None

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.3.5 - TRAPPOLA ESAME!
══════════════════════════════════════════════════════════════════════════════
x = 10

def func():
    print(x)
    x = 20

func()

Cosa succede?
A) Stampa 10
B) Stampa 20
C) UnboundLocalError
D) NameError

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.3.6
══════════════════════════════════════════════════════════════════════════════
def outer():
    x = 1
    def inner():
        nonlocal x
        x = 2
    inner()
    return x

print(outer())

Stampa:
A) 1
B) 2
C) Error
D) None

Tua risposta: ___
"""


RISPOSTE_4_3 = """
══════════════════════════════════════════════════════════════════════════════
RISPOSTE QUIZ 4.3
══════════════════════════════════════════════════════════════════════════════

4.3.1: B) None
       Funzione senza return (o con solo pass) restituisce None

4.3.2: C) <class 'tuple'>
       return 1, 2, 3 restituisce una tupla (1, 2, 3)

4.3.3: B) 20
          10
       x = 20 dentro func crea una variabile LOCALE.
       La x globale resta 10.

4.3.4: B) 20
       global x permette di modificare la variabile globale.

4.3.5: C) UnboundLocalError
       L'assegnazione x = 20 rende x locale per TUTTA la funzione,
       ma il print(x) avviene PRIMA dell'assegnazione!

4.3.6: B) 2
       nonlocal x modifica la x della funzione outer.
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 4.5: TUPLES
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    4.5 TEORIA: TUPLE                                         │
└──────────────────────────────────────────────────────────────────────────────┘

Le TUPLE sono come le liste, ma IMMUTABILI.
"""

# Creazione
t1 = (1, 2, 3)
t2 = 1, 2, 3       # Parentesi opzionali!
t3 = tuple([1, 2, 3])  # Da lista
empty = ()
single = (1,)      # VIRGOLA necessaria per tupla singola!

# ATTENZIONE:
not_a_tuple = (1)  # Questo è solo il numero 1!
print(type(not_a_tuple))  # <class 'int'>
print(type(single))       # <class 'tuple'>


"""
OPERAZIONI (come le liste, ma senza modifica):
──────────────────────────────────────────────
"""
t = (1, 2, 3, 4, 5)

# Indexing
print(t[0])    # 1
print(t[-1])   # 5

# Slicing
print(t[1:4])  # (2, 3, 4)

# Lunghezza
print(len(t))  # 5

# in
print(3 in t)  # True

# Concatenazione
print((1, 2) + (3, 4))  # (1, 2, 3, 4)

# Ripetizione
print((1, 2) * 3)  # (1, 2, 1, 2, 1, 2)

# ERRORE: le tuple sono IMMUTABILI
# t[0] = 10  # TypeError!


"""
TUPLE UNPACKING:
────────────────
"""
t = (1, 2, 3)
a, b, c = t
print(a, b, c)  # 1 2 3

# Con *
first, *rest = (1, 2, 3, 4)
print(first)  # 1
print(rest)   # [2, 3, 4] - è una LISTA!

*start, last = (1, 2, 3, 4)
print(start)  # [1, 2, 3]
print(last)   # 4


"""
TUPLE CON ELEMENTI MUTABILI - TRAPPOLA!
───────────────────────────────────────
"""
t = ([1, 2], [3, 4])
# t[0] = [5, 6]  # ERRORE: non puoi riassegnare
t[0].append(3)   # OK: modifichi la lista DENTRO la tupla
print(t)  # ([1, 2, 3], [3, 4])


# ══════════════════════════════════════════════════════════════════════════════
#                     SECTION 4.5 - QUIZ
# ══════════════════════════════════════════════════════════════════════════════

QUIZ_4_5 = """
══════════════════════════════════════════════════════════════════════════════
QUIZ 4.5.1 - CRITICO!
══════════════════════════════════════════════════════════════════════════════
print(type((1)))

Stampa:
A) <class 'tuple'>
B) <class 'int'>
C) <class 'list'>
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.5.2 - CRITICO!
══════════════════════════════════════════════════════════════════════════════
print(type((1,)))

Stampa:
A) <class 'tuple'>
B) <class 'int'>
C) <class 'list'>
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.5.3
══════════════════════════════════════════════════════════════════════════════
t = (1, 2, 3)
t[0] = 10
print(t)

Cosa succede?
A) (10, 2, 3)
B) TypeError
C) (1, 2, 3)
D) None

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.5.4
══════════════════════════════════════════════════════════════════════════════
first, *rest = (1, 2, 3, 4)
print(type(rest))

Stampa:
A) <class 'tuple'>
B) <class 'list'>
C) <class 'int'>
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.5.5 - TRAPPOLA!
══════════════════════════════════════════════════════════════════════════════
t = ([1, 2], [3, 4])
t[0].append(5)
print(t)

Stampa:
A) TypeError
B) ([1, 2, 5], [3, 4])
C) ([1, 2], [3, 4])
D) ([1, 2], [3, 4, 5])

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.5.6
══════════════════════════════════════════════════════════════════════════════
print((1, 2) + (3,))

Stampa:
A) (1, 2, 3)
B) (4, 2)
C) Error
D) (1, 2, (3,))

Tua risposta: ___
"""


RISPOSTE_4_5 = """
══════════════════════════════════════════════════════════════════════════════
RISPOSTE QUIZ 4.5
══════════════════════════════════════════════════════════════════════════════

4.5.1: B) <class 'int'>
       (1) è solo il numero 1 tra parentesi, NON una tupla!

4.5.2: A) <class 'tuple'>
       (1,) con la virgola È una tupla!

4.5.3: B) TypeError
       Le tuple sono IMMUTABILI, non puoi modificare elementi.

4.5.4: B) <class 'list'>
       *rest nell'unpacking crea sempre una LISTA, non tupla!

4.5.5: B) ([1, 2, 5], [3, 4])
       La tupla è immutabile, ma gli oggetti al suo interno no!
       Puoi modificare la lista, non puoi riassegnare t[0].

4.5.6: A) (1, 2, 3)
       Concatenazione di tuple.
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 4.6: DICTIONARIES
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    4.6 TEORIA: DIZIONARI                                     │
└──────────────────────────────────────────────────────────────────────────────┘

I DIZIONARI sono collezioni di coppie chiave-valore.
"""

# Creazione
d1 = {"name": "Marco", "age": 30}
d2 = dict(name="Marco", age=30)
empty = {}

# Accesso
print(d1["name"])     # Marco
# print(d1["city"])   # KeyError!

# get() - sicuro, restituisce default se chiave non esiste
print(d1.get("name"))        # Marco
print(d1.get("city"))        # None
print(d1.get("city", "N/A")) # N/A (default personalizzato)


"""
MODIFICA:
─────────
"""
d = {"a": 1, "b": 2}

# Aggiungere/modificare
d["c"] = 3
d["a"] = 10
print(d)  # {'a': 10, 'b': 2, 'c': 3}

# Rimuovere
del d["b"]
print(d)  # {'a': 10, 'c': 3}

# pop() - rimuove e restituisce il valore
value = d.pop("a")
print(value)  # 10
print(d)      # {'c': 3}


"""
CHIAVI VALIDE:
──────────────
Le chiavi devono essere HASHABLE (immutabili):
✅ str, int, float, tuple (di immutabili), bool
❌ list, dict, set
"""

valid = {
    "string": 1,
    42: 2,
    (1, 2): 3,
    True: 4
}

# ERRORE:
# invalid = {[1, 2]: 1}  # TypeError: unhashable type: 'list'


"""
ITERAZIONE:
───────────
"""
d = {"a": 1, "b": 2, "c": 3}

# Chiavi (default)
for key in d:
    print(key)  # a, b, c

# Chiavi (esplicito)
for key in d.keys():
    print(key)

# Valori
for value in d.values():
    print(value)  # 1, 2, 3

# Coppie
for key, value in d.items():
    print(f"{key}: {value}")


"""
in - verifica CHIAVI, non valori:
─────────────────────────────────
"""
d = {"a": 1, "b": 2}
print("a" in d)  # True (chiave)
print(1 in d)    # False (1 è un valore, non una chiave!)


"""
METODI UTILI:
─────────────
"""
d = {"a": 1, "b": 2}

print(d.keys())    # dict_keys(['a', 'b'])
print(d.values())  # dict_values([1, 2])
print(d.items())   # dict_items([('a', 1), ('b', 2)])

# update() - unisce dizionari
d.update({"c": 3, "a": 10})
print(d)  # {'a': 10, 'b': 2, 'c': 3}

# | (Python 3.9+) - merge
d1 = {"a": 1}
d2 = {"b": 2}
d3 = d1 | d2  # {'a': 1, 'b': 2}


"""
CHIAVI DUPLICATE - L'ULTIMA VINCE:
──────────────────────────────────
"""
d = {"a": 1, "a": 2, "a": 3}
print(d)  # {'a': 3}


# ══════════════════════════════════════════════════════════════════════════════
#                     SECTION 4.6 - QUIZ
# ══════════════════════════════════════════════════════════════════════════════

QUIZ_4_6 = """
══════════════════════════════════════════════════════════════════════════════
QUIZ 4.6.1
══════════════════════════════════════════════════════════════════════════════
d = {"a": 1, "b": 2}
print(d["c"])

Cosa succede?
A) None
B) KeyError
C) 0
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.6.2
══════════════════════════════════════════════════════════════════════════════
d = {"a": 1, "b": 2}
print(d.get("c"))

Stampa:
A) KeyError
B) None
C) "c"
D) 0

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.6.3
══════════════════════════════════════════════════════════════════════════════
d = {"a": 1, "b": 2}
print("b" in d)

Stampa:
A) True
B) False
C) 2
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.6.4 - CRITICO!
══════════════════════════════════════════════════════════════════════════════
d = {"a": 1, "b": 2}
print(2 in d)

Stampa:
A) True
B) False
C) Error
D) "b"

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.6.5 - CRITICO!
══════════════════════════════════════════════════════════════════════════════
d = {"a": 1, "a": 2, "a": 3}
print(d["a"])

Stampa:
A) 1
B) 2
C) 3
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.6.6
══════════════════════════════════════════════════════════════════════════════
d = {[1, 2]: "value"}

Cosa succede?
A) Crea il dizionario
B) TypeError
C) KeyError
D) None

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.6.7
══════════════════════════════════════════════════════════════════════════════
d = {"x": 1}
for k in d:
    print(k)

Stampa:
A) 1
B) x
C) ('x', 1)
D) {'x': 1}

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.6.8
══════════════════════════════════════════════════════════════════════════════
d = {"a": 1}
d["b"] = 2
d["a"] = 10
print(len(d))

Stampa:
A) 1
B) 2
C) 3
D) 4

Tua risposta: ___
"""


RISPOSTE_4_6 = """
══════════════════════════════════════════════════════════════════════════════
RISPOSTE QUIZ 4.6
══════════════════════════════════════════════════════════════════════════════

4.6.1: B) KeyError
       d["c"] solleva KeyError se la chiave non esiste.

4.6.2: B) None
       get() restituisce None se la chiave non esiste (default).

4.6.3: A) True
       "b" È una chiave del dizionario.

4.6.4: B) False
       in verifica le CHIAVI, non i valori!
       2 è un valore, non una chiave.

4.6.5: C) 3
       Con chiavi duplicate, l'ultimo valore vince.

4.6.6: B) TypeError
       Le liste non sono hashable, non possono essere chiavi!

4.6.7: B) x
       Iterare su un dict itera sulle CHIAVI.

4.6.8: B) 2
       d["a"] = 10 MODIFICA, non aggiunge. Solo "b" è nuovo.
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 4.7: EXCEPTIONS
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    4.7 TEORIA: ECCEZIONI                                     │
└──────────────────────────────────────────────────────────────────────────────┘

ECCEZIONI COMUNI:
─────────────────
"""
# ValueError: valore inappropriato
# int("hello")

# TypeError: tipo inappropriato
# "2" + 2

# IndexError: indice fuori range
# [1,2,3][10]

# KeyError: chiave non trovata
# {}["a"]

# ZeroDivisionError
# 1 / 0

# NameError: variabile non definita
# print(undefined_var)

# AttributeError: attributo/metodo non esiste
# "hello".appendd()


"""
TRY/EXCEPT:
───────────
"""
try:
    x = int("hello")
except:
    print("Errore!")


# Catturare eccezione specifica
try:
    x = int("hello")
except ValueError:
    print("Valore non valido!")


# Eccezioni multiple
try:
    x = int(input())
    y = 10 / x
except ValueError:
    print("Non è un numero!")
except ZeroDivisionError:
    print("Non dividere per zero!")


# Catturare multiple in una riga
try:
    # ...
    pass
except (ValueError, TypeError):
    print("Errore di valore o tipo!")


# Accedere all'oggetto eccezione
try:
    x = 1 / 0
except ZeroDivisionError as e:
    print(f"Errore: {e}")  # Errore: division by zero


"""
TRY/EXCEPT/ELSE/FINALLY:
────────────────────────
"""
try:
    x = int("10")
except ValueError:
    print("Errore!")
else:
    print("Nessun errore!")  # Eseguito se NESSUNA eccezione
finally:
    print("Sempre eseguito!")  # SEMPRE eseguito

# Output:
# Nessun errore!
# Sempre eseguito!


"""
⚠️ FINALLY viene eseguito SEMPRE, anche con return!
────────────────────────────────────────────────────
"""

def test():
    try:
        return 1
    finally:
        print("Finally!")  # Eseguito prima del return!

result = test()  # Stampa "Finally!", poi result = 1


"""
RAISE - Sollevare eccezioni:
────────────────────────────
"""

def divide(a, b):
    if b == 0:
        raise ValueError("Divisore non può essere zero!")
    return a / b

# raise senza argomenti ri-solleva l'ultima eccezione
try:
    x = 1 / 0
except:
    print("Gestito parzialmente")
    # raise  # Ri-solleva l'eccezione


"""
GERARCHIA ECCEZIONI (parziale):
───────────────────────────────
BaseException
├── SystemExit
├── KeyboardInterrupt
└── Exception
    ├── ValueError
    ├── TypeError
    ├── IndexError
    ├── KeyError
    ├── ZeroDivisionError
    ├── FileNotFoundError
    └── ...

except Exception cattura TUTTE le eccezioni normali (non SystemExit, etc.)
"""


# ══════════════════════════════════════════════════════════════════════════════
#                     SECTION 4.7 - QUIZ
# ══════════════════════════════════════════════════════════════════════════════

QUIZ_4_7 = """
══════════════════════════════════════════════════════════════════════════════
QUIZ 4.7.1
══════════════════════════════════════════════════════════════════════════════
try:
    x = int("hello")
except ValueError:
    print("A")
except:
    print("B")

Stampa:
A) A
B) B
C) AB
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.7.2 - CRITICO!
══════════════════════════════════════════════════════════════════════════════
try:
    x = int("10")
except:
    print("A")
else:
    print("B")
finally:
    print("C")

Stampa:
A) A C
B) B C
C) C
D) A B C

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.7.3 - CRITICO!
══════════════════════════════════════════════════════════════════════════════
try:
    x = int("hello")
except:
    print("A")
else:
    print("B")
finally:
    print("C")

Stampa:
A) A C
B) B C
C) A B C
D) C

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.7.4
══════════════════════════════════════════════════════════════════════════════
def func():
    try:
        return 1
    finally:
        return 2

print(func())

Stampa:
A) 1
B) 2
C) 1 2
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.7.5
══════════════════════════════════════════════════════════════════════════════
print([1, 2, 3][10])

Quale eccezione?
A) ValueError
B) TypeError
C) IndexError
D) KeyError

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.7.6
══════════════════════════════════════════════════════════════════════════════
print({"a": 1}["b"])

Quale eccezione?
A) ValueError
B) TypeError
C) IndexError
D) KeyError

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 4.7.7
══════════════════════════════════════════════════════════════════════════════
print(int("3.14"))

Quale eccezione?
A) ValueError
B) TypeError
C) FloatError
D) Nessuna, stampa 3

Tua risposta: ___
"""


RISPOSTE_4_7 = """
══════════════════════════════════════════════════════════════════════════════
RISPOSTE QUIZ 4.7
══════════════════════════════════════════════════════════════════════════════

4.7.1: A) A
       int("hello") solleva ValueError, gestito dal primo except.

4.7.2: B) B C
       Nessuna eccezione → else eseguito, finally SEMPRE eseguito.

4.7.3: A) A C
       Eccezione → except eseguito, else NON eseguito, finally SEMPRE.

4.7.4: B) 2
       return nel finally SOVRASCRIVE il return nel try!

4.7.5: C) IndexError
       Indice fuori range in una lista.

4.7.6: D) KeyError
       Chiave non trovata in un dizionario.

4.7.7: A) ValueError
       int() non può convertire una stringa con punto decimale.
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    MODULE 4 - TEST FINALE
# ══════════════════════════════════════════════════════════════════════════════

MODULE_4_TEST = """
══════════════════════════════════════════════════════════════════════════════
                         MODULE 4 - TEST FINALE
                    40 domande - Target: 70% (28/40)
                     Tempo consigliato: 45 minuti
══════════════════════════════════════════════════════════════════════════════

Q1. def func(): pass; print(func()) = ?
    A) pass    B) None    C) Error    D) ()

Q2. def f(a,b=10): return a+b; print(f(5)) = ?
    A) 5    B) 10    C) 15    D) Error

Q3. def f(*args): return type(args); print(f()) = ?
    A) list    B) tuple    C) dict    D) Error

Q4. def f(**kw): return type(kw); print(f()) = ?
    A) list    B) tuple    C) dict    D) Error

Q5. def add(x,lst=[]): lst.append(x); return lst
    add(1); print(add(2)) = ?
    A) [2]    B) [1,2]    C) [1]    D) Error

Q6. x=10; def f(): x=20; f(); print(x) = ?
    A) 10    B) 20    C) Error    D) None

Q7. x=10; def f(): global x; x=20; f(); print(x) = ?
    A) 10    B) 20    C) Error    D) None

Q8. x=5; def f(): print(x); x=10; f() → ?
    A) 5    B) 10    C) UnboundLocalError    D) None

Q9. print(type((1))) = ?
    A) int    B) tuple    C) list    D) Error

Q10. print(type((1,))) = ?
     A) int    B) tuple    C) list    D) Error

Q11. t=(1,2,3); t[0]=10 → ?
     A) (10,2,3)    B) TypeError    C) (1,2,3)    D) None

Q12. t=([1,2],); t[0].append(3); print(t) = ?
     A) TypeError    B) ([1,2,3],)    C) ([1,2],)    D) Error

Q13. a,*b=(1,2,3,4); print(type(b)) = ?
     A) tuple    B) list    C) int    D) Error

Q14. d={"a":1}; print(d["b"]) → ?
     A) None    B) KeyError    C) 0    D) Error

Q15. d={"a":1}; print(d.get("b")) = ?
     A) KeyError    B) None    C) "b"    D) Error

Q16. d={"a":1}; print("a" in d) = ?
     A) True    B) False    C) 1    D) Error

Q17. d={"a":1}; print(1 in d) = ?
     A) True    B) False    C) "a"    D) Error

Q18. d={"a":1,"a":2}; print(d["a"]) = ?
     A) 1    B) 2    C) [1,2]    D) Error

Q19. d={[1]:1} → ?
     A) Crea dict    B) TypeError    C) KeyError    D) None

Q20. for k in {"a":1}: print(k) = ?
     A) 1    B) a    C) ('a',1)    D) Error

Q21. try: int("x"); except: print("A"); else: print("B") = ?
     A) A    B) B    C) AB    D) Error

Q22. try: int("5"); except: print("A"); else: print("B") = ?
     A) A    B) B    C) AB    D) Error

Q23. try: x=1; finally: print("F"); print(x) = ?
     A) F 1    B) 1 F    C) F    D) Error

Q24. def f(): try: return 1; finally: return 2; print(f()) = ?
     A) 1    B) 2    C) 12    D) Error

Q25. [1,2][10] solleva?
     A) ValueError    B) IndexError    C) KeyError    D) TypeError

Q26. {}["x"] solleva?
     A) ValueError    B) IndexError    C) KeyError    D) TypeError

Q27. int("3.14") solleva?
     A) ValueError    B) TypeError    C) FloatError    D) Nessuna

Q28. def f(a,b,c): return a+b+c; f(1,c=3,b=2) = ?
     A) 6    B) Error    C) 123    D) None

Q29. def f(): return 1,2; x=f(); type(x) = ?
     A) int    B) list    C) tuple    D) Error

Q30. def outer(): x=1; def inner(): nonlocal x; x=2; inner(); return x
     outer() = ?
     A) 1    B) 2    C) Error    D) None

Q31. t=(1,2)+(3,); print(t) = ?
     A) (1,2,3)    B) (4,2)    C) Error    D) (1,2,(3,))

Q32. print(len({"a":1,"b":2,"a":3})) = ?
     A) 2    B) 3    C) 4    D) Error

Q33. d={"a":1}; d.update({"b":2}); len(d) = ?
     A) 1    B) 2    C) 3    D) Error

Q34. first,*rest=(1,); print(rest) = ?
     A) ()    B) []    C) None    D) Error

Q35. try: 1/0; except ZeroDivisionError: print("A"); except: print("B") = ?
     A) A    B) B    C) AB    D) Error

Q36. bool(()) and bool({}) = ?
     A) True    B) False    C) ()    D) {}

Q37. d={}; d[1]=1; d["1"]=2; len(d) = ?
     A) 1    B) 2    C) Error    D) 3

Q38. (1,2,3)[1:] = ?
     A) (1,2)    B) (2,3)    C) [2,3]    D) Error

Q39. def f(x=[]): x.append(1); return len(x)
     f(),f(),f() = ?
     A) (1,1,1)    B) (1,2,3)    C) (3,3,3)    D) Error

Q40. try: raise ValueError; except Exception: print("A") = ?
     A) A    B) Error    C) ValueError    D) None


══════════════════════════════════════════════════════════════════════════════
                              FINE TEST
══════════════════════════════════════════════════════════════════════════════
"""


MODULE_4_TEST_ANSWERS = """
══════════════════════════════════════════════════════════════════════════════
                    MODULE 4 - RISPOSTE TEST FINALE
══════════════════════════════════════════════════════════════════════════════

Q1:  B) None
Q2:  C) 15
Q3:  B) tuple
Q4:  C) dict
Q5:  B) [1,2] (default mutabile!)
Q6:  A) 10 (x locale in f)
Q7:  B) 20 (global)
Q8:  C) UnboundLocalError
Q9:  A) int ((1) non è tupla!)
Q10: B) tuple ((1,) è tupla)
Q11: B) TypeError (immutabile)
Q12: B) ([1,2,3],)
Q13: B) list (*rest è sempre lista)
Q14: B) KeyError
Q15: B) None
Q16: A) True
Q17: B) False (in verifica chiavi!)
Q18: B) 2 (ultimo valore vince)
Q19: B) TypeError (lista non hashable)
Q20: B) a (itera su chiavi)
Q21: A) A (eccezione → except)
Q22: B) B (nessuna eccezione → else)
Q23: A) F 1 (finally poi continua)
Q24: B) 2 (finally sovrascrive)
Q25: B) IndexError
Q26: C) KeyError
Q27: A) ValueError
Q28: A) 6
Q29: C) tuple
Q30: B) 2
Q31: A) (1,2,3)
Q32: A) 2 (chiave duplicata)
Q33: B) 2
Q34: B) [] (lista vuota)
Q35: A) A (primo match)
Q36: B) False (False and ...)
Q37: B) 2 (1 e "1" sono chiavi diverse!)
Q38: B) (2,3)
Q39: B) (1,2,3) (default mutabile!)
Q40: A) A (ValueError è subclass di Exception)

PUNTEGGIO:
36-40: Eccellente! Pronto per PCEP
32-35: Ottimo!
28-31: Buono, target raggiunto
<28:   Rivedi teoria
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    ESECUZIONE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 78)
    print("PYTHON ESSENTIALS 1 - MODULE 4")
    print("Functions, Tuples, Dictionaries, Exceptions")
    print("=" * 78)
    print("""
    
    ULTIMO MODULO PE1! Completa Block 3 e 4 del PCEP.
    
    CONTENUTO:
    ──────────
    - Section 4.2: Parameters & Arguments (6 quiz)
    - Section 4.3: Return & Scope (6 quiz)
    - Section 4.5: Tuples (6 quiz)
    - Section 4.6: Dictionaries (8 quiz)
    - Section 4.7: Exceptions (7 quiz)
    - Test finale: 40 domande
    
    TRAPPOLE CRITICHE:
    ──────────────────
    ⚠️  Default mutabile (lista come default)
    ⚠️  UnboundLocalError (assegnazione rende locale)
    ⚠️  (1) vs (1,) - int vs tuple!
    ⚠️  in nei dict verifica CHIAVI, non valori
    ⚠️  finally sovrascrive return
    
    COMANDI:
    ────────
    print(QUIZ_4_2)      # Parameters
    print(QUIZ_4_3)      # Scope
    print(QUIZ_4_5)      # Tuples
    print(QUIZ_4_6)      # Dictionaries
    print(QUIZ_4_7)      # Exceptions
    print(MODULE_4_TEST) # Test finale
    
    """)
    print("\n" + "=" * 78)
    print("PE1 COMPLETATO!")
    print("Sei pronto per l'esame PCEP-30-02!")
    print("=" * 78)
