"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║              INTEGRAZIONI SYLLABUS PYTHON INSTITUTE                          ║
║              Copertura 100% - Elementi Mancanti                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Questo file integra gli elementi del syllabus ufficiale Python Institute
che non erano completamente coperti nei moduli principali.

CONTENUTO:
├── PCEP Integration: List Comprehensions (complete)
├── PCAP Integration: __pycache__, bytearray
├── PCPP1 Integration: Composition, Copy, Shelve, Exception Chaining
└── Quiz di verifica per ogni sezione

═══════════════════════════════════════════════════════════════════════════════
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    PCEP INTEGRATION: LIST COMPREHENSIONS
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│             PCEP 3.1 - LIST COMPREHENSIONS (PCEP-30-02)                      │
└──────────────────────────────────────────────────────────────────────────────┘

Le list comprehensions sono un modo compatto per creare liste.

SINTASSI BASE:
    [espressione for elemento in iterabile]

SINTASSI CON CONDIZIONE:
    [espressione for elemento in iterabile if condizione]
"""

# ═══════════════════════════════════════════════════════════════════════════
# ESEMPI BASE
# ═══════════════════════════════════════════════════════════════════════════

# Modo tradizionale
quadrati_tradizionale = []
for x in range(10):
    quadrati_tradizionale.append(x ** 2)

# Con list comprehension (EQUIVALENTE)
quadrati = [x ** 2 for x in range(10)]
print(quadrati)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# Stringhe
parole = ["hello", "world", "python"]
maiuscole = [p.upper() for p in parole]
print(maiuscole)  # ['HELLO', 'WORLD', 'PYTHON']

# Con funzione
def doppio(n):
    return n * 2

doppi = [doppio(x) for x in range(5)]
print(doppi)  # [0, 2, 4, 6, 8]


# ═══════════════════════════════════════════════════════════════════════════
# CON CONDIZIONE (if)
# ═══════════════════════════════════════════════════════════════════════════

# Solo numeri pari
pari = [x for x in range(20) if x % 2 == 0]
print(pari)  # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

# Solo stringhe lunghe
parole = ["a", "abc", "abcde", "ab"]
lunghe = [p for p in parole if len(p) > 2]
print(lunghe)  # ['abc', 'abcde']

# Filtrare numeri negativi
numeri = [-5, 3, -2, 8, -1, 7]
positivi = [n for n in numeri if n > 0]
print(positivi)  # [3, 8, 7]


# ═══════════════════════════════════════════════════════════════════════════
# CON if-else (espressione condizionale)
# ═══════════════════════════════════════════════════════════════════════════

# NOTA: if-else va PRIMA del for!
# Sintassi: [valore_se_vero if condizione else valore_se_falso for x in iter]

numeri = [1, 2, 3, 4, 5]
pari_dispari = ["pari" if x % 2 == 0 else "dispari" for x in numeri]
print(pari_dispari)  # ['dispari', 'pari', 'dispari', 'pari', 'dispari']

# Sostituire negativi con 0
numeri = [-5, 3, -2, 8, -1]
non_negativi = [n if n >= 0 else 0 for n in numeri]
print(non_negativi)  # [0, 3, 0, 8, 0]


# ═══════════════════════════════════════════════════════════════════════════
# NESTED LIST COMPREHENSIONS
# ═══════════════════════════════════════════════════════════════════════════

# Creare matrice 3x3
matrice = [[i * j for j in range(1, 4)] for i in range(1, 4)]
print(matrice)  # [[1, 2, 3], [2, 4, 6], [3, 6, 9]]

# Appiattire matrice (flatten)
matrice = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [elem for riga in matrice for elem in riga]
print(flat)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Prodotto cartesiano
colori = ['rosso', 'verde']
taglie = ['S', 'M', 'L']
combinazioni = [(c, t) for c in colori for t in taglie]
print(combinazioni)
# [('rosso', 'S'), ('rosso', 'M'), ('rosso', 'L'), 
#  ('verde', 'S'), ('verde', 'M'), ('verde', 'L')]


# ═══════════════════════════════════════════════════════════════════════════
# QUIZ LIST COMPREHENSIONS
# ═══════════════════════════════════════════════════════════════════════════

QUIZ_LIST_COMPREHENSIONS = """
Q1. [x ** 2 for x in range(5)] produce:
    A) [1, 4, 9, 16, 25]
    B) [0, 1, 4, 9, 16]
    C) [0, 2, 4, 6, 8]
    D) Errore

Q2. [x for x in range(10) if x % 2 == 0] produce:
    A) [1, 3, 5, 7, 9]
    B) [0, 2, 4, 6, 8]
    C) [2, 4, 6, 8, 10]
    D) []

Q3. Quale è corretto per if-else in list comprehension?
    A) [x if x > 0 for x in lista else 0]
    B) [x for x in lista if x > 0 else 0]
    C) [x if x > 0 else 0 for x in lista]
    D) [for x in lista: x if x > 0 else 0]

Q4. [c.upper() for c in "hello"] produce:
    A) "HELLO"
    B) ['H', 'E', 'L', 'L', 'O']
    C) ['HELLO']
    D) Errore

Q5. [x for x in range(10) if x > 5 if x < 8] produce:
    A) [6, 7]
    B) [5, 6, 7, 8]
    C) Errore (due if)
    D) []

Q6. [[j for j in range(3)] for i in range(2)] produce:
    A) [0, 1, 2, 0, 1, 2]
    B) [[0, 1, 2], [0, 1, 2]]
    C) [[0, 0], [1, 1], [2, 2]]
    D) Errore

Q7. [x * 2 for x in [1, 2, 3]] è equivalente a:
    A) list(map(lambda x: x * 2, [1, 2, 3]))
    B) filter(lambda x: x * 2, [1, 2, 3])
    C) reduce(lambda x: x * 2, [1, 2, 3])
    D) [1, 2, 3].map(x * 2)

Q8. Quale appiattisce [[1,2], [3,4]] in [1,2,3,4]?
    A) [x for x in lista]
    B) [x for sublist in lista for x in sublist]
    C) [[x for x in sublist] for sublist in lista]
    D) [sublist for x in lista]

RISPOSTE: Q1:B, Q2:B, Q3:C, Q4:B, Q5:A, Q6:B, Q7:A, Q8:B
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    PCAP INTEGRATION: __pycache__
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│             PCAP 1.5 - __pycache__ DIRECTORY (PCAP-31-03)                    │
└──────────────────────────────────────────────────────────────────────────────┘

Quando Python importa un modulo, lo compila in bytecode e lo salva
nella cartella __pycache__ per velocizzare le importazioni successive.
"""

# ═══════════════════════════════════════════════════════════════════════════
# TEORIA __pycache__
# ═══════════════════════════════════════════════════════════════════════════

"""
COSA È __pycache__?
───────────────────
- Directory creata automaticamente da Python 3
- Contiene file .pyc (Python Compiled)
- I file .pyc sono bytecode compilato

STRUTTURA DEI FILE:
───────────────────
mymodule.py  →  __pycache__/mymodule.cpython-310.pyc
                              ↑          ↑
                              nome    versione Python

PERCHÉ ESISTE?
──────────────
1. Velocizza le importazioni successive
2. Non deve ricompilare ogni volta
3. Bytecode è indipendente dalla piattaforma

QUANDO VIENE AGGIORNATO?
────────────────────────
Python ricompila se:
- Il file .py è più recente del .pyc
- La versione di Python è cambiata

ESEMPIO STRUTTURA:
──────────────────
myproject/
├── main.py
├── mymodule.py
└── __pycache__/
    └── mymodule.cpython-310.pyc

DISABILITARE __pycache__:
─────────────────────────
# Variabile d'ambiente
export PYTHONDONTWRITEBYTECODE=1

# Flag da linea di comando
python -B script.py

# Nel codice (non raccomandato)
import sys
sys.dont_write_bytecode = True

PULIRE __pycache__:
───────────────────
# Linux/Mac
find . -type d -name __pycache__ -exec rm -rf {} +

# Python
import shutil
shutil.rmtree('__pycache__')

IMPORTANTE PER GIT:
───────────────────
Aggiungi __pycache__ a .gitignore!
"""

# ═══════════════════════════════════════════════════════════════════════════
# QUIZ __pycache__
# ═══════════════════════════════════════════════════════════════════════════

QUIZ_PYCACHE = """
Q1. __pycache__ contiene:
    A) File sorgente .py
    B) File bytecode .pyc
    C) File eseguibili
    D) File di configurazione

Q2. Il formato del nome file in __pycache__ è:
    A) modulo.pyc
    B) modulo.cpython-XX.pyc (XX = versione)
    C) modulo.compiled
    D) modulo.bytecode

Q3. Python ricompila un modulo quando:
    A) Il .py è più recente del .pyc
    B) La versione Python è cambiata
    C) Entrambe A e B
    D) Mai, usa sempre il .pyc

Q4. Per disabilitare __pycache__, usa:
    A) PYTHONDONTWRITEBYTECODE=1
    B) python -B script.py
    C) Entrambe A e B
    D) Non si può disabilitare

Q5. __pycache__ dovrebbe essere in .gitignore?
    A) Sì
    B) No
    C) Dipende
    D) Non esiste .gitignore per Python

RISPOSTE: Q1:B, Q2:B, Q3:C, Q4:C, Q5:A
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    PCAP INTEGRATION: bytearray
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│             PCAP 5.5 - BYTEARRAY (PCAP-31-03)                                │
└──────────────────────────────────────────────────────────────────────────────┘

bytearray è una sequenza MUTABILE di byte (0-255).
Differisce da bytes che è IMMUTABILE.
"""

# ═══════════════════════════════════════════════════════════════════════════
# CREAZIONE BYTEARRAY
# ═══════════════════════════════════════════════════════════════════════════

# Da stringa (richiede encoding)
ba1 = bytearray("Hello", "utf-8")
print(ba1)  # bytearray(b'Hello')

# Da lista di interi (0-255)
ba2 = bytearray([72, 101, 108, 108, 111])  # ASCII codes per "Hello"
print(ba2)  # bytearray(b'Hello')

# Da bytes
ba3 = bytearray(b"Hello")
print(ba3)  # bytearray(b'Hello')

# Vuoto con dimensione
ba4 = bytearray(5)  # 5 byte inizializzati a 0
print(ba4)  # bytearray(b'\x00\x00\x00\x00\x00')


# ═══════════════════════════════════════════════════════════════════════════
# MUTABILITÀ (differenza da bytes)
# ═══════════════════════════════════════════════════════════════════════════

# bytearray è MUTABILE
ba = bytearray(b"Hello")
ba[0] = 74  # ASCII per 'J'
print(ba)   # bytearray(b'Jello')

# bytes è IMMUTABILE
# b = b"Hello"
# b[0] = 74  # TypeError: 'bytes' object does not support item assignment


# ═══════════════════════════════════════════════════════════════════════════
# METODI COMUNI
# ═══════════════════════════════════════════════════════════════════════════

ba = bytearray(b"Hello World")

# Metodi simili a list
ba.append(33)              # Aggiunge byte (!)
ba.extend(b" Python")      # Estende con bytes
ba.insert(0, 42)           # Inserisce a posizione
ba.pop()                   # Rimuove e ritorna ultimo byte
ba.remove(32)              # Rimuove prima occorrenza

# Metodi simili a str
ba = bytearray(b"hello world")
print(ba.upper())          # bytearray(b'HELLO WORLD')
print(ba.replace(b"hello", b"hi"))  # bytearray(b'hi world')
print(ba.find(b"world"))   # 6
print(ba.split())          # [bytearray(b'hello'), bytearray(b'world')]


# ═══════════════════════════════════════════════════════════════════════════
# USO CON FILE I/O
# ═══════════════════════════════════════════════════════════════════════════

# Lettura in buffer bytearray
buffer = bytearray(100)  # Buffer di 100 byte

# with open("file.bin", "rb") as f:
#     num_bytes = f.readinto(buffer)  # Legge direttamente nel buffer!
#     print(f"Letti {num_bytes} byte")

# Scrittura da bytearray
data = bytearray(b"Binary data to write")
# with open("output.bin", "wb") as f:
#     f.write(data)


# ═══════════════════════════════════════════════════════════════════════════
# CONVERSIONI
# ═══════════════════════════════════════════════════════════════════════════

ba = bytearray(b"Hello")

# bytearray -> bytes (immutabile)
b = bytes(ba)

# bytearray -> str
s = ba.decode("utf-8")
print(s)  # "Hello"

# bytearray -> list
lst = list(ba)
print(lst)  # [72, 101, 108, 108, 111]

# hex representation
print(ba.hex())  # '48656c6c6f'


# ═══════════════════════════════════════════════════════════════════════════
# QUIZ BYTEARRAY
# ═══════════════════════════════════════════════════════════════════════════

QUIZ_BYTEARRAY = """
Q1. bytearray è:
    A) Immutabile    B) Mutabile    C) Dipende    D) Non è una sequenza

Q2. bytearray([72, 101, 108, 108, 111]) rappresenta:
    A) Numeri    B) "Hello"    C) Errore    D) [72, 101, 108, 108, 111]

Q3. Quale è valido?
    A) bytearray("Hello")
    B) bytearray("Hello", "utf-8")
    C) bytearray(Hello)
    D) bytearray[Hello]

Q4. Per modificare un byte in bytes:
    A) bytes[0] = 65
    B) Non è possibile (immutabile)
    C) bytes.set(0, 65)
    D) bytes[0:1] = 65

Q5. bytearray(5) crea:
    A) [0, 0, 0, 0, 0]
    B) bytearray di 5 byte a zero
    C) Errore
    D) bytearray(b'5')

Q6. ba.decode("utf-8") restituisce:
    A) bytes    B) str    C) list    D) int

Q7. f.readinto(buffer) è utile perché:
    A) È più veloce
    B) Riusa memoria esistente
    C) Entrambe A e B
    D) Non esiste questo metodo

RISPOSTE: Q1:B, Q2:B, Q3:B, Q4:B, Q5:B, Q6:B, Q7:C
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    PCPP1 INTEGRATION: COMPOSITION
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│             PCPP1 - COMPOSITION vs INHERITANCE (PCPP-32-10x)                 │
└──────────────────────────────────────────────────────────────────────────────┘

COMPOSITION (HAS-A): Un oggetto "ha" altri oggetti come attributi
INHERITANCE (IS-A): Un oggetto "è" un tipo di un altro oggetto

"Favor composition over inheritance" - Design Principle
"""

# ═══════════════════════════════════════════════════════════════════════════
# ESEMPIO INHERITANCE (IS-A)
# ═══════════════════════════════════════════════════════════════════════════

class Animal:
    def speak(self):
        pass

class Dog(Animal):  # Dog IS-A Animal
    def speak(self):
        return "Woof!"

class Cat(Animal):  # Cat IS-A Animal
    def speak(self):
        return "Meow!"


# ═══════════════════════════════════════════════════════════════════════════
# ESEMPIO COMPOSITION (HAS-A)
# ═══════════════════════════════════════════════════════════════════════════

class Engine:
    def __init__(self, horsepower):
        self.horsepower = horsepower
    
    def start(self):
        return "Engine started"

class Wheels:
    def __init__(self, count=4):
        self.count = count
    
    def rotate(self):
        return f"{self.count} wheels rotating"

class Car:
    def __init__(self):
        # Car HAS-A Engine (composition)
        self.engine = Engine(200)
        # Car HAS-A Wheels (composition)
        self.wheels = Wheels(4)
    
    def drive(self):
        return f"{self.engine.start()}, {self.wheels.rotate()}"

car = Car()
print(car.drive())  # "Engine started, 4 wheels rotating"


# ═══════════════════════════════════════════════════════════════════════════
# QUANDO USARE COSA
# ═══════════════════════════════════════════════════════════════════════════

"""
USA INHERITANCE QUANDO:
───────────────────────
- C'è una vera relazione "IS-A"
- Vuoi usare polimorfismo
- Le sottoclassi sono varianti della superclasse
- Esempio: Dog IS-A Animal

USA COMPOSITION QUANDO:
───────────────────────
- C'è una relazione "HAS-A"
- Vuoi flessibilità (cambiare componenti a runtime)
- Vuoi evitare accoppiamento stretto
- Vuoi riusare codice senza ereditare
- Esempio: Car HAS-A Engine

PROBLEMI DI INHERITANCE:
────────────────────────
1. Accoppiamento stretto con superclasse
2. Fragile base class problem
3. Difficile cambiare a runtime
4. Può violare encapsulation
"""


# ═══════════════════════════════════════════════════════════════════════════
# DEPENDENCY INJECTION (pattern correlato)
# ═══════════════════════════════════════════════════════════════════════════

class Logger:
    def log(self, message):
        print(f"LOG: {message}")

class FileLogger:
    def log(self, message):
        print(f"FILE LOG: {message}")

class Service:
    def __init__(self, logger):  # Dependency Injection
        self.logger = logger     # Composition with flexibility
    
    def do_something(self):
        self.logger.log("Doing something")

# Puoi cambiare il logger senza modificare Service
service1 = Service(Logger())
service2 = Service(FileLogger())


# ═══════════════════════════════════════════════════════════════════════════
# QUIZ COMPOSITION
# ═══════════════════════════════════════════════════════════════════════════

QUIZ_COMPOSITION = """
Q1. "Dog IS-A Animal" suggerisce:
    A) Composition    B) Inheritance    C) Aggregation    D) Association

Q2. "Car HAS-A Engine" suggerisce:
    A) Inheritance    B) Composition    C) Polimorfismo    D) Encapsulation

Q3. Composition è preferita quando:
    A) Vuoi flessibilità
    B) Vuoi evitare accoppiamento
    C) Vuoi riusare senza ereditare
    D) Tutte le precedenti

Q4. Quale è un problema di inheritance?
    A) Troppa flessibilità
    B) Fragile base class problem
    C) Nessun riuso di codice
    D) Non supporta polimorfismo

Q5. Dependency Injection usa:
    A) Solo inheritance
    B) Composition
    C) Solo classi astratte
    D) Global variables

RISPOSTE: Q1:B, Q2:B, Q3:D, Q4:B, Q5:B
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    PCPP1 INTEGRATION: SHALLOW/DEEP COPY
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│             PCPP1 - SHALLOW AND DEEP COPY (PCPP-32-10x)                      │
└──────────────────────────────────────────────────────────────────────────────┘

SHALLOW COPY: Copia l'oggetto ma NON gli oggetti interni (condivisi)
DEEP COPY: Copia l'oggetto E tutti gli oggetti interni (indipendenti)
"""

import copy

# ═══════════════════════════════════════════════════════════════════════════
# ASSIGNMENT (NO COPY!)
# ═══════════════════════════════════════════════════════════════════════════

original = [[1, 2, 3], [4, 5, 6]]
assigned = original  # NON è una copia! Stesso oggetto

assigned[0][0] = 999
print(original)  # [[999, 2, 3], [4, 5, 6]] - MODIFICATO!


# ═══════════════════════════════════════════════════════════════════════════
# SHALLOW COPY
# ═══════════════════════════════════════════════════════════════════════════

original = [[1, 2, 3], [4, 5, 6]]

# Modi per creare shallow copy:
shallow1 = original.copy()           # Metodo .copy()
shallow2 = list(original)            # Costruttore
shallow3 = original[:]               # Slicing
shallow4 = copy.copy(original)       # Modulo copy

# Shallow copy: lista esterna copiata, liste interne CONDIVISE
shallow1[0][0] = 999
print(original)  # [[999, 2, 3], [4, 5, 6]] - MODIFICATO!

# Ma aggiungere elementi alla lista esterna è indipendente
original = [[1, 2, 3], [4, 5, 6]]
shallow = original.copy()
shallow.append([7, 8, 9])
print(original)  # [[1, 2, 3], [4, 5, 6]] - NON modificato


# ═══════════════════════════════════════════════════════════════════════════
# DEEP COPY
# ═══════════════════════════════════════════════════════════════════════════

original = [[1, 2, 3], [4, 5, 6]]
deep = copy.deepcopy(original)

# Deep copy: tutto è indipendente
deep[0][0] = 999
print(original)  # [[1, 2, 3], [4, 5, 6]] - NON modificato!


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZZAZIONE DIFFERENZE
# ═══════════════════════════════════════════════════════════════════════════

"""
ORIGINAL          SHALLOW COPY         DEEP COPY
─────────         ────────────         ─────────
   │                   │                  │
   v                   v                  v
┌─────────┐       ┌─────────┐        ┌─────────┐
│ lista   │       │ lista   │        │ lista   │
│ esterna │       │ esterna │        │ esterna │
└────┬────┘       └────┬────┘        └────┬────┘
     │                 │                  │
     │    ┌────────────┘                  │
     │    │                               │
     v    v                               v
┌─────────┐                          ┌─────────┐
│ [1,2,3] │ <── CONDIVISA            │ [1,2,3] │ <── INDIPENDENTE
│ [4,5,6] │                          │ [4,5,6] │
└─────────┘                          └─────────┘
"""


# ═══════════════════════════════════════════════════════════════════════════
# CON OGGETTI CUSTOM
# ═══════════════════════════════════════════════════════════════════════════

class Address:
    def __init__(self, city):
        self.city = city

class Person:
    def __init__(self, name, address):
        self.name = name
        self.address = address  # Oggetto composto

# Shallow copy
p1 = Person("Alice", Address("Rome"))
p2 = copy.copy(p1)

p2.name = "Bob"           # Indipendente
p2.address.city = "Milan" # CONDIVISO!

print(p1.name)           # "Alice" (indipendente)
print(p1.address.city)   # "Milan" (modificato!)

# Deep copy
p1 = Person("Alice", Address("Rome"))
p3 = copy.deepcopy(p1)

p3.address.city = "Milan"
print(p1.address.city)   # "Rome" (indipendente!)


# ═══════════════════════════════════════════════════════════════════════════
# CUSTOMIZZARE COPY
# ═══════════════════════════════════════════════════════════════════════════

class CustomClass:
    def __init__(self, data):
        self.data = data
    
    def __copy__(self):
        """Chiamato da copy.copy()"""
        print("Shallow copy called")
        return CustomClass(self.data)
    
    def __deepcopy__(self, memo):
        """Chiamato da copy.deepcopy()"""
        print("Deep copy called")
        return CustomClass(copy.deepcopy(self.data, memo))


# ═══════════════════════════════════════════════════════════════════════════
# QUIZ SHALLOW/DEEP COPY
# ═══════════════════════════════════════════════════════════════════════════

QUIZ_COPY = """
Q1. a = b (assignment) crea:
    A) Shallow copy    B) Deep copy    C) Nessuna copia    D) Errore

Q2. Shallow copy di lista nested condivide:
    A) La lista esterna    B) Le liste interne    C) Niente    D) Tutto

Q3. Per deep copy si usa:
    A) .copy()    B) copy.copy()    C) copy.deepcopy()    D) [:]

Q4. lista.copy() è:
    A) Deep copy    B) Shallow copy    C) Errore    D) Assignment

Q5. Quale NON crea shallow copy?
    A) lista[:]    B) list(lista)    C) lista.copy()    D) lista = altra

Q6. __deepcopy__ riceve:
    A) Niente    B) self    C) self e memo    D) Solo memo

Q7. Quando usare deep copy?
    A) Liste semplici
    B) Strutture nested che devono essere indipendenti
    C) Sempre
    D) Mai

RISPOSTE: Q1:C, Q2:B, Q3:C, Q4:B, Q5:D, Q6:C, Q7:B
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    PCPP1 INTEGRATION: SHELVE MODULE
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│             PCPP1 - SHELVE MODULE (PCPP-32-10x)                              │
└──────────────────────────────────────────────────────────────────────────────┘

shelve fornisce un dizionario persistente (salvato su disco).
Usa pickle internamente per serializzare oggetti.
"""

import shelve

# ═══════════════════════════════════════════════════════════════════════════
# USO BASE
# ═══════════════════════════════════════════════════════════════════════════

# Aprire/creare uno shelf
# shelf = shelve.open('mydata')  # Crea mydata.db (o simile)

# Usare come dizionario
# shelf['name'] = 'Alice'
# shelf['scores'] = [95, 87, 92]
# shelf['config'] = {'theme': 'dark', 'language': 'it'}

# Leggere
# print(shelf['name'])  # 'Alice'

# Chiudere (IMPORTANTE!)
# shelf.close()


# ═══════════════════════════════════════════════════════════════════════════
# USO CON CONTEXT MANAGER (RACCOMANDATO)
# ═══════════════════════════════════════════════════════════════════════════

SHELVE_EXAMPLE = '''
# Scrittura
with shelve.open('mydata') as shelf:
    shelf['user'] = {'name': 'Alice', 'age': 30}
    shelf['settings'] = {'theme': 'dark'}

# Lettura
with shelve.open('mydata') as shelf:
    user = shelf['user']
    print(user['name'])  # Alice
    
    # Iterare
    for key in shelf:
        print(key, shelf[key])
'''


# ═══════════════════════════════════════════════════════════════════════════
# ATTENZIONE: MUTABILITÀ
# ═══════════════════════════════════════════════════════════════════════════

"""
PROBLEMA: Modificare oggetti mutabili in-place NON salva!

# SBAGLIATO - modifica non salvata
with shelve.open('mydata') as shelf:
    shelf['scores'] = [1, 2, 3]
    shelf['scores'].append(4)  # NON SALVATO!

# CORRETTO - usa writeback=True
with shelve.open('mydata', writeback=True) as shelf:
    shelf['scores'] = [1, 2, 3]
    shelf['scores'].append(4)  # Salvato automaticamente

# ALTERNATIVA - riassegnare esplicitamente
with shelve.open('mydata') as shelf:
    scores = shelf['scores']
    scores.append(4)
    shelf['scores'] = scores  # Riassegna per salvare
"""


# ═══════════════════════════════════════════════════════════════════════════
# METODI DISPONIBILI
# ═══════════════════════════════════════════════════════════════════════════

"""
shelf.keys()      - Ritorna le chiavi
shelf.values()    - Ritorna i valori
shelf.items()     - Ritorna coppie (key, value)
shelf.get(key)    - Come dict.get()
shelf.pop(key)    - Rimuove e ritorna
shelf.clear()     - Svuota
shelf.sync()      - Forza scrittura su disco
shelf.close()     - Chiude il file
"""


# ═══════════════════════════════════════════════════════════════════════════
# SHELVE vs PICKLE vs JSON
# ═══════════════════════════════════════════════════════════════════════════

"""
┌────────────┬─────────────┬─────────────────┬──────────────────┐
│ Carattere  │ shelve      │ pickle          │ JSON             │
├────────────┼─────────────┼─────────────────┼──────────────────┤
│ Interfaccia│ Dict-like   │ dump/load       │ dump/load        │
│ Persistenza│ Automatica  │ Manuale         │ Manuale          │
│ Formato    │ DB files    │ Binary          │ Text             │
│ Tipi Python│ Tutti       │ Tutti           │ Limitati         │
│ Leggibile  │ No          │ No              │ Sì               │
│ Sicurezza  │ Bassa       │ Bassa           │ Alta             │
└────────────┴─────────────┴─────────────────┴──────────────────┘
"""


# ═══════════════════════════════════════════════════════════════════════════
# QUIZ SHELVE
# ═══════════════════════════════════════════════════════════════════════════

QUIZ_SHELVE = """
Q1. shelve fornisce:
    A) Un database SQL
    B) Un dizionario persistente
    C) Un file JSON
    D) Un server

Q2. shelve usa internamente:
    A) JSON    B) CSV    C) pickle    D) XML

Q3. Per salvare modifiche a oggetti mutabili:
    A) Succede automaticamente
    B) Usare writeback=True
    C) Non è possibile
    D) Usare sync()

Q4. Il modo raccomandato di usare shelve è:
    A) shelf = shelve.open()
    B) with shelve.open() as shelf:
    C) shelve.load()
    D) import shelf

Q5. shelve.keys() ritorna:
    A) Lista    B) KeysView    C) Dict    D) Set

Q6. Quale NON è un metodo di shelve?
    A) sync()    B) close()    C) append()    D) clear()

RISPOSTE: Q1:B, Q2:C, Q3:B, Q4:B, Q5:B, Q6:C
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    PCPP1 INTEGRATION: EXCEPTION CHAINING
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│             PCPP1 - EXCEPTION CHAINING (PCPP-32-10x)                         │
└──────────────────────────────────────────────────────────────────────────────┘

Exception chaining permette di collegare eccezioni per mostrare
la catena di errori che ha portato all'eccezione finale.

__cause__: Eccezione impostata esplicitamente con "raise ... from ..."
__context__: Eccezione che era attiva quando è stata sollevata la nuova
"""

# ═══════════════════════════════════════════════════════════════════════════
# IMPLICIT CHAINING (__context__)
# ═══════════════════════════════════════════════════════════════════════════

def implicit_chain():
    try:
        x = 1 / 0
    except ZeroDivisionError:
        # Solleva nuova eccezione mentre si gestisce un'altra
        # Python imposta automaticamente __context__
        raise ValueError("Cannot process")

# try:
#     implicit_chain()
# except ValueError as e:
#     print(f"Exception: {e}")
#     print(f"Context: {e.__context__}")  # ZeroDivisionError


# ═══════════════════════════════════════════════════════════════════════════
# EXPLICIT CHAINING (__cause__) - "raise ... from ..."
# ═══════════════════════════════════════════════════════════════════════════

def explicit_chain():
    try:
        x = int("not a number")
    except ValueError as original:
        # Chaining esplicito con "from"
        raise RuntimeError("Failed to parse") from original

# try:
#     explicit_chain()
# except RuntimeError as e:
#     print(f"Exception: {e}")
#     print(f"Cause: {e.__cause__}")  # ValueError


# ═══════════════════════════════════════════════════════════════════════════
# SUPPRESS CONTEXT - "raise ... from None"
# ═══════════════════════════════════════════════════════════════════════════

def suppress_context():
    try:
        x = 1 / 0
    except ZeroDivisionError:
        # Sopprime il contesto - nasconde l'eccezione originale
        raise ValueError("Invalid operation") from None

# Output NON mostra "During handling of the above exception..."


# ═══════════════════════════════════════════════════════════════════════════
# TRACEBACK COMPLETO
# ═══════════════════════════════════════════════════════════════════════════

"""
IMPLICIT CHAINING OUTPUT:
─────────────────────────
Traceback (most recent call last):
  File "...", line 3, in implicit_chain
    x = 1 / 0
ZeroDivisionError: division by zero

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "...", line 5, in implicit_chain
    raise ValueError("Cannot process")
ValueError: Cannot process


EXPLICIT CHAINING OUTPUT:
─────────────────────────
Traceback (most recent call last):
  File "...", line 3, in explicit_chain
    x = int("not a number")
ValueError: invalid literal for int() with base 10: 'not a number'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "...", line 5, in explicit_chain
    raise RuntimeError("Failed to parse") from original
RuntimeError: Failed to parse
"""


# ═══════════════════════════════════════════════════════════════════════════
# USO PRATICO
# ═══════════════════════════════════════════════════════════════════════════

class DatabaseError(Exception):
    pass

class ConnectionError(Exception):
    pass

def connect_to_database(host):
    try:
        # Simula errore di connessione
        raise OSError(f"Cannot reach {host}")
    except OSError as e:
        # Wrappa in eccezione di dominio
        raise ConnectionError(f"Database connection failed") from e

def get_user(user_id):
    try:
        connect_to_database("localhost")
    except ConnectionError as e:
        raise DatabaseError(f"Cannot get user {user_id}") from e


# ═══════════════════════════════════════════════════════════════════════════
# ISPEZIONARE LA CATENA
# ═══════════════════════════════════════════════════════════════════════════

import traceback

def inspect_chain(exc):
    """Ispeziona la catena di eccezioni"""
    print(f"Exception: {type(exc).__name__}: {exc}")
    
    if exc.__cause__:
        print(f"Explicit cause (__cause__): {exc.__cause__}")
    
    if exc.__context__:
        print(f"Implicit context (__context__): {exc.__context__}")
    
    # __suppress_context__ indica se usato "from None"
    if exc.__suppress_context__:
        print("Context was suppressed (from None)")


# ═══════════════════════════════════════════════════════════════════════════
# QUIZ EXCEPTION CHAINING
# ═══════════════════════════════════════════════════════════════════════════

QUIZ_EXCEPTION_CHAINING = """
Q1. __cause__ è impostato da:
    A) Python automaticamente
    B) raise ... from exception
    C) except ... as
    D) try ... finally

Q2. __context__ è impostato:
    A) Solo esplicitamente
    B) Automaticamente durante handling
    C) Mai
    D) Solo con from None

Q3. "raise X from None" serve a:
    A) Causare X
    B) Sopprimere il contesto
    C) Creare catena esplicita
    D) Niente

Q4. Quale messaggio indica explicit chaining?
    A) "During handling of the above exception..."
    B) "The above exception was the direct cause..."
    C) "Exception occurred"
    D) "Traceback"

Q5. Exception chaining è utile per:
    A) Nascondere errori
    B) Mostrare la catena di errori
    C) Performance
    D) Type checking

Q6. __suppress_context__ è True quando:
    A) Usato raise ... from exception
    B) Usato raise ... from None
    C) Sempre
    D) Mai

RISPOSTE: Q1:B, Q2:B, Q3:B, Q4:B, Q5:B, Q6:B
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    RIEPILOGO E TEST FINALE
# ══════════════════════════════════════════════════════════════════════════════

TEST_FINALE_INTEGRAZIONI = """
╔══════════════════════════════════════════════════════════════════════════════╗
║              TEST FINALE - INTEGRAZIONI SYLLABUS (35 domande)                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Tempo: 30 minuti | Pass: 70% (25/35)                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

LIST COMPREHENSIONS (8 domande)
───────────────────────────────
Q1. [x**2 for x in range(4)] → ?
    A) [0, 1, 4, 9]  B) [1, 4, 9, 16]  C) [0, 2, 4, 6]  D) Errore

Q2. [x for x in range(10) if x % 3 == 0] → ?
    A) [0, 3, 6, 9]  B) [3, 6, 9]  C) [1, 4, 7]  D) []

Q3. [x if x > 0 else 0 for x in [-1, 2, -3, 4]] → ?
    A) [2, 4]  B) [0, 2, 0, 4]  C) [-1, 2, -3, 4]  D) Errore

Q4. [[i*j for j in range(3)] for i in range(2)] → ?
    A) [0,0,0,0,1,2]  B) [[0,0,0],[0,1,2]]  C) [[0,1,2],[0,1,2]]  D) Errore

Q5. Quale appiattisce [[1,2],[3,4]]?
    A) [x for x in lst]
    B) [x for sub in lst for x in sub]
    C) [[x] for x in lst]
    D) list(lst)

Q6. {x: x**2 for x in range(3)} crea:
    A) Lista  B) Set  C) Dizionario  D) Tupla

Q7. [c for c in "abc"] → ?
    A) "abc"  B) ['a','b','c']  C) ['abc']  D) Errore

Q8. [(x,y) for x in [1,2] for y in [3,4]] ha:
    A) 2 elementi  B) 4 elementi  C) 6 elementi  D) 8 elementi

__pycache__ (5 domande)
───────────────────────
Q9. __pycache__ contiene file:
    A) .py  B) .pyc  C) .exe  D) .txt

Q10. Il nome file include:
     A) Solo nome modulo  B) Nome + versione Python  C) Solo versione  D) Timestamp

Q11. Per disabilitare __pycache__:
     A) PYTHONDONTWRITEBYTECODE=1  B) python -B  C) Entrambe  D) Nessuna

Q12. __pycache__ dovrebbe essere in .gitignore?
     A) Sì  B) No  C) Dipende  D) Non esiste

Q13. Il bytecode viene rigenerato quando:
     A) .py cambia  B) Versione Python cambia  C) Entrambe  D) Mai

BYTEARRAY (5 domande)
─────────────────────
Q14. bytearray è:
     A) Immutabile  B) Mutabile  C) Sia A che B  D) Nessuna

Q15. bytearray("test") richiede:
     A) Niente  B) Encoding  C) Int  D) List

Q16. bytearray([65, 66, 67]).decode() → ?
     A) "ABC"  B) [65,66,67]  C) 65  D) Errore

Q17. Differenza tra bytes e bytearray?
     A) bytes mutabile  B) bytearray mutabile  C) Nessuna  D) bytes più veloce

Q18. f.readinto(buffer) usa:
     A) bytes  B) bytearray  C) str  D) list

COMPOSITION (4 domande)
───────────────────────
Q19. "Car HAS-A Engine" indica:
     A) Inheritance  B) Composition  C) Polimorfismo  D) Astrazione

Q20. Composition favorisce:
     A) Accoppiamento stretto  B) Flessibilità  C) Velocità  D) Memoria

Q21. "Dog IS-A Animal" indica:
     A) Composition  B) Inheritance  C) Aggregation  D) Association

Q22. Dependency Injection usa:
     A) Solo inheritance  B) Composition  C) Solo ABC  D) Metaclass

SHALLOW/DEEP COPY (5 domande)
─────────────────────────────
Q23. a = b crea:
     A) Shallow copy  B) Deep copy  C) Nessuna copia  D) Errore

Q24. lista.copy() crea:
     A) Deep copy  B) Shallow copy  C) Reference  D) Errore

Q25. copy.deepcopy() copia:
     A) Solo livello 1  B) Tutti i livelli  C) Niente  D) Solo primitivi

Q26. Per oggetti nested indipendenti:
     A) .copy()  B) deepcopy()  C) [:]  D) list()

Q27. __deepcopy__ riceve:
     A) self  B) memo  C) self e memo  D) Niente

SHELVE (4 domande)
──────────────────
Q28. shelve fornisce:
     A) SQL DB  B) Dict persistente  C) JSON file  D) Server

Q29. shelve usa internamente:
     A) JSON  B) pickle  C) CSV  D) XML

Q30. Per modifiche a mutabili usare:
     A) sync()  B) writeback=True  C) commit()  D) save()

Q31. shelve si chiude con:
     A) with statement  B) .close()  C) Entrambe  D) Automatico

EXCEPTION CHAINING (4 domande)
──────────────────────────────
Q32. __cause__ è impostato da:
     A) Python  B) raise...from  C) except  D) finally

Q33. "raise X from None" serve a:
     A) Causare X  B) Sopprimere contesto  C) Catena  D) Niente

Q34. __context__ è impostato:
     A) Esplicitamente  B) Automaticamente  C) Mai  D) Solo from

Q35. __suppress_context__ True quando:
     A) raise...from exc  B) raise...from None  C) Sempre  D) Mai


═══════════════════════════════════════════════════════════════════════════════
                              RISPOSTE
═══════════════════════════════════════════════════════════════════════════════

Q1:A   Q2:A   Q3:B   Q4:B   Q5:B   Q6:C   Q7:B   Q8:B
Q9:B   Q10:B  Q11:C  Q12:A  Q13:C  Q14:B  Q15:B  Q16:A
Q17:B  Q18:B  Q19:B  Q20:B  Q21:B  Q22:B  Q23:C  Q24:B
Q25:B  Q26:B  Q27:C  Q28:B  Q29:B  Q30:B  Q31:C  Q32:B
Q33:B  Q34:B  Q35:B

PUNTEGGIO:
──────────
32-35: Eccellente! Pronto per gli esami!
28-31: Ottimo! Piccola revisione.
25-27: Buono (70% pass)
<25:   Rivedi le sezioni deboli.

═══════════════════════════════════════════════════════════════════════════════
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    ESECUZIONE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 78)
    print("INTEGRAZIONI SYLLABUS PYTHON INSTITUTE")
    print("Copertura 100% elementi mancanti")
    print("=" * 78)
    print("""
    CONTENUTO:
    ──────────
    ✅ PCEP: List Comprehensions
    ✅ PCAP: __pycache__, bytearray  
    ✅ PCPP1: Composition, Shallow/Deep Copy, Shelve, Exception Chaining
    
    COMANDI:
    ────────
    print(QUIZ_LIST_COMPREHENSIONS)
    print(QUIZ_PYCACHE)
    print(QUIZ_BYTEARRAY)
    print(QUIZ_COMPOSITION)
    print(QUIZ_COPY)
    print(QUIZ_SHELVE)
    print(QUIZ_EXCEPTION_CHAINING)
    print(TEST_FINALE_INTEGRAZIONI)
    """)
