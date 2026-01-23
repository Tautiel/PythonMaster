"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    PYTHON ADVANCED (PA) - MODULE 2                           ║
║           Best Practices, PEP 8, Type Hints, SOLID Principles                ║
║                                                                              ║
║                     Allineato al Syllabus PCPP1-32-10x                       ║
║                         OpenEDG Python Institute                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

PCPP1 Exam Block 2: Best Practices and Standardization (20%)

STRUTTURA MODULO:
├── Section 2.1: PEP 8 Style Guide
├── Section 2.2: PEP 20 (Zen of Python)
├── Section 2.3: PEP 257 (Docstrings)
├── Section 2.4: Type Hints (PEP 484)
├── Section 2.5: SOLID Principles
├── Section 2.6: Code Quality Tools
├── Section 2.7: Project Structure
├── Labs (10 esercizi pratici)
└── Module 2 Quiz (35 domande)

TEMPO STIMATO: 6-8 ore

═══════════════════════════════════════════════════════════════════════════════
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 2.1: PEP 8 STYLE GUIDE
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    2.1 TEORIA: PEP 8 STYLE GUIDE                             │
└──────────────────────────────────────────────────────────────────────────────┘

PEP 8 è la guida di stile ufficiale per codice Python.
"Readability counts" - Un codice leggibile è manutenibile.

INDENTAZIONE:
─────────────
- Usa 4 SPAZI per livello (mai TAB!)
- Le continuation lines devono essere allineate
"""

# CORRETTO
def long_function_name(
        var_one, var_two,
        var_three, var_four):
    print(var_one)

# CORRETTO (hanging indent)
def long_function_name(
    var_one, var_two,
    var_three, var_four
):
    print(var_one)


"""
LUNGHEZZA LINEE:
────────────────
- MAX 79 caratteri per codice
- MAX 72 caratteri per docstrings/commenti
"""

# CORRETTO - spezza linee lunghe
my_list = [
    'item1', 'item2', 'item3',
    'item4', 'item5', 'item6'
]

# Con backslash (meno preferito)
long_string = "This is a very long string that " \
              "continues on the next line"

# Con parentesi (preferito)
long_string = (
    "This is a very long string that "
    "continues on the next line"
)


"""
IMPORT:
───────
- Uno per riga
- Ordine: standard library, third-party, local
- Mai usare: from module import *
"""

# CORRETTO
import os
import sys

from typing import List, Dict

from mypackage import mymodule

# SBAGLIATO
import os, sys  # Mai multipli sulla stessa riga
from os import *  # Mai wildcard import


"""
WHITESPACE:
───────────
"""

# CORRETTO
spam(ham[1], {eggs: 2})
foo = (0,)
if x == 4: print(x)
x = 1
y = 2
long_variable = 3

# SBAGLIATO
spam( ham[ 1 ], { eggs: 2 } )  # Spazi dentro parentesi
x             = 1  # Allineamento con spazi extra


"""
NAMING CONVENTIONS:
───────────────────
"""

# Variabili e funzioni: snake_case
my_variable = 1
def my_function():
    pass

# Classi: PascalCase (CapWords)
class MyClass:
    pass

# Costanti: SCREAMING_SNAKE_CASE
MAX_SIZE = 100
DEFAULT_COLOR = 'blue'

# Protetto (convenzione): _leading_underscore
_internal_variable = 'private'

# Name mangling: __double_leading
class Example:
    __mangled = 'very private'  # Diventa _Example__mangled

# Magic methods: __double_both__
def __init__(self):
    pass


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 2.1                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_2_1 = """
Q1. Quanti spazi per indentazione secondo PEP 8?
    A) 2    B) 4    C) Tab    D) A scelta

Q2. Lunghezza massima linea raccomandata:
    A) 79 caratteri    B) 100 caratteri    C) 120 caratteri    D) Illimitata

Q3. Naming convention per costanti:
    A) camelCase    B) PascalCase    C) SCREAMING_SNAKE_CASE    D) snake_case

Q4. Naming convention per classi:
    A) snake_case    B) PascalCase    C) UPPER_CASE    D) mixedCase

Q5. import os, sys è:
    A) Corretto    B) Sbagliato secondo PEP 8    C) Obbligatorio    D) Deprecato

Q6. from module import * è:
    A) Consigliato    B) Da evitare    C) Obbligatorio    D) Più veloce

Q7. _variable indica:
    A) Variabile privata (convenzione)    B) Variabile di classe    C) Costante    D) Errore

Q8. __variable causa:
    A) Errore    B) Name mangling    C) Variabile globale    D) Costante
"""

ANSWERS_2_1 = """
RISPOSTE QUIZ 2.1:
Q1: B - 4 spazi, mai tab
Q2: A - 79 caratteri per codice
Q3: C - SCREAMING_SNAKE_CASE per costanti
Q4: B - PascalCase (CapWords) per classi
Q5: B - Sbagliato, un import per riga
Q6: B - Da evitare, inquina il namespace
Q7: A - Convenzione per "protected"
Q8: B - Name mangling (_ClassName__variable)
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 2.2: PEP 20 (ZEN OF PYTHON)
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    2.2 TEORIA: ZEN OF PYTHON                                 │
└──────────────────────────────────────────────────────────────────────────────┘
"""

import this  # Stampa lo Zen of Python

ZEN_OF_PYTHON = """
The Zen of Python, by Tim Peters

Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!
"""

"""
PRINCIPI CHIAVE:
────────────────

1. "Explicit is better than implicit"
"""
# SBAGLIATO - implicito
from module import *

# CORRETTO - esplicito
from module import specific_function

"""
2. "Simple is better than complex"
"""
# SBAGLIATO - complesso
result = (lambda x: x*2)(5)

# CORRETTO - semplice
def double(x):
    return x * 2
result = double(5)

"""
3. "Flat is better than nested"
"""
# SBAGLIATO - troppo annidato
def process(data):
    if data:
        if len(data) > 0:
            if data[0] != None:
                return data[0]
    return None

# CORRETTO - flat con early return
def process(data):
    if not data:
        return None
    if len(data) == 0:
        return None
    if data[0] is None:
        return None
    return data[0]

"""
4. "Errors should never pass silently"
"""
# SBAGLIATO - errori silenziati
try:
    risky_operation()
except:  # Bare except!
    pass

# CORRETTO - gestione esplicita
try:
    risky_operation()
except SpecificError as e:
    logging.error(f"Operation failed: {e}")
    raise


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 2.2                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_2_2 = """
Q1. Come si accede allo Zen of Python?
    A) print(zen)    B) import this    C) python --zen    D) help(zen)

Q2. "Explicit is better than implicit" suggerisce:
    A) Usare sempre import *
    B) Specificare esattamente cosa si importa
    C) Nascondere i dettagli
    D) Usare abbreviazioni

Q3. "Flat is better than nested" suggerisce:
    A) Più indentazione possibile
    B) Evitare troppi livelli di annidamento
    C) Non usare funzioni
    D) Solo un file per progetto

Q4. except: pass è:
    A) Best practice    B) Da evitare (silenzia errori)    C) Obbligatorio    D) Deprecato

Q5. Chi ha scritto lo Zen of Python?
    A) Guido van Rossum    B) Tim Peters    C) Raymond Hettinger    D) Anonymous
"""

ANSWERS_2_2 = """
RISPOSTE QUIZ 2.2:
Q1: B - import this
Q2: B - Specificare esattamente (no wildcard import)
Q3: B - Evitare troppi livelli di annidamento (early return)
Q4: B - Da evitare, gli errori non devono passare silenziosamente
Q5: B - Tim Peters
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 2.3: PEP 257 (DOCSTRINGS)
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    2.3 TEORIA: DOCSTRINGS                                    │
└──────────────────────────────────────────────────────────────────────────────┘

Le docstrings documentano moduli, classi, funzioni.
Si accede con __doc__ o help().
"""

def example_function(param1, param2):
    """
    Breve descrizione della funzione.
    
    Descrizione più lunga che spiega cosa fa la funzione,
    come usarla, e altri dettagli importanti.
    
    Args:
        param1 (int): Descrizione del primo parametro.
        param2 (str): Descrizione del secondo parametro.
    
    Returns:
        bool: Descrizione del valore restituito.
    
    Raises:
        ValueError: Se param1 è negativo.
    
    Example:
        >>> example_function(1, "test")
        True
    """
    if param1 < 0:
        raise ValueError("param1 must be non-negative")
    return True


class ExampleClass:
    """
    Breve descrizione della classe.
    
    Descrizione più lunga della classe e del suo scopo.
    
    Attributes:
        attr1 (int): Descrizione dell'attributo.
        attr2 (str): Descrizione dell'attributo.
    """
    
    def __init__(self, attr1, attr2):
        """
        Inizializza ExampleClass.
        
        Args:
            attr1 (int): Valore iniziale per attr1.
            attr2 (str): Valore iniziale per attr2.
        """
        self.attr1 = attr1
        self.attr2 = attr2


"""
STILI DI DOCSTRING:
───────────────────

1. Google Style (mostrato sopra)
2. NumPy Style
3. Sphinx/reStructuredText
"""

# NUMPY STYLE
def numpy_style_function(param1, param2):
    """
    Breve descrizione.
    
    Parameters
    ----------
    param1 : int
        Descrizione del primo parametro.
    param2 : str
        Descrizione del secondo parametro.
    
    Returns
    -------
    bool
        Descrizione del valore restituito.
    """
    return True


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 2.3                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_2_3 = """
Q1. Come si accede alla docstring di una funzione?
    A) func.doc    B) func.__doc__    C) doc(func)    D) getdoc(func)

Q2. Una docstring deve essere:
    A) Un commento #
    B) Una stringa come prima istruzione
    C) Una variabile
    D) Un decoratore

Q3. PEP 257 specifica:
    A) La sintassi Python    B) Le convenzioni per docstrings    C) I type hints    D) Lo stile

Q4. Args: nella docstring indica:
    A) Gli argomenti della funzione
    B) Gli errori possibili
    C) I valori di ritorno
    D) Gli attributi della classe

Q5. help(func) mostra:
    A) Il codice sorgente    B) La docstring    C) Gli errori    D) I test
"""

ANSWERS_2_3 = """
RISPOSTE QUIZ 2.3:
Q1: B - func.__doc__ (o help(func))
Q2: B - Una stringa come prima istruzione del blocco
Q3: B - Convenzioni per docstrings
Q4: A - Gli argomenti/parametri della funzione
Q5: B - La docstring (documentazione)
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 2.4: TYPE HINTS (PEP 484)
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    2.4 TEORIA: TYPE HINTS                                    │
└──────────────────────────────────────────────────────────────────────────────┘

I type hints (PEP 484) permettono di annotare i tipi.
Non sono obbligatori e non vengono controllati a runtime!
"""

from typing import List, Dict, Tuple, Optional, Union, Callable, Any


# TYPE HINTS BASE
def greet(name: str) -> str:
    return f"Hello, {name}"


def add(a: int, b: int) -> int:
    return a + b


# VARIABILI CON TYPE HINTS
age: int = 30
name: str = "Marco"
is_active: bool = True


# COLLECTIONS
def process_items(items: List[int]) -> Dict[str, int]:
    return {"sum": sum(items), "count": len(items)}


def get_coordinates() -> Tuple[float, float]:
    return (1.0, 2.0)


# OPTIONAL (può essere None)
def find_user(user_id: int) -> Optional[str]:
    """Restituisce None se utente non trovato."""
    if user_id > 0:
        return "User"
    return None


# UNION (più tipi possibili)
def process(value: Union[int, str]) -> str:
    return str(value)


# Python 3.10+ syntax
def process_new(value: int | str) -> str:
    return str(value)


# CALLABLE
def apply_func(func: Callable[[int, int], int], a: int, b: int) -> int:
    """func è una funzione che prende 2 int e restituisce int."""
    return func(a, b)


# GENERICS
from typing import TypeVar, Generic

T = TypeVar('T')

class Box(Generic[T]):
    def __init__(self, content: T):
        self.content = content
    
    def get(self) -> T:
        return self.content


# CLASS METHODS
class MyClass:
    def __init__(self, value: int) -> None:
        self.value = value
    
    def process(self, multiplier: float) -> float:
        return self.value * multiplier
    
    @classmethod
    def create(cls, value: int) -> 'MyClass':
        return cls(value)


"""
TYPE CHECKING CON MYPY:
───────────────────────
mypy controlla i tipi STATICAMENTE (prima dell'esecuzione).

$ pip install mypy
$ mypy script.py
"""


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 2.4                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_2_4 = """
Q1. I type hints sono controllati a runtime?
    A) Sì, sempre    B) No, sono solo annotazioni    C) Solo con flag    D) Solo in debug

Q2. Optional[int] equivale a:
    A) int    B) None    C) Union[int, None]    D) List[int]

Q3. def f() -> None: significa:
    A) La funzione non esiste
    B) La funzione non restituisce nulla
    C) Errore
    D) La funzione è privata

Q4. Callable[[int, int], int] descrive:
    A) Una lista    B) Una funzione con 2 int che restituisce int    C) Un errore    D) Un dizionario

Q5. Per verificare i type hints usi:
    A) python --check    B) mypy    C) pylint    D) pytest

Q6. List[str] significa:
    A) Una stringa    B) Una lista di stringhe    C) Una lista o stringa    D) Errore

Q7. In Python 3.10+, Union[int, str] si scrive:
    A) int | str    B) int & str    C) int + str    D) int, str

Q8. -> dopo i parametri indica:
    A) Il corpo della funzione    B) Il tipo di ritorno    C) Un decoratore    D) Un commento
"""

ANSWERS_2_4 = """
RISPOSTE QUIZ 2.4:
Q1: B - No, sono solo annotazioni per documentazione e tool esterni
Q2: C - Union[int, None] - può essere int o None
Q3: B - La funzione non restituisce nulla (return None implicito)
Q4: B - Una funzione che prende 2 int e restituisce int
Q5: B - mypy è il type checker standard
Q6: B - Una lista che contiene stringhe
Q7: A - int | str (union type syntax)
Q8: B - Il tipo di ritorno della funzione
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 2.5: SOLID PRINCIPLES
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    2.5 TEORIA: SOLID PRINCIPLES                              │
└──────────────────────────────────────────────────────────────────────────────┘

SOLID = 5 principi per codice manutenibile e scalabile.
"""

from abc import ABC, abstractmethod


"""
S - SINGLE RESPONSIBILITY PRINCIPLE
───────────────────────────────────
Una classe dovrebbe avere una sola ragione per cambiare.
"""

# SBAGLIATO - troppe responsabilità
class UserBad:
    def __init__(self, name):
        self.name = name
    
    def save_to_database(self):
        # Logica database
        pass
    
    def send_email(self):
        # Logica email
        pass
    
    def generate_report(self):
        # Logica report
        pass


# CORRETTO - responsabilità separate
class User:
    def __init__(self, name):
        self.name = name

class UserRepository:
    def save(self, user):
        pass

class EmailService:
    def send(self, user, message):
        pass

class ReportGenerator:
    def generate(self, user):
        pass


"""
O - OPEN/CLOSED PRINCIPLE
─────────────────────────
Aperto per estensione, chiuso per modifica.
"""

# SBAGLIATO - modifica necessaria per nuovi tipi
class DiscountCalculatorBad:
    def calculate(self, customer_type, amount):
        if customer_type == "regular":
            return amount * 0.1
        elif customer_type == "premium":
            return amount * 0.2
        # Devo modificare per ogni nuovo tipo!


# CORRETTO - estensibile senza modifica
class Discount(ABC):
    @abstractmethod
    def calculate(self, amount: float) -> float:
        pass

class RegularDiscount(Discount):
    def calculate(self, amount):
        return amount * 0.1

class PremiumDiscount(Discount):
    def calculate(self, amount):
        return amount * 0.2

class VIPDiscount(Discount):  # Nuova classe, nessuna modifica!
    def calculate(self, amount):
        return amount * 0.3


"""
L - LISKOV SUBSTITUTION PRINCIPLE
─────────────────────────────────
Le sottoclassi devono essere sostituibili alle classi base.
"""

# SBAGLIATO - viola LSP
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height

class SquareBad(Rectangle):
    def __init__(self, side):
        super().__init__(side, side)
    
    # Problema: se modifico width, height non cambia!


# CORRETTO - interfaccia comune
class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass

class RectangleGood(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height

class SquareGood(Shape):
    def __init__(self, side):
        self.side = side
    
    def area(self):
        return self.side ** 2


"""
I - INTERFACE SEGREGATION PRINCIPLE
───────────────────────────────────
Interfacce specifiche sono meglio di una interfaccia generale.
"""

# SBAGLIATO - interfaccia troppo grande
class WorkerBad(ABC):
    @abstractmethod
    def work(self): pass
    
    @abstractmethod
    def eat(self): pass
    
    @abstractmethod
    def sleep(self): pass

# Un robot non può mangiare o dormire!


# CORRETTO - interfacce segregate
class Workable(ABC):
    @abstractmethod
    def work(self): pass

class Eatable(ABC):
    @abstractmethod
    def eat(self): pass

class Human(Workable, Eatable):
    def work(self):
        pass
    
    def eat(self):
        pass

class Robot(Workable):
    def work(self):
        pass
    # Non deve implementare eat!


"""
D - DEPENDENCY INVERSION PRINCIPLE
──────────────────────────────────
Dipendi da astrazioni, non da implementazioni concrete.
"""

# SBAGLIATO - dipendenza da classe concreta
class MySQLDatabase:
    def save(self, data):
        pass

class UserServiceBad:
    def __init__(self):
        self.db = MySQLDatabase()  # Dipendenza concreta!
    
    def save_user(self, user):
        self.db.save(user)


# CORRETTO - dipendenza da astrazione
class Database(ABC):
    @abstractmethod
    def save(self, data): pass

class MySQLDatabaseGood(Database):
    def save(self, data):
        pass

class PostgreSQLDatabase(Database):
    def save(self, data):
        pass

class UserServiceGood:
    def __init__(self, db: Database):  # Dependency Injection
        self.db = db
    
    def save_user(self, user):
        self.db.save(user)

# Posso usare qualsiasi database!
service = UserServiceGood(MySQLDatabaseGood())
service = UserServiceGood(PostgreSQLDatabase())


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 2.5                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_2_5 = """
Q1. Cosa significa la S in SOLID?
    A) Simple Responsibility    B) Single Responsibility    C) Strict Responsibility    D) Separate

Q2. Open/Closed significa:
    A) File aperti/chiusi
    B) Aperto per estensione, chiuso per modifica
    C) Classi pubbliche/private
    D) Database open/closed

Q3. Liskov Substitution dice che:
    A) Le sottoclassi devono essere diverse
    B) Le sottoclassi devono essere sostituibili alle classi base
    C) Non usare ereditarietà
    D) Usare solo composizione

Q4. Interface Segregation preferisce:
    A) Una interfaccia grande
    B) Molte interfacce piccole e specifiche
    C) Nessuna interfaccia
    D) Solo classi concrete

Q5. Dependency Inversion consiglia:
    A) Dipendere da classi concrete
    B) Dipendere da astrazioni
    C) Non avere dipendenze
    D) Usare variabili globali

Q6. SOLID aiuta a creare codice:
    A) Più veloce    B) Più manutenibile    C) Più corto    D) Più complesso

Q7. Dependency Injection è:
    A) Passare le dipendenze dall'esterno
    B) Creare dipendenze internamente
    C) Un design pattern
    D) A e C

Q8. Una classe con 10 metodi non correlati viola:
    A) Open/Closed    B) Single Responsibility    C) Liskov    D) Nessuno
"""

ANSWERS_2_5 = """
RISPOSTE QUIZ 2.5:
Q1: B - Single Responsibility Principle
Q2: B - Aperto per estensione, chiuso per modifica
Q3: B - Le sottoclassi devono essere sostituibili senza rompere il codice
Q4: B - Molte interfacce piccole e specifiche
Q5: B - Dipendere da astrazioni (interfacce/ABC)
Q6: B - Più manutenibile e scalabile
Q7: D - Passare dipendenze dall'esterno (è anche un design pattern)
Q8: B - Single Responsibility (troppe responsabilità)
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 2.6: CODE QUALITY TOOLS
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    2.6 TEORIA: CODE QUALITY TOOLS                            │
└──────────────────────────────────────────────────────────────────────────────┘

Tool per verificare e migliorare la qualità del codice.
"""

TOOLS_REFERENCE = """
LINTERS (controllano stile e errori):
─────────────────────────────────────
pylint      - Completo, configurabile
flake8      - Combinazione di pyflakes, pycodestyle, mccabe
pycodestyle - Solo PEP 8 (ex pep8)
pyflakes    - Solo errori logici
ruff        - Nuovo, velocissimo (scritto in Rust)

FORMATTERS (formattano automaticamente):
────────────────────────────────────────
black       - "The uncompromising formatter"
autopep8    - Corregge violazioni PEP 8
yapf        - Google, configurabile
isort       - Ordina gli import

TYPE CHECKERS:
──────────────
mypy        - Standard de facto
pyright     - Microsoft, veloce
pyre        - Facebook

SECURITY:
─────────
bandit      - Trova vulnerabilità di sicurezza

USAGE:
──────
$ pip install pylint flake8 black mypy
$ pylint script.py
$ flake8 script.py
$ black script.py
$ mypy script.py
"""


"""
CONFIGURAZIONE:
───────────────
Puoi configurare i tool con:
- pyproject.toml (moderno, consigliato)
- setup.cfg
- file specifici (.pylintrc, .flake8, mypy.ini)
"""

PYPROJECT_EXAMPLE = """
# pyproject.toml

[tool.black]
line-length = 88
target-version = ['py310']

[tool.isort]
profile = "black"

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_ignores = true

[tool.pylint.messages_control]
disable = ["missing-docstring", "too-few-public-methods"]
"""


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 2.6                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_2_6 = """
Q1. pylint è:
    A) Un formatter    B) Un linter    C) Un type checker    D) Un debugger

Q2. black è:
    A) Un linter    B) Un formatter    C) Un type checker    D) Un test runner

Q3. mypy controlla:
    A) Lo stile    B) I type hints    C) La formattazione    D) I test

Q4. flake8 combina:
    A) mypy + black
    B) pyflakes + pycodestyle + mccabe
    C) pylint + pytest
    D) ruff + isort

Q5. isort serve per:
    A) Ordinare gli import    B) Ordinare le funzioni    C) Ordinare i file    D) Ordinare i test

Q6. pyproject.toml è usato per:
    A) Solo pytest    B) Configurazione tool vari    C) Solo mypy    D) Solo black

Q7. bandit trova:
    A) Bug di stile    B) Vulnerabilità di sicurezza    C) Type errors    D) Test falliti

Q8. Il tool più veloce tra i linter moderni è:
    A) pylint    B) flake8    C) ruff    D) pycodestyle
"""

ANSWERS_2_6 = """
RISPOSTE QUIZ 2.6:
Q1: B - Linter (controlla errori e stile)
Q2: B - Formatter (formatta automaticamente)
Q3: B - Type hints (type checker)
Q4: B - pyflakes + pycodestyle + mccabe
Q5: A - Ordinare gli import
Q6: B - Configurazione centralizzata per vari tool
Q7: B - Vulnerabilità di sicurezza
Q8: C - ruff (scritto in Rust, molto veloce)
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 2.7: PROJECT STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    2.7 TEORIA: PROJECT STRUCTURE                             │
└──────────────────────────────────────────────────────────────────────────────┘
"""

PROJECT_STRUCTURE = """
STRUTTURA PROGETTO CONSIGLIATA:
───────────────────────────────

my_project/
├── src/
│   └── my_package/
│       ├── __init__.py
│       ├── core.py
│       ├── utils.py
│       └── models/
│           ├── __init__.py
│           └── user.py
├── tests/
│   ├── __init__.py
│   ├── test_core.py
│   └── test_utils.py
├── docs/
│   └── index.md
├── pyproject.toml
├── README.md
├── LICENSE
└── .gitignore


FILE IMPORTANTI:
────────────────

pyproject.toml - Configurazione progetto (PEP 518/621)
README.md      - Documentazione principale
LICENSE        - Licenza del progetto
.gitignore     - File da ignorare in git
requirements.txt - Dipendenze (o in pyproject.toml)


__init__.py:
────────────
Rende una directory un package Python.
Può essere vuoto o contenere inizializzazioni.
"""


PYPROJECT_FULL_EXAMPLE = """
# pyproject.toml completo

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "my_package"
version = "0.1.0"
description = "My awesome package"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [
    {name = "Marco", email = "marco@example.com"}
]
dependencies = [
    "requests>=2.28.0",
    "pandas>=1.5.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    "mypy>=1.0.0",
]

[project.scripts]
my-cli = "my_package.cli:main"
"""


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 2.7                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_2_7 = """
Q1. __init__.py serve per:
    A) Inizializzare variabili    B) Rendere una directory un package    C) Testare    D) Documentare

Q2. I test vanno tipicamente in:
    A) src/    B) tests/    C) docs/    D) La root del progetto

Q3. pyproject.toml è definito da:
    A) PEP 8    B) PEP 518/621    C) PEP 484    D) PEP 257

Q4. [project.scripts] definisce:
    A) Test    B) Comandi CLI    C) Documentazione    D) Dipendenze

Q5. [project.optional-dependencies] è per:
    A) Dipendenze obbligatorie
    B) Dipendenze opzionali (es. dev, test)
    C) Dipendenze Python
    D) Dipendenze di sistema
"""

ANSWERS_2_7 = """
RISPOSTE QUIZ 2.7:
Q1: B - Rende una directory un package Python
Q2: B - tests/ (separati dal codice sorgente)
Q3: B - PEP 518 (build-system) e PEP 621 (metadata)
Q4: B - Comandi CLI installabili
Q5: B - Dipendenze opzionali come dev tools
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    MODULE 2 FINAL TEST
# ══════════════════════════════════════════════════════════════════════════════

MODULE_2_FINAL_TEST = """
═══════════════════════════════════════════════════════════════════════════════
                    PA MODULE 2 - TEST FINALE (35 domande)
                             Tempo: 40 minuti
═══════════════════════════════════════════════════════════════════════════════

Q1.  PEP 8 raccomanda __ spazi per indentazione: A) 2  B) 4  C) Tab  D) 8

Q2.  Lunghezza max linea PEP 8: A) 79  B) 100  C) 120  D) Illimitata

Q3.  Naming per costanti: A) camelCase  B) snake_case  C) UPPER_CASE  D) PascalCase

Q4.  Naming per classi: A) snake_case  B) PascalCase  C) UPPER_CASE  D) camelCase

Q5.  import this mostra: A) Errore  B) Zen of Python  C) PEP 8  D) Help

Q6.  "Explicit is better than implicit" vieta: A) Variabili  B) from x import *  C) Classi  D) Funzioni

Q7.  Docstring si accede con: A) __doc__  B) __str__  C) __repr__  D) __help__

Q8.  Type hints sono verificati a runtime? A) Sì  B) No  C) Dipende  D) Solo debug

Q9.  Optional[int] equivale a: A) int  B) None  C) Union[int,None]  D) List[int]

Q10. Callable[[int],str] descrive: A) Lista  B) Funzione int→str  C) Classe  D) Dizionario

Q11. mypy è: A) Formatter  B) Linter  C) Type checker  D) Debugger

Q12. S in SOLID: A) Simple  B) Single Responsibility  C) Strict  D) Separate

Q13. O in SOLID: A) Object  B) Open/Closed  C) Override  D) Operate

Q14. L in SOLID: A) Liskov Substitution  B) Logic  C) Layer  D) Link

Q15. I in SOLID: A) Inheritance  B) Interface Segregation  C) Import  D) Instance

Q16. D in SOLID: A) Database  B) Dependency Inversion  C) Debug  D) Design

Q17. Dependency Injection è: A) Creare dipendenze dentro  B) Passare dipendenze fuori  C) Bug  D) Pattern

Q18. pylint è: A) Formatter  B) Linter  C) Type checker  D) Test runner

Q19. black è: A) Formatter  B) Linter  C) Type checker  D) Debugger

Q20. flake8 include: A) mypy  B) pyflakes+pycodestyle  C) black  D) pytest

Q21. __init__.py rende: A) File eseguibile  B) Directory un package  C) Classe  D) Test

Q22. Tests vanno in: A) src/  B) tests/  C) docs/  D) Root

Q23. pyproject.toml è: A) PEP 8  B) PEP 518/621  C) PEP 484  D) PEP 257

Q24. _variable indica: A) Privato  B) Protetto (convenzione)  C) Costante  D) Globale

Q25. __variable causa: A) Errore  B) Name mangling  C) Globale  D) Pubblico

Q26. ruff è: A) Linter veloce  B) Type checker  C) Formatter  D) Test runner

Q27. bandit cerca: A) Bug stile  B) Security issues  C) Type errors  D) Test

Q28. isort ordina: A) Funzioni  B) Import  C) Classi  D) Test

Q29. Google style docstring usa: A) Args:  B) Parameters:  C) Params:  D) Input:

Q30. NumPy style docstring usa: A) Args:  B) Parameters\n----------  C) Params:  D) Input:

Q31. -> in def f() -> int: indica: A) Corpo  B) Return type  C) Decoratore  D) Commento

Q32. List[str] è: A) Stringa  B) Lista di str  C) Lista o str  D) Errore

Q33. int | str (Python 3.10+) equivale a: A) int+str  B) Union[int,str]  C) int&str  D) Errore

Q34. [tool.black] in pyproject.toml configura: A) Test  B) Formatter black  C) Linter  D) Type checker

Q35. project.scripts in pyproject.toml definisce: A) Test  B) CLI commands  C) Docs  D) Dependencies


═══════════════════════════════════════════════════════════════════════════════
"""

MODULE_2_FINAL_ANSWERS = """
═══════════════════════════════════════════════════════════════════════════════
                    PA MODULE 2 - RISPOSTE TEST FINALE
═══════════════════════════════════════════════════════════════════════════════

Q1: B    Q2: A    Q3: C    Q4: B    Q5: B
Q6: B    Q7: A    Q8: B    Q9: C    Q10: B
Q11: C   Q12: B   Q13: B   Q14: A   Q15: B
Q16: B   Q17: B   Q18: B   Q19: A   Q20: B
Q21: B   Q22: B   Q23: B   Q24: B   Q25: B
Q26: A   Q27: B   Q28: B   Q29: A   Q30: B
Q31: B   Q32: B   Q33: B   Q34: B   Q35: B

PUNTEGGIO:
──────────
32-35: Eccellente!
28-31: Ottimo!
25-27: Buono (70%)
<25:   Rivedi le sezioni deboli
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    ESECUZIONE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 78)
    print("PYTHON ADVANCED - MODULE 2")
    print("Best Practices, PEP 8, Type Hints, SOLID")
    print("=" * 78)
    print("""
    COME USARE:
    ───────────
    print(QUIZ_2_1)   → Quiz PEP 8
    print(QUIZ_2_2)   → Quiz Zen of Python
    print(QUIZ_2_3)   → Quiz Docstrings
    print(QUIZ_2_4)   → Quiz Type Hints
    print(QUIZ_2_5)   → Quiz SOLID
    print(QUIZ_2_6)   → Quiz Code Quality Tools
    print(QUIZ_2_7)   → Quiz Project Structure
    print(MODULE_2_FINAL_TEST)    → Test finale
    print(MODULE_2_FINAL_ANSWERS) → Risposte test
    """)
