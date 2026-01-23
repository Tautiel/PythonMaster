"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    PYTHON ADVANCED (PA) - MODULE 1                           ║
║           Advanced OOP: Decorators, Metaclasses, ABC, Descriptors            ║
║                                                                              ║
║                     Allineato al Syllabus PCPP1-32-10x                       ║
║                         OpenEDG Python Institute                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

PCPP1 Exam Block 1: Advanced OOP (25% dell'esame)

STRUTTURA MODULO:
├── Section 1.1: Decorators Fundamentals
├── Section 1.2: Advanced Decorators
├── Section 1.3: Class Decorators
├── Section 1.4: Metaclasses Fundamentals
├── Section 1.5: Metaclasses in Practice
├── Section 1.6: Abstract Base Classes (ABC)
├── Section 1.7: Descriptors
├── Section 1.8: __slots__ and Memory Optimization
├── Section 1.9: Method Resolution Order (MRO)
├── Section 1.10: Mixins and Multiple Inheritance
├── Labs (15 esercizi pratici)
└── Module 1 Quiz (40 domande)

TEMPO STIMATO: 8-10 ore

═══════════════════════════════════════════════════════════════════════════════
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 1.1: DECORATORS FUNDAMENTALS
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    1.1 TEORIA: DECORATORS FUNDAMENTALS                       │
└──────────────────────────────────────────────────────────────────────────────┘

COS'È UN DECORATOR?
───────────────────
Un decorator è una funzione che:
1. Prende una funzione come argomento
2. Restituisce una nuova funzione (solitamente modificata)

È un'applicazione del pattern "Higher-Order Functions" (funzioni che
operano su altre funzioni).

CONCETTO CHIAVE: In Python, le funzioni sono FIRST-CLASS OBJECTS.
Questo significa che possono essere:
- Assegnate a variabili
- Passate come argomenti
- Restituite da altre funzioni
"""

# FUNZIONI COME FIRST-CLASS OBJECTS

def greet(name):
    return f"Hello, {name}"

# Assegnare funzione a variabile
say_hello = greet
print(say_hello("Marco"))  # "Hello, Marco"

# Passare funzione come argomento
def call_twice(func, arg):
    func(arg)
    func(arg)

call_twice(print, "Hi!")  # Stampa "Hi!" due volte

# Restituire funzione da funzione
def create_multiplier(n):
    def multiplier(x):
        return x * n
    return multiplier

double = create_multiplier(2)
print(double(5))  # 10


"""
DECORATOR PATTERN BASE:
───────────────────────
"""

def simple_decorator(func):
    """
    Template base di un decorator.
    
    func: la funzione originale da "decorare"
    wrapper: la nuova funzione che sostituisce l'originale
    """
    def wrapper(*args, **kwargs):
        # Codice PRIMA della funzione originale
        print(f"Calling {func.__name__}")
        
        # Chiamata alla funzione originale
        result = func(*args, **kwargs)
        
        # Codice DOPO la funzione originale
        print(f"{func.__name__} returned {result}")
        
        return result
    return wrapper


# Applicazione MANUALE del decorator
def add(a, b):
    return a + b

add = simple_decorator(add)  # Sostituisce add con wrapper
print(add(2, 3))


# Applicazione con SINTASSI @ (syntactic sugar)
@simple_decorator
def multiply(a, b):
    return a * b

# multiply = simple_decorator(multiply)  <-- equivalente
print(multiply(2, 3))


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 1.1                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_1_1 = """
Q1. Un decorator è una funzione che:
    A) Crea una classe
    B) Prende una funzione e restituisce una funzione
    C) Decora l'output sulla console
    D) Modifica le variabili globali

Q2. Cosa significa "@decorator" sopra una funzione?
    A) Un commento speciale
    B) func = decorator(func)
    C) decorator = func(decorator)
    D) Crea una nuova funzione chiamata decorator

Q3. Perché la wrapper function usa *args, **kwargs?
    A) Per questioni di performance
    B) Per accettare qualsiasi numero di argomenti
    C) È obbligatorio per i decorators
    D) Per compatibilità con Python 2

Q4. Cosa stampa questo codice?
    def d(f):
        def w():
            print("A")
            f()
            print("B")
        return w
    
    @d
    def hi():
        print("Hi")
    
    hi()
    
    A) Hi        B) A Hi B      C) A B Hi      D) Error

Q5. Qual è l'output?
    def outer(func):
        def inner():
            return func() * 2
        return inner
    
    @outer
    def get_five():
        return 5
    
    print(get_five())
    
    A) 5         B) 10          C) 25          D) Error
"""

ANSWERS_1_1 = """
RISPOSTE QUIZ 1.1:
Q1: B - Un decorator prende una funzione e restituisce una funzione
Q2: B - @decorator equivale a func = decorator(func)
Q3: B - *args, **kwargs permette alla wrapper di accettare qualsiasi argomento
Q4: B - A Hi B (wrapper stampa A, chiama f(), stampa B)
Q5: B - 10 (get_five restituisce 5, decorator moltiplica per 2)
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 1.2: ADVANCED DECORATORS
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    1.2 TEORIA: ADVANCED DECORATORS                           │
└──────────────────────────────────────────────────────────────────────────────┘

DECORATORS CON ARGOMENTI:
─────────────────────────
Per passare argomenti a un decorator, serve un livello extra di nesting.
"""

def repeat(times):
    """Decorator factory: crea decorators personalizzati."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = None
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)  # repeat(3) restituisce decorator, che poi decora hi
def hi():
    print("Hi!")
    return "done"

hi()  # Stampa "Hi!" tre volte


"""
PRESERVARE I METADATA CON @wraps:
─────────────────────────────────
Senza @wraps, la funzione decorata perde __name__, __doc__, etc.
"""

from functools import wraps

def bad_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def good_decorator(func):
    @wraps(func)  # Copia metadata da func a wrapper
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@bad_decorator
def my_func_bad():
    """Documentazione importante"""
    pass

@good_decorator
def my_func_good():
    """Documentazione importante"""
    pass

print(my_func_bad.__name__)   # 'wrapper' - PERSO!
print(my_func_good.__name__)  # 'my_func_good' - PRESERVATO!
print(my_func_good.__doc__)   # 'Documentazione importante' - PRESERVATO!


"""
STACKING DECORATORS:
────────────────────
Puoi applicare più decorators. Vengono applicati dal basso verso l'alto.
"""

def bold(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return f"<b>{func(*args, **kwargs)}</b>"
    return wrapper

def italic(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return f"<i>{func(*args, **kwargs)}</i>"
    return wrapper

@bold
@italic
def greet(name):
    return f"Hello, {name}"

# Ordine: greet → italic(greet) → bold(italic(greet))
print(greet("Marco"))  # <b><i>Hello, Marco</i></b>


"""
DECORATOR PER TIMING:
─────────────────────
"""

import time

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.4f}s")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(0.1)
    return "done"


"""
DECORATOR PER CACHING (MEMOIZATION):
────────────────────────────────────
"""

def memoize(func):
    cache = {}
    @wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper

@memoize
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Senza memoization: O(2^n)
# Con memoization: O(n)
print(fibonacci(100))  # Velocissimo grazie al caching!


"""
NOTA: Python ha functools.lru_cache per questo!
"""

from functools import lru_cache

@lru_cache(maxsize=None)  # Cache illimitata
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 1.2                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_1_2 = """
Q1. Per creare un decorator con argomenti, quanti livelli di nesting servono?
    A) 1         B) 2           C) 3           D) Nessuno

Q2. Cosa fa @wraps(func)?
    A) Chiama func automaticamente
    B) Copia i metadata di func su wrapper
    C) Rende wrapper più veloce
    D) Crea una copia di func

Q3. Con @a @b @c def f(): ..., quale decorator viene applicato per primo?
    A) a         B) b           C) c           D) Tutti insieme

Q4. lru_cache serve per:
    A) Eliminare funzioni lente
    B) Memorizzare risultati di chiamate precedenti
    C) Limitare gli argomenti
    D) Registrare gli errori

Q5. Qual è l'output?
    def d(times):
        def decorator(f):
            def w():
                return f() * times
            return w
        return decorator
    
    @d(3)
    def five():
        return 5
    
    print(five())
    
    A) 5         B) 15          C) 3           D) Error
"""

ANSWERS_1_2 = """
RISPOSTE QUIZ 1.2:
Q1: C - 3 livelli: decorator_factory → decorator → wrapper
Q2: B - @wraps copia __name__, __doc__, etc. dalla funzione originale
Q3: C - I decorators vengono applicati dal basso verso l'alto
Q4: B - lru_cache memorizza risultati per evitare ricalcoli
Q5: B - 15 (five() restituisce 5, moltiplicato per 3)
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 1.3: CLASS DECORATORS
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    1.3 TEORIA: CLASS DECORATORS                              │
└──────────────────────────────────────────────────────────────────────────────┘

I decorators possono essere applicati anche alle CLASSI!
"""

# DECORATOR COME FUNZIONE CHE MODIFICA UNA CLASSE
def add_method(cls):
    """Aggiunge un metodo alla classe."""
    def say_hi(self):
        return f"Hi from {self.__class__.__name__}"
    cls.say_hi = say_hi
    return cls

@add_method
class MyClass:
    pass

obj = MyClass()
print(obj.say_hi())  # "Hi from MyClass"


# SINGLETON PATTERN CON DECORATOR
def singleton(cls):
    """Assicura che esista una sola istanza della classe."""
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class Database:
    def __init__(self, url):
        print(f"Connecting to {url}")
        self.url = url

db1 = Database("localhost")  # "Connecting to localhost"
db2 = Database("other")      # Niente! Restituisce db1
print(db1 is db2)            # True


# DECORATOR COME CLASSE (con __call__)
class CountCalls:
    """Decorator implementato come classe."""
    def __init__(self, func):
        self.func = func
        self.count = 0
    
    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"Call #{self.count}")
        return self.func(*args, **kwargs)

@CountCalls
def say_hello():
    print("Hello!")

say_hello()  # Call #1, Hello!
say_hello()  # Call #2, Hello!
print(say_hello.count)  # 2


"""
dataclass DECORATOR (Python 3.7+):
──────────────────────────────────
@dataclass genera automaticamente __init__, __repr__, __eq__, etc.
"""

from dataclasses import dataclass, field
from typing import List

@dataclass
class Person:
    name: str
    age: int
    email: str = ""  # Valore default
    
    # Campo calcolato
    def is_adult(self) -> bool:
        return self.age >= 18

p1 = Person("Marco", 30)
p2 = Person("Marco", 30)
print(p1)            # Person(name='Marco', age=30, email='')
print(p1 == p2)      # True (grazie a __eq__ generato)


@dataclass(frozen=True)  # Immutabile
class Point:
    x: float
    y: float

pt = Point(1.0, 2.0)
# pt.x = 3.0  # FrozenInstanceError!


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 1.3                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_1_3 = """
Q1. Un class decorator:
    A) Può solo modificare classi esistenti
    B) Può restituire una nuova classe o modificare quella esistente
    C) Non può aggiungere metodi
    D) Funziona solo su classi built-in

Q2. Per implementare un decorator come classe, quale metodo è necessario?
    A) __init__  B) __call__   C) __new__    D) __del__

Q3. Il decorator @singleton:
    A) Crea infinite istanze
    B) Assicura che esista una sola istanza
    C) Blocca la creazione di istanze
    D) Rende la classe astratta

Q4. @dataclass genera automaticamente:
    A) Solo __init__
    B) __init__, __repr__, __eq__
    C) Solo metodi privati
    D) Nessun metodo

Q5. @dataclass(frozen=True) crea:
    A) Una classe che non può essere importata
    B) Una classe immutabile
    C) Una classe astratta
    D) Una classe senza attributi
"""

ANSWERS_1_3 = """
RISPOSTE QUIZ 1.3:
Q1: B - Un class decorator può restituire una nuova classe o modificare l'esistente
Q2: B - __call__ rende l'istanza del decorator callable
Q3: B - Singleton assicura una sola istanza
Q4: B - @dataclass genera __init__, __repr__, __eq__ e altri
Q5: B - frozen=True crea una classe immutabile (non si possono modificare attributi)
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 1.4: METACLASSES FUNDAMENTALS
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    1.4 TEORIA: METACLASSES FUNDAMENTALS                      │
└──────────────────────────────────────────────────────────────────────────────┘

COS'È UNA METACLASS?
────────────────────
In Python, TUTTO è un oggetto, incluse le classi!

- Un'ISTANZA è creata da una CLASSE
- Una CLASSE è creata da una METACLASS

La metaclass default è `type`.
"""

# Le classi sono oggetti!
class MyClass:
    pass

print(type(MyClass))  # <class 'type'>
print(isinstance(MyClass, type))  # True

# Anche int, str, list sono istanze di type
print(type(int))   # <class 'type'>
print(type(str))   # <class 'type'>


"""
CREARE CLASSI CON type():
─────────────────────────
type() può creare classi dinamicamente:
    type(name, bases, attrs)
"""

# Modo tradizionale
class Dog:
    species = "Canis familiaris"
    def bark(self):
        return "Woof!"

# Modo equivalente con type()
def bark(self):
    return "Woof!"

Dog2 = type(
    'Dog2',                      # Nome della classe
    (),                          # Tuple delle classi base
    {'species': 'Canis familiaris', 'bark': bark}  # Attributi
)

d1 = Dog()
d2 = Dog2()
print(d1.bark())  # Woof!
print(d2.bark())  # Woof!


"""
CREARE UNA METACLASS CUSTOM:
────────────────────────────
Una metaclass è una classe che eredita da `type`.
"""

class MyMeta(type):
    """Metaclass che modifica la creazione delle classi."""
    
    def __new__(mcs, name, bases, attrs):
        """
        Chiamato quando viene CREATA una classe.
        
        mcs: la metaclass stessa
        name: nome della classe
        bases: tuple delle classi base
        attrs: dizionario degli attributi
        """
        print(f"Creating class: {name}")
        
        # Aggiungi un attributo automaticamente
        attrs['_created_by'] = 'MyMeta'
        
        # Chiama il __new__ di type per creare la classe
        return super().__new__(mcs, name, bases, attrs)


class MyClass(metaclass=MyMeta):  # Output: Creating class: MyClass
    pass

print(MyClass._created_by)  # 'MyMeta'


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 1.4                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_1_4 = """
Q1. In Python, le classi sono istanze di:
    A) object   B) type       C) class      D) meta

Q2. type('A', (), {}) crea:
    A) Una stringa
    B) Una tupla
    C) Una classe chiamata 'A'
    D) Un dizionario

Q3. Per creare una metaclass custom, devi ereditare da:
    A) object   B) type       C) meta       D) class

Q4. Il metodo __new__ di una metaclass è chiamato:
    A) Quando crei un'istanza
    B) Quando definisci una classe che usa quella metaclass
    C) Mai automaticamente
    D) Solo in Python 2

Q5. type(int) restituisce:
    A) 'int'    B) int        C) <class 'type'>   D) None
"""

ANSWERS_1_4 = """
RISPOSTE QUIZ 1.4:
Q1: B - Le classi sono istanze di type (la metaclass default)
Q2: C - type(name, bases, attrs) crea una nuova classe
Q3: B - Una metaclass deve ereditare da type
Q4: B - __new__ è chiamato quando si definisce una classe con quella metaclass
Q5: C - type(int) restituisce <class 'type'> perché int è una classe
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 1.5: METACLASSES IN PRACTICE
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    1.5 TEORIA: METACLASSES IN PRACTICE                       │
└──────────────────────────────────────────────────────────────────────────────┘

CASI D'USO DELLE METACLASSES:
─────────────────────────────
1. Registrare automaticamente classi
2. Validare la struttura delle classi
3. Modificare attributi automaticamente
4. Implementare pattern (Singleton, etc.)
"""

# 1. REGISTRY PATTERN
class PluginMeta(type):
    """Metaclass che registra tutte le sottoclassi."""
    registry = {}
    
    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)
        # Non registrare la classe base
        if bases:  # Se ha classi parent
            mcs.registry[name] = cls
        return cls


class Plugin(metaclass=PluginMeta):
    """Classe base per i plugin."""
    pass


class PluginA(Plugin):
    pass


class PluginB(Plugin):
    pass


print(PluginMeta.registry)  # {'PluginA': <class...>, 'PluginB': <class...>}


# 2. VALIDAZIONE
class ValidatedMeta(type):
    """Metaclass che valida che le classi abbiano certi attributi."""
    required = ['name', 'process']
    
    def __new__(mcs, name, bases, attrs):
        # Non validare la classe base
        if bases:
            for attr in mcs.required:
                if attr not in attrs:
                    raise TypeError(f"Class {name} must define '{attr}'")
        return super().__new__(mcs, name, bases, attrs)


class Handler(metaclass=ValidatedMeta):
    pass


class MyHandler(Handler):
    name = "MyHandler"
    
    def process(self):
        pass

# class BadHandler(Handler):  # TypeError: must define 'name' and 'process'
#     pass


# 3. SINGLETON CON METACLASS
class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Config(metaclass=SingletonMeta):
    def __init__(self):
        self.data = {}


c1 = Config()
c2 = Config()
print(c1 is c2)  # True


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 1.5                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_1_5 = """
Q1. Il Registry Pattern con metaclass serve per:
    A) Eliminare classi inutilizzate
    B) Registrare automaticamente tutte le sottoclassi
    C) Bloccare l'ereditarietà
    D) Creare dizionari

Q2. Per implementare Singleton con metaclass, si sovrascrive:
    A) __new__
    B) __init__
    C) __call__
    D) __del__

Q3. Una metaclass può sollevare TypeError per:
    A) Validare che una classe abbia certi attributi
    B) Bloccare l'esecuzione
    C) Creare errori a runtime
    D) Velocizzare il codice

Q4. Quale metodo viene chiamato quando si FA obj = MyClass()?
    A) __new__ della metaclass
    B) __call__ della metaclass
    C) __init__ della metaclass
    D) Nessuno

Q5. PluginMeta.registry contiene:
    A) Tutte le istanze create
    B) Tutte le classi che usano PluginMeta
    C) Solo la classe base Plugin
    D) Un singolo plugin
"""

ANSWERS_1_5 = """
RISPOSTE QUIZ 1.5:
Q1: B - Registry registra automaticamente sottoclassi
Q2: C - __call__ è chiamato quando istanzi (MyClass())
Q3: A - Possiamo validare struttura delle classi
Q4: B - __call__ della metaclass gestisce l'istanziazione
Q5: B - Contiene tutte le classi con quella metaclass (esclusa base)
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 1.6: ABSTRACT BASE CLASSES (ABC)
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    1.6 TEORIA: ABSTRACT BASE CLASSES                         │
└──────────────────────────────────────────────────────────────────────────────┘

ABC = Abstract Base Class
È una classe che non può essere istanziata direttamente e che definisce
un'interfaccia che le sottoclassi DEVONO implementare.
"""

from abc import ABC, abstractmethod


class Animal(ABC):
    """Classe astratta - non può essere istanziata."""
    
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def speak(self):
        """Le sottoclassi DEVONO implementare questo metodo."""
        pass
    
    @abstractmethod
    def move(self):
        """Le sottoclassi DEVONO implementare questo metodo."""
        pass
    
    # Metodi NON astratti possono essere ereditati
    def describe(self):
        return f"I am {self.name}"


# animal = Animal("Test")  # TypeError: Can't instantiate abstract class


class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"
    
    def move(self):
        return f"{self.name} runs"


class Bird(Animal):
    def speak(self):
        return f"{self.name} says Tweet!"
    
    def move(self):
        return f"{self.name} flies"


dog = Dog("Rex")
print(dog.speak())     # "Rex says Woof!"
print(dog.describe())  # "I am Rex" (metodo ereditato)


"""
@abstractmethod CON @property:
──────────────────────────────
Puoi anche definire property astratte!
"""

class Shape(ABC):
    @property
    @abstractmethod
    def area(self):
        """Le sottoclassi devono definire area come property."""
        pass


class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    @property
    def area(self):
        return 3.14159 * self.radius ** 2


"""
VIRTUAL SUBCLASS CON register():
────────────────────────────────
Puoi registrare una classe come "virtuale subclass" senza ereditarietà.
"""

class MyCollection(ABC):
    @abstractmethod
    def __len__(self):
        pass


# Registra list come subclass virtuale
MyCollection.register(list)

print(isinstance([], MyCollection))  # True!


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 1.6                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_1_6 = """
Q1. ABC sta per:
    A) Always Base Class
    B) Abstract Base Class
    C) Automatic Build Class
    D) Advanced Base Code

Q2. Cosa succede se provi a istanziare una classe astratta?
    A) Funziona normalmente
    B) TypeError
    C) None
    D) Crea un'istanza vuota

Q3. @abstractmethod significa:
    A) Il metodo è privato
    B) Le sottoclassi devono implementare questo metodo
    C) Il metodo non può essere chiamato
    D) Il metodo è statico

Q4. Puoi combinare @property e @abstractmethod?
    A) No, mai
    B) Sì, @property deve venire prima
    C) Sì, @abstractmethod deve venire prima
    D) Solo in Python 2

Q5. ABC.register(MyClass) serve per:
    A) Eliminare una classe
    B) Rendere MyClass una virtual subclass
    C) Bloccare l'ereditarietà
    D) Creare un singleton
"""

ANSWERS_1_6 = """
RISPOSTE QUIZ 1.6:
Q1: B - Abstract Base Class
Q2: B - TypeError: Can't instantiate abstract class
Q3: B - Le sottoclassi devono implementare i metodi astratti
Q4: B - @property deve venire prima di @abstractmethod
Q5: B - register() crea una virtual subclass senza vera ereditarietà
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 1.7: DESCRIPTORS
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    1.7 TEORIA: DESCRIPTORS                                   │
└──────────────────────────────────────────────────────────────────────────────┘

Un DESCRIPTOR è un oggetto che definisce come vengono get/set gli attributi.

Il DESCRIPTOR PROTOCOL consiste in:
- __get__(self, obj, objtype=None) -> valore
- __set__(self, obj, value)
- __delete__(self, obj)

Data Descriptor: implementa __set__ e/o __delete__
Non-Data Descriptor: implementa solo __get__
"""

class Validator:
    """Descriptor per validare valori."""
    
    def __init__(self, min_value=None, max_value=None):
        self.min_value = min_value
        self.max_value = max_value
    
    def __set_name__(self, owner, name):
        """Chiamato quando il descriptor è assegnato a un attributo di classe."""
        self.name = name
        self.private_name = f'_{name}'
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.private_name, None)
    
    def __set__(self, obj, value):
        if self.min_value is not None and value < self.min_value:
            raise ValueError(f"{self.name} must be >= {self.min_value}")
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f"{self.name} must be <= {self.max_value}")
        setattr(obj, self.private_name, value)


class Person:
    age = Validator(min_value=0, max_value=150)
    height = Validator(min_value=0, max_value=300)
    
    def __init__(self, name, age, height):
        self.name = name
        self.age = age
        self.height = height


p = Person("Marco", 30, 180)
print(p.age)     # 30
# p.age = -5     # ValueError: age must be >= 0
# p.age = 200    # ValueError: age must be <= 150


"""
PROPERTY È UN DESCRIPTOR!
─────────────────────────
@property è implementato internamente come descriptor.
"""

class Temperature:
    def __init__(self, celsius=0):
        self._celsius = celsius
    
    @property
    def celsius(self):
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Below absolute zero!")
        self._celsius = value
    
    @property
    def fahrenheit(self):
        return self._celsius * 9/5 + 32


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 1.7                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_1_7 = """
Q1. Il Descriptor Protocol include:
    A) Solo __get__
    B) __get__, __set__, __delete__
    C) __init__, __call__
    D) Solo __set__

Q2. Un Data Descriptor deve implementare:
    A) Solo __get__
    B) __set__ e/o __delete__
    C) Tutti e tre i metodi
    D) Nessun metodo

Q3. __set_name__ è chiamato quando:
    A) Si setta un valore
    B) Il descriptor è assegnato come attributo di classe
    C) L'istanza viene creata
    D) Mai automaticamente

Q4. @property internamente è:
    A) Un decorator semplice
    B) Un descriptor
    C) Una metaclass
    D) Una funzione built-in

Q5. In __get__(self, obj, objtype), se obj è None significa:
    A) Errore
    B) L'attributo è stato acceduto dalla classe, non da un'istanza
    C) Il descriptor non esiste
    D) Il valore è None
"""

ANSWERS_1_7 = """
RISPOSTE QUIZ 1.7:
Q1: B - __get__, __set__, __delete__
Q2: B - Data descriptor implementa __set__ e/o __delete__
Q3: B - __set_name__ è chiamato quando il descriptor diventa attributo di classe
Q4: B - property è implementato come descriptor
Q5: B - obj=None significa accesso dalla classe (MyClass.attr) non istanza
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 1.8: __slots__ AND MEMORY OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    1.8 TEORIA: __slots__                                     │
└──────────────────────────────────────────────────────────────────────────────┘

Normalmente, gli attributi di un'istanza sono memorizzati in __dict__.
__slots__ permette di:
1. Risparmiare memoria (niente __dict__)
2. Velocizzare l'accesso agli attributi
3. Prevenire la creazione di attributi arbitrari
"""

# SENZA __slots__
class PointNormal:
    def __init__(self, x, y):
        self.x = x
        self.y = y

p = PointNormal(1, 2)
print(p.__dict__)  # {'x': 1, 'y': 2}
p.z = 3  # Posso aggiungere attributi liberamente


# CON __slots__
class PointSlots:
    __slots__ = ('x', 'y')  # Solo questi attributi sono permessi
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

ps = PointSlots(1, 2)
# print(ps.__dict__)  # AttributeError: no __dict__!
# ps.z = 3  # AttributeError: 'PointSlots' object has no attribute 'z'


"""
RISPARMIO MEMORIA:
──────────────────
Per migliaia di istanze, __slots__ può risparmiare molta memoria.
"""

import sys

class WithDict:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class WithSlots:
    __slots__ = ('x', 'y', 'z')
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

obj_dict = WithDict(1, 2, 3)
obj_slots = WithSlots(1, 2, 3)

# La differenza è significativa con molte istanze
print(f"WithDict size: ~{sys.getsizeof(obj_dict) + sys.getsizeof(obj_dict.__dict__)} bytes")


"""
EREDITARIETÀ CON __slots__:
───────────────────────────
Attenzione: se la classe parent non ha __slots__, il child avrà __dict__.
"""

class Parent:
    __slots__ = ('x',)

class Child(Parent):
    __slots__ = ('y',)  # Aggiungi solo i NUOVI attributi

c = Child()
c.x = 1  # OK
c.y = 2  # OK
# c.z = 3  # AttributeError


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 1.8                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_1_8 = """
Q1. __slots__ serve per:
    A) Creare più attributi
    B) Limitare gli attributi permessi e risparmiare memoria
    C) Velocizzare i metodi
    D) Bloccare l'ereditarietà

Q2. Una classe con __slots__ ha __dict__?
    A) Sì, sempre
    B) No, a meno che non includa '__dict__' in __slots__
    C) Dipende dalla versione Python
    D) Solo se eredita da object

Q3. Posso aggiungere attributi arbitrari a una classe con __slots__?
    A) Sì, liberamente
    B) No, solo quelli in __slots__
    C) Solo se sono privati
    D) Solo a runtime

Q4. In ereditarietà, il child deve definire in __slots__:
    A) Tutti gli attributi (anche del parent)
    B) Solo i nuovi attributi
    C) Nessun attributo
    D) Solo attributi privati

Q5. Il vantaggio principale di __slots__ è:
    A) Codice più leggibile
    B) Risparmio memoria con molte istanze
    C) Compatibilità Python 2
    D) Supporto multithread
"""

ANSWERS_1_8 = """
RISPOSTE QUIZ 1.8:
Q1: B - __slots__ limita attributi e risparmia memoria
Q2: B - No __dict__ a meno che sia esplicitamente in __slots__
Q3: B - Solo gli attributi elencati in __slots__
Q4: B - Solo i nuovi attributi, non quelli del parent
Q5: B - Risparmio memoria significativo con molte istanze
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 1.9: METHOD RESOLUTION ORDER (MRO)
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    1.9 TEORIA: METHOD RESOLUTION ORDER                       │
└──────────────────────────────────────────────────────────────────────────────┘

MRO = l'ordine in cui Python cerca metodi/attributi nelle classi parent.
Python usa l'algoritmo C3 Linearization.
"""

class A:
    def method(self):
        return "A"

class B(A):
    def method(self):
        return "B"

class C(A):
    def method(self):
        return "C"

class D(B, C):  # Diamond inheritance
    pass

d = D()
print(d.method())  # "B"

# Vedi MRO
print(D.__mro__)
# (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)

# Oppure
print(D.mro())


"""
REGOLE C3 LINEARIZATION:
────────────────────────
1. Una classe viene prima dei suoi parent
2. L'ordine dei parent è preservato
3. Parent comuni appaiono dopo tutti i loro figli
"""

class X: pass
class Y: pass
class Z: pass
class A(X, Y): pass
class B(Y, Z): pass
class M(A, B): pass

print(M.mro())
# [M, A, X, B, Y, Z, object]


"""
super() E MRO:
──────────────
super() non chiama necessariamente il parent diretto!
Segue MRO.
"""

class Base:
    def __init__(self):
        print("Base.__init__")

class A(Base):
    def __init__(self):
        print("A.__init__")
        super().__init__()  # Chiama il prossimo in MRO

class B(Base):
    def __init__(self):
        print("B.__init__")
        super().__init__()

class C(A, B):
    def __init__(self):
        print("C.__init__")
        super().__init__()

# MRO: C -> A -> B -> Base -> object
c = C()
# Output:
# C.__init__
# A.__init__
# B.__init__  <-- A.super() chiama B, non Base!
# Base.__init__


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 1.9                                          │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_1_9 = """
Q1. MRO sta per:
    A) Method Return Order
    B) Method Resolution Order
    C) Module Resolution Order
    D) Main Return Object

Q2. Per vedere MRO di una classe:
    A) print(cls.mro())
    B) print(cls.__order__)
    C) print(mro(cls))
    D) print(cls.resolution())

Q3. class D(B, C): ... - quale viene cercato prima, B o C?
    A) C     B) B     C) Casuale     D) Dipende da object

Q4. super() chiama:
    A) Sempre il parent diretto
    B) Il prossimo nella MRO
    C) Sempre object
    D) Tutti i parent

Q5. In diamond inheritance A→B→D, A→C→D, MRO di D è:
    A) D, A, B, C, object
    B) D, B, C, A, object
    C) D, C, B, A, object
    D) D, A, object
"""

ANSWERS_1_9 = """
RISPOSTE QUIZ 1.9:
Q1: B - Method Resolution Order
Q2: A - cls.mro() o cls.__mro__
Q3: B - L'ordine segue la definizione: prima B, poi C
Q4: B - super() chiama il prossimo nella MRO, non necessariamente il parent diretto
Q5: B - D, B, C, A, object (C3 linearization)
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 1.10: MIXINS AND MULTIPLE INHERITANCE
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    1.10 TEORIA: MIXINS                                       │
└──────────────────────────────────────────────────────────────────────────────┘

Un MIXIN è una classe che fornisce funzionalità aggiuntive
ma non è pensata per essere istanziata da sola.
"""

class JSONMixin:
    """Mixin che aggiunge serializzazione JSON."""
    def to_json(self):
        import json
        return json.dumps(self.__dict__)
    
    @classmethod
    def from_json(cls, json_str):
        import json
        data = json.loads(json_str)
        return cls(**data)


class LoggingMixin:
    """Mixin che aggiunge logging."""
    def log(self, message):
        print(f"[{self.__class__.__name__}] {message}")


class Person(JSONMixin, LoggingMixin):
    def __init__(self, name, age):
        self.name = name
        self.age = age


p = Person("Marco", 30)
print(p.to_json())  # {"name": "Marco", "age": 30}
p.log("Created!")   # [Person] Created!


"""
CONVENZIONI MIXIN:
──────────────────
1. Nome che finisce in 'Mixin'
2. Non ha __init__ (o chiama super().__init__)
3. Non ha stato proprio
4. Fornisce metodi utili
"""


"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    QUIZ SECTION 1.10                                         │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_1_10 = """
Q1. Un mixin è:
    A) Una classe base obbligatoria
    B) Una classe che aggiunge funzionalità senza essere istanziata da sola
    C) Un tipo di decorator
    D) Una metaclass

Q2. I mixin dovrebbero avere __init__?
    A) Sì, obbligatorio
    B) No, o deve chiamare super().__init__()
    C) Solo se privati
    D) Solo in Python 2

Q3. class MyClass(A, MixinB, MixinC): ... - i mixin sono:
    A) A
    B) MixinB e MixinC
    C) Tutti
    D) Nessuno

Q4. Un mixin dovrebbe avere stato proprio?
    A) Sì, sempre
    B) No, fornisce solo metodi
    C) Solo attributi di classe
    D) Solo se serializzabile

Q5. JSONMixin.to_json() usa tipicamente:
    A) self.data
    B) self.__dict__
    C) cls.__dict__
    D) __slots__
"""

ANSWERS_1_10 = """
RISPOSTE QUIZ 1.10:
Q1: B - Un mixin aggiunge funzionalità senza essere istanziato da solo
Q2: B - No __init__, o deve chiamare super().__init__() per cooperare
Q3: B - MixinB e MixinC sono i mixin (convenzione nome + scopo)
Q4: B - I mixin non dovrebbero avere stato proprio
Q5: B - self.__dict__ contiene gli attributi dell'istanza
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    LABS - ESERCIZI PRATICI
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    LABS: 15 ESERCIZI PRATICI                                 │
└──────────────────────────────────────────────────────────────────────────────┘
"""

LABS = """
═══════════════════════════════════════════════════════════════════════════════
                              LABS MODULE 1
═══════════════════════════════════════════════════════════════════════════════

LAB 1: Crea un decorator @validate_positive che verifica che tutti gli
       argomenti numerici siano positivi.

LAB 2: Crea un decorator @retry(max_attempts=3) che riprova una funzione
       se solleva un'eccezione.

LAB 3: Crea un decorator @deprecated(message) che stampa un warning
       quando la funzione viene chiamata.

LAB 4: Crea una metaclass AutoRepr che genera automaticamente __repr__
       per tutte le classi che la usano.

LAB 5: Crea una classe astratta Shape con metodi astratti area() e 
       perimeter(). Implementa Rectangle, Circle, Triangle.

LAB 6: Crea un descriptor Typed che valida il tipo di un attributo.

LAB 7: Crea una classe Point con __slots__ e metodi __add__, __sub__, __eq__.

LAB 8: Crea un sistema di plugin con metaclass Registry.

LAB 9: Implementa una classe Configuration come Singleton.

LAB 10: Crea mixin SerializableMixin e ComparableMixin.

LAB 11: Crea un decorator @trace che stampa chiamate e ritorni.

LAB 12: Crea una metaclass che forza tutti i metodi ad avere docstring.

LAB 13: Implementa un descriptor CachedProperty che calcola una volta sola.

LAB 14: Crea una gerarchia di classi per un sistema bancario con ABC.

LAB 15: Implementa un decorator di classe @total_ordering semplificato.
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    MODULE 1 FINAL TEST
# ══════════════════════════════════════════════════════════════════════════════

MODULE_1_FINAL_TEST = """
═══════════════════════════════════════════════════════════════════════════════
                    PA MODULE 1 - TEST FINALE (40 domande)
                             Tempo: 45 minuti
═══════════════════════════════════════════════════════════════════════════════

SEZIONE A: DECORATORS (10 domande)
──────────────────────────────────

Q1. @decorator equivale a:
    A) decorator(func)    B) func(decorator)    C) func = decorator(func)    D) Error

Q2. @wraps(func) preserva:
    A) Il codice    B) I metadata (__name__, __doc__)    C) Gli argomenti    D) Niente

Q3. Per un decorator con argomenti servono:
    A) 1 livello    B) 2 livelli    C) 3 livelli    D) 0 livelli

Q4. lru_cache è un esempio di:
    A) Singleton    B) Memoization    C) Factory    D) Observer

Q5. @a @b def f(): ... - ordine di applicazione:
    A) a poi b    B) b poi a    C) Simultaneo    D) Random

Q6. Un decorator può restituire:
    A) Solo funzioni    B) Qualsiasi callable    C) Solo None    D) Solo stringhe

Q7. functools.wraps è:
    A) Un decorator    B) Una funzione    C) Una classe    D) Un modulo

Q8. __call__ rende un oggetto:
    A) Iterabile    B) Callable    C) Hashable    D) Comparable

Q9. @dataclass genera:
    A) Solo __init__    B) __init__, __repr__, __eq__    C) Niente    D) Solo __str__

Q10. @dataclass(frozen=True) crea:
     A) Una classe veloce    B) Una classe immutabile    C) Una classe vuota    D) Un errore


SEZIONE B: METACLASSES (10 domande)
───────────────────────────────────

Q11. type(MyClass) per una classe normale restituisce:
     A) 'MyClass'    B) MyClass    C) <class 'type'>    D) object

Q12. type('A', (B,), {'x': 1}) crea:
     A) Una stringa    B) Una classe A che eredita da B    C) Un dizionario    D) Error

Q13. Una metaclass deve ereditare da:
     A) object    B) type    C) ABC    D) class

Q14. __new__ di una metaclass è chiamato:
     A) Per ogni istanza    B) Quando si definisce una classe    C) Mai    D) Manualmente

Q15. __call__ di una metaclass è chiamato:
     A) Quando si definisce la classe    B) Quando si istanzia    C) Mai    D) Per ogni metodo

Q16. Il Registry Pattern con metaclass:
     A) Elimina classi    B) Registra sottoclassi    C) Crea singleton    D) Valida tipi

Q17. SingletonMeta sovrascrive:
     A) __new__    B) __init__    C) __call__    D) __del__

Q18. Per validare struttura classi, la metaclass:
     A) Solleva TypeError in __new__    B) Modifica __init__    C) Blocca import    D) Niente

Q19. PluginMeta.registry contiene:
     A) Istanze    B) Classi    C) Funzioni    D) Moduli

Q20. __init_subclass__ è:
     A) Un decorator    B) Un hook per sottoclassi    C) Una metaclass    D) Un metodo statico


SEZIONE C: ABC E DESCRIPTORS (10 domande)
─────────────────────────────────────────

Q21. ABC sta per:
     A) Always Base Class    B) Abstract Base Class    C) Auto Build Class    D) Any Base Code

Q22. @abstractmethod significa:
     A) Metodo privato    B) Metodo da implementare    C) Metodo statico    D) Metodo finale

Q23. Istanziare una ABC con metodi astratti:
     A) Funziona    B) TypeError    C) None    D) Crea istanza vuota

Q24. ABC.register(cls) crea:
     A) Una copia    B) Una virtual subclass    C) Un singleton    D) Un errore

Q25. Descriptor Protocol include:
     A) __get__ solo    B) __get__, __set__, __delete__    C) __call__    D) __init__

Q26. Data Descriptor implementa:
     A) Solo __get__    B) __set__ e/o __delete__    C) Solo __init__    D) Niente

Q27. __set_name__(self, owner, name) - owner è:
     A) L'istanza    B) La classe    C) Il descriptor    D) Il valore

Q28. @property è implementato come:
     A) Decorator    B) Metaclass    C) Descriptor    D) ABC

Q29. In __get__(self, obj, objtype), obj=None significa:
     A) Errore    B) Accesso dalla classe    C) Valore None    D) Descriptor non valido

Q30. Validator descriptor serve per:
     A) Creare classi    B) Validare valori di attributi    C) Logging    D) Caching


SEZIONE D: __slots__, MRO, MIXINS (10 domande)
──────────────────────────────────────────────

Q31. __slots__ serve per:
     A) Più attributi    B) Limitare attributi e risparmiare memoria    C) Velocità    D) Debug

Q32. Classe con __slots__ ha __dict__?
     A) Sì    B) No (a meno di '__dict__' in slots)    C) Dipende    D) Solo se eredita

Q33. MRO sta per:
     A) Method Return Order    B) Method Resolution Order    C) Main Resource Object    D) Module Run Order

Q34. cls.mro() restituisce:
     A) Stringa    B) Lista di classi    C) Dizionario    D) Tupla

Q35. super() chiama:
     A) Sempre parent diretto    B) Prossimo in MRO    C) Sempre object    D) Tutti i parent

Q36. Diamond inheritance A→B→D, A→C→D - MRO di D:
     A) D, A, B, C, object    B) D, B, C, A, object    C) D, C, B, A, object    D) Errore

Q37. Un mixin dovrebbe:
     A) Avere __init__ complesso    B) Fornire metodi senza stato    C) Essere astratto    D) Avere slots

Q38. class X(A, MixinB): - ordine consigliato:
     A) Mixin prima    B) Classe principale prima    C) Casuale    D) Solo un parent

Q39. JSONMixin.to_json() tipicamente usa:
     A) cls.data    B) self.__dict__    C) __slots__    D) globals()

Q40. Multiple inheritance in Python:
     A) Non supportata    B) Supportata con MRO C3    C) Solo 2 parent    D) Solo con metaclass


═══════════════════════════════════════════════════════════════════════════════
                              FINE TEST
                     Calcola punteggio prima delle risposte!
═══════════════════════════════════════════════════════════════════════════════
"""

MODULE_1_FINAL_ANSWERS = """
═══════════════════════════════════════════════════════════════════════════════
                    PA MODULE 1 - RISPOSTE TEST FINALE
═══════════════════════════════════════════════════════════════════════════════

SEZIONE A: DECORATORS
Q1: C    Q2: B    Q3: C    Q4: B    Q5: B
Q6: B    Q7: A    Q8: B    Q9: B    Q10: B

SEZIONE B: METACLASSES
Q11: C    Q12: B    Q13: B    Q14: B    Q15: B
Q16: B    Q17: C    Q18: A    Q19: B    Q20: B

SEZIONE C: ABC E DESCRIPTORS
Q21: B    Q22: B    Q23: B    Q24: B    Q25: B
Q26: B    Q27: B    Q28: C    Q29: B    Q30: B

SEZIONE D: __slots__, MRO, MIXINS
Q31: B    Q32: B    Q33: B    Q34: B    Q35: B
Q36: B    Q37: B    Q38: B    Q39: B    Q40: B


PUNTEGGIO:
──────────
36-40: Eccellente! Pronto per PCPP1
32-35: Ottimo! Rivedi gli errori
28-31: Buono! Target 70% raggiunto
<28:   Riguarda le sezioni con più errori


SEZIONI DA RIVEDERE:
────────────────────
A (Q1-10):  Decorators
B (Q11-20): Metaclasses
C (Q21-30): ABC e Descriptors
D (Q31-40): __slots__, MRO, Mixins
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    ESECUZIONE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 78)
    print("PYTHON ADVANCED - MODULE 1")
    print("Advanced OOP: Decorators, Metaclasses, ABC, Descriptors")
    print("=" * 78)
    print("""
    
    CONTENUTO DEL MODULO:
    ─────────────────────
    10 Sezioni di teoria con quiz
    15 Labs pratici
    Test finale (40 domande)
    
    COME USARE:
    ───────────
    print(QUIZ_1_1)   → Quiz sezione 1.1
    print(ANSWERS_1_1) → Risposte
    print(LABS)        → Esercizi pratici
    print(MODULE_1_FINAL_TEST)    → Test finale
    print(MODULE_1_FINAL_ANSWERS) → Risposte test
    
    """)
