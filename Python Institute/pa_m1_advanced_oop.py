#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PYTHON PROFESSIONAL 1 - MODULE 1                          ║
║                    ADVANCED OOP                                               ║
║                    PCPP1-32-101 Section 1: 35% (14 domande) - BIGGEST!       ║
╚══════════════════════════════════════════════════════════════════════════════╝

SYLLABUS:
├── PCPP1 1.1 - Classes, instances, attributes, methods
├── PCPP1 1.2 - Shallow vs deep copies (copy module)
├── PCPP1 1.3 - Serialization: pickle, shelve
├── PCPP1 1.4 - Decorators, class/static methods, abstract classes
├── PCPP1 1.5 - Metaprogramming, metaclasses
├── PCPP1 1.6 - Exception chaining (__context__, __cause__)
└── PCPP1 1.7 - Subclassing built-in types
"""

import copy
import pickle
import shelve
from abc import ABC, abstractmethod

# ══════════════════════════════════════════════════════════════════════════════
# 1.1 COPY MODULE - SHALLOW VS DEEP (ESAME!)
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("1.1 SHALLOW VS DEEP COPY (ESAME!)")
print("=" * 70)

print("""
📋 SHALLOW COPY (copy.copy)
   - Crea nuovo oggetto container
   - MA gli elementi interni sono RIFERIMENTI agli originali
   - Modifiche agli oggetti nested si riflettono in entrambi

📋 DEEP COPY (copy.deepcopy)
   - Crea nuovo oggetto container
   - E copia RICORSIVAMENTE tutti gli oggetti nested
   - Completamente indipendente dall'originale
""")

# Esempio
original = [[1, 2, 3], [4, 5, 6]]

shallow = copy.copy(original)
deep = copy.deepcopy(original)

print(f"Original: {original}")
print(f"Shallow:  {shallow}")
print(f"Deep:     {deep}")

# Modifica elemento nested
original[0][0] = 999

print(f"\nDopo original[0][0] = 999:")
print(f"Original: {original}")
print(f"Shallow:  {shallow}")  # MODIFICATO!
print(f"Deep:     {deep}")     # INVARIATO!

# Con classi custom
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __repr__(self):
        return f"Point({self.x}, {self.y})"

class Line:
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __repr__(self):
        return f"Line({self.start}, {self.end})"

p1 = Point(0, 0)
p2 = Point(10, 10)
line_orig = Line(p1, p2)

line_shallow = copy.copy(line_orig)
line_deep = copy.deepcopy(line_orig)

print(f"\nline_orig.start is line_shallow.start: {line_orig.start is line_shallow.start}")  # True
print(f"line_orig.start is line_deep.start: {line_orig.start is line_deep.start}")  # False

# ══════════════════════════════════════════════════════════════════════════════
# 1.2 PICKLE SERIALIZATION (ESAME!)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("1.2 PICKLE SERIALIZATION (ESAME!)")
print("=" * 70)

print("""
📋 PICKLE
   - Serializza oggetti Python in formato binario
   - Può serializzare quasi TUTTO (classi, funzioni, etc.)
   - NON sicuro: non caricare pickle da fonti non fidate!

📋 FUNZIONI:
   pickle.dumps(obj)      → bytes (serializza)
   pickle.loads(bytes)    → obj (deserializza)
   pickle.dump(obj, file) → scrive su file
   pickle.load(file)      → legge da file
""")

# Esempio con dumps/loads
data = {
    'name': 'Marco',
    'scores': [95, 87, 92],
    'active': True
}

# Serializza a bytes
serialized = pickle.dumps(data)
print(f"Serialized type: {type(serialized)}")
print(f"Serialized[:50]: {serialized[:50]}")

# Deserializza
restored = pickle.loads(serialized)
print(f"Restored: {restored}")
print(f"data == restored: {data == restored}")

# Con classi custom
class Player:
    def __init__(self, name, level):
        self.name = name
        self.level = level
    def __repr__(self):
        return f"Player('{self.name}', {self.level})"

player = Player("Hero", 42)
player_bytes = pickle.dumps(player)
player_restored = pickle.loads(player_bytes)
print(f"\nOriginal: {player}")
print(f"Restored: {player_restored}")

# Con file
print("""
# Scrittura su file
with open('data.pickle', 'wb') as f:
    pickle.dump(data, f)

# Lettura da file
with open('data.pickle', 'rb') as f:
    loaded = pickle.load(f)
""")

# ══════════════════════════════════════════════════════════════════════════════
# 1.3 SHELVE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("1.3 SHELVE")
print("=" * 70)

print("""
📋 SHELVE
   - Database persistente key-value
   - Chiavi: stringhe
   - Valori: qualsiasi oggetto Python (usa pickle internamente)
   - Si usa come un dizionario!

# Scrittura
with shelve.open('mydata') as db:
    db['user'] = {'name': 'Marco', 'age': 25}
    db['scores'] = [95, 87, 92]

# Lettura
with shelve.open('mydata') as db:
    print(db['user'])
    print(list(db.keys()))

# ⚠️ ATTENZIONE con oggetti mutabili:
with shelve.open('mydata', writeback=True) as db:
    db['scores'].append(100)  # Funziona solo con writeback=True
""")

# ══════════════════════════════════════════════════════════════════════════════
# 1.4 DECORATORS (ESAME!)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("1.4 DECORATORS (ESAME!)")
print("=" * 70)

# Basic decorator
def logger(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"Finished {func.__name__}")
        return result
    return wrapper

@logger
def greet(name):
    return f"Hello, {name}!"

print(greet("Marco"))

# Decorator with arguments
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def say_hello():
    print("Hello!")

print("\n@repeat(3):")
say_hello()

# Preserving metadata with functools.wraps
from functools import wraps

def my_decorator(func):
    @wraps(func)  # Preserva __name__, __doc__
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

# ══════════════════════════════════════════════════════════════════════════════
# 1.5 CLASS AND STATIC METHODS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("1.5 CLASS AND STATIC METHODS")
print("=" * 70)

class MyClass:
    class_var = 0
    
    def __init__(self, value):
        self.instance_var = value
    
    def instance_method(self):
        """Riceve self, accede a instance e class vars"""
        return f"Instance: {self.instance_var}, Class: {self.class_var}"
    
    @classmethod
    def class_method(cls):
        """Riceve cls (la classe), NO accesso a instance vars"""
        return f"Class var: {cls.class_var}"
    
    @staticmethod
    def static_method():
        """NON riceve self o cls, utility function"""
        return "I'm a static method"

obj = MyClass(42)
print(f"instance_method(): {obj.instance_method()}")
print(f"class_method(): MyClass.class_method() = {MyClass.class_method()}")
print(f"static_method(): MyClass.static_method() = {MyClass.static_method()}")

# Factory method pattern
class Date:
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day
    
    @classmethod
    def from_string(cls, date_str):
        """Factory method"""
        year, month, day = map(int, date_str.split('-'))
        return cls(year, month, day)

date = Date.from_string("2025-01-26")
print(f"\nDate.from_string('2025-01-26'): {date.year}/{date.month}/{date.day}")

# ══════════════════════════════════════════════════════════════════════════════
# 1.6 PROPERTY DECORATORS (ESAME!)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("1.6 PROPERTY DECORATORS (ESAME!)")
print("=" * 70)

class Temperature:
    def __init__(self, celsius=0):
        self._celsius = celsius
    
    @property
    def celsius(self):
        """Getter"""
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        """Setter con validazione"""
        if value < -273.15:
            raise ValueError("Temperature below absolute zero!")
        self._celsius = value
    
    @celsius.deleter
    def celsius(self):
        """Deleter"""
        print("Deleting temperature")
        del self._celsius
    
    @property
    def fahrenheit(self):
        """Read-only property (no setter)"""
        return self._celsius * 9/5 + 32

temp = Temperature(25)
print(f"temp.celsius = {temp.celsius}")
print(f"temp.fahrenheit = {temp.fahrenheit}")

temp.celsius = 30  # Chiama setter
print(f"After temp.celsius = 30: {temp.celsius}")

# temp.fahrenheit = 100  # AttributeError - no setter!

# ══════════════════════════════════════════════════════════════════════════════
# 1.7 ABSTRACT CLASSES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("1.7 ABSTRACT CLASSES")
print("=" * 70)

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass
    
    @abstractmethod
    def perimeter(self):
        pass
    
    def description(self):
        """Metodo concreto (non abstract)"""
        return f"I am a {self.__class__.__name__}"

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)

# shape = Shape()  # TypeError - can't instantiate abstract class
rect = Rectangle(4, 5)
print(f"rect.area() = {rect.area()}")
print(f"rect.description() = {rect.description()}")

# ══════════════════════════════════════════════════════════════════════════════
# 1.8 EXCEPTION CHAINING (ESAME!)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("1.8 EXCEPTION CHAINING (ESAME!)")
print("=" * 70)

print("""
📋 EXCEPTION ATTRIBUTES:
   __cause__     → Eccezione esplicita (raise ... from ...)
   __context__   → Eccezione implicita (durante handling)
   __traceback__ → Traceback object
""")

# Explicit chaining (raise from)
def process_data():
    try:
        result = 1 / 0
    except ZeroDivisionError as e:
        raise ValueError("Invalid data") from e

print("Explicit chaining (raise from):")
try:
    process_data()
except ValueError as e:
    print(f"  Caught: {e}")
    print(f"  __cause__: {e.__cause__}")

# Implicit chaining (during except)
def implicit_chain():
    try:
        result = 1 / 0
    except ZeroDivisionError:
        raise KeyError("Something went wrong")  # No 'from'

print("\nImplicit chaining:")
try:
    implicit_chain()
except KeyError as e:
    print(f"  Caught: {e}")
    print(f"  __context__: {e.__context__}")

# Suppressing context (from None)
def suppress_context():
    try:
        result = 1 / 0
    except ZeroDivisionError:
        raise ValueError("Clean error") from None  # Suppresses context

# ══════════════════════════════════════════════════════════════════════════════
# 1.9 SUBCLASSING BUILT-IN TYPES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("1.9 SUBCLASSING BUILT-IN TYPES")
print("=" * 70)

# Subclassing list
class TrackedList(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.access_count = 0
    
    def __getitem__(self, index):
        self.access_count += 1
        return super().__getitem__(index)

tl = TrackedList([1, 2, 3, 4, 5])
print(f"tl[0] = {tl[0]}")
print(f"tl[2] = {tl[2]}")
print(f"Access count: {tl.access_count}")

# Subclassing dict
class DefaultDict(dict):
    def __init__(self, default_factory, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_factory = default_factory
    
    def __missing__(self, key):
        self[key] = self.default_factory()
        return self[key]

dd = DefaultDict(list)
dd['a'].append(1)
dd['a'].append(2)
dd['b'].append(3)
print(f"\nDefaultDict: {dict(dd)}")

# ══════════════════════════════════════════════════════════════════════════════
# 1.10 METACLASSES (ADVANCED)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("1.10 METACLASSES (ADVANCED)")
print("=" * 70)

print("""
📋 METACLASS
   - Classe di una classe
   - Controlla come le classi vengono create
   - type è la metaclass di default
   - MyClass = type('MyClass', (BaseClass,), {'attr': value})
""")

# Custom metaclass
class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    def __init__(self):
        self.connection = "Connected"

db1 = Database()
db2 = Database()
print(f"db1 is db2: {db1 is db2}")  # True - same instance

# ══════════════════════════════════════════════════════════════════════════════
# QUIZ
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("QUIZ DI VERIFICA")
print("=" * 70)

print("""
Q1. copy.copy() crea?
    A) Deep copy  B) Shallow copy  C) Reference  D) Clone
    → RISPOSTA: B

Q2. pickle.dumps() restituisce?
    A) String  B) Dict  C) Bytes  D) File
    → RISPOSTA: C

Q3. @classmethod riceve come primo parametro?
    A) self  B) cls  C) None  D) *args
    → RISPOSTA: B

Q4. @staticmethod riceve?
    A) self  B) cls  C) Nessuno  D) Both
    → RISPOSTA: C

Q5. @property crea?
    A) Attribute  B) Getter  C) Method  D) Variable
    → RISPOSTA: B

Q6. raise ValueError from e imposta?
    A) __context__  B) __cause__  C) __traceback__  D) __error__
    → RISPOSTA: B

Q7. ABC sta per?
    A) Abstract Base Class  B) Any Base Class  C) Another Base Class
    → RISPOSTA: A

Q8. type è?
    A) Built-in function  B) Metaclass  C) Both  D) Neither
    → RISPOSTA: C
""")

print("\n" + "=" * 70)
print("ADVANCED OOP MODULE COMPLETATO!")
print("=" * 70)
