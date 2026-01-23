"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    150 ESERCIZI OOP PROGRESSIVI                              ║
║                                                                              ║
║                 Da Classi Base a Metaclassi                                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

STRUTTURA:
- Level 1 (1-30): Classi Base
- Level 2 (31-60): Ereditarietà
- Level 3 (61-90): Polimorfismo e Metodi Speciali
- Level 4 (91-120): Design Patterns
- Level 5 (121-150): Metaclassi e Decoratori Avanzati

═══════════════════════════════════════════════════════════════════════════════
"""

# ══════════════════════════════════════════════════════════════════════════════
#                    LEVEL 1: CLASSI BASE (1-30)
# ══════════════════════════════════════════════════════════════════════════════

"""
EXERCISE 1: Simple Class
───────────────────────────────────────────────────────────────────────────────
Crea una classe Person con attributi name e age.

Test:
    p = Person("Alice", 30)
    print(p.name)  # "Alice"
    print(p.age)   # 30
"""

class Person:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age


"""
EXERCISE 2: Class with Method
───────────────────────────────────────────────────────────────────────────────
Aggiungi un metodo introduce() che restituisce "Hi, I'm {name}".
"""

class Person2:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age
    
    def introduce(self) -> str:
        return f"Hi, I'm {self.name}"


"""
EXERCISE 3: Private Attributes
───────────────────────────────────────────────────────────────────────────────
Crea una classe BankAccount con _balance privato e metodi deposit/withdraw.
"""

class BankAccount:
    def __init__(self, initial_balance: float = 0):
        self._balance = initial_balance
    
    def deposit(self, amount: float):
        if amount > 0:
            self._balance += amount
    
    def withdraw(self, amount: float) -> bool:
        if 0 < amount <= self._balance:
            self._balance -= amount
            return True
        return False
    
    def get_balance(self) -> float:
        return self._balance


"""
EXERCISE 4: Property Decorator
───────────────────────────────────────────────────────────────────────────────
Riscrivi BankAccount usando @property per balance.
"""

class BankAccount2:
    def __init__(self, initial_balance: float = 0):
        self._balance = initial_balance
    
    @property
    def balance(self) -> float:
        return self._balance
    
    @balance.setter
    def balance(self, value: float):
        if value >= 0:
            self._balance = value


"""
EXERCISE 5: Class Variable
───────────────────────────────────────────────────────────────────────────────
Crea una classe Counter che conta quante istanze sono state create.
"""

class Counter:
    count = 0
    
    def __init__(self):
        Counter.count += 1
    
    @classmethod
    def get_count(cls) -> int:
        return cls.count


"""
EXERCISE 6: Static Method
───────────────────────────────────────────────────────────────────────────────
Crea una classe MathUtils con metodi statici per operazioni matematiche.
"""

class MathUtils:
    @staticmethod
    def add(a: float, b: float) -> float:
        return a + b
    
    @staticmethod
    def is_even(n: int) -> bool:
        return n % 2 == 0
    
    @staticmethod
    def factorial(n: int) -> int:
        if n <= 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result


"""
EXERCISE 7: Class Method
───────────────────────────────────────────────────────────────────────────────
Crea una classe Date con class method from_string per parsing "YYYY-MM-DD".
"""

class Date:
    def __init__(self, year: int, month: int, day: int):
        self.year = year
        self.month = month
        self.day = day
    
    @classmethod
    def from_string(cls, date_string: str) -> 'Date':
        year, month, day = map(int, date_string.split('-'))
        return cls(year, month, day)
    
    def __str__(self) -> str:
        return f"{self.year:04d}-{self.month:02d}-{self.day:02d}"


"""
EXERCISE 8: Rectangle
───────────────────────────────────────────────────────────────────────────────
Crea una classe Rectangle con area() e perimeter().
"""

class Rectangle:
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
    
    def area(self) -> float:
        return self.width * self.height
    
    def perimeter(self) -> float:
        return 2 * (self.width + self.height)


"""
EXERCISE 9: Circle
───────────────────────────────────────────────────────────────────────────────
Crea una classe Circle con area e circumference (usa math.pi).
"""

import math

class Circle:
    def __init__(self, radius: float):
        self.radius = radius
    
    @property
    def area(self) -> float:
        return math.pi * self.radius ** 2
    
    @property
    def circumference(self) -> float:
        return 2 * math.pi * self.radius


"""
EXERCISE 10: Temperature
───────────────────────────────────────────────────────────────────────────────
Crea una classe Temperature che gestisce conversioni C/F.
"""

class Temperature:
    def __init__(self, celsius: float = 0):
        self._celsius = celsius
    
    @property
    def celsius(self) -> float:
        return self._celsius
    
    @celsius.setter
    def celsius(self, value: float):
        self._celsius = value
    
    @property
    def fahrenheit(self) -> float:
        return self._celsius * 9/5 + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value: float):
        self._celsius = (value - 32) * 5/9


# ══════════════════════════════════════════════════════════════════════════════
#                    LEVEL 2: EREDITARIETÀ (31-60)
# ══════════════════════════════════════════════════════════════════════════════

"""
EXERCISE 31: Basic Inheritance
───────────────────────────────────────────────────────────────────────────────
Crea Animal (base) e Dog/Cat (derivate) con speak().
"""

class Animal:
    def __init__(self, name: str):
        self.name = name
    
    def speak(self) -> str:
        return "Some sound"


class Dog(Animal):
    def speak(self) -> str:
        return f"{self.name} says Woof!"


class Cat(Animal):
    def speak(self) -> str:
        return f"{self.name} says Meow!"


"""
EXERCISE 32: Super() Usage
───────────────────────────────────────────────────────────────────────────────
Crea Employee (base) e Manager (derivata) che estende __init__.
"""

class Employee:
    def __init__(self, name: str, salary: float):
        self.name = name
        self.salary = salary


class Manager(Employee):
    def __init__(self, name: str, salary: float, department: str):
        super().__init__(name, salary)
        self.department = department
        self.team = []
    
    def add_employee(self, employee: Employee):
        self.team.append(employee)


"""
EXERCISE 33: Method Override
───────────────────────────────────────────────────────────────────────────────
Crea Shape (base) e Square/Circle con area() override.
"""

class Shape:
    def area(self) -> float:
        raise NotImplementedError("Subclasses must implement area()")


class Square(Shape):
    def __init__(self, side: float):
        self.side = side
    
    def area(self) -> float:
        return self.side ** 2


class CircleShape(Shape):
    def __init__(self, radius: float):
        self.radius = radius
    
    def area(self) -> float:
        return math.pi * self.radius ** 2


"""
EXERCISE 34: Multiple Inheritance
───────────────────────────────────────────────────────────────────────────────
Crea Flyable e Swimmable mixins, e Duck che eredita entrambi.
"""

class Flyable:
    def fly(self) -> str:
        return f"{self.name} is flying"


class Swimmable:
    def swim(self) -> str:
        return f"{self.name} is swimming"


class Duck(Animal, Flyable, Swimmable):
    def speak(self) -> str:
        return f"{self.name} says Quack!"


"""
EXERCISE 35: Abstract Base Class
───────────────────────────────────────────────────────────────────────────────
Crea una classe astratta Vehicle con metodo astratto start().
"""

from abc import ABC, abstractmethod


class Vehicle(ABC):
    def __init__(self, brand: str):
        self.brand = brand
    
    @abstractmethod
    def start(self) -> str:
        pass


class Car(Vehicle):
    def start(self) -> str:
        return f"{self.brand} car engine starting..."


class Motorcycle(Vehicle):
    def start(self) -> str:
        return f"{self.brand} motorcycle revving..."


# ══════════════════════════════════════════════════════════════════════════════
#                    LEVEL 3: METODI SPECIALI (61-90)
# ══════════════════════════════════════════════════════════════════════════════

"""
EXERCISE 61: __str__ and __repr__
───────────────────────────────────────────────────────────────────────────────
Implementa __str__ e __repr__ per una classe Book.
"""

class Book:
    def __init__(self, title: str, author: str, year: int):
        self.title = title
        self.author = author
        self.year = year
    
    def __str__(self) -> str:
        return f"{self.title} by {self.author} ({self.year})"
    
    def __repr__(self) -> str:
        return f"Book(title='{self.title}', author='{self.author}', year={self.year})"


"""
EXERCISE 62: __eq__ and __hash__
───────────────────────────────────────────────────────────────────────────────
Implementa uguaglianza per una classe Point.
"""

class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Point):
            return NotImplemented
        return self.x == other.x and self.y == other.y
    
    def __hash__(self) -> int:
        return hash((self.x, self.y))


"""
EXERCISE 63: __lt__ and Comparison
───────────────────────────────────────────────────────────────────────────────
Implementa confronti per una classe Student (ordina per grade).
"""

from functools import total_ordering


@total_ordering
class Student:
    def __init__(self, name: str, grade: float):
        self.name = name
        self.grade = grade
    
    def __eq__(self, other) -> bool:
        return self.grade == other.grade
    
    def __lt__(self, other) -> bool:
        return self.grade < other.grade


"""
EXERCISE 64: __add__ and Arithmetic
───────────────────────────────────────────────────────────────────────────────
Implementa operazioni aritmetiche per una classe Vector.
"""

class Vector:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def __add__(self, other: 'Vector') -> 'Vector':
        return Vector(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: 'Vector') -> 'Vector':
        return Vector(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float) -> 'Vector':
        return Vector(self.x * scalar, self.y * scalar)
    
    def __rmul__(self, scalar: float) -> 'Vector':
        return self.__mul__(scalar)
    
    def __repr__(self) -> str:
        return f"Vector({self.x}, {self.y})"


"""
EXERCISE 65: __len__ and __getitem__
───────────────────────────────────────────────────────────────────────────────
Implementa una classe CustomList che supporta len() e indexing.
"""

class CustomList:
    def __init__(self, items=None):
        self._items = list(items) if items else []
    
    def __len__(self) -> int:
        return len(self._items)
    
    def __getitem__(self, index):
        return self._items[index]
    
    def __setitem__(self, index, value):
        self._items[index] = value
    
    def __iter__(self):
        return iter(self._items)
    
    def append(self, item):
        self._items.append(item)


"""
EXERCISE 66: __contains__
───────────────────────────────────────────────────────────────────────────────
Implementa 'in' operator per una classe Range personalizzata.
"""

class MyRange:
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end
    
    def __contains__(self, item: int) -> bool:
        return self.start <= item < self.end
    
    def __iter__(self):
        current = self.start
        while current < self.end:
            yield current
            current += 1


"""
EXERCISE 67: __call__
───────────────────────────────────────────────────────────────────────────────
Crea una classe Multiplier che può essere chiamata come funzione.
"""

class Multiplier:
    def __init__(self, factor: float):
        self.factor = factor
    
    def __call__(self, value: float) -> float:
        return value * self.factor


"""
EXERCISE 68: Context Manager (__enter__/__exit__)
───────────────────────────────────────────────────────────────────────────────
Crea un context manager Timer che misura tempo di esecuzione.
"""

import time


class Timer:
    def __init__(self, name: str = "Timer"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        print(f"{self.name}: {self.elapsed:.4f} seconds")
        return False
    
    @property
    def elapsed(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time


# ══════════════════════════════════════════════════════════════════════════════
#                    LEVEL 4: DESIGN PATTERNS (91-120)
# ══════════════════════════════════════════════════════════════════════════════

"""
EXERCISE 91: Singleton Pattern
───────────────────────────────────────────────────────────────────────────────
Implementa Singleton usando __new__.
"""

class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


"""
EXERCISE 92: Factory Pattern
───────────────────────────────────────────────────────────────────────────────
Implementa una Factory per creare diversi tipi di documenti.
"""

class Document(ABC):
    @abstractmethod
    def create(self) -> str:
        pass


class PDFDocument(Document):
    def create(self) -> str:
        return "Creating PDF document"


class WordDocument(Document):
    def create(self) -> str:
        return "Creating Word document"


class DocumentFactory:
    @staticmethod
    def create_document(doc_type: str) -> Document:
        if doc_type == 'pdf':
            return PDFDocument()
        elif doc_type == 'word':
            return WordDocument()
        else:
            raise ValueError(f"Unknown document type: {doc_type}")


"""
EXERCISE 93: Observer Pattern
───────────────────────────────────────────────────────────────────────────────
Implementa Observer pattern per notifiche.
"""

class Subject:
    def __init__(self):
        self._observers = []
        self._state = None
    
    def attach(self, observer):
        self._observers.append(observer)
    
    def detach(self, observer):
        self._observers.remove(observer)
    
    def notify(self):
        for observer in self._observers:
            observer.update(self._state)
    
    @property
    def state(self):
        return self._state
    
    @state.setter
    def state(self, value):
        self._state = value
        self.notify()


class Observer(ABC):
    @abstractmethod
    def update(self, state):
        pass


class ConcreteObserver(Observer):
    def __init__(self, name: str):
        self.name = name
    
    def update(self, state):
        print(f"{self.name} received update: {state}")


"""
EXERCISE 94: Strategy Pattern
───────────────────────────────────────────────────────────────────────────────
Implementa diverse strategie di sorting.
"""

class SortStrategy(ABC):
    @abstractmethod
    def sort(self, data: list) -> list:
        pass


class BubbleSort(SortStrategy):
    def sort(self, data: list) -> list:
        arr = data.copy()
        n = len(arr)
        for i in range(n):
            for j in range(0, n-i-1):
                if arr[j] > arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
        return arr


class QuickSort(SortStrategy):
    def sort(self, data: list) -> list:
        if len(data) <= 1:
            return data
        pivot = data[len(data) // 2]
        left = [x for x in data if x < pivot]
        middle = [x for x in data if x == pivot]
        right = [x for x in data if x > pivot]
        return self.sort(left) + middle + self.sort(right)


class Sorter:
    def __init__(self, strategy: SortStrategy):
        self._strategy = strategy
    
    def sort(self, data: list) -> list:
        return self._strategy.sort(data)


"""
EXERCISE 95: Decorator Pattern
───────────────────────────────────────────────────────────────────────────────
Implementa decoratori per aggiungere funzionalità a un Coffee.
"""

class Coffee(ABC):
    @abstractmethod
    def cost(self) -> float:
        pass
    
    @abstractmethod
    def description(self) -> str:
        pass


class SimpleCoffee(Coffee):
    def cost(self) -> float:
        return 2.0
    
    def description(self) -> str:
        return "Simple coffee"


class CoffeeDecorator(Coffee):
    def __init__(self, coffee: Coffee):
        self._coffee = coffee


class MilkDecorator(CoffeeDecorator):
    def cost(self) -> float:
        return self._coffee.cost() + 0.5
    
    def description(self) -> str:
        return f"{self._coffee.description()}, milk"


class SugarDecorator(CoffeeDecorator):
    def cost(self) -> float:
        return self._coffee.cost() + 0.2
    
    def description(self) -> str:
        return f"{self._coffee.description()}, sugar"


# ══════════════════════════════════════════════════════════════════════════════
#                    LEVEL 5: METACLASSI E DECORATORI (121-150)
# ══════════════════════════════════════════════════════════════════════════════

"""
EXERCISE 121: Simple Metaclass
───────────────────────────────────────────────────────────────────────────────
Crea una metaclasse che aggiunge un attributo a tutte le classi.
"""

class AutoIdMeta(type):
    _counter = 0
    
    def __new__(mcs, name, bases, attrs):
        attrs['class_id'] = AutoIdMeta._counter
        AutoIdMeta._counter += 1
        return super().__new__(mcs, name, bases, attrs)


class MyClass1(metaclass=AutoIdMeta):
    pass


class MyClass2(metaclass=AutoIdMeta):
    pass


"""
EXERCISE 122: Registry Metaclass
───────────────────────────────────────────────────────────────────────────────
Crea una metaclasse che registra automaticamente tutte le sottoclassi.
"""

class RegistryMeta(type):
    registry = {}
    
    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)
        if name != 'Base':  # Don't register base class
            mcs.registry[name] = cls
        return cls


class PluginBase(metaclass=RegistryMeta):
    pass


class Plugin1(PluginBase):
    pass


class Plugin2(PluginBase):
    pass


"""
EXERCISE 123: Validation Metaclass
───────────────────────────────────────────────────────────────────────────────
Crea una metaclasse che valida che tutte le classi abbiano docstring.
"""

class DocstringMeta(type):
    def __new__(mcs, name, bases, attrs):
        if not attrs.get('__doc__'):
            raise TypeError(f"Class {name} must have a docstring")
        return super().__new__(mcs, name, bases, attrs)


"""
EXERCISE 124: Function Decorator with Arguments
───────────────────────────────────────────────────────────────────────────────
Crea un decoratore retry che riprova n volte in caso di eccezione.
"""

def retry(max_attempts: int = 3, delay: float = 1.0):
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator


"""
EXERCISE 125: Class Decorator
───────────────────────────────────────────────────────────────────────────────
Crea un class decorator che aggiunge logging automatico.
"""

def log_methods(cls):
    """Decorator che logga tutte le chiamate ai metodi."""
    for name, method in cls.__dict__.items():
        if callable(method) and not name.startswith('_'):
            setattr(cls, name, _make_logged(method, name))
    return cls


def _make_logged(method, name):
    def wrapper(self, *args, **kwargs):
        print(f"Calling {name} with args={args}, kwargs={kwargs}")
        result = method(self, *args, **kwargs)
        print(f"{name} returned {result}")
        return result
    return wrapper


"""
EXERCISE 126: Descriptor Protocol
───────────────────────────────────────────────────────────────────────────────
Implementa un descriptor per validazione di tipo.
"""

class TypedAttribute:
    def __init__(self, name: str, expected_type: type):
        self.name = name
        self.expected_type = expected_type
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.name)
    
    def __set__(self, obj, value):
        if not isinstance(value, self.expected_type):
            raise TypeError(
                f"{self.name} must be {self.expected_type.__name__}, "
                f"got {type(value).__name__}"
            )
        obj.__dict__[self.name] = value


class PersonTyped:
    name = TypedAttribute('name', str)
    age = TypedAttribute('age', int)
    
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age


# ══════════════════════════════════════════════════════════════════════════════
#                    TEST RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_tests():
    """Esegui test per verificare le soluzioni."""
    print("=" * 70)
    print("RUNNING OOP EXERCISES TESTS")
    print("=" * 70)
    
    # Level 1
    print("\nLevel 1: Basic Classes")
    p = Person("Alice", 30)
    assert p.name == "Alice"
    assert p.age == 30
    print("  ✓ Exercise 1: Person")
    
    p2 = Person2("Bob", 25)
    assert p2.introduce() == "Hi, I'm Bob"
    print("  ✓ Exercise 2: Person with method")
    
    ba = BankAccount(100)
    ba.deposit(50)
    assert ba.get_balance() == 150
    print("  ✓ Exercise 3: BankAccount")
    
    # Level 2
    print("\nLevel 2: Inheritance")
    dog = Dog("Rex")
    assert dog.speak() == "Rex says Woof!"
    print("  ✓ Exercise 31: Animal inheritance")
    
    mgr = Manager("Alice", 100000, "Engineering")
    mgr.add_employee(Employee("Bob", 50000))
    assert len(mgr.team) == 1
    print("  ✓ Exercise 32: Super() usage")
    
    # Level 3
    print("\nLevel 3: Special Methods")
    book = Book("1984", "Orwell", 1949)
    assert str(book) == "1984 by Orwell (1949)"
    print("  ✓ Exercise 61: __str__ and __repr__")
    
    v1 = Vector(1, 2)
    v2 = Vector(3, 4)
    v3 = v1 + v2
    assert v3.x == 4 and v3.y == 6
    print("  ✓ Exercise 64: Vector arithmetic")
    
    # Level 4
    print("\nLevel 4: Design Patterns")
    s1 = Singleton()
    s2 = Singleton()
    assert s1 is s2
    print("  ✓ Exercise 91: Singleton")
    
    pdf = DocumentFactory.create_document('pdf')
    assert pdf.create() == "Creating PDF document"
    print("  ✓ Exercise 92: Factory")
    
    # Level 5
    print("\nLevel 5: Metaclasses")
    assert MyClass1.class_id == 0
    assert MyClass2.class_id == 1
    print("  ✓ Exercise 121: AutoId Metaclass")
    
    assert 'Plugin1' in RegistryMeta.registry
    assert 'Plugin2' in RegistryMeta.registry
    print("  ✓ Exercise 122: Registry Metaclass")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
