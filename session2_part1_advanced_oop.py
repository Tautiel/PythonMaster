"""
🚀 SESSIONE 2 - PARTE 1: ADVANCED OOP & DESIGN PATTERNS
========================================================
Super Intensive Python Master Course
Durata: 90 minuti di OOP avanzato
"""

import abc
import typing
from typing import Any, List, Dict, Optional, Protocol, TypeVar, Generic
from dataclasses import dataclass, field
from functools import singledispatch, wraps
from collections import defaultdict
import inspect
import weakref

print("="*80)
print("🎯 SESSIONE 2: ADVANCED OOP & DESIGN PATTERNS")
print("="*80)

# ==============================================================================
# SEZIONE 1: INHERITANCE & MRO
# ==============================================================================

def section1_inheritance_mro():
    """Inheritance avanzata e Method Resolution Order"""
    
    print("\n" + "="*60)
    print("🧬 SEZIONE 1: INHERITANCE & MRO")
    print("="*60)
    
    # 1.1 MULTIPLE INHERITANCE
    print("\n🔀 1.1 MULTIPLE INHERITANCE")
    print("-"*40)
    
    class A:
        def method(self):
            print("A.method")
            
    class B:
        def method(self):
            print("B.method")
            
    class C(A, B):  # Ordine importante!
        pass
    
    class D(B, A):  # Ordine diverso
        pass
    
    c = C()
    d = D()
    
    print("class C(A, B):")
    c.method()  # Chiama A.method
    
    print("\nclass D(B, A):")
    d.method()  # Chiama B.method
    
    # 1.2 METHOD RESOLUTION ORDER (MRO)
    print("\n📜 1.2 METHOD RESOLUTION ORDER")
    print("-"*40)
    
    class Animal:
        def speak(self):
            print("Animal speaks")
    
    class Mammal(Animal):
        def speak(self):
            print("Mammal speaks")
            super().speak()
    
    class Bird(Animal):
        def speak(self):
            print("Bird speaks")
            super().speak()
    
    class Bat(Mammal, Bird):  # Un pipistrello!
        def speak(self):
            print("Bat speaks")
            super().speak()
    
    bat = Bat()
    
    print("MRO di Bat:")
    for cls in Bat.__mro__:
        print(f"  → {cls.__name__}")
    
    print("\nChiamata bat.speak():")
    bat.speak()
    
    # 1.3 DIAMOND PROBLEM
    print("\n💎 1.3 DIAMOND PROBLEM")
    print("-"*40)
    
    class Top:
        def __init__(self):
            print("Top.__init__")
            
    class Left(Top):
        def __init__(self):
            super().__init__()
            print("Left.__init__")
            
    class Right(Top):
        def __init__(self):
            super().__init__()
            print("Right.__init__")
            
    class Bottom(Left, Right):
        def __init__(self):
            super().__init__()
            print("Bottom.__init__")
    
    print("Creazione Bottom (Diamond):")
    bottom = Bottom()
    print(f"\nMRO: {[c.__name__ for c in Bottom.__mro__]}")
    
    # 1.4 SUPER() AVANZATO
    print("\n⚡ 1.4 SUPER() COOPERATIVO")
    print("-"*40)
    
    class Base:
        def __init__(self, value):
            self.value = value
            print(f"Base.__init__({value})")
    
    class A(Base):
        def __init__(self, value, a_param):
            super().__init__(value)
            self.a_param = a_param
            print(f"A.__init__({a_param})")
    
    class B(Base):
        def __init__(self, value, b_param):
            super().__init__(value)
            self.b_param = b_param
            print(f"B.__init__({b_param})")
    
    class C(A, B):
        def __init__(self, value, a_param, b_param, c_param):
            # Super() cooperativo con kwargs
            super().__init__(value, a_param)  
            # Nota: B.__init__ non viene chiamato automaticamente!
            B.__init__(self, value, b_param)
            self.c_param = c_param
            print(f"C.__init__({c_param})")
    
    print("Inizializzazione complessa:")
    c = C(100, "a", "b", "c")

# ==============================================================================
# SEZIONE 2: ABSTRACT CLASSES & PROTOCOLS
# ==============================================================================

def section2_abstract_protocols():
    """Abstract Base Classes e Protocols"""
    
    print("\n" + "="*60)
    print("🎨 SEZIONE 2: ABSTRACT CLASSES & PROTOCOLS")
    print("="*60)
    
    # 2.1 ABSTRACT BASE CLASSES
    print("\n🏗️ 2.1 ABSTRACT BASE CLASSES")
    print("-"*40)
    
    from abc import ABC, abstractmethod
    
    class Vehicle(ABC):
        """Classe astratta per veicoli"""
        
        def __init__(self, brand: str, model: str):
            self.brand = brand
            self.model = model
        
        @abstractmethod
        def start_engine(self):
            """Deve essere implementato"""
            pass
        
        @abstractmethod
        def stop_engine(self):
            """Deve essere implementato"""
            pass
        
        @property
        @abstractmethod
        def max_speed(self) -> float:
            """Proprietà astratta"""
            pass
        
        def honk(self):
            """Metodo concreto condiviso"""
            print("Beep beep!")
    
    class Car(Vehicle):
        """Implementazione concreta"""
        
        def __init__(self, brand: str, model: str, horsepower: int):
            super().__init__(brand, model)
            self.horsepower = horsepower
            self._engine_on = False
        
        def start_engine(self):
            self._engine_on = True
            print(f"{self.brand} {self.model} engine started!")
        
        def stop_engine(self):
            self._engine_on = False
            print(f"{self.brand} {self.model} engine stopped!")
        
        @property
        def max_speed(self) -> float:
            return self.horsepower * 1.5
    
    # Test
    car = Car("Tesla", "Model S", 400)
    car.start_engine()
    print(f"Max speed: {car.max_speed} km/h")
    car.honk()
    car.stop_engine()
    
    # Non puoi istanziare Vehicle direttamente!
    try:
        vehicle = Vehicle("Generic", "Model")
    except TypeError as e:
        print(f"\n❌ Cannot instantiate abstract class: {e}")
    
    # 2.2 PROTOCOLS (Structural Subtyping)
    print("\n🔌 2.2 PROTOCOLS (Duck Typing Formale)")
    print("-"*40)
    
    from typing import Protocol
    
    class Drawable(Protocol):
        """Protocol per oggetti disegnabili"""
        def draw(self) -> str:
            ...
    
    class Resizable(Protocol):
        """Protocol per oggetti ridimensionabili"""
        def resize(self, factor: float) -> None:
            ...
    
    # Non serve ereditare!
    class Circle:
        def __init__(self, radius: float):
            self.radius = radius
        
        def draw(self) -> str:
            return f"○ (r={self.radius})"
        
        def resize(self, factor: float) -> None:
            self.radius *= factor
    
    class Square:
        def __init__(self, side: float):
            self.side = side
        
        def draw(self) -> str:
            return f"□ (s={self.side})"
        
        def resize(self, factor: float) -> None:
            self.side *= factor
    
    def render(shape: Drawable) -> None:
        """Accetta qualsiasi Drawable"""
        print(f"Drawing: {shape.draw()}")
    
    def scale(shape: Resizable, factor: float) -> None:
        """Accetta qualsiasi Resizable"""
        shape.resize(factor)
        print(f"Resized by {factor}x")
    
    # Test
    circle = Circle(5)
    square = Square(10)
    
    render(circle)  # OK - ha metodo draw()
    render(square)  # OK - ha metodo draw()
    
    scale(circle, 2)  # OK - ha metodo resize()
    scale(square, 0.5)  # OK - ha metodo resize()
    
    # 2.3 ABSTRACT PROPERTIES E METODI
    print("\n🏠 2.3 ABSTRACT PROPERTIES")
    print("-"*40)
    
    class DataProcessor(ABC):
        """Processor astratto con properties"""
        
        @property
        @abstractmethod
        def input_format(self) -> str:
            """Formato input accettato"""
            pass
        
        @property
        @abstractmethod
        def output_format(self) -> str:
            """Formato output prodotto"""
            pass
        
        @abstractmethod
        def process(self, data: Any) -> Any:
            """Processa i dati"""
            pass
        
        @classmethod
        @abstractmethod
        def from_config(cls, config: Dict):
            """Factory method astratto"""
            pass
        
        @staticmethod
        @abstractmethod
        def validate(data: Any) -> bool:
            """Validazione statica"""
            pass
    
    class JSONProcessor(DataProcessor):
        """Implementazione per JSON"""
        
        @property
        def input_format(self) -> str:
            return "JSON"
        
        @property
        def output_format(self) -> str:
            return "JSON"
        
        def process(self, data: Any) -> Any:
            import json
            if isinstance(data, str):
                return json.loads(data)
            return json.dumps(data)
        
        @classmethod
        def from_config(cls, config: Dict):
            return cls()
        
        @staticmethod
        def validate(data: Any) -> bool:
            import json
            try:
                if isinstance(data, str):
                    json.loads(data)
                else:
                    json.dumps(data)
                return True
            except:
                return False
    
    processor = JSONProcessor()
    print(f"Processor: {processor.input_format} → {processor.output_format}")
    result = processor.process('{"key": "value"}')
    print(f"Processed: {result}")

# ==============================================================================
# SEZIONE 3: DESIGN PATTERNS
# ==============================================================================

def section3_design_patterns():
    """Design Patterns principali in Python"""
    
    print("\n" + "="*60)
    print("🎨 SEZIONE 3: DESIGN PATTERNS")
    print("="*60)
    
    # 3.1 SINGLETON PATTERN
    print("\n🔐 3.1 SINGLETON PATTERN")
    print("-"*40)
    
    class SingletonMeta(type):
        """Metaclass per Singleton"""
        _instances = {}
        
        def __call__(cls, *args, **kwargs):
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
            return cls._instances[cls]
    
    class DatabaseConnection(metaclass=SingletonMeta):
        """Solo una connessione DB"""
        def __init__(self):
            print("Creating database connection...")
            self.connection = "Connected to DB"
        
        def query(self, sql: str):
            return f"Executing: {sql}"
    
    # Test Singleton
    db1 = DatabaseConnection()
    db2 = DatabaseConnection()
    print(f"db1 is db2? {db1 is db2}")  # True!
    
    # 3.2 FACTORY PATTERN
    print("\n🏭 3.2 FACTORY PATTERN")
    print("-"*40)
    
    class Animal(ABC):
        @abstractmethod
        def speak(self) -> str:
            pass
    
    class Dog(Animal):
        def speak(self) -> str:
            return "Woof!"
    
    class Cat(Animal):
        def speak(self) -> str:
            return "Meow!"
    
    class AnimalFactory:
        """Factory per creare animali"""
        
        @staticmethod
        def create_animal(animal_type: str) -> Animal:
            animals = {
                "dog": Dog,
                "cat": Cat
            }
            
            animal_class = animals.get(animal_type.lower())
            if not animal_class:
                raise ValueError(f"Unknown animal type: {animal_type}")
            
            return animal_class()
    
    # Test Factory
    dog = AnimalFactory.create_animal("dog")
    cat = AnimalFactory.create_animal("cat")
    print(f"Dog says: {dog.speak()}")
    print(f"Cat says: {cat.speak()}")
    
    # 3.3 OBSERVER PATTERN
    print("\n👁️ 3.3 OBSERVER PATTERN")
    print("-"*40)
    
    class Subject:
        """Soggetto osservabile"""
        def __init__(self):
            self._observers: List = []
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
    
    # Test Observer
    subject = Subject()
    obs1 = ConcreteObserver("Observer1")
    obs2 = ConcreteObserver("Observer2")
    
    subject.attach(obs1)
    subject.attach(obs2)
    
    subject.state = "NEW_STATE"
    
    # 3.4 DECORATOR PATTERN (non il Python decorator)
    print("\n🎁 3.4 DECORATOR PATTERN")
    print("-"*40)
    
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
        
        def cost(self) -> float:
            return self._coffee.cost()
        
        def description(self) -> str:
            return self._coffee.description()
    
    class MilkDecorator(CoffeeDecorator):
        def cost(self) -> float:
            return self._coffee.cost() + 0.5
        
        def description(self) -> str:
            return f"{self._coffee.description()} + milk"
    
    class SugarDecorator(CoffeeDecorator):
        def cost(self) -> float:
            return self._coffee.cost() + 0.2
        
        def description(self) -> str:
            return f"{self._coffee.description()} + sugar"
    
    # Test Decorator Pattern
    coffee = SimpleCoffee()
    print(f"{coffee.description()}: ${coffee.cost()}")
    
    coffee_with_milk = MilkDecorator(coffee)
    print(f"{coffee_with_milk.description()}: ${coffee_with_milk.cost()}")
    
    coffee_full = SugarDecorator(MilkDecorator(coffee))
    print(f"{coffee_full.description()}: ${coffee_full.cost()}")
    
    # 3.5 STRATEGY PATTERN
    print("\n🎯 3.5 STRATEGY PATTERN")
    print("-"*40)
    
    class SortStrategy(ABC):
        @abstractmethod
        def sort(self, data: List) -> List:
            pass
    
    class BubbleSort(SortStrategy):
        def sort(self, data: List) -> List:
            result = data.copy()
            n = len(result)
            for i in range(n):
                for j in range(0, n-i-1):
                    if result[j] > result[j+1]:
                        result[j], result[j+1] = result[j+1], result[j]
            return result
    
    class QuickSort(SortStrategy):
        def sort(self, data: List) -> List:
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
        
        @property
        def strategy(self) -> SortStrategy:
            return self._strategy
        
        @strategy.setter
        def strategy(self, strategy: SortStrategy):
            self._strategy = strategy
        
        def sort(self, data: List) -> List:
            return self._strategy.sort(data)
    
    # Test Strategy
    data = [3, 1, 4, 1, 5, 9, 2, 6]
    
    sorter = Sorter(BubbleSort())
    print(f"BubbleSort: {sorter.sort(data)}")
    
    sorter.strategy = QuickSort()
    print(f"QuickSort: {sorter.sort(data)}")

# ==============================================================================
# SEZIONE 4: ADVANCED OOP FEATURES
# ==============================================================================

def section4_advanced_features():
    """Features OOP avanzate di Python"""
    
    print("\n" + "="*60)
    print("🚀 SEZIONE 4: ADVANCED OOP FEATURES")
    print("="*60)
    
    # 4.1 MIXINS
    print("\n🧩 4.1 MIXINS")
    print("-"*40)
    
    class TimestampMixin:
        """Aggiunge timestamp automatici"""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            from datetime import datetime
            self.created_at = datetime.now()
            self.updated_at = datetime.now()
        
        def touch(self):
            from datetime import datetime
            self.updated_at = datetime.now()
    
    class SerializableMixin:
        """Aggiunge serializzazione JSON"""
        def to_dict(self):
            return {
                key: value for key, value in self.__dict__.items()
                if not key.startswith('_')
            }
        
        def to_json(self):
            import json
            from datetime import datetime
            
            def json_encoder(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return str(obj)
            
            return json.dumps(self.to_dict(), default=json_encoder)
    
    class User(TimestampMixin, SerializableMixin):
        def __init__(self, name: str, email: str):
            super().__init__()
            self.name = name
            self.email = email
    
    # Test Mixins
    user = User("Alice", "alice@example.com")
    print(f"User created at: {user.created_at}")
    print(f"User as JSON: {user.to_json()}")
    
    # 4.2 DESCRIPTORS AVANZATI
    print("\n📝 4.2 DESCRIPTORS AVANZATI")
    print("-"*40)
    
    class TypedProperty:
        """Descriptor con type checking"""
        def __init__(self, expected_type, default=None):
            self.expected_type = expected_type
            self.default = default
            self.data = weakref.WeakKeyDictionary()
        
        def __get__(self, obj, objtype=None):
            if obj is None:
                return self
            return self.data.get(obj, self.default)
        
        def __set__(self, obj, value):
            if not isinstance(value, self.expected_type):
                raise TypeError(f"Expected {self.expected_type.__name__}")
            self.data[obj] = value
        
        def __delete__(self, obj):
            del self.data[obj]
    
    class Product:
        name = TypedProperty(str)
        price = TypedProperty(float, 0.0)
        quantity = TypedProperty(int, 0)
        
        def __init__(self, name: str, price: float, quantity: int):
            self.name = name
            self.price = price
            self.quantity = quantity
    
    # Test Descriptors
    product = Product("Laptop", 999.99, 10)
    print(f"Product: {product.name}, ${product.price}, qty: {product.quantity}")
    
    try:
        product.price = "invalid"
    except TypeError as e:
        print(f"❌ Type error: {e}")
    
    # 4.3 OPERATOR OVERLOADING
    print("\n➕ 4.3 OPERATOR OVERLOADING")
    print("-"*40)
    
    class Vector:
        """Vector con operatori personalizzati"""
        def __init__(self, x: float, y: float):
            self.x = x
            self.y = y
        
        def __repr__(self):
            return f"Vector({self.x}, {self.y})"
        
        def __add__(self, other):
            return Vector(self.x + other.x, self.y + other.y)
        
        def __sub__(self, other):
            return Vector(self.x - other.x, self.y - other.y)
        
        def __mul__(self, scalar):
            return Vector(self.x * scalar, self.y * scalar)
        
        def __rmul__(self, scalar):
            return self.__mul__(scalar)
        
        def __eq__(self, other):
            return self.x == other.x and self.y == other.y
        
        def __abs__(self):
            return (self.x ** 2 + self.y ** 2) ** 0.5
        
        def __bool__(self):
            return bool(abs(self))
    
    # Test operators
    v1 = Vector(3, 4)
    v2 = Vector(1, 2)
    
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print(f"v1 + v2 = {v1 + v2}")
    print(f"v1 - v2 = {v1 - v2}")
    print(f"v1 * 2 = {v1 * 2}")
    print(f"3 * v1 = {3 * v1}")
    print(f"|v1| = {abs(v1)}")
    
    # 4.4 CONTEXT MANAGERS AS CLASSES
    print("\n🔒 4.4 CONTEXT MANAGERS AS CLASSES")
    print("-"*40)
    
    class DatabaseTransaction:
        """Context manager per transazioni"""
        def __init__(self, connection):
            self.connection = connection
            self.transaction = None
        
        def __enter__(self):
            self.transaction = "BEGIN TRANSACTION"
            print(self.transaction)
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type:
                print("ROLLBACK")
                return False  # Propaga eccezione
            else:
                print("COMMIT")
                return True
        
        def execute(self, query: str):
            print(f"  Executing: {query}")
    
    # Test context manager
    with DatabaseTransaction("db_connection") as tx:
        tx.execute("INSERT INTO users VALUES (...)")
        tx.execute("UPDATE products SET ...")

# ==============================================================================
# SEZIONE 5: GENERICS E TYPE HINTS AVANZATI
# ==============================================================================

def section5_generics():
    """Generics e Type Hints avanzati"""
    
    print("\n" + "="*60)
    print("🧬 SEZIONE 5: GENERICS E TYPE HINTS")
    print("="*60)
    
    # 5.1 GENERIC CLASSES
    print("\n📦 5.1 GENERIC CLASSES")
    print("-"*40)
    
    T = TypeVar('T')
    
    class Stack(Generic[T]):
        """Stack generico type-safe"""
        def __init__(self):
            self._items: List[T] = []
        
        def push(self, item: T) -> None:
            self._items.append(item)
        
        def pop(self) -> T:
            if not self._items:
                raise IndexError("Stack is empty")
            return self._items.pop()
        
        def peek(self) -> Optional[T]:
            return self._items[-1] if self._items else None
        
        def __len__(self) -> int:
            return len(self._items)
        
        def __repr__(self) -> str:
            return f"Stack({self._items})"
    
    # Test Generic Stack
    int_stack: Stack[int] = Stack()
    int_stack.push(1)
    int_stack.push(2)
    int_stack.push(3)
    print(f"Int stack: {int_stack}")
    print(f"Pop: {int_stack.pop()}")
    
    str_stack: Stack[str] = Stack()
    str_stack.push("hello")
    str_stack.push("world")
    print(f"String stack: {str_stack}")
    
    # 5.2 BOUNDED TYPE VARIABLES
    print("\n🔗 5.2 BOUNDED TYPE VARIABLES")
    print("-"*40)
    
    from typing import TypeVar, Protocol
    
    class Comparable(Protocol):
        def __lt__(self, other) -> bool: ...
        def __le__(self, other) -> bool: ...
    
    C = TypeVar('C', bound=Comparable)
    
    def find_max(items: List[C]) -> Optional[C]:
        """Trova il massimo in una lista di Comparable"""
        if not items:
            return None
        
        max_item = items[0]
        for item in items[1:]:
            if item > max_item:
                max_item = item
        
        return max_item
    
    # Test bounded types
    numbers = [3, 1, 4, 1, 5, 9]
    words = ["python", "java", "rust", "go"]
    
    print(f"Max number: {find_max(numbers)}")
    print(f"Max word: {find_max(words)}")
    
    # 5.3 GENERIC FUNCTIONS
    print("\n🔧 5.3 GENERIC FUNCTIONS")
    print("-"*40)
    
    K = TypeVar('K')
    V = TypeVar('V')
    
    def reverse_dict(d: Dict[K, V]) -> Dict[V, K]:
        """Inverte chiavi e valori di un dizionario"""
        return {v: k for k, v in d.items()}
    
    def merge_dicts(d1: Dict[K, V], d2: Dict[K, V]) -> Dict[K, V]:
        """Unisce due dizionari dello stesso tipo"""
        result = d1.copy()
        result.update(d2)
        return result
    
    # Test generic functions
    original = {"a": 1, "b": 2, "c": 3}
    reversed_dict = reverse_dict(original)
    print(f"Original: {original}")
    print(f"Reversed: {reversed_dict}")
    
    dict1 = {"x": 10, "y": 20}
    dict2 = {"z": 30, "w": 40}
    merged = merge_dicts(dict1, dict2)
    print(f"Merged: {merged}")

# ==============================================================================
# MAIN - Menu per le sezioni
# ==============================================================================

def main():
    """Menu principale per OOP avanzato"""
    
    print("\n" + "="*60)
    print("🎓 ADVANCED OOP - SCEGLI SEZIONE")
    print("="*60)
    
    sections = [
        ("Inheritance & MRO", section1_inheritance_mro),
        ("Abstract Classes & Protocols", section2_abstract_protocols),
        ("Design Patterns", section3_design_patterns),
        ("Advanced OOP Features", section4_advanced_features),
        ("Generics & Type Hints", section5_generics)
    ]
    
    print("\n0. Esegui TUTTO")
    for i, (name, _) in enumerate(sections, 1):
        print(f"{i}. {name}")
    
    choice = input("\nScegli (0-5): ")
    
    try:
        choice = int(choice)
        if choice == 0:
            for name, func in sections:
                input(f"\n➡️ Press ENTER for: {name}")
                func()
        elif 1 <= choice <= len(sections):
            sections[choice-1][1]()
        else:
            print("Scelta non valida")
    except (ValueError, IndexError):
        print("Scelta non valida")
    
    print("\n" + "="*60)
    print("✅ PARTE 1 COMPLETATA!")
    print("Prossimo: session2_part2_concurrency.py")
    print("="*60)

if __name__ == "__main__":
    main()
