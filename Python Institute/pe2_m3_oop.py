#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PYTHON ESSENTIALS 2 - MODULE 3                            ║
║                    OBJECT-ORIENTED PROGRAMMING                                ║
║                    PCAP-31-03 Section 4: 34% (12 domande) - BIGGEST!         ║
╚══════════════════════════════════════════════════════════════════════════════╝

SYLLABUS:
├── PCAP 4.1 - OOP concepts: class, object, property, method, encapsulation
├── PCAP 4.2 - Instance vs class variables, __dict__, name mangling
├── PCAP 4.3 - Methods and self parameter
├── PCAP 4.4 - Introspection: hasattr(), __name__, __module__, __bases__
├── PCAP 4.5 - Inheritance: single, multiple, isinstance(), MRO, polymorphism
└── PCAP 4.6 - Constructors (__init__)
"""

# ══════════════════════════════════════════════════════════════════════════════
# 4.1 OOP BASIC CONCEPTS
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("4.1 OOP CONCEPTS")
print("=" * 70)

print("""
📋 TERMINOLOGIA FONDAMENTALE:

CLASS (Classe)
   - Blueprint/template per creare oggetti
   - Definisce attributi e metodi
   
OBJECT (Oggetto/Istanza)
   - Istanza concreta di una classe
   - Creato dalla classe
   
ATTRIBUTE (Attributo)
   - Variabile appartenente a oggetto o classe
   - Instance variable: appartiene all'oggetto
   - Class variable: condivisa tra tutte le istanze

METHOD (Metodo)
   - Funzione definita dentro una classe
   - Primo parametro è sempre 'self'

ENCAPSULATION (Incapsulamento)
   - Nascondere i dettagli interni
   - Esporre solo l'interfaccia necessaria

INHERITANCE (Ereditarietà)
   - Una classe può ereditare da un'altra
   - Subclass eredita attributi/metodi della superclass
""")

# ══════════════════════════════════════════════════════════════════════════════
# 4.2 DEFINING CLASSES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("4.2 DEFINING CLASSES")
print("=" * 70)

class Dog:
    # Class variable (condivisa)
    species = "Canis familiaris"
    count = 0
    
    # Constructor
    def __init__(self, name, age):
        # Instance variables (uniche per oggetto)
        self.name = name
        self.age = age
        Dog.count += 1
    
    # Instance method
    def bark(self):
        return f"{self.name} says Woof!"
    
    # Method with parameters
    def birthday(self):
        self.age += 1
        return f"{self.name} is now {self.age}"

# Creating objects
dog1 = Dog("Buddy", 3)
dog2 = Dog("Max", 5)

print(f"dog1.name = {dog1.name}")
print(f"dog2.name = {dog2.name}")
print(f"dog1.bark() = {dog1.bark()}")
print(f"Dog.species = {Dog.species}")
print(f"dog1.species = {dog1.species}")  # Accesso via istanza
print(f"Dog.count = {Dog.count}")

# ══════════════════════════════════════════════════════════════════════════════
# 4.3 INSTANCE vs CLASS VARIABLES (ESAME!)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("4.3 INSTANCE vs CLASS VARIABLES (ESAME!)")
print("=" * 70)

class Counter:
    # Class variable
    total = 0
    
    def __init__(self, start):
        # Instance variable
        self.value = start
        Counter.total += 1

c1 = Counter(10)
c2 = Counter(20)

print(f"c1.value = {c1.value}")     # 10 (instance)
print(f"c2.value = {c2.value}")     # 20 (instance)
print(f"Counter.total = {Counter.total}")  # 2 (class - condivisa!)

# ⚠️ ATTENZIONE: Shadowing!
print("\n⚠️ SHADOWING:")
c1.total = 100  # Crea instance variable che "shadowa" class variable!
print(f"c1.total = {c1.total}")       # 100 (instance!)
print(f"c2.total = {c2.total}")       # 2 (class)
print(f"Counter.total = {Counter.total}")  # 2 (class - non modificata!)

# ══════════════════════════════════════════════════════════════════════════════
# 4.4 THE __dict__ ATTRIBUTE (ESAME!)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("4.4 THE __dict__ ATTRIBUTE (ESAME!)")
print("=" * 70)

class Person:
    species = "Human"
    
    def __init__(self, name, age):
        self.name = name
        self.age = age

p = Person("Marco", 25)

# __dict__ dell'OGGETTO (solo instance variables)
print(f"p.__dict__ = {p.__dict__}")
# {'name': 'Marco', 'age': 25}

# __dict__ della CLASSE (class variables e metodi)
print(f"\nPerson.__dict__.keys() = {list(Person.__dict__.keys())}")
# ['__module__', '__dict__', '__weakref__', '__doc__', 'species', '__init__']

# Aggiungere attributi dinamicamente
p.email = "marco@email.com"
print(f"\nDopo p.email = ...: p.__dict__ = {p.__dict__}")

# ══════════════════════════════════════════════════════════════════════════════
# 4.5 NAME MANGLING (__private)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("4.5 NAME MANGLING (__private)")
print("=" * 70)

class BankAccount:
    def __init__(self, balance):
        self.__balance = balance  # Private (name mangling)
    
    def get_balance(self):
        return self.__balance

acc = BankAccount(1000)
print(f"acc.get_balance() = {acc.get_balance()}")

# acc.__balance  # AttributeError!
# Ma accessibile tramite name mangling:
print(f"acc._BankAccount__balance = {acc._BankAccount__balance}")

print("""
📋 CONVENZIONI NAMING:
   name      → Public
   _name     → Protected (convenzione, accessibile)
   __name    → Private (name mangling → _ClassName__name)
   __name__  → Special/Magic (es. __init__, __str__)
""")

# ══════════════════════════════════════════════════════════════════════════════
# 4.6 THE self PARAMETER
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("4.6 THE self PARAMETER")
print("=" * 70)

print("""
📋 self:
   - Primo parametro di ogni instance method
   - Riferimento all'istanza corrente
   - NON è una keyword, ma convenzione fortissima
   - Python lo passa automaticamente
""")

class Example:
    def method(self):
        print(f"self = {self}")
        print(f"type(self) = {type(self)}")

e = Example()
e.method()

# Equivalente a:
Example.method(e)

# ══════════════════════════════════════════════════════════════════════════════
# 4.7 INTROSPECTION (ESAME!)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("4.7 INTROSPECTION (ESAME!)")
print("=" * 70)

class Animal:
    pass

class Dog(Animal):
    pass

d = Dog()

# hasattr() - Verifica se attributo esiste
print(f"hasattr(d, 'bark') = {hasattr(d, 'bark')}")

# setattr() - Imposta attributo
setattr(d, 'name', 'Buddy')
print(f"Dopo setattr: d.name = {d.name}")

# getattr() - Ottiene attributo
print(f"getattr(d, 'name') = {getattr(d, 'name')}")
print(f"getattr(d, 'age', 'N/A') = {getattr(d, 'age', 'N/A')}")  # Default

# delattr() - Elimina attributo
delattr(d, 'name')
print(f"Dopo delattr: hasattr(d, 'name') = {hasattr(d, 'name')}")

# Special attributes
print(f"\nDog.__name__ = {Dog.__name__}")        # 'Dog'
print(f"Dog.__module__ = {Dog.__module__}")    # '__main__'
print(f"Dog.__bases__ = {Dog.__bases__}")      # (<class 'Animal'>,)

# ══════════════════════════════════════════════════════════════════════════════
# 4.8 INHERITANCE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("4.8 INHERITANCE")
print("=" * 70)

# Single inheritance
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        return "Some sound"

class Dog(Animal):
    def speak(self):  # Override
        return "Woof!"
    
    def fetch(self):  # Nuovo metodo
        return f"{self.name} fetches!"

class Cat(Animal):
    def speak(self):
        return "Meow!"

dog = Dog("Buddy")
cat = Cat("Whiskers")

print(f"dog.speak() = {dog.speak()}")
print(f"cat.speak() = {cat.speak()}")
print(f"dog.fetch() = {dog.fetch()}")

# isinstance() - Verifica tipo
print(f"\nisinstance(dog, Dog) = {isinstance(dog, Dog)}")      # True
print(f"isinstance(dog, Animal) = {isinstance(dog, Animal)}")  # True
print(f"isinstance(dog, Cat) = {isinstance(dog, Cat)}")        # False

# issubclass() - Verifica gerarchia classi
print(f"\nissubclass(Dog, Animal) = {issubclass(Dog, Animal)}")  # True
print(f"issubclass(Dog, Cat) = {issubclass(Dog, Cat)}")          # False

# ══════════════════════════════════════════════════════════════════════════════
# 4.9 MULTIPLE INHERITANCE & MRO
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("4.9 MULTIPLE INHERITANCE & MRO")
print("=" * 70)

class A:
    def method(self):
        return "A"

class B(A):
    def method(self):
        return "B"

class C(A):
    def method(self):
        return "C"

class D(B, C):  # Multiple inheritance
    pass

d = D()
print(f"d.method() = {d.method()}")  # 'B' - primo nella MRO

# MRO - Method Resolution Order
print(f"\nD.__mro__ = {D.__mro__}")
# (D, B, C, A, object)

print(f"D.mro() = {D.mro()}")

# ══════════════════════════════════════════════════════════════════════════════
# 4.10 super()
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("4.10 super()")
print("=" * 70)

class Parent:
    def __init__(self, name):
        self.name = name
    
    def greet(self):
        return f"Hello, I'm {self.name}"

class Child(Parent):
    def __init__(self, name, age):
        super().__init__(name)  # Chiama Parent.__init__
        self.age = age
    
    def greet(self):
        parent_greet = super().greet()  # Chiama Parent.greet
        return f"{parent_greet}, age {self.age}"

c = Child("Marco", 25)
print(f"c.greet() = {c.greet()}")

# ══════════════════════════════════════════════════════════════════════════════
# 4.11 POLYMORPHISM
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("4.11 POLYMORPHISM")
print("=" * 70)

class Shape:
    def area(self):
        raise NotImplementedError

class Rectangle(Shape):
    def __init__(self, w, h):
        self.w = w
        self.h = h
    def area(self):
        return self.w * self.h

class Circle(Shape):
    def __init__(self, r):
        self.r = r
    def area(self):
        return 3.14159 * self.r ** 2

# Polymorphism - stesso metodo, comportamento diverso
shapes = [Rectangle(4, 5), Circle(3)]
for shape in shapes:
    print(f"{type(shape).__name__}.area() = {shape.area():.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# 4.12 OVERRIDING __str__
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("4.12 OVERRIDING __str__")
print("=" * 70)

class Product:
    def __init__(self, name, price):
        self.name = name
        self.price = price
    
    def __str__(self):
        return f"Product({self.name}, ${self.price})"
    
    def __repr__(self):
        return f"Product('{self.name}', {self.price})"

p = Product("Laptop", 999)
print(f"print(p) = {p}")              # Usa __str__
print(f"repr(p) = {repr(p)}")         # Usa __repr__

# ══════════════════════════════════════════════════════════════════════════════
# QUIZ
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("QUIZ DI VERIFICA")
print("=" * 70)

print("""
Q1. Quale variabile è condivisa tra tutte le istanze?
    A) Instance variable  B) Class variable  C) Local variable  D) Global
    → RISPOSTA: B

Q2. obj.__dict__ contiene?
    A) Class variables  B) Instance variables  C) Methods  D) All
    → RISPOSTA: B

Q3. __name diventa (name mangling in class MyClass)?
    A) __name  B) _name  C) _MyClass__name  D) MyClass__name
    → RISPOSTA: C

Q4. isinstance(dog, Animal) con Dog(Animal) restituisce?
    A) True  B) False  C) Error  D) None
    → RISPOSTA: A

Q5. Quale attributo mostra la gerarchia delle classi parent?
    A) __dict__  B) __bases__  C) __class__  D) __mro__
    → RISPOSTA: B (direct parents) o D (full order)

Q6. MRO di D(B, C) con B(A), C(A)?
    A) D, A, B, C  B) D, B, C, A  C) D, C, B, A  D) Error
    → RISPOSTA: B

Q7. super() in Child.__init__ chiama?
    A) object.__init__  B) Parent.__init__  C) Child.__init__  D) Error
    → RISPOSTA: B

Q8. Quale metodo è chiamato da print(obj)?
    A) __repr__  B) __str__  C) __print__  D) __display__
    → RISPOSTA: B
""")

print("\n" + "=" * 70)
print("MODULO 3 COMPLETATO! → Prossimo: pe2_m4_generators_files.py")
print("=" * 70)
