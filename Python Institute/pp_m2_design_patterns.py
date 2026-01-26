#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PYTHON PROFESSIONAL 2 - MODULE 2                          ║
║                    DESIGN PATTERNS                                            ║
║                    PCPP2 Prep - Design Patterns expected ~20% of exam        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import copy

# ══════════════════════════════════════════════════════════════════════════════
# 2.1 CREATIONAL PATTERNS
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("2.1 CREATIONAL PATTERNS")
print("=" * 70)

# SINGLETON
print("\n📋 SINGLETON - Only one instance exists")

class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

s1 = Singleton()
s2 = Singleton()
print(f"Singleton: s1 is s2 = {s1 is s2}")  # True

# FACTORY METHOD
print("\n📋 FACTORY METHOD - Creates objects without specifying exact class")

class Animal(ABC):
    @abstractmethod
    def speak(self): pass

class Dog(Animal):
    def speak(self): return "Woof!"

class Cat(Animal):
    def speak(self): return "Meow!"

class AnimalFactory:
    @staticmethod
    def create(animal_type: str) -> Animal:
        if animal_type == "dog":
            return Dog()
        elif animal_type == "cat":
            return Cat()
        raise ValueError(f"Unknown: {animal_type}")

dog = AnimalFactory.create("dog")
print(f"Factory: dog.speak() = {dog.speak()}")

# ABSTRACT FACTORY
print("\n📋 ABSTRACT FACTORY - Creates families of related objects")

class GUIFactory(ABC):
    @abstractmethod
    def create_button(self): pass
    @abstractmethod
    def create_checkbox(self): pass

class WindowsFactory(GUIFactory):
    def create_button(self): return "Windows Button"
    def create_checkbox(self): return "Windows Checkbox"

class MacFactory(GUIFactory):
    def create_button(self): return "Mac Button"
    def create_checkbox(self): return "Mac Checkbox"

# BUILDER
print("\n📋 BUILDER - Constructs complex objects step by step")

class Pizza:
    def __init__(self):
        self.dough = None
        self.sauce = None
        self.toppings = []
    def __str__(self):
        return f"Pizza({self.dough}, {self.sauce}, {self.toppings})"

class PizzaBuilder:
    def __init__(self):
        self.pizza = Pizza()
    def set_dough(self, dough):
        self.pizza.dough = dough
        return self
    def set_sauce(self, sauce):
        self.pizza.sauce = sauce
        return self
    def add_topping(self, topping):
        self.pizza.toppings.append(topping)
        return self
    def build(self):
        return self.pizza

pizza = (PizzaBuilder()
         .set_dough("thin")
         .set_sauce("tomato")
         .add_topping("cheese")
         .add_topping("mushrooms")
         .build())
print(f"Builder: {pizza}")

# PROTOTYPE
print("\n📋 PROTOTYPE - Clone existing objects")

class Prototype:
    def clone(self):
        return copy.deepcopy(self)

class Document(Prototype):
    def __init__(self, content):
        self.content = content

doc1 = Document("Hello")
doc2 = doc1.clone()
doc2.content = "World"
print(f"Prototype: doc1={doc1.content}, doc2={doc2.content}")

# ══════════════════════════════════════════════════════════════════════════════
# 2.2 STRUCTURAL PATTERNS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("2.2 STRUCTURAL PATTERNS")
print("=" * 70)

# ADAPTER
print("\n📋 ADAPTER - Converts interface to another")

class OldPrinter:
    def print_old(self, text):
        return f"[OLD] {text}"

class PrinterAdapter:
    def __init__(self, old_printer):
        self.old = old_printer
    def print(self, text):
        return self.old.print_old(text)

adapter = PrinterAdapter(OldPrinter())
print(f"Adapter: {adapter.print('Hello')}")

# DECORATOR (Pattern, not Python decorator)
print("\n📋 DECORATOR PATTERN - Adds behavior dynamically")

class Coffee(ABC):
    @abstractmethod
    def cost(self): pass

class SimpleCoffee(Coffee):
    def cost(self): return 2.0

class MilkDecorator(Coffee):
    def __init__(self, coffee):
        self.coffee = coffee
    def cost(self):
        return self.coffee.cost() + 0.5

class SugarDecorator(Coffee):
    def __init__(self, coffee):
        self.coffee = coffee
    def cost(self):
        return self.coffee.cost() + 0.2

coffee = SugarDecorator(MilkDecorator(SimpleCoffee()))
print(f"Decorator: coffee.cost() = {coffee.cost()}")

# FACADE
print("\n📋 FACADE - Simplified interface to complex subsystem")

class SubsystemA:
    def operation_a(self): return "A"

class SubsystemB:
    def operation_b(self): return "B"

class Facade:
    def __init__(self):
        self.a = SubsystemA()
        self.b = SubsystemB()
    def operation(self):
        return f"{self.a.operation_a()}{self.b.operation_b()}"

print(f"Facade: {Facade().operation()}")

# PROXY
print("\n📋 PROXY - Controls access to object")

class RealSubject:
    def request(self):
        return "Real response"

class Proxy:
    def __init__(self):
        self._real = None
    def request(self):
        if self._real is None:
            print("  Proxy: Creating real subject (lazy)")
            self._real = RealSubject()
        return self._real.request()

proxy = Proxy()
print(f"Proxy: {proxy.request()}")

# ══════════════════════════════════════════════════════════════════════════════
# 2.3 BEHAVIORAL PATTERNS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("2.3 BEHAVIORAL PATTERNS")
print("=" * 70)

# OBSERVER
print("\n📋 OBSERVER - Notify dependents of state changes")

class Subject:
    def __init__(self):
        self._observers = []
        self._state = None
    def attach(self, observer):
        self._observers.append(observer)
    def notify(self):
        for obs in self._observers:
            obs.update(self._state)
    def set_state(self, state):
        self._state = state
        self.notify()

class Observer:
    def __init__(self, name):
        self.name = name
    def update(self, state):
        print(f"  {self.name} received: {state}")

subject = Subject()
subject.attach(Observer("A"))
subject.attach(Observer("B"))
subject.set_state("new state")

# STRATEGY
print("\n📋 STRATEGY - Interchangeable algorithms")

class SortStrategy(ABC):
    @abstractmethod
    def sort(self, data): pass

class QuickSort(SortStrategy):
    def sort(self, data):
        return sorted(data)

class BubbleSort(SortStrategy):
    def sort(self, data):
        return sorted(data)  # Simplified

class Sorter:
    def __init__(self, strategy):
        self.strategy = strategy
    def sort(self, data):
        return self.strategy.sort(data)

sorter = Sorter(QuickSort())
print(f"Strategy: {sorter.sort([3,1,2])}")

# COMMAND
print("\n📋 COMMAND - Encapsulate request as object")

class Command(ABC):
    @abstractmethod
    def execute(self): pass

class LightOnCommand(Command):
    def __init__(self, light):
        self.light = light
    def execute(self):
        self.light.on()

class Light:
    def on(self): print("  Light is ON")
    def off(self): print("  Light is OFF")

class RemoteControl:
    def __init__(self):
        self.command = None
    def set_command(self, cmd):
        self.command = cmd
    def press(self):
        self.command.execute()

light = Light()
remote = RemoteControl()
remote.set_command(LightOnCommand(light))
remote.press()

# STATE
print("\n📋 STATE - Object behavior changes with state")

class State(ABC):
    @abstractmethod
    def handle(self): pass

class OnState(State):
    def handle(self): return "ON"

class OffState(State):
    def handle(self): return "OFF"

class Switch:
    def __init__(self):
        self.state = OffState()
    def toggle(self):
        if isinstance(self.state, OffState):
            self.state = OnState()
        else:
            self.state = OffState()
    def status(self):
        return self.state.handle()

switch = Switch()
print(f"State: {switch.status()}")
switch.toggle()
print(f"State: {switch.status()}")

# ══════════════════════════════════════════════════════════════════════════════
# QUIZ
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("QUIZ")
print("=" * 70)

print("""
Q1. Singleton ensures?
    → Only one instance exists

Q2. Factory Method purpose?
    → Create objects without specifying exact class

Q3. Adapter pattern does?
    → Converts one interface to another

Q4. Decorator pattern (not Python) does?
    → Adds behavior dynamically to objects

Q5. Observer pattern is for?
    → Notifying dependents of state changes

Q6. Strategy pattern allows?
    → Interchangeable algorithms at runtime

Q7. Facade provides?
    → Simplified interface to complex subsystem

Q8. Command pattern does?
    → Encapsulates request as object
""")

print("\n" + "=" * 70)
print("DESIGN PATTERNS MODULE COMPLETE!")
print("=" * 70)
