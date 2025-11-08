"""
🚀 SESSIONE 1 - PARTE 3: CONCETTI AVANZATI E GOTCHAS
====================================================
Super Intensive Python Master Course
Durata: 60 minuti di concetti avanzati
"""

import sys
import gc
import time
import asyncio
from typing import Any, List, Dict, Optional
from functools import wraps, lru_cache
from contextlib import contextmanager
import threading
import multiprocessing

print("="*80)
print("🧠 PARTE 3: CONCETTI AVANZATI E GOTCHAS")
print("="*80)

# ==============================================================================
# SEZIONE 1: PYTHON GOTCHAS - Le trappole da evitare
# ==============================================================================

def section1_python_gotchas():
    """I problemi più comuni e come evitarli"""
    
    print("\n" + "="*60)
    print("⚠️ PYTHON GOTCHAS - TRAPPOLE COMUNI")
    print("="*60)
    
    # GOTCHA 1: Mutable Default Arguments
    print("\n🔴 GOTCHA 1: Mutable Default Arguments")
    print("-"*40)
    
    # Il problema
    def bad_append(item, lst=[]):  # ❌ PERICOLOSO!
        lst.append(item)
        return lst
    
    result1 = bad_append(1)
    result2 = bad_append(2)
    
    print("❌ PROBLEMA:")
    print(f"  Prima chiamata: bad_append(1) = {result1}")
    print(f"  Seconda chiamata: bad_append(2) = {result2}")
    print(f"  Stessa lista? {result1 is result2} ← SI! Bug!")
    
    # La soluzione
    def good_append(item, lst=None):  # ✅ CORRETTO
        if lst is None:
            lst = []
        lst.append(item)
        return lst
    
    result1 = good_append(1)
    result2 = good_append(2)
    
    print("\n✅ SOLUZIONE:")
    print(f"  Prima chiamata: good_append(1) = {result1}")
    print(f"  Seconda chiamata: good_append(2) = {result2}")
    print(f"  Stessa lista? {result1 is result2} ← NO! Corretto!")
    
    # GOTCHA 2: Late Binding Closures
    print("\n🔴 GOTCHA 2: Late Binding Closures")
    print("-"*40)
    
    # Il problema
    funcs = []
    for i in range(3):
        funcs.append(lambda: i)  # ❌ Tutti printano 2!
    
    print("❌ PROBLEMA:")
    print(f"  Funzioni create in loop:")
    for j, f in enumerate(funcs):
        print(f"    func[{j}]() = {f()} ← Tutti uguali!")
    
    # La soluzione
    funcs = []
    for i in range(3):
        funcs.append(lambda x=i: x)  # ✅ Capture value
    
    print("\n✅ SOLUZIONE:")
    print(f"  Con default parameter:")
    for j, f in enumerate(funcs):
        print(f"    func[{j}]() = {f()} ← Corretti!")
    
    # GOTCHA 3: Modifying List While Iterating
    print("\n🔴 GOTCHA 3: Modificare lista durante iterazione")
    print("-"*40)
    
    # Il problema
    numbers = [1, 2, 3, 4, 5, 6]
    print(f"Lista originale: {numbers}")
    
    # ❌ SBAGLIATO - Salta elementi!
    numbers_copy = numbers.copy()
    for i, num in enumerate(numbers_copy):
        if num % 2 == 0:
            del numbers_copy[i]  # Bug! Indices cambiano
    
    print(f"❌ Tentativo di rimuovere pari: {numbers_copy} ← Sbagliato!")
    
    # ✅ CORRETTO - List comprehension
    numbers = [1, 2, 3, 4, 5, 6]
    numbers = [num for num in numbers if num % 2 != 0]
    print(f"✅ List comprehension: {numbers} ← Corretto!")
    
    # GOTCHA 4: Integer Caching
    print("\n🔴 GOTCHA 4: Integer Caching")
    print("-"*40)
    
    a = 256
    b = 256
    print(f"a = 256, b = 256")
    print(f"  a is b? {a is b} ← Stesso oggetto (cached)")
    
    a = 257
    b = 257
    print(f"\na = 257, b = 257")
    print(f"  a is b? {a is b} ← Oggetti diversi!")
    print(f"  Python caches integers from -5 to 256")
    
    # GOTCHA 5: Class Variables vs Instance Variables
    print("\n🔴 GOTCHA 5: Class vs Instance Variables")
    print("-"*40)
    
    class BadClass:
        shared = []  # ❌ Class variable - CONDIVISA!
        
        def add(self, item):
            self.shared.append(item)
    
    obj1 = BadClass()
    obj2 = BadClass()
    
    obj1.add(1)
    print(f"❌ obj1.shared: {obj1.shared}")
    print(f"❌ obj2.shared: {obj2.shared} ← Modificato anche questo!")
    
    class GoodClass:
        def __init__(self):
            self.not_shared = []  # ✅ Instance variable
        
        def add(self, item):
            self.not_shared.append(item)
    
    obj1 = GoodClass()
    obj2 = GoodClass()
    
    obj1.add(1)
    print(f"\n✅ obj1.not_shared: {obj1.not_shared}")
    print(f"✅ obj2.not_shared: {obj2.not_shared} ← Non modificato!")
    
    # GOTCHA 6: Mutable Operations
    print("\n🔴 GOTCHA 6: += con liste")
    print("-"*40)
    
    # Comportamento diverso!
    def modify_list_inplace(lst):
        lst += [4]  # Modifica in-place!
        return lst
    
    def modify_list_new(lst):
        lst = lst + [4]  # Crea nuova lista!
        return lst
    
    original = [1, 2, 3]
    result = modify_list_inplace(original.copy())
    print(f"lst += [4] modifica in-place: {result}")
    
    original = [1, 2, 3]
    result = modify_list_new(original)
    print(f"lst = lst + [4] crea nuova: {result}")
    print(f"Originale rimane: {original}")

# ==============================================================================
# SEZIONE 2: MEMORY MANAGEMENT E PERFORMANCE
# ==============================================================================

def section2_memory_performance():
    """Gestione memoria e ottimizzazione performance"""
    
    print("\n" + "="*60)
    print("💾 MEMORY MANAGEMENT E PERFORMANCE")
    print("="*60)
    
    # Reference Counting
    print("\n📊 REFERENCE COUNTING")
    print("-"*40)
    
    x = [1, 2, 3]
    print(f"Lista creata: x = [1, 2, 3]")
    print(f"  Refcount: {sys.getrefcount(x) - 1}")
    print(f"  Size: {sys.getsizeof(x)} bytes")
    
    y = x
    print(f"\ny = x")
    print(f"  Refcount dopo: {sys.getrefcount(x) - 1}")
    
    del y
    print(f"\ndel y")
    print(f"  Refcount dopo: {sys.getrefcount(x) - 1}")
    
    # Garbage Collection
    print("\n♻️ GARBAGE COLLECTION")
    print("-"*40)
    
    class Node:
        def __init__(self, value):
            self.value = value
            self.ref = None
    
    # Circular reference
    print("Creazione riferimento circolare:")
    a = Node(1)
    b = Node(2)
    a.ref = b
    b.ref = a
    
    print(f"  Oggetti tracked: {len(gc.get_objects())}")
    
    del a, b
    print(f"  Dopo del a, b: {len(gc.get_objects())}")
    
    collected = gc.collect()
    print(f"  Garbage collected: {collected} oggetti")
    print(f"  Dopo GC: {len(gc.get_objects())}")
    
    # Memory Optimization: Slots
    print("\n💡 OPTIMIZATION: __slots__")
    print("-"*40)
    
    class RegularClass:
        def __init__(self, x, y):
            self.x = x
            self.y = y
    
    class SlottedClass:
        __slots__ = ['x', 'y']  # No __dict__, risparmia memoria
        
        def __init__(self, x, y):
            self.x = x
            self.y = y
    
    regular = RegularClass(1, 2)
    slotted = SlottedClass(1, 2)
    
    print(f"Regular class:")
    print(f"  Has __dict__: {hasattr(regular, '__dict__')}")
    print(f"  Size of __dict__: {sys.getsizeof(regular.__dict__)} bytes")
    
    print(f"\nSlotted class:")
    print(f"  Has __dict__: {hasattr(slotted, '__dict__')}")
    print(f"  Memory saved: ~40% per instance")
    
    # Generators vs Lists
    print("\n⚡ GENERATORS vs LISTS")
    print("-"*40)
    
    # List - tutto in memoria
    def get_squares_list(n):
        return [x**2 for x in range(n)]
    
    # Generator - lazy evaluation
    def get_squares_gen(n):
        return (x**2 for x in range(n))
    
    n = 1000000
    
    # Misura memoria
    list_squares = get_squares_list(n)
    gen_squares = get_squares_gen(n)
    
    print(f"Per {n:,} elementi:")
    print(f"  List size: {sys.getsizeof(list_squares):,} bytes")
    print(f"  Generator size: {sys.getsizeof(gen_squares)} bytes")
    print(f"  Risparmio: {sys.getsizeof(list_squares) / sys.getsizeof(gen_squares):.0f}x")
    
    # String Interning
    print("\n📝 STRING INTERNING")
    print("-"*40)
    
    a = "hello"
    b = "hello"
    print(f"Stringhe semplici:")
    print(f"  a = 'hello', b = 'hello'")
    print(f"  a is b? {a is b} ← Stesso oggetto (interned)")
    
    a = "hello world!"
    b = "hello world!"
    print(f"\nStringhe con spazi:")
    print(f"  a = 'hello world!', b = 'hello world!'")
    print(f"  a is b? {a is b} ← Oggetti diversi")
    
    # Force interning
    a = sys.intern("hello world!")
    b = sys.intern("hello world!")
    print(f"\nDopo sys.intern():")
    print(f"  a is b? {a is b} ← Stesso oggetto!")

# ==============================================================================
# SEZIONE 3: DECORATORS AVANZATI
# ==============================================================================

def section3_advanced_decorators():
    """Decorators avanzati e pattern"""
    
    print("\n" + "="*60)
    print("🎨 DECORATORS AVANZATI")
    print("="*60)
    
    # Decorator con parametri
    print("\n🔧 DECORATOR CON PARAMETRI")
    print("-"*40)
    
    def retry(max_attempts=3, delay=1):
        """Decorator che ritenta in caso di errore"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        print(f"  Attempt {attempt + 1} failed: {e}")
                        if attempt < max_attempts - 1:
                            time.sleep(delay)
                
                raise last_exception
            return wrapper
        return decorator
    
    @retry(max_attempts=3, delay=0.1)
    def unstable_function():
        """Simula funzione che può fallire"""
        import random
        if random.random() < 0.7:  # 70% failure rate
            raise ValueError("Random failure")
        return "Success!"
    
    print("Testing retry decorator:")
    try:
        result = unstable_function()
        print(f"  Result: {result}")
    except:
        print(f"  Failed after all attempts")
    
    # Class Decorator
    print("\n🏗️ CLASS DECORATOR")
    print("-"*40)
    
    def dataclass_simple(cls):
        """Simplified dataclass decorator"""
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        def __repr__(self):
            items = [f"{k}={v!r}" for k, v in self.__dict__.items()]
            return f"{cls.__name__}({', '.join(items)})"
        
        cls.__init__ = __init__
        cls.__repr__ = __repr__
        return cls
    
    @dataclass_simple
    class Person:
        pass
    
    p = Person(name="Alice", age=30)
    print(f"Person created: {p}")
    
    # Property Decorator
    print("\n🏠 PROPERTY DECORATOR")
    print("-"*40)
    
    class Temperature:
        def __init__(self, celsius=0):
            self._celsius = celsius
        
        @property
        def celsius(self):
            return self._celsius
        
        @celsius.setter
        def celsius(self, value):
            if value < -273.15:
                raise ValueError("Temperature below absolute zero!")
            self._celsius = value
        
        @property
        def fahrenheit(self):
            return self._celsius * 9/5 + 32
        
        @fahrenheit.setter
        def fahrenheit(self, value):
            self.celsius = (value - 32) * 5/9
    
    temp = Temperature()
    print("Temperature class:")
    temp.celsius = 25
    print(f"  Celsius: {temp.celsius}°C")
    print(f"  Fahrenheit: {temp.fahrenheit}°F")
    
    temp.fahrenheit = 86
    print(f"  After setting 86°F:")
    print(f"  Celsius: {temp.celsius:.1f}°C")
    
    # Caching Decorator
    print("\n💾 CACHING DECORATOR")
    print("-"*40)
    
    def memoize(func):
        """Cache results of function calls"""
        cache = {}
        
        @wraps(func)
        def wrapper(*args):
            if args in cache:
                print(f"  Cache hit for {args}")
                return cache[args]
            
            print(f"  Computing for {args}")
            result = func(*args)
            cache[args] = result
            return result
        
        wrapper.cache = cache  # Expose cache
        return wrapper
    
    @memoize
    def fibonacci(n):
        if n < 2:
            return n
        return fibonacci(n - 1) + fibonacci(n - 2)
    
    print("Fibonacci with memoization:")
    print(f"  fib(10) = {fibonacci(10)}")
    print(f"  Cache size: {len(fibonacci.cache)}")
    print(f"  fib(10) again = {fibonacci(10)} (from cache)")

# ==============================================================================
# SEZIONE 4: CONTEXT MANAGERS
# ==============================================================================

def section4_context_managers():
    """Context managers e with statement"""
    
    print("\n" + "="*60)
    print("🔒 CONTEXT MANAGERS")
    print("="*60)
    
    # Basic Context Manager
    print("\n📁 BASIC CONTEXT MANAGER")
    print("-"*40)
    
    class FileManager:
        """Context manager for file operations"""
        def __init__(self, filename, mode):
            self.filename = filename
            self.mode = mode
            self.file = None
        
        def __enter__(self):
            print(f"  Opening {self.filename}")
            self.file = open(self.filename, self.mode)
            return self.file
        
        def __exit__(self, exc_type, exc_value, traceback):
            print(f"  Closing {self.filename}")
            if self.file:
                self.file.close()
            
            if exc_type:
                print(f"  Exception occurred: {exc_value}")
            
            return False  # Don't suppress exceptions
    
    # Test it
    with FileManager("test.txt", "w") as f:
        f.write("Hello, World!")
    
    # contextlib decorator
    print("\n🎯 CONTEXTMANAGER DECORATOR")
    print("-"*40)
    
    @contextmanager
    def timer(name):
        """Time a code block"""
        print(f"  Starting {name}")
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            print(f"  {name} took {elapsed:.4f} seconds")
    
    with timer("Heavy computation"):
        time.sleep(0.1)  # Simulate work
    
    # Multiple Context Managers
    print("\n🔗 MULTIPLE CONTEXT MANAGERS")
    print("-"*40)
    
    @contextmanager
    def resource_a():
        print("  Acquiring resource A")
        yield "A"
        print("  Releasing resource A")
    
    @contextmanager
    def resource_b():
        print("  Acquiring resource B")
        yield "B"
        print("  Releasing resource B")
    
    with resource_a() as a, resource_b() as b:
        print(f"  Using resources: {a} and {b}")
    
    # Cleanup
    import os
    if os.path.exists("test.txt"):
        os.remove("test.txt")

# ==============================================================================
# SEZIONE 5: METACLASSES E DESCRIPTORS
# ==============================================================================

def section5_metaclasses():
    """Metaclasses e descriptors - advanced OOP"""
    
    print("\n" + "="*60)
    print("🧬 METACLASSES E DESCRIPTORS")
    print("="*60)
    
    # Simple Metaclass
    print("\n🔮 METACLASS EXAMPLE")
    print("-"*40)
    
    class SingletonMeta(type):
        """Metaclass for singleton pattern"""
        _instances = {}
        
        def __call__(cls, *args, **kwargs):
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
            return cls._instances[cls]
    
    class Database(metaclass=SingletonMeta):
        def __init__(self):
            print("  Creating database connection")
            self.connection = "Connected"
    
    # Test singleton
    db1 = Database()
    db2 = Database()
    print(f"  db1 is db2? {db1 is db2} ← Same instance!")
    
    # Descriptor
    print("\n📝 DESCRIPTOR PROTOCOL")
    print("-"*40)
    
    class ValidatedAttribute:
        """Descriptor for validated attributes"""
        def __init__(self, min_value=None, max_value=None):
            self.min_value = min_value
            self.max_value = max_value
        
        def __set_name__(self, owner, name):
            self.name = name
        
        def __get__(self, obj, objtype=None):
            if obj is None:
                return self
            return obj.__dict__.get(self.name)
        
        def __set__(self, obj, value):
            if self.min_value is not None and value < self.min_value:
                raise ValueError(f"{self.name} must be >= {self.min_value}")
            if self.max_value is not None and value > self.max_value:
                raise ValueError(f"{self.name} must be <= {self.max_value}")
            obj.__dict__[self.name] = value
    
    class Person:
        age = ValidatedAttribute(min_value=0, max_value=150)
        
        def __init__(self, name, age):
            self.name = name
            self.age = age  # Uses descriptor
    
    # Test descriptor
    try:
        p = Person("Alice", 30)
        print(f"  Person created: {p.name}, age {p.age}")
        
        p.age = -5  # Should raise error
    except ValueError as e:
        print(f"  Validation error: {e}")

# ==============================================================================
# MASTER CHALLENGE: Sistema di Monitoraggio
# ==============================================================================

def master_challenge():
    """Challenge finale che combina tutto"""
    
    print("\n" + "="*60)
    print("🏆 MASTER CHALLENGE: SISTEMA DI MONITORAGGIO")
    print("="*60)
    
    print("""
    📋 IL TUO COMPITO:
    
    Crea un sistema che:
    1. Monitora file system per cambiamenti
    2. Monitora performance sistema (CPU, RAM)
    3. Genera report automatici
    4. Usa decorators per logging
    5. Usa context managers per risorse
    6. Implementa pattern Singleton
    7. Gestisce errori con retry
    
    TEMPLATE INIZIALE:
    """)
    
    print("""
    class SystemMonitor(metaclass=SingletonMeta):
        '''Sistema di monitoraggio universale'''
        
        def __init__(self):
            self.metrics = []
            self.alerts = []
        
        @retry(max_attempts=3)
        @timer
        @log_execution
        def monitor_system(self):
            '''Monitora CPU e RAM'''
            # TODO: Implementa
            pass
        
        @contextmanager
        def monitoring_session(self):
            '''Context manager per sessione'''
            # TODO: Implementa
            pass
        
        def generate_report(self):
            '''Genera report HTML/JSON'''
            # TODO: Implementa
            pass
    
    # IMPLEMENTALO!
    """)
    
    print("\n💡 SUGGERIMENTI:")
    print("  • Usa psutil per system metrics")
    print("  • Usa watchdog per file monitoring")
    print("  • Usa jinja2 per HTML reports")
    print("  • Usa asyncio per operazioni async")
    print("  • Usa SQLite per storage")

# ==============================================================================
# BONUS: ASYNC/AWAIT PREVIEW
# ==============================================================================

def bonus_async_preview():
    """Preview di programmazione asincrona"""
    
    print("\n" + "="*60)
    print("⚡ BONUS: ASYNC/AWAIT PREVIEW")
    print("="*60)
    
    print("""
    📝 ASYNC/AWAIT - Il futuro di Python
    
    Async permette di eseguire operazioni I/O-bound in modo concorrente
    senza threads. Perfetto per:
    - Web scraping multiplo
    - API calls parallele  
    - WebSocket connections
    - Database queries parallele
    """)
    
    # Esempio base
    print("\n💻 ESEMPIO BASE:")
    print("-"*40)
    
    example_code = '''
    import asyncio
    
    async def fetch_data(id):
        """Simula fetch asincrono"""
        print(f"  Fetching data {id}...")
        await asyncio.sleep(1)  # Simula I/O
        return f"Data {id}"
    
    async def main():
        """Esegue tasks in parallelo"""
        # Crea tasks
        tasks = [fetch_data(i) for i in range(3)]
        
        # Esegue in parallelo
        results = await asyncio.gather(*tasks)
        
        print(f"  Results: {results}")
    
    # Run
    asyncio.run(main())
    '''
    
    print(example_code)
    
    # Esegui esempio
    import asyncio
    
    async def fetch_data(id):
        print(f"  Starting fetch {id}")
        await asyncio.sleep(0.5)
        print(f"  Completed fetch {id}")
        return f"Data-{id}"
    
    async def demo():
        print("\n🚀 ESECUZIONE PARALLELA:")
        start = time.time()
        
        # Parallelo
        tasks = [fetch_data(i) for i in range(3)]
        results = await asyncio.gather(*tasks)
        
        elapsed = time.time() - start
        print(f"\n  Results: {results}")
        print(f"  Time: {elapsed:.2f}s (not 1.5s!)")
    
    # Esegui
    asyncio.run(demo())
    
    print("""
    
    ⚡ VANTAGGI ASYNC:
    • Non blocca il thread principale
    • Gestisce migliaia di connessioni
    • Perfetto per I/O-bound tasks
    • Più efficiente dei threads
    
    📚 APPROFONDIREMO NELLA SESSIONE 3!
    """)

# ==============================================================================
# MAIN - Menu per le sezioni
# ==============================================================================

def main():
    """Menu principale"""
    
    print("\n" + "="*60)
    print("🧠 CONCETTI AVANZATI - SCEGLI SEZIONE")
    print("="*60)
    
    sections = [
        ("Python Gotchas", section1_python_gotchas),
        ("Memory & Performance", section2_memory_performance),
        ("Advanced Decorators", section3_advanced_decorators),
        ("Context Managers", section4_context_managers),
        ("Metaclasses", section5_metaclasses),
        ("Master Challenge", master_challenge),
        ("Async Preview (Bonus)", bonus_async_preview)
    ]
    
    print("\n0. Esegui TUTTO")
    for i, (name, _) in enumerate(sections, 1):
        print(f"{i}. {name}")
    
    choice = input("\nScegli (0-7): ")
    
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
    print("✅ PARTE 3 COMPLETATA!")
    print("="*60)
    
    print("""
    🎓 RIEPILOGO SESSIONE 1:
    
    ✅ Python Internals padroneggiati
    ✅ Tutti i tipi di dati e strutture
    ✅ Control flow completo
    ✅ Funzioni avanzate e decorators
    ✅ 3 progetti completi costruiti
    ✅ Gotchas e best practices
    ✅ Memory management
    ✅ Context managers
    ✅ Metaclasses base
    
    📈 LIVELLO RAGGIUNTO: INTERMEDIATE++
    
    🚀 PROSSIMA SESSIONE:
    - Advanced OOP
    - Concurrency (threading, async)
    - Networking
    - Databases avanzati
    - Testing professionale
    """)

if __name__ == "__main__":
    main()
