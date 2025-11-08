"""
🚀 SESSIONE 1 - PARTE 1: PYTHON FUNDAMENTALS COMPLETI
=====================================================
Super Intensive Python Master Course
Durata: 90 minuti di teoria dettagliata
"""

import sys
import gc
import dis
import math
from decimal import Decimal, getcontext
from fractions import Fraction
from collections import ChainMap, namedtuple
from typing import List, Dict, Optional, Union, Callable, TypeVar
import copy

print("="*80)
print("🎯 SESSIONE 1: PYTHON COMPLETE CORE MASTERY")
print("="*80)

# ==============================================================================
# SEZIONE 1: PYTHON INTERNALS - Come funziona davvero
# ==============================================================================

def section1_python_internals():
    """
    Comprendi come Python gestisce memoria, oggetti e riferimenti
    """
    print("\n" + "="*60)
    print("📚 SEZIONE 1: PYTHON INTERNALS")
    print("="*60)
    
    # 1.1 OBJECT MODEL
    print("\n🔍 1.1 OBJECT MODEL - Tutto è un oggetto!")
    print("-"*40)
    x = 42
    print(f"x = 42")
    print(f"├── ID (indirizzo memoria): {id(x)}")
    print(f"├── Tipo: {type(x)}")
    print(f"├── Valore: {x}")
    print(f"├── Size in bytes: {sys.getsizeof(x)}")
    print(f"└── Refcount: {sys.getrefcount(x) - 1}")
    
    # Dimostra che è un riferimento
    y = x  # NON copia il valore, copia il RIFERIMENTO
    print(f"\ny = x  # Cosa succede?")
    print(f"├── x is y? {x is y} (stesso oggetto!)")
    print(f"├── ID di x: {id(x)}")
    print(f"├── ID di y: {id(y)}")
    print(f"└── Refcount ora: {sys.getrefcount(x) - 1} (aumentato!)")
    
    # 1.2 MUTABLE vs IMMUTABLE
    print("\n🔄 1.2 MUTABLE vs IMMUTABLE - Differenza critica")
    print("-"*40)
    
    # Immutable
    print("IMMUTABLE (int, str, tuple):")
    a = 100
    print(f"  a = 100, ID: {id(a)}")
    a = a + 1  # Crea NUOVO oggetto!
    print(f"  a = a + 1")
    print(f"  a = {a}, ID: {id(a)} ← DIVERSO! Nuovo oggetto creato")
    
    # Mutable
    print("\nMUTABLE (list, dict, set):")
    lista1 = [1, 2, 3]
    print(f"  lista1 = [1, 2, 3], ID: {id(lista1)}")
    lista2 = lista1  # Riferimento, non copia!
    lista2.append(4)
    print(f"  lista2 = lista1; lista2.append(4)")
    print(f"  lista1 = {lista1}, ID: {id(lista1)} ← STESSO!")
    print(f"  lista2 = {lista2}, ID: {id(lista2)} ← STESSO!")
    
    # 1.3 COPIE - Shallow vs Deep
    print("\n📋 1.3 SHALLOW vs DEEP COPY")
    print("-"*40)
    
    original = [[1, 2], [3, 4]]
    shallow = original.copy()
    deep = copy.deepcopy(original)
    
    print(f"Original: {original}")
    original[0].append(3)
    print(f"Dopo original[0].append(3):")
    print(f"  Original: {original}")
    print(f"  Shallow:  {shallow} ← inner list modificata!")
    print(f"  Deep:     {deep} ← non modificata!")
    
    # 1.4 BYTECODE
    print("\n💾 1.4 BYTECODE - Come Python compila")
    print("-"*40)
    
    def example(a, b):
        return a + b * 2
    
    print("def example(a, b):")
    print("    return a + b * 2")
    print("\nBytecode generato:")
    dis.dis(example)
    
    # 1.5 GARBAGE COLLECTION
    print("\n♻️ 1.5 GARBAGE COLLECTION")
    print("-"*40)
    
    class Node:
        def __init__(self, value):
            self.value = value
            self.ref = None
    
    # Circular reference
    a = Node(1)
    b = Node(2)
    a.ref = b
    b.ref = a  # Circular!
    
    print("Creato riferimento circolare:")
    print("a → b → a")
    print(f"Oggetti prima del GC: {len(gc.get_objects())}")
    
    del a, b  # Cancella riferimenti
    collected = gc.collect()  # Forza garbage collection
    
    print(f"Oggetti raccolti dal GC: {collected}")
    print(f"Oggetti dopo il GC: {len(gc.get_objects())}")

# ==============================================================================
# SEZIONE 2: TIPI DI DATI - Tutti i dettagli
# ==============================================================================

def section2_data_types():
    """
    Ogni tipo di dato con TUTTI i suoi metodi e segreti
    """
    print("\n" + "="*60)
    print("📊 SEZIONE 2: TIPI DI DATI COMPLETI")
    print("="*60)
    
    # 2.1 INTEGERS
    print("\n🔢 2.1 INTEGERS - Illimitati e potenti")
    print("-"*40)
    
    # Numeri enormi
    huge = 10**100
    print(f"Python gestisce numeri ENORMI:")
    print(f"  10^100 = ...{str(huge)[-20:]}")
    print(f"  Bit necessari: {huge.bit_length()}")
    
    # Sistemi numerici
    print("\nSistemi numerici:")
    num = 42
    print(f"  Decimale:     {num}")
    print(f"  Binario:      0b{bin(num)[2:]} = {0b101010}")
    print(f"  Ottale:       0o{oct(num)[2:]} = {0o52}")
    print(f"  Esadecimale:  0x{hex(num)[2:].upper()} = {0x2A}")
    
    # Operazioni bitwise
    print("\nOperazioni bitwise:")
    a, b = 12, 5  # 1100, 0101
    print(f"  {a:04b} & {b:04b} = {a & b:04b} (AND)")
    print(f"  {a:04b} | {b:04b} = {a | b:04b} (OR)")
    print(f"  {a:04b} ^ {b:04b} = {a ^ b:04b} (XOR)")
    print(f"  {a:04b} << 2 = {a << 2:06b} (Left shift)")
    
    # 2.2 FLOATS
    print("\n💫 2.2 FLOATS - Precisione e problemi")
    print("-"*40)
    
    # Il problema classico
    result = 0.1 + 0.2
    print("Il problema della precisione:")
    print(f"  0.1 + 0.2 = {result}")
    print(f"  È uguale a 0.3? {result == 0.3}")
    print(f"  Differenza: {result - 0.3}")
    
    # Soluzione con Decimal
    getcontext().prec = 50
    d1 = Decimal('0.1')
    d2 = Decimal('0.2')
    print(f"\nSoluzione con Decimal:")
    print(f"  Decimal('0.1') + Decimal('0.2') = {d1 + d2}")
    
    # Valori speciali
    print("\nValori speciali:")
    print(f"  Infinity: {float('inf')}")
    print(f"  -Infinity: {float('-inf')}")
    print(f"  NaN: {float('nan')}")
    print(f"  Inf > 999999? {float('inf') > 999999}")
    print(f"  NaN == NaN? {float('nan') == float('nan')} (sempre False!)")
    
    # 2.3 STRINGS
    print("\n📝 2.3 STRINGS - Unicode e metodi")
    print("-"*40)
    
    text = "Python 🐍 Master"
    print(f"Testo: '{text}'")
    print(f"  Length: {len(text)}")
    print(f"  Bytes (UTF-8): {len(text.encode('utf-8'))}")
    
    # Metodi principali
    s = "  Python Master  "
    print(f"\nMetodi di trasformazione:")
    print(f"  Original:     '{s}'")
    print(f"  .strip():     '{s.strip()}'")
    print(f"  .upper():     '{s.upper()}'")
    print(f"  .lower():     '{s.lower()}'")
    print(f"  .title():     '{s.title()}'")
    print(f"  .replace():   '{s.replace('Python', 'Java')}'")
    print(f"  .center(20):  '{s.strip().center(20, '*')}'")
    
    # F-strings avanzate
    print("\nF-strings avanzate:")
    name, score = "Marco", 95.456
    print(f"  Allineamento:  '{name:>10}' '{name:^10}' '{name:<10}'")
    print(f"  Decimali:      {score:.2f}")
    print(f"  Percentuale:   {0.856:.1%}")
    print(f"  Separatori:    {1234567:,}")
    print(f"  Debug mode:    {name=}, {score=}")
    
    # 2.4 BOOLEANS e Truthy/Falsy
    print("\n✅ 2.4 BOOLEANS - Truthy e Falsy")
    print("-"*40)
    
    print("Valori FALSY (considerati False):")
    falsy_values = [False, None, 0, 0.0, '', [], {}, set(), ()]
    for val in falsy_values:
        print(f"  {str(val):10} → bool({str(val)}) = {bool(val)}")
    
    print("\nValori TRUTHY (tutto il resto):")
    truthy_values = [True, 1, -1, 'Hello', [0], {0: 0}, (0,)]
    for val in truthy_values:
        print(f"  {str(val):10} → bool({str(val)}) = {bool(val)}")
    
    # Short-circuit
    print("\nShort-circuit evaluation:")
    print("  False and print('Non stampato') → niente")
    result = False and print('Non stampato')
    print("  True or print('Non stampato') → niente")
    result = True or print('Non stampato')

# ==============================================================================
# SEZIONE 3: STRUTTURE DATI COMPLETE
# ==============================================================================

def section3_data_structures():
    """
    Liste, Dizionari, Tuple, Set con tutti i metodi
    """
    print("\n" + "="*60)
    print("📚 SEZIONE 3: STRUTTURE DATI COMPLETE")
    print("="*60)
    
    # 3.1 LISTE
    print("\n📋 3.1 LISTE - Array dinamici")
    print("-"*40)
    
    # Creazione
    lst = [1, 2, 3, 4, 5]
    print(f"Lista originale: {lst}")
    
    # Tutti i metodi
    print("\nMetodi di modifica:")
    lst.append(6)
    print(f"  .append(6):    {lst}")
    lst.extend([7, 8])
    print(f"  .extend([7,8]): {lst}")
    lst.insert(0, 0)
    print(f"  .insert(0, 0): {lst}")
    removed = lst.pop()
    print(f"  .pop():        {lst} (removed: {removed})")
    lst.remove(0)
    print(f"  .remove(0):    {lst}")
    
    # Slicing
    print("\nSlicing:")
    lst = list(range(10))
    print(f"  lst = {lst}")
    print(f"  lst[2:5]   = {lst[2:5]}")
    print(f"  lst[:3]    = {lst[:3]}")
    print(f"  lst[7:]    = {lst[7:]}")
    print(f"  lst[::2]   = {lst[::2]}")
    print(f"  lst[::-1]  = {lst[::-1]}")
    print(f"  lst[2:8:2] = {lst[2:8:2]}")
    
    # List comprehensions
    print("\nList comprehensions:")
    squares = [x**2 for x in range(5)]
    print(f"  Squares: {squares}")
    evens = [x for x in range(10) if x % 2 == 0]
    print(f"  Evens: {evens}")
    matrix = [[i*j for j in range(3)] for i in range(3)]
    print(f"  Matrix: {matrix}")
    
    # 3.2 DIZIONARI
    print("\n📖 3.2 DIZIONARI - Hash tables")
    print("-"*40)
    
    # Creazione
    d = {'name': 'Marco', 'age': 30, 'city': 'Milano'}
    print(f"Dizionario: {d}")
    
    # Metodi
    print("\nMetodi principali:")
    print(f"  .keys():   {list(d.keys())}")
    print(f"  .values(): {list(d.values())}")
    print(f"  .items():  {list(d.items())}")
    
    # Get sicuro
    print(f"  .get('name'):     {d.get('name')}")
    print(f"  .get('missing'):  {d.get('missing', 'default')}")
    
    # Update
    d.update({'age': 31, 'job': 'Developer'})
    print(f"  .update():  {d}")
    
    # Dict comprehensions
    print("\nDict comprehensions:")
    squares = {x: x**2 for x in range(5)}
    print(f"  Squares: {squares}")
    
    # Dictionary merging (3.9+)
    d1 = {'a': 1, 'b': 2}
    d2 = {'b': 3, 'c': 4}
    d3 = d1 | d2
    print(f"  {d1} | {d2} = {d3}")
    
    # 3.3 TUPLE
    print("\n📦 3.3 TUPLE - Immutabili")
    print("-"*40)
    
    t = (1, 2, 3)
    print(f"Tupla: {t}")
    print(f"  Immutabile: t[0] = 99 → TypeError!")
    
    # Named tuples
    Point = namedtuple('Point', ['x', 'y'])
    p = Point(11, 22)
    print(f"  Named tuple: Point(x={p.x}, y={p.y})")
    
    # Unpacking
    a, b, c = (1, 2, 3)
    print(f"  Unpacking: a={a}, b={b}, c={c}")
    
    first, *middle, last = (1, 2, 3, 4, 5)
    print(f"  Extended: first={first}, middle={middle}, last={last}")
    
    # 3.4 SET
    print("\n🔢 3.4 SET - Insiemi matematici")
    print("-"*40)
    
    a = {1, 2, 3, 4}
    b = {3, 4, 5, 6}
    print(f"Set A: {a}")
    print(f"Set B: {b}")
    
    print("\nOperazioni:")
    print(f"  A | B (union):      {a | b}")
    print(f"  A & B (intersect):  {a & b}")
    print(f"  A - B (difference): {a - b}")
    print(f"  A ^ B (symmetric):  {a ^ b}")
    
    # Frozen sets
    fs = frozenset([1, 2, 3])
    print(f"  Frozen set: {fs} (immutabile)")

# ==============================================================================
# SEZIONE 4: CONTROL FLOW COMPLETO
# ==============================================================================

def section4_control_flow():
    """
    If/elif/else, loops, comprehensions - tutti i pattern
    """
    print("\n" + "="*60)
    print("🔀 SEZIONE 4: CONTROL FLOW COMPLETO")
    print("="*60)
    
    # 4.1 IF/ELIF/ELSE Patterns
    print("\n🎯 4.1 IF/ELIF/ELSE - Tutti i pattern")
    print("-"*40)
    
    # Pattern 1: Guard Clauses
    def process_data(data):
        if data is None:
            return "No data"
        if not isinstance(data, list):
            return "Invalid type"
        if len(data) == 0:
            return "Empty list"
        return f"Processing {len(data)} items"
    
    print("Guard Clauses (early return):")
    print(f"  None: {process_data(None)}")
    print(f"  'str': {process_data('string')}")
    print(f"  []: {process_data([])}")
    print(f"  [1,2,3]: {process_data([1,2,3])}")
    
    # Pattern 2: Ternary
    x = 10
    status = "positive" if x > 0 else "non-positive"
    print(f"\nTernary: x={x} → {status}")
    
    # Pattern 3: Chained comparisons
    y = 5
    print(f"\nChained: 0 <= {y} < 10? {0 <= y < 10}")
    
    # Pattern 4: all/any
    conditions = [True, True, False, True]
    print(f"\nall({conditions}): {all(conditions)}")
    print(f"any({conditions}): {any(conditions)}")
    
    # 4.2 FOR LOOPS
    print("\n🔄 4.2 FOR LOOPS - Tecniche avanzate")
    print("-"*40)
    
    # Enumerate
    items = ['a', 'b', 'c']
    print("Enumerate:")
    for i, item in enumerate(items, start=1):
        print(f"  {i}. {item}")
    
    # Zip
    names = ['Alice', 'Bob']
    ages = [25, 30]
    print("\nZip:")
    for name, age in zip(names, ages):
        print(f"  {name}: {age} anni")
    
    # Dictionary iteration
    d = {'a': 1, 'b': 2}
    print("\nDict iteration:")
    for key, value in d.items():
        print(f"  {key} = {value}")
    
    # 4.3 WHILE e Walrus
    print("\n⏰ 4.3 WHILE patterns")
    print("-"*40)
    
    # Walrus operator (3.8+)
    print("Walrus operator esempio:")
    data = [1, 2, 3, 0, 4]
    i = 0
    while (val := data[i]) != 0:
        print(f"  Processing: {val}")
        i += 1
        if i >= len(data):
            break
    
    # 4.4 COMPREHENSIONS
    print("\n🎨 4.4 COMPREHENSIONS - Tutti i tipi")
    print("-"*40)
    
    # List comprehension
    squares = [x**2 for x in range(5)]
    print(f"List comp: {squares}")
    
    # Dict comprehension
    d = {x: x**2 for x in range(3)}
    print(f"Dict comp: {d}")
    
    # Set comprehension
    s = {x % 3 for x in range(10)}
    print(f"Set comp: {s}")
    
    # Generator expression
    gen = (x**2 for x in range(5))
    print(f"Generator: {list(gen)}")
    
    # Nested with conditions
    result = [x for x in range(20) if x % 2 == 0 if x % 3 == 0]
    print(f"Multi-condition: {result}")

# ==============================================================================
# SEZIONE 5: FUNZIONI - Dal base all'avanzato
# ==============================================================================

def section5_functions():
    """
    Funzioni complete: parametri, closures, decorators, generators
    """
    print("\n" + "="*60)
    print("⚡ SEZIONE 5: FUNZIONI COMPLETE")
    print("="*60)
    
    # 5.1 PARAMETRI
    print("\n📝 5.1 TUTTI I TIPI DI PARAMETRI")
    print("-"*40)
    
    # Tutti i tipi insieme
    def full_function(pos_only, /, standard, *args, kw_only, **kwargs):
        """
        pos_only: solo posizionale (prima di /)
        standard: posizionale o keyword
        *args: argomenti variabili
        kw_only: solo keyword (dopo *)
        **kwargs: keyword variabili
        """
        print(f"  pos_only: {pos_only}")
        print(f"  standard: {standard}")
        print(f"  args: {args}")
        print(f"  kw_only: {kw_only}")
        print(f"  kwargs: {kwargs}")
    
    print("Chiamata complessa:")
    full_function(1, 2, 3, 4, kw_only=5, extra1=6, extra2=7)
    
    # Default mutable - il problema
    def bad_append(item, lst=[]):  # ❌ PERICOLOSO!
        lst.append(item)
        return lst
    
    print("\n⚠️ Mutable default problem:")
    print(f"  Prima chiamata: {bad_append(1)}")
    print(f"  Seconda chiamata: {bad_append(2)} ← STESSA LISTA!")
    
    # Soluzione
    def good_append(item, lst=None):  # ✅ CORRETTO
        if lst is None:
            lst = []
        lst.append(item)
        return lst
    
    print("\n✅ Soluzione corretta:")
    print(f"  Prima chiamata: {good_append(1)}")
    print(f"  Seconda chiamata: {good_append(2)}")
    
    # 5.2 CLOSURES
    print("\n🔒 5.2 CLOSURES e NESTED FUNCTIONS")
    print("-"*40)
    
    def outer(x):
        """x è 'captured' dalla closure"""
        def inner(y):
            return x + y  # Accede a x dell'outer scope
        return inner
    
    add_five = outer(5)
    print(f"Closure esempio:")
    print(f"  add_five(3) = {add_five(3)}")
    print(f"  add_five(7) = {add_five(7)}")
    
    # nonlocal
    def counter():
        count = 0
        def increment():
            nonlocal count  # Modifica variabile esterna
            count += 1
            return count
        return increment
    
    c = counter()
    print(f"\nCounter con nonlocal:")
    print(f"  {c()}, {c()}, {c()}")
    
    # 5.3 DECORATORS
    print("\n🎨 5.3 DECORATORS")
    print("-"*40)
    
    def timer(func):
        """Decorator che misura il tempo"""
        import time
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            print(f"  {func.__name__} took {time.time()-start:.4f}s")
            return result
        return wrapper
    
    @timer
    def slow_function():
        import time
        time.sleep(0.1)
        return "Done"
    
    print("Timer decorator:")
    result = slow_function()
    
    # Decorator con parametri
    def repeat(times=3):
        def decorator(func):
            def wrapper(*args, **kwargs):
                for i in range(times):
                    result = func(*args, **kwargs)
                return result
            return wrapper
        return decorator
    
    @repeat(times=3)
    def greet():
        print("  Hello!")
        return "Greeted"
    
    print("\nRepeat decorator:")
    greet()
    
    # 5.4 GENERATORS
    print("\n⚡ 5.4 GENERATORS - Lazy evaluation")
    print("-"*40)
    
    def fibonacci(n):
        """Generator per Fibonacci"""
        a, b = 0, 1
        count = 0
        while count < n:
            yield a
            a, b = b, a + b
            count += 1
    
    print("Fibonacci generator:")
    fib = fibonacci(10)
    print(f"  Primi 10: {list(fib)}")
    
    # Generator expression
    squares = (x**2 for x in range(5))
    print(f"\nGenerator expression:")
    print(f"  {next(squares)}, {next(squares)}, {next(squares)}")
    
    # 5.5 LAMBDA
    print("\n⚡ 5.5 LAMBDA e FUNCTIONAL")
    print("-"*40)
    
    # Lambda base
    square = lambda x: x**2
    print(f"Lambda: square(5) = {square(5)}")
    
    # Con map, filter, reduce
    numbers = [1, 2, 3, 4, 5]
    
    squared = list(map(lambda x: x**2, numbers))
    print(f"\nMap: {numbers} → {squared}")
    
    evens = list(filter(lambda x: x % 2 == 0, numbers))
    print(f"Filter: {numbers} → {evens}")
    
    from functools import reduce
    product = reduce(lambda x, y: x * y, numbers)
    print(f"Reduce: {numbers} → {product}")
    
    # Sorting con lambda
    people = [('Alice', 25), ('Bob', 30), ('Charlie', 20)]
    people.sort(key=lambda p: p[1])
    print(f"\nSort by age: {people}")

# ==============================================================================
# MAIN - Esegui tutte le sezioni
# ==============================================================================

def main():
    """Esegue tutte le sezioni della lezione"""
    
    sections = [
        ("Python Internals", section1_python_internals),
        ("Data Types", section2_data_types),
        ("Data Structures", section3_data_structures),
        ("Control Flow", section4_control_flow),
        ("Functions", section5_functions)
    ]
    
    print("\n" + "="*80)
    print("🎯 BENVENUTO AL PYTHON MASTER COURSE - SESSIONE 1")
    print("="*80)
    print("\nScegli cosa studiare:")
    print("0. TUTTO (completo)")
    for i, (name, _) in enumerate(sections, 1):
        print(f"{i}. {name}")
    
    choice = input("\nInserisci numero (0-5): ")
    
    try:
        choice = int(choice)
        if choice == 0:
            for name, func in sections:
                input(f"\n🎯 Premi ENTER per: {name}")
                func()
        elif 1 <= choice <= len(sections):
            sections[choice-1][1]()
        else:
            print("Scelta non valida")
    except (ValueError, IndexError):
        print("Scelta non valida")
    
    print("\n" + "="*80)
    print("✅ SESSIONE COMPLETATA!")
    print("Prossimo step: Esegui session1_part2_projects.py per i progetti")
    print("="*80)

if __name__ == "__main__":
    main()
