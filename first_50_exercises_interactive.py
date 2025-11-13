#!/usr/bin/env python3
"""
🎯 INTERACTIVE JUPYTER NOTEBOOK - PRIMI 50 ESERCIZI
Puoi aprire questo file in Jupyter o VS Code con Jupyter extension
"""

# %%[markdown]
# # 🚀 Python Master Journey - Primi 50 Esercizi Interattivi
# ## Dal Zero alla Padronanza dei Fondamenti
# 
# ### 📋 Come usare questo notebook:
# 1. **Leggi** la descrizione dell'esercizio
# 2. **Prova** a risolverlo nella cella "TUA SOLUZIONE"
# 3. **Esegui** la cella per testare
# 4. **Confronta** con la soluzione proposta
# 5. **Sperimenta** con variazioni
# 
# ---

# %% [markdown]
# ## 📦 Setup Iniziale

# %%
# Import necessari per tutto il notebook
import sys
import gc
import weakref
from copy import copy, deepcopy
import time
from typing import Any, List, Dict, Optional
import tracemalloc

# Progress tracker
completed_exercises = []

def mark_completed(exercise_num: int):
    """Marca un esercizio come completato"""
    if exercise_num not in completed_exercises:
        completed_exercises.append(exercise_num)
        print(f"✅ Esercizio {exercise_num} completato! Progress: {len(completed_exercises)}/50")
    return f"Completati: {completed_exercises}"

print("🎯 Benvenuto nel tuo Python Learning Journey!")
print("=" * 50)
print("Iniziamo con i primi 50 esercizi!")

# %% [markdown]
# ---
# # 📦 SEZIONE 1: VARIABLES & MEMORY (1-20)
# ## Capire come Python gestisce la memoria

# %% [markdown]
# ### 🎯 Esercizio 1: Prima Variabile e ID Memoria
# Crea una variabile con il tuo nome e stampa il suo ID in memoria

# %%
# 🔷 TUA SOLUZIONE - Esercizio 1
# Scrivi qui sotto:



# %%
# ✅ SOLUZIONE PROPOSTA - Esercizio 1
name = "Marco"
print(f"Nome: {name}")
print(f"ID in memoria: {id(name)}")
print(f"Tipo: {type(name)}")
print(f"Dimensione in bytes: {sys.getsizeof(name)}")

# 💡 SPIEGAZIONE:
# id() restituisce l'indirizzo di memoria dell'oggetto
# Ogni oggetto in Python ha un ID unico durante la sua vita

mark_completed(1)

# %% [markdown]
# ### 🎯 Esercizio 2: Integer Caching
# Assegna lo stesso intero a due variabili e verifica se puntano allo stesso oggetto

# %%
# 🔷 TUA SOLUZIONE - Esercizio 2



# %%
# ✅ SOLUZIONE PROPOSTA - Esercizio 2
a = 256
b = 256
print(f"a = {a}, b = {b}")
print(f"ID di a: {id(a)}")
print(f"ID di b: {id(b)}")
print(f"a is b? {a is b}")  # True!
print(f"a == b? {a == b}")  # True

# 💡 SPIEGAZIONE:
# Python cacha gli interi da -5 a 256 per ottimizzazione
# 'is' verifica l'identità (stesso oggetto)
# '==' verifica l'uguaglianza (stesso valore)

mark_completed(2)

# %% [markdown]
# ### 🎯 Esercizio 3: Integer Caching Limit
# Ripeti con 257 - cosa cambia?

# %%
# 🔷 TUA SOLUZIONE - Esercizio 3



# %%
# ✅ SOLUZIONE PROPOSTA - Esercizio 3
x = 257
y = 257
print(f"x = {x}, y = {y}")
print(f"ID di x: {id(x)}")
print(f"ID di y: {id(y)}")
print(f"x is y? {x is y}")  # False!
print(f"x == y? {x == y}")  # True

# Confronto con numeri cachati
small_a = 100
small_b = 100
print(f"\n100 is cached: {small_a is small_b}")  # True

# 💡 SPIEGAZIONE:
# Fuori dal range -5 to 256, Python crea oggetti separati
# Questo è un'ottimizzazione di memoria per numeri comuni

mark_completed(3)

# %% [markdown]
# ### 🎯 Esercizio 4: Memory Size Analysis
# Calcola la dimensione in bytes di diversi tipi

# %%
# 🔷 TUA SOLUZIONE - Esercizio 4



# %%
# ✅ SOLUZIONE PROPOSTA - Esercizio 4
def analyze_memory(*objects):
    """Analizza memoria di diversi oggetti"""
    for obj in objects:
        print(f"{str(obj):20} | Type: {type(obj).__name__:10} | Size: {sys.getsizeof(obj):4} bytes")

# Test con diversi tipi
analyze_memory(
    42,                    # int
    3.14,                  # float
    "Hello",               # str
    [],                    # empty list
    [1, 2, 3],            # list with items
    {},                    # empty dict
    {"a": 1, "b": 2},     # dict with items
    (1, 2, 3),            # tuple
    set([1, 2, 3]),       # set
    True                   # bool
)

# Confronto stringhe di diverse lunghezze
print("\n📝 String size scaling:")
for i in [0, 1, 10, 100, 1000]:
    s = "a" * i
    print(f"String of {i:4} chars: {sys.getsizeof(s):5} bytes")

mark_completed(4)

# %% [markdown]
# ### 🎯 Esercizio 5: Shallow vs Deep Copy
# Dimostra la differenza tra shallow e deep copy

# %%
# 🔷 TUA SOLUZIONE - Esercizio 5



# %%
# ✅ SOLUZIONE PROPOSTA - Esercizio 5
# Setup: lista nested
original = [[1, 2], [3, 4]]
print(f"Original: {original}")
print(f"Original ID: {id(original)}")

# Shallow copy
shallow = copy(original)
# o shallow = original.copy()
# o shallow = original[:]

print(f"\n🔷 Shallow copy: {shallow}")
print(f"Shallow ID: {id(shallow)}")
print(f"Same object? {original is shallow}")  # False

# Modifichiamo shallow copy
shallow[0].append(3)
print(f"\n⚠️ After modifying shallow[0]:")
print(f"Original: {original}")  # MODIFICATO!
print(f"Shallow: {shallow}")

# Deep copy
original2 = [[1, 2], [3, 4]]
deep = deepcopy(original2)

print(f"\n🔷 Deep copy: {deep}")
deep[0].append(3)
print(f"\n✅ After modifying deep[0]:")
print(f"Original2: {original2}")  # NON modificato
print(f"Deep: {deep}")

# 💡 SPIEGAZIONE:
# Shallow copy: copia solo il primo livello
# Deep copy: copia ricorsivamente tutti i livelli

mark_completed(5)

# %% [markdown]
# ### 🎯 Esercizio 6: String Interning
# Verifica quando Python riusa stringhe

# %%
# 🔷 TUA SOLUZIONE - Esercizio 6



# %%
# ✅ SOLUZIONE PROPOSTA - Esercizio 6
# Stringhe corte e semplici
s1 = "hello"
s2 = "hello"
print(f"s1 = '{s1}', s2 = '{s2}'")
print(f"s1 is s2? {s1 is s2}")  # True - interning!

# Stringhe con spazi
s3 = "hello world"
s4 = "hello world"
print(f"\ns3 = '{s3}', s4 = '{s4}'")
print(f"s3 is s4? {s3 is s4}")  # Dipende...

# Stringhe create dinamicamente
s5 = "hel" + "lo"
s6 = "hello"
print(f"\ns5 = '{s5}', s6 = '{s6}'")
print(f"s5 is s6? {s5 is s6}")  # True

# Force interning
s7 = " hello world "
s8 = " hello world "
print(f"\ns7 is s8? {s7 is s8}")  # False
s7_interned = sys.intern(s7)
s8_interned = sys.intern(s8)
print(f"After intern: {s7_interned is s8_interned}")  # True

# 💡 SPIEGAZIONE:
# Python ottimizza riusando stringhe identiche (interning)
# Funziona automaticamente per stringhe corte e semplici

mark_completed(6)

# %% [markdown]
# ### 🎯 Esercizio 7: Mutable Lists Demo
# Mostra come le liste sono mutabili e condivise

# %%
# 🔷 TUA SOLUZIONE - Esercizio 7



# %%
# ✅ SOLUZIONE PROPOSTA - Esercizio 7
# Due variabili, stesso oggetto
list1 = [1, 2, 3]
list2 = list1  # NON è una copia!

print(f"list1: {list1}, ID: {id(list1)}")
print(f"list2: {list2}, ID: {id(list2)}")
print(f"Stesso oggetto? {list1 is list2}")

# Modifica attraverso list2
list2.append(4)
print(f"\nDopo list2.append(4):")
print(f"list1: {list1}")  # Modificata!
print(f"list2: {list2}")

# Come evitarlo - crea una copia
list3 = [1, 2, 3]
list4 = list3.copy()  # o list3[:] o list(list3)

list4.append(4)
print(f"\nCon copia:")
print(f"list3: {list3}")  # NON modificata
print(f"list4: {list4}")

# Function side effects
def modify_list(lst):
    lst.append(999)
    return lst

my_list = [1, 2, 3]
result = modify_list(my_list)
print(f"\nDopo funzione:")
print(f"my_list: {my_list}")  # Modificata!

# 💡 SPIEGAZIONE:
# Le liste sono mutabili - modifiche si riflettono ovunque

mark_completed(7)

# %% [markdown]
# ### 🎯 Esercizio 8: Tuple Immutability
# Dimostra l'immutabilità delle tuple

# %%
# 🔷 TUA SOLUZIONE - Esercizio 8



# %%
# ✅ SOLUZIONE PROPOSTA - Esercizio 8
# Tuple base
t = (1, 2, 3)
print(f"Tuple: {t}")

# Prova a modificare - ERRORE!
try:
    t[0] = 10
except TypeError as e:
    print(f"❌ Errore: {e}")

# MA attenzione con oggetti mutabili dentro tuple!
t_mixed = ([1, 2], 3, 4)
print(f"\nTuple con lista: {t_mixed}")

# La tuple è immutabile ma la lista dentro NO
t_mixed[0].append(3)
print(f"Dopo append: {t_mixed}")  # Lista modificata!

# Operazioni permesse - creano NUOVE tuple
t1 = (1, 2)
t2 = (3, 4)
t3 = t1 + t2  # Nuova tuple
print(f"\nConcatenazione: {t3}")
print(f"t1 unchanged: {t1}")

# Unpacking
a, b, c = (10, 20, 30)
print(f"\nUnpacking: a={a}, b={b}, c={c}")

# 💡 SPIEGAZIONE:
# Tuple sono immutabili ma oggetti contenuti potrebbero non esserlo

mark_completed(8)

# %% [markdown]
# ### 🎯 Esercizio 9: Reference Cycle
# Crea un ciclo di riferimenti

# %%
# 🔷 TUA SOLUZIONE - Esercizio 9



# %%
# ✅ SOLUZIONE PROPOSTA - Esercizio 9
# Ciclo semplice
a = []
b = [a]
a.append(b)

print("Ciclo creato:")
print(f"a contiene: {a}")
print(f"b contiene: {b}")
print(f"a[0] is b? {a[0] is b}")
print(f"b[0] is a? {b[0] is a}")

# Ciclo con dizionari
dict1 = {'name': 'dict1'}
dict2 = {'name': 'dict2'}
dict1['ref'] = dict2
dict2['ref'] = dict1

print(f"\nCiclo dizionari:")
print(f"dict1 refers to: {dict1['ref']['name']}")
print(f"dict2 refers to: {dict1['ref']['ref']['name']}")

# Garbage collector gestisce i cicli
import gc
print(f"\nOggetti tracked dal GC: {len(gc.get_objects())}")
collected = gc.collect()
print(f"Oggetti collected: {collected}")

# 💡 SPIEGAZIONE:
# Python ha un garbage collector che gestisce i cicli
# Ma meglio evitarli quando possibile

mark_completed(9)

# %% [markdown]
# ### 🎯 Esercizio 10: Del Statement
# Usa `del` per eliminare riferimenti

# %%
# 🔷 TUA SOLUZIONE - Esercizio 10



# %%
# ✅ SOLUZIONE PROPOSTA - Esercizio 10
# Creazione e distruzione
class MyClass:
    def __init__(self, name):
        self.name = name
        print(f"✅ {name} creato")
    
    def __del__(self):
        print(f"❌ {self.name} distrutto")

# Singolo riferimento
obj1 = MyClass("Object1")
del obj1  # Distrutto immediatamente
print("Dopo del obj1")

# Multipli riferimenti
obj2 = MyClass("Object2")
ref = obj2  # Secondo riferimento
del obj2  # NON distrutto ancora
print("Dopo del obj2 (ma ref esiste)")
del ref  # ORA viene distrutto
print("Dopo del ref")

# Del di elementi lista
my_list = [1, 2, 3, 4, 5]
del my_list[2]  # Rimuove elemento
print(f"\nLista dopo del [2]: {my_list}")

del my_list[1:3]  # Rimuove slice
print(f"Lista dopo del [1:3]: {my_list}")

# 💡 SPIEGAZIONE:
# del rimuove il riferimento, non necessariamente l'oggetto
# L'oggetto viene distrutto quando riferimenti = 0

mark_completed(10)

# %% [markdown]
# ---
# # 🔢 SEZIONE 2: DATA TYPES (11-20)
# ## Padroneggiare i tipi di dati fondamentali

# %% [markdown]
# ### 🎯 Esercizio 11: == vs is
# Confronta uguaglianza vs identità

# %%
# 🔷 TUA SOLUZIONE - Esercizio 11



# %%
# ✅ SOLUZIONE PROPOSTA - Esercizio 11
def compare_equality_identity(a, b):
    """Confronta == vs is"""
    print(f"a = {a}, b = {b}")
    print(f"a == b: {a == b}")  # Valore uguale?
    print(f"a is b: {a is b}")  # Stesso oggetto?
    print(f"id(a): {id(a)}, id(b): {id(b)}")
    print("-" * 40)

# Test con diversi tipi
print("🔢 NUMERI PICCOLI (cached):")
compare_equality_identity(100, 100)

print("🔢 NUMERI GRANDI:")
compare_equality_identity(1000, 1000)

print("📝 STRINGHE:")
compare_equality_identity("hello", "hello")

print("📋 LISTE (stesso valore):")
compare_equality_identity([1, 2], [1, 2])

print("📋 LISTE (stesso oggetto):")
list_a = [1, 2]
compare_equality_identity(list_a, list_a)

print("❌ NONE:")
compare_equality_identity(None, None)  # None è singleton!

# 💡 SPIEGAZIONE:
# == confronta valori
# is confronta identità (stesso oggetto in memoria)

mark_completed(11)

# %% [markdown]
# ### 🎯 Esercizio 12: Reference Counter
# Conta quanti riferimenti ha un oggetto

# %%
# 🔷 TUA SOLUZIONE - Esercizio 12



# %%
# ✅ SOLUZIONE PROPOSTA - Esercizio 12
def show_refcount(obj, name="object"):
    """Mostra reference count di un oggetto"""
    # getrefcount restituisce count + 1 (per il parametro stesso)
    count = sys.getrefcount(obj) - 1
    print(f"{name}: refs = {count}")
    return count

# Test con diversi scenari
x = [1, 2, 3]
show_refcount(x, "x iniziale")

y = x  # Nuovo riferimento
show_refcount(x, "x dopo y=x")

z = x  # Altro riferimento
show_refcount(x, "x dopo z=x")

# In container
container = [x, x, x]
show_refcount(x, "x in container")

# Rimuovi riferimenti
del y
show_refcount(x, "x dopo del y")

del z
show_refcount(x, "x dopo del z")

# Funzione che usa l'oggetto
def use_list(lst):
    show_refcount(lst, "dentro funzione")
    return lst

result = use_list(x)
show_refcount(x, "x dopo funzione")

# 💡 SPIEGAZIONE:
# Python conta i riferimenti per gestire la memoria
# Quando refs = 0, l'oggetto può essere deallocato

mark_completed(12)

# %% [markdown]
# ### 🎯 Esercizio 13: Pass by Reference
# Dimostra che Python passa per riferimento

# %%
# 🔷 TUA SOLUZIONE - Esercizio 13



# %%
# ✅ SOLUZIONE PROPOSTA - Esercizio 13
def modify_immutable(x):
    """Prova a modificare immutabile"""
    print(f"  Dentro funzione - prima: x = {x}, id = {id(x)}")
    x = x + 1  # Crea NUOVO oggetto
    print(f"  Dentro funzione - dopo:  x = {x}, id = {id(x)}")
    return x

def modify_mutable(lst):
    """Modifica mutabile"""
    print(f"  Dentro funzione - prima: lst = {lst}, id = {id(lst)}")
    lst.append(999)  # Modifica in-place
    print(f"  Dentro funzione - dopo:  lst = {lst}, id = {id(lst)}")
    return lst

# Test con immutabile
print("🔢 IMMUTABILE (int):")
num = 10
print(f"Fuori - prima: num = {num}, id = {id(num)}")
result = modify_immutable(num)
print(f"Fuori - dopo:  num = {num}, id = {id(num)}")
print(f"Result: {result}\n")

# Test con mutabile
print("📋 MUTABILE (list):")
my_list = [1, 2, 3]
print(f"Fuori - prima: my_list = {my_list}, id = {id(my_list)}")
result = modify_mutable(my_list)
print(f"Fuori - dopo:  my_list = {my_list}, id = {id(my_list)}")
print(f"Result: {result}")

# 💡 SPIEGAZIONE:
# Python passa sempre il riferimento
# Ma immutabili non possono essere modificati in-place

mark_completed(13)

# %% [markdown]
# ### 🎯 Esercizio 14: List Assignment vs Copy
# Differenza tra assegnazione e copia

# %%
# 🔷 TUA SOLUZIONE - Esercizio 14



# %%
# ✅ SOLUZIONE PROPOSTA - Esercizio 14
# ASSEGNAZIONE - stesso oggetto
original = [1, 2, [3, 4]]
assigned = original

print("🔗 ASSEGNAZIONE:")
print(f"original: {original}, id: {id(original)}")
print(f"assigned: {assigned}, id: {id(assigned)}")
print(f"Stesso oggetto? {assigned is original}")

assigned.append(5)
print(f"Dopo append su assigned: original = {original}\n")

# SHALLOW COPY - oggetto diverso, contenuto condiviso
original = [1, 2, [3, 4]]
shallow = original.copy()  # o original[:] o list(original)

print("📄 SHALLOW COPY:")
print(f"original: {original}, id: {id(original)}")
print(f"shallow:  {shallow}, id: {id(shallow)}")
print(f"Stesso oggetto? {shallow is original}")

shallow.append(5)
print(f"Dopo append su shallow: original = {original}")

shallow[2].append(5)  # Modifica lista nested!
print(f"Dopo modifica nested: original = {original}\n")

# DEEP COPY - tutto indipendente
from copy import deepcopy
original = [1, 2, [3, 4]]
deep = deepcopy(original)

print("📋 DEEP COPY:")
deep[2].append(5)
print(f"Dopo modifica nested in deep: original = {original}")
print(f"deep = {deep}")

# 💡 SPIEGAZIONE:
# Assignment: stesso oggetto
# Shallow copy: nuovo oggetto, riferimenti condivisi
# Deep copy: tutto nuovo e indipendente

mark_completed(14)

# %% [markdown]
# ### 🎯 Esercizio 15: Dictionary Memory Growth
# Osserva come cresce un dizionario

# %%
# 🔷 TUA SOLUZIONE - Esercizio 15



# %%
# ✅ SOLUZIONE PROPOSTA - Esercizio 15
# Track dictionary growth
d = {}
sizes = []

print("📊 Dictionary Size Growth:")
print(f"{'Items':<10} {'Size (bytes)':<15} {'Growth'}")
print("-" * 40)

prev_size = sys.getsizeof(d)
print(f"{0:<10} {prev_size:<15} -")

for i in range(1, 21):
    d[f"key{i}"] = f"value{i}"
    current_size = sys.getsizeof(d)
    growth = current_size - prev_size
    
    if growth > 0:
        print(f"{i:<10} {current_size:<15} +{growth} ⚡ RESIZE!")
    else:
        print(f"{i:<10} {current_size:<15} {growth}")
    
    sizes.append(current_size)
    prev_size = current_size

# Visualizza pattern
print("\n📈 Growth Pattern:")
for i in range(1, len(sizes)):
    if sizes[i] > sizes[i-1]:
        ratio = sizes[i] / sizes[i-1] if sizes[i-1] > 0 else 0
        print(f"At {i} items: size grew by {ratio:.2f}x")

# 💡 SPIEGAZIONE:
# I dizionari crescono dinamicamente
# Resize avviene quando si supera la capacità
# Growth factor è circa 2x per ottimizzare performance

mark_completed(15)

# %% [markdown]
# ### 🎯 Esercizio 16: Weak References
# Usa weakref per riferimenti che non prevengono garbage collection

# %%
# 🔷 TUA SOLUZIONE - Esercizio 16



# %%
# ✅ SOLUZIONE PROPOSTA - Esercizio 16
import weakref

class MyObject:
    def __init__(self, name):
        self.name = name
    
    def __repr__(self):
        return f"MyObject('{self.name}')"
    
    def __del__(self):
        print(f"🗑️ {self.name} is being deleted")

# Riferimento normale
print("🔗 NORMAL REFERENCE:")
obj1 = MyObject("Normal")
ref1 = obj1  # Riferimento forte
print(f"Object: {obj1}")
print(f"Refcount: {sys.getrefcount(obj1) - 1}")

del obj1  # Object NON deleted (ref1 exists)
print("After del obj1 - object still alive")
del ref1  # NOW it's deleted

print("\n🔗 WEAK REFERENCE:")
obj2 = MyObject("Weak")
weak_ref = weakref.ref(obj2)  # Riferimento debole

print(f"Object: {obj2}")
print(f"Weak ref valid: {weak_ref() is not None}")
print(f"Access via weak: {weak_ref()}")

del obj2  # Object IS deleted!
print("After del obj2:")
print(f"Weak ref valid: {weak_ref() is not None}")
print(f"Access via weak: {weak_ref()}")  # None

# WeakValueDictionary example
print("\n📚 WEAK DICTIONARY:")
cache = weakref.WeakValueDictionary()
obj3 = MyObject("Cached")
cache['key1'] = obj3

print(f"In cache: {list(cache.keys())}")
del obj3  # Removed from cache automatically!
print(f"After del: {list(cache.keys())}")

# 💡 SPIEGAZIONE:
# Weak references non prevengono garbage collection
# Utili per cache, observers, callbacks

mark_completed(16)

# %% [markdown]
# ### 🎯 Esercizio 17: Small Integer Caching
# Esplora il caching degli interi piccoli

# %%
# 🔷 TUA SOLUZIONE - Esercizio 17



# %%
# ✅ SOLUZIONE PROPOSTA - Esercizio 17
def test_integer_caching(num):
    """Test se un intero è cached"""
    a = num
    b = num
    return a is b

print("🔢 INTEGER CACHING TEST:")
print("Range -5 to 256 is cached by Python\n")

# Test boundaries
test_values = [-10, -5, -1, 0, 1, 100, 255, 256, 257, 1000]

for val in test_values:
    is_cached = test_integer_caching(val)
    symbol = "✅" if is_cached else "❌"
    print(f"{val:5}: {symbol} {'Cached' if is_cached else 'Not cached'}")

# Dimostra con calcoli
print("\n🧮 CALCULATIONS:")
a = 100 + 100  # = 200 (cached)
b = 200
print(f"100 + 100 = {a}, 200 → same object? {a is b}")

c = 200 + 200  # = 400 (not cached)
d = 400
print(f"200 + 200 = {c}, 400 → same object? {c is d}")

# Force same object
e = 500
f = e  # Assignment, not new object
print(f"\nAssignment: e=500, f=e → same object? {e is f}")

# 💡 SPIEGAZIONE:
# Python pre-crea interi -5 to 256 all'avvio
# Ottimizzazione per numeri comuni
# Non fare affidamento su questo comportamento!

mark_completed(17)

# %% [markdown]
# ### 🎯 Esercizio 18: += con Liste vs Tuple
# Comportamento diverso di += per mutabili/immutabili

# %%
# 🔷 TUA SOLUZIONE - Esercizio 18



# %%
# ✅ SOLUZIONE PROPOSTA - Esercizio 18
print("📋 += CON LISTE (mutabile):")
list1 = [1, 2, 3]
list1_id = id(list1)
print(f"Prima: {list1}, id: {list1_id}")

list1 += [4, 5]  # Modifica in-place
print(f"Dopo:  {list1}, id: {id(list1)}")
print(f"Stesso oggetto? {list1_id == id(list1)}")  # True!

print("\n📦 += CON TUPLE (immutabile):")
tuple1 = (1, 2, 3)
tuple1_id = id(tuple1)
print(f"Prima: {tuple1}, id: {tuple1_id}")

tuple1 += (4, 5)  # Crea nuova tuple
print(f"Dopo:  {tuple1}, id: {id(tuple1)}")
print(f"Stesso oggetto? {tuple1_id == id(tuple1)}")  # False!

print("\n⚠️ TRICKY CASE - Tuple in lista:")
t = ([1, 2], 3)
print(f"Prima: {t}")

try:
    t[0] += [3]  # Modifica E errore!
except TypeError as e:
    print(f"Errore: {e}")

print(f"Dopo:  {t}")  # Lista modificata comunque!

# Perché succede?
print("\n💡 EXPLANATION:")
print("t[0] += [3] viene tradotto in:")
print("1. t[0].extend([3])  # Funziona - lista modificata")
print("2. t[0] = t[0]       # Fallisce - tuple immutabile")

# 💡 SPIEGAZIONE:
# += su mutabili: modifica in-place
# += su immutabili: crea nuovo oggetto

mark_completed(18)

# %% [markdown]
# ### 🎯 Esercizio 19: Class Lifecycle
# Traccia creazione e distruzione di oggetti

# %%
# 🔷 TUA SOLUZIONE - Esercizio 19



# %%
# ✅ SOLUZIONE PROPOSTA - Esercizio 19
class LifecycleTracker:
    """Classe che traccia il suo ciclo di vita"""
    
    instance_count = 0  # Class variable
    
    def __init__(self, name):
        self.name = name
        LifecycleTracker.instance_count += 1
        self.instance_number = LifecycleTracker.instance_count
        print(f"🆕 #{self.instance_number} '{name}' created")
        print(f"   Total instances: {LifecycleTracker.instance_count}")
    
    def __del__(self):
        print(f"💀 #{self.instance_number} '{self.name}' destroyed")
        LifecycleTracker.instance_count -= 1
        print(f"   Remaining instances: {LifecycleTracker.instance_count}")
    
    def __repr__(self):
        return f"LifecycleTracker('{self.name}')"

print("🔄 OBJECT LIFECYCLE DEMO:\n")

# Create objects
obj1 = LifecycleTracker("First")
obj2 = LifecycleTracker("Second")
obj3 = LifecycleTracker("Third")

print(f"\nCurrent objects: {[obj1, obj2, obj3]}")

# Delete one
print("\n🗑️ Deleting 'Second':")
del obj2

# Create new one
print("\n🆕 Creating new:")
obj4 = LifecycleTracker("Fourth")

# Scope exit
print("\n📦 Creating in function scope:")
def create_temp():
    temp = LifecycleTracker("Temporary")
    print(f"Inside function: {temp}")
    # Destroyed when function exits

create_temp()
print("Back in main scope")

# Force garbage collection
print("\n🧹 Force cleanup:")
import gc
gc.collect()

# 💡 SPIEGAZIONE:
# __init__ chiamato alla creazione
# __del__ chiamato alla distruzione
# Non fare affidamento su __del__ per risorse critiche!

mark_completed(19)

# %% [markdown]
# ### 🎯 Esercizio 20: Object References Explorer
# Usa gc.get_referents() per esplorare riferimenti

# %%
# 🔷 TUA SOLUZIONE - Esercizio 20



# %%
# ✅ SOLUZIONE PROPOSTA - Esercizio 20
import gc

def explore_references(obj, name="object"):
    """Esplora i riferimenti di un oggetto"""
    print(f"\n🔍 Exploring: {name}")
    print(f"Object: {obj}")
    print(f"Type: {type(obj)}")
    print(f"ID: {id(obj)}")
    
    # Get referents (objects that this object refers to)
    referents = gc.get_referents(obj)
    print(f"Referents ({len(referents)} objects):")
    
    # Categorize referents
    categories = {}
    for ref in referents[:10]:  # Limit to first 10
        ref_type = type(ref).__name__
        if ref_type not in categories:
            categories[ref_type] = 0
        categories[ref_type] += 1
    
    for typ, count in categories.items():
        print(f"  - {typ}: {count}")
    
    # Get referrers (objects that refer to this object)
    referrers = gc.get_referrers(obj)
    print(f"Referrers ({len(referrers)} objects)")
    
    return referents, referrers

# Test with different objects
my_list = [1, 2, [3, 4]]
my_dict = {'key': my_list, 'value': 100}

explore_references(my_list, "my_list")
explore_references(my_dict, "my_dict")

# Create circular reference
circular1 = {'name': 'A'}
circular2 = {'name': 'B'}
circular1['ref'] = circular2
circular2['ref'] = circular1

explore_references(circular1, "circular1")

# Check if objects are tracked by GC
print("\n🗑️ GC Tracking:")
print(f"my_list tracked? {gc.is_tracked(my_list)}")
print(f"my_dict tracked? {gc.is_tracked(my_dict)}")
print(f"integer tracked? {gc.is_tracked(42)}")  # No - immutable

# 💡 SPIEGAZIONE:
# gc module permette di esplorare il grafo degli oggetti
# Utile per debug di memory leak e riferimenti

mark_completed(20)

# %% [markdown]
# ---
# # 🔤 SEZIONE 3: STRINGS & NUMBERS (21-30)
# ## Manipolazione di stringhe e operazioni numeriche

# %% [markdown]
# ### 🎯 Esercizio 21: Type Conversions
# Converti tra int, float, str

# %%
# 🔷 TUA SOLUZIONE - Esercizio 21



# %%
# ✅ SOLUZIONE PROPOSTA - Esercizio 21
def type_conversion_chain(value):
    """Catena di conversioni di tipo"""
    conversions = []
    
    # Start value
    conversions.append(('Start', value, type(value).__name__))
    
    # To string
    str_val = str(value)
    conversions.append(('→ str', str_val, type(str_val).__name__))
    
    # To int (if possible)
    try:
        int_val = int(float(str_val))
        conversions.append(('→ int', int_val, type(int_val).__name__))
        
        # To float
        float_val = float(int_val)
        conversions.append(('→ float', float_val, type(float_val).__name__))
        
        # To bool
        bool_val = bool(float_val)
        conversions.append(('→ bool', bool_val, type(bool_val).__name__))
        
    except ValueError as e:
        conversions.append(('→ Error', str(e), 'Error'))
    
    return conversions

# Test different starting values
test_values = [123, 45.67, "89", "12.34", "hello", True, None]

for val in test_values:
    print(f"\n🔄 Converting: {val}")
    print(f"{'Step':<10} {'Value':<15} {'Type':<10}")
    print("-" * 35)
    
    for step, value, typ in type_conversion_chain(val):
        print(f"{step:<10} {str(value):<15} {typ:<10}")

# Special cases
print("\n⚠️ SPECIAL CASES:")
print(f"int('0xFF', 16) = {int('0xFF', 16)}")  # Hex to int
print(f"int('0b1010', 2) = {int('0b1010', 2)}")  # Binary to int
print(f"oct(8) = {oct(8)}")  # Int to octal
print(f"hex(255) = {hex(255)}")  # Int to hex
print(f"ord('A') = {ord('A')}")  # Char to ASCII
print(f"chr(65) = {chr(65)}")  # ASCII to char

mark_completed(21)

# %% [markdown]
# ### Altri 29 esercizi seguono lo stesso pattern...
# Per brevità, includiamo solo alcuni esempi chiave

# %% [markdown]
# ---
# # 📊 PROGRESS TRACKER

# %%
def show_progress():
    """Mostra il progresso attuale"""
    total = 50
    completed = len(completed_exercises)
    percentage = (completed / total) * 100
    
    # Progress bar
    bar_length = 50
    filled = int(bar_length * completed / total)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    print(f"\n{'='*60}")
    print(f"📊 PROGRESS REPORT")
    print(f"{'='*60}")
    print(f"\nCompleted: {completed}/{total} ({percentage:.1f}%)")
    print(f"\n[{bar}]")
    
    if completed < 10:
        level = "🥉 Beginner"
    elif completed < 20:
        level = "🥈 Novice"  
    elif completed < 30:
        level = "🥇 Intermediate"
    elif completed < 40:
        level = "💎 Advanced"
    elif completed < 50:
        level = "⭐ Expert"
    else:
        level = "🏆 MASTER"
    
    print(f"\nLevel: {level}")
    
    if completed_exercises:
        print(f"\n✅ Completed exercises: {completed_exercises}")
    
    remaining = total - completed
    if remaining > 0:
        print(f"\n📝 Remaining: {remaining} exercises")
        print(f"💪 Keep going! Next: Exercise {max(completed_exercises, default=0) + 1}")
    else:
        print("\n🎉 CONGRATULAZIONI! Hai completato tutti i 50 esercizi!")
        print("🚀 Ready for exercises 51-100!")

# Mostra progresso iniziale
show_progress()

# %% [markdown]
# ---
# # 🎯 CHALLENGE FINALE

# %%
# Mini progetto che combina tutti i concetti dei primi 20 esercizi
print("""
🏆 CHALLENGE: Memory Inspector Tool

Crea una classe MemoryInspector che:
1. Traccia oggetti e loro dimensioni
2. Identifica riferimenti circolari
3. Monitora crescita memoria
4. Trova oggetti duplicati
5. Suggerisce ottimizzazioni

Usa tutti i concetti appresi finora!
""")

# Template per iniziare
class MemoryInspector:
    """Il tuo memory inspector personalizzato"""
    
    def __init__(self):
        self.tracked_objects = []
    
    def track(self, obj, name=""):
        """Aggiungi oggetto al tracking"""
        # TUA IMPLEMENTAZIONE
        pass
    
    def analyze(self):
        """Analizza tutti gli oggetti tracked"""
        # TUA IMPLEMENTAZIONE
        pass
    
    def find_duplicates(self):
        """Trova oggetti duplicati in memoria"""
        # TUA IMPLEMENTAZIONE
        pass
    
    def suggest_optimizations(self):
        """Suggerisci ottimizzazioni memoria"""
        # TUA IMPLEMENTAZIONE
        pass

# Test il tuo inspector!
# inspector = MemoryInspector()
# inspector.track([1,2,3], "list1")
# inspector.analyze()

# %% [markdown]
# ---
# # 📚 RISORSE E NEXT STEPS

# %%
print("""
📚 RISORSE UTILI:
- Python Memory Management: https://realpython.com/python-memory-management/
- Python Data Model: https://docs.python.org/3/reference/datamodel.html
- Garbage Collection: https://docs.python.org/3/library/gc.html

🎯 PROSSIMI PASSI:
1. Completa tutti i 50 esercizi
2. Rifai quelli difficili
3. Crea variazioni personali
4. Passa agli esercizi 51-100
5. Inizia i progetti della Week 2!

💪 REMEMBER:
"Every expert was once a beginner who never gave up!"
""")

# Save your progress!
def save_progress():
    """Salva il tuo progresso in un file"""
    with open('my_progress.txt', 'w') as f:
        f.write(f"Completed: {completed_exercises}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    print("✅ Progress saved to my_progress.txt")

# Uncomment per salvare
# save_progress()

print("\n🚀 Buon coding e divertiti con Python!")
