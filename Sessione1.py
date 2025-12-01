# ========================================================================
# SEZIONE 1 VARIABLES E MEMORY (GIORNI 1-2)
# ========================================================================

# ESERCIZIO 1 VARIABLE INSPECTOR

print("-" * 50)
print("ESERCIZI MEMORY E VARIABLES")
print("-" * 50)

def variable_inspector(var, name = "variable"):
    """Analizza una variabile e mostra :
    Nome, Valore, Tipo, ID memoria, Mutabilità"""
    
    print(f"Nome: {name}")
    print(f"Valore: {var}")
    print(f"Tipo: {type(var).__name__}")
    print(f"ID memoria: {id(var)}")
    
    
    # Check mutabilità
    
    immutable_types = (int, float, str, tuple, bool, frozenset)
    is_mutable = not isinstance(var, immutable_types)
    print(f"Mutable: {'SI' if is_mutable else 'NO'}")
    print()
    return var

# Test esercizio

if __name__ == "__main__":
    #Testa con diversi tipi
    x = 42
    variable_inspector(x, "x")
    
    y = "Python"
    variable_inspector(y, "y")
    
    z = [1,2,3]
    variable_inspector(z, "z")
    
# ESERCIZIO 2 REFERENCE COUNTER

print("Reference counter")

def test_references():
    
    """Dimostra quando Python crea nuovi oggetti vs riusi riferimenti"""
    
    # Integers (caches -5 to 256) 
    a = 100 
    b = 100
    print("a = 100, b = 100")
    print(f"a is b?: {a is b}")
    print(f"ID a: {id(a)}, ID b: {id(b)}")
    
    # Large integers
    x = 1000
    y = 1000
    print(f"x = 1000, y = 1000")
    print(f"x is y?: {x is y}")
    print(f"ID x: {id(x)}, ID y: {id(y)}")
    
    
    # Lists ( sempre nuovi oggetti)
    list1 = [1, 2, 3]
    list2 = [1, 2, 3]
    print(" list1 = [1, 2, 3], list2 = [1, 2, 3]")
    print(f"list is list2?: {list1 is list2}")
    print(f"list1 == list2: {list1 == list2}")
    
    # TUO turno: Aggiungi test per stringhe
    stringa1 = "Hello World"
    stringa2 = "Hello World"
    print("stringa1 = Hello World, stringa2 = Hello World")
    print(f"stringa1 is stringa2?: {stringa1 is stringa2}")
    print(f"stringa1 == stringa2?: {stringa1 == stringa2}")
    
test_references()

