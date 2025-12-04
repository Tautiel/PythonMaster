import copy    

# ========================================================================
# SEZIONE 1 VARIABLES E MEMORY (GIORNI 1-2)
# ========================================================================

# ESERCIZIO 1 VARIABLE INSPECTOR


print("-" * 50)
print("ESERCIZI MEMORY E VARIABLES")
print("-" * 50)


def variable_inspector(var, name="variable"):
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
    # Testa con diversi tipi
    x = 42
    variable_inspector(x, "x")

    y = "Python"
    variable_inspector(y, "y")

    z = [1, 2, 3]
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


# ==========================================================================
# ESERCIZIO 1.1: Hello Variables
# ==========================================================================


def esercizio_1_1():
    name = "Marco"
    age = 35
    height = 1.70

    print(f"Buongiorno mi chiamo {name}, ho {age} anni e sono alto circa {height}m")

    print(f"tipo di name: {type(name)}")
    print(f"tipo di age: {type(age)}")
    print(f"tipo di height: {type(height)}")
    return name, age, height


if __name__ == "__main__":
    result = esercizio_1_1()
    assert isinstance(result[0], str), "name deve essere una stringa"
    assert isinstance(result[1], int), "age deve essere un int"
    assert isinstance(result[2], float), "height deve un float"
    print("Eserizio 1.1 completato")

esercizio_1_1()

# ================================================================
# ESERCIZIO 1.2 Multiple Assignment
# ================================================================


def esercizio1_2():
    
    a, b, c = 1, 2, 3
    print(f"a = {a}, b = {b}, c= {c}")
    
    x, y = 10, 20
    print(f"Prima dello Swap x = {x}, y = {y}")
    
    x, y = y, x
    print(f"Dopo lo Swap x = {x}, y = {y}")
    
    return a, b, c, x, y

# Test

if __name__ == "__main__":
    a, b, c, x, y = esercizio1_2()
    assert (a, b, c) == (1, 2, 3), "assegnazione multipla fallita"
    assert (x, y) == (20, 10), "swap fallito"
    print("Esercizio 1.2 completato")
    

esercizio1_2()

# ===================================================================
# ESERCIZZIO 1.3 Variable Name Conviention
# ===================================================================


def esercizio1_3():
    b = 5
    mio_nome = "Marco"
    is_active = True
    _cache = {}
    _internal_use = 0

    for i in range(3):
        print(f"Iterazione: {i}")
    
    for index, value in enumerate([10, 20 ,30, 40]):
        print(f"index = {index}, value = {value} ")
    
    first, _, third = (1, 2, 3)
    print(f"first = {first}, third = {third}")
    
    return mio_nome


if __name__ == "__main__":
    result = esercizio1_3()
    print("Esercizio 1.3 completato")
    
    
# ==============================================
# ESERCIZIO 3 Mutable vs Immutable
# ==============================================

# Immutable (string)
def immutable_mutable():
    s = "Hello"
    print(f"String originale: {s}, ID: {id(s)}")
    s = s + "World"
    print(f"Dopo concatenazione = {s}")
    print("Nuovo oggetto creato!")
    # Mutable (list)
    lst = [1, 2, 3]
    print(f"Lista originale: {lst}, ID = {id(lst)}")
    lst.append(55)
    print(f"Dopo appen: {lst}, ID = {id(lst)}")
    print("Stesso oggetto modificato")
    tpl = (10, 11, 12)
    print(f"LA tupla è ora: {tpl}, ID: {id(tpl)}")


if __name__ == "__main__":
    result = immutable_mutable()
    print("Esercizio 3 completato")
    
# ==============================================================
# ESERCIZIO 2.1 Integers Operations 
# ==============================================================


def integers_operations():
    x = 100
    y = 400
    z = y // x
    d = y / x
    
    print(f"L'operatore di divisione y//x resituisce questo risultato: {z}")
    print(f"ID: {z}, Type: {type(z)}")
    print(f"L'operatore di divisiopne / restituisce questo risultato: {d}")
    print(f"ID: {d}, Type: {type(d)}")
    
    a, b = 17, 5
    
    print(f"{a} // {b} = {a // b}")
    print(f"{a} / {b} = {a / b}")
    print(f"{a} % {b} = {a % b}" )
    # Relazione fondamentale 
    assert a == (a //b) * b + (a % b), "Relazione violata"
    print(f"Verificata: {a} = {a // b} * {b} + {a % b}")
    
    quoziente, resto = divmod(a, b)
    print(f"divmod({a}, {b}) = ({quoziente}, {resto})")
    
    # Potenza
    print(f"2 ** 10 = {2 ** 10}")
    print(f"2 ** 100 = {2 ** 100}")
    
    # pow con modulo utile in crittografia
    print(f"pow(2, 10, 1000) = {pow(2, 10, 1000)}")


    # Applicazione trading per calcoli lotti
    capitale = 10000
    price_per_shares = 156.78
    
    shares_can_buy = capitale // price_per_shares
    remaining_cash = capitale % price_per_shares
    
    print(f"Capitale: ${capitale}")
    print(f"Prezzo azione: ${price_per_shares}")
    print(f"Azioni acquistabili: {int(shares_can_buy)}")
    print(f"Cash rimanente: ${remaining_cash}")
    
    return shares_can_buy


if __name__ == "__main__":
    result = integers_operations()
    print("Esercizio 2.1 completato")
    
    
# MATEMATICA ( 4 OPERAZIONI BASE)


def concetto_1_operazioni_base():
    """Addizione"""
    capitale_iniziale = 10000
    deposito = 300
    capitale_totale = capitale_iniziale + deposito 
    print(f"Capitale iniziale è: {capitale_iniziale}")
    print(f"Deposito è: {deposito}")
    print(f"Totale: { capitale_iniziale} + {deposito} = {capitale_totale}$") 
    """Sottrazione"""
    prezzo_di_acquisto = 100
    prezzo_di_vendita = 120
    guadagno = prezzo_di_vendita - prezzo_di_acquisto 
    print(f"Prezzo di acquisto: {prezzo_di_acquisto}")
    print(f"prezzo di vendita: {prezzo_di_vendita}")
    print(f"Guadagno: {prezzo_di_vendita} - {prezzo_di_acquisto} = {guadagno}")   
    prezzi_vendita_male = 80
    perdita = prezzi_vendita_male - prezzo_di_acquisto
    print(f"Se vendi a: {prezzi_vendita_male}")
    print(f"Risultato = {prezzi_vendita_male} - {prezzo_di_acquisto} = {perdita}")

if __name__ == "__main__":
    result = concetto_1_operazioni_base()
    print("Esercizio 4 operazioni completato")

def test_copy_behavior():
    """
    Differenza tra shallow e deep copy
    """
    # Lista con sottolista
    ls1 = [[1, 2], [3, 4], [5, 6]]
    
    # Shallow copy
    shallow = ls1.copy()
    shallow[0].append(3) # Modifica la sottolista
    print(f"ls1 dopo shallow copy: {ls1}")
    print(f"Shallow copy: {shallow}")
    print("Le sottoliste sono condivise!\n")
    
    # Deep copy
    ls_2 = [[1, 2], [3, 4], [4, 5]]
    deep = copy.deepcopy(ls_2)
    deep[0].append(3)
    
    print(f"ls_2 dopo deep copy: {ls_2}")
    print(f"Deep copy: {deep}")
    print("Copie totalmente indipendenti!\n")
    
    # Dizionari annidati esempio
    dz_1 = {"Utente_1": 
        {"occhi":"marroni", 
        "altezza": 1.85,
        "capelli": "neri"
        },
        "Utente_2":{
            "occhi": "azzurri",
            "altezza": 1.70,
            "capelli": "biondi"
            
        },
    }
    
    deep_dz_1 = copy.deepcopy(dz_1)
    print(f"dz_! dopo deep copy: {dz_1}")
    print(f"Deep copy: {deep_dz_1}")
    print("Copie totalmente indipendenti!\n")   
test_copy_behavior()
