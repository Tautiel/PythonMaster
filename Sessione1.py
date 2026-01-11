import copy
from collections import Counter, deque


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


# Esercio 5: Variable Scope


global_var = "Sono globale"

def test_scope():
    """
    Local vs global
    """
    
    local_var = "Sono locale"
    
    def inner_function():
        nonlocal local_var
        local_var = "Modificata da inner"
        inner_var = "Sono in inner"
        print(f"Inner - local_var: {local_var}")
        print(f"Inner - global_var: {global_var}")
        
    print(f"Outer - local_var prima: {local_var}")
    inner_function()
    print(f"Outer local_var dopo: {local_var}")


test_scope()

# SEZIONE 2

# Esercizio 7: Number Systems
 
def number_systems(n):
    """
    Mostra un numero in diversi sistemi numerici
    """
    if isinstance(n, str):
        n = int(n)
        
    print(f"Decimale: {n}")
    print(f"binario: {bin(n)}")
    print(f"Ottale: {oct(n)}")
    print(f"Esadecimale: {hex(n)}")
    
    
    if n == hex(n) or bin(n):
        print(f"Conversione da binario/hex a decimale: {n}")


number_systems(6)

# ESERCIZIO 9 PRECISION FLOAT

def float_precision():
    a = 0.1 + 0.1 + 0.1
    b = 0.3
    print(f"0.1 + 0.1 + 0.1 = {a}")
    print(f"0.3 = {b}")
    print(f"Sono uguali? {a == b}")
    print(f"Differenza: {abs(a - b)}")
    
    #Soluzione con decimal
    
    from decimal import Decimal
    x = Decimal('0.1') + Decimal('0.1') + Decimal('0.1')
    y = Decimal('0.3')
    print(f"\ncon Decimal:")
    print(f"Sono uguali? {x == y}")
    
    d = Decimal('0.1') / Decimal('0.1') / Decimal('0.1')
    g = Decimal('0.3')
    print(f"\ncon Decimanl:")
    print(f"Sono uguali? {x == y}")
    
float_precision()     

# ESERCIZIO 10 BOOLEAN PUZZLE

# Valori da testare AND e OR

def boolean_puzzle():
    
    values = [True and True,
              True and False,
              True or True,
              True or False]
    
    print(f"Test della verità:")
    for val in values:
        print(f"{repr(val):4} -> {bool(val)}")
        
    print("Logic operations:")
    print(f"True and False : {True and False}")
    print(f"True or False : {True or False}")
    print(f"not True : {not True}")


boolean_puzzle()

def list_operations():
    
    li_op= [1, 2 ,2 ,3, 4, 4, 5, 6, 6, 7 ,8, 8,]
    
    it = {}
    res = []
    
    for li in li_op:
        if li not in it:
            res.append(li)
            it[li] = True
            
    print(f"La lista iterata senza duplicati è : {res}")
    
    
list_operations()

def dict_operations():
    person = {
        'name': 'Marco',
        'age': 35,
        'skills': ['Python', 'Trading'],
        'location': 'Milano'
    }
    print(f"Dizionario: {person}")
    # Accesso sicuro 
    print(f"Nome: {person.get('name', 'N/A')}")
    print(f"Email: {person.get('email', 'Non presente')}")
    # Update
    person.update({'email': 'marco@example.com', 'age': 36})
    print(f"Dopo update: {person}")
    # Keys, value, item
    print(f"Chiavi: {list(person.keys())}")
    print(f"Valori: {list(person.values())}")
    # Dictionaries comprehension
    squared_numbers = {x: x**2 for x in range(5)}
    print(f"Quadrati: {squared_numbers}")
    # Inverti chiavi e valori
    new_person = {}
    for key, value in person.items():
        if isinstance(value, (list, dict, set)):
            continue
        new_person[value] = key
    print(f"Prima del ciclo for: {person}")
    print(f"Dopo il ciclo for: {new_person}")
dict_operations()

def set_oprations():
    # Trading assets
    portfolio_a = {'BTC', 'ETH', 'ADA', 'DOT'}
    portfolio_b = {'BTC', 'BNB', 'ADA', 'SOL'}
    
    print(f"Porfolio A: {portfolio_a}")
    print(f"portfolio B: {portfolio_b}")
    
    # Operazioni
    
    print(f"Unione (tutti): {portfolio_a | portfolio_b}")
    print(f"Intersezione (comuni): {portfolio_a & portfolio_b}")
    print(f"Differenza (solo in A): {portfolio_a - portfolio_b}")
    print(f"Differenza simmetrica: {portfolio_a ^ portfolio_b}")
    
    # Controlli
    
    print(f"''BTC' in A?{'BTC' in portfolio_a}")
    print(f"A è un subset di B? {portfolio_a.issubset(portfolio_b)}")
    
    portfolio_c = {'ETH', 'XRP', 'SOL', 'MATIC'}
    
    print(f"Unisco i tre portfolio: {portfolio_a | portfolio_b | portfolio_c}")


set_oprations()

def tuple_operation():
    # Coordinate
    point = (10, 20, 30)
    x, y, z = point
    print(f"Coordinate: x = {x}, y = {y}, z = {z}")
    
    # Swap senza temp
    a, b = 5, 10
    print(f"Prima: A = {a}, b = {b}")
    
    # Extended unpacking
    numbers = (1, 2, 3, 4, 5, 6)
    first, *middle, last = numbers 
    print(f"First: {first}, Middle: {middle}, last: {last}")
    
    # Named tuple
    from collections import namedtuple
    Trade = namedtuple('Trade', ['symbol', 'price', 'quantity'])
    trade = Trade('BTC', 45000, 0.5)
    print(f"Trade: {trade.symbol} @ ${trade.price}")
    
    # Funzione che ritorna multiple values
    
    def get_stats(values):
        minimo = min(values)
        massimo = max(values)
        totale = sum(values)
        return minimo, massimo, totale   # <-- tupla
    
    stats = get_stats(numbers)
    print(f"Statistiche (min, max, sum): {stats}")
    
    # Oppure unpacking
    mn, mx, sm = get_stats(numbers)
    print(f"Minimo: {mn}, Massimo: {mx}, Somma: {sm}")


tuple_operation()

def nested_structures():
    
    trading_data = {
        'BTC': {
            'price': 80000,
            'volume': 1000000000,
            'trades': [
                {'time': '10:00', 'price': 89956, 'qty': 0.1},
                {'time': '10:05', 'price': 91456, 'qty': 0.2}    
            ]
        },
        'ETH': {
            'price': 3000,
            'volume': 500000,
            'trades': [
               {'time': '10:00', 'price': 2547, 'qty': 1.0},
               {'time': '10:05', 'price': 3054, 'qty': 0.5}
            ]
        }
    }
    
    # Accesso ai dati
    btc_price = trading_data['BTC']['price']
    print(f"BTC price: ${btc_price}")
    
    # Primo trade ETH
    first_trade_eth = trading_data['ETH']['trades'][0]
    print(f"First ETH trade: : {first_trade_eth}")
    
    # Calcolo valore totale trade
    total_btc_value = sum(
        trades['price'] * trades['qty']
        for trades in trading_data['BTC']['trades']
    )
    print(f"Total BTC traded value: ${total_btc_value}")
    
    # Trova il prezzo medio di BTC
    
    mid_price_btc = sum(
        trades['price'] * trades['qty'] for trades in trading_data['BTC']['trades']) / sum(trades['qty'] for trades in trading_data['BTC']['trades'])
    
    print(f"Accesso al prezzo medio dei trades di BTC {mid_price_btc}")
    

nested_structures()

def list_basic_operations():
    
    prices = [100, 5.5, 5, 357, 55, 3000, 22, 0.11, 5.12, 250]
    
    prices.append(35000)
    
    prices.insert(2, 2000)
    
    first =  prices.pop(0)
    
    index_3000 = prices.index(3000)
    
    count_2000 = prices.count(2000)
    
    prices.sort()
    
    prices.reverse()
    
    print(f"lista finale {prices}")
    print(f"primo rimosso {first}")
    print(f"index di 3000 {index_3000}")
    
    return prices

list_basic_operations()

def advanced_slicing():
    
    data = list(range(1,50))
    first_5 = data[:5]
    last_5 = data[-5:]
    middle = data [15:25]
    every_second = data[::2]
    every_third_from_second = data[1::3]
    revesed_list = data[::-1]
    last_10_reversed = data[-10:][::-1]
    
    print(f"primi 5:{first_5}, ultimi 5:{last_5}, ultimi 10 inverso:{last_10_reversed}")
    
advanced_slicing()

def list_comprehension_practices():
    
    squares = [x**2 for x in range(1,11)]
    print(f"Quadrati: {squares}")
    
    evens = [x for x in range(1, 21) if x % 2 == 0]
    print(f"pari {evens}")
    
    prices = [100 ,250, 75, 300, 150]
    discouted = [p * 0.9 for p in prices]
    print(f"prezzi scontati:{discouted}")
    
    premium_disounted = [p *0.8 for p in prices if p > 100]
    print(f"prezzi premium: {premium_disounted}")
    

list_comprehension_practices()

from collections import Counter


def counter_dict_operation():
    
    token_list = ["BTC", "ETH", "ADA", "TRX", "BTC", "BTC", "HNT", "HBAR", "HNT", "ADA", "ADA", "ETH", "ETH"]
    
    token_dict = Counter(token_list)  
    
    print(f"Prima del modulo Counter : {token_list}")
    print(f"Dopo il modulo Counter {token_dict}")
    
    add_btc = Counter({"BTC" : 3}) + Counter({"BTC" : 6})
    
    print(F"Dopo somma Counter di BTC: {add_btc}")
    
    
counter_dict_operation()

def deque_exercise():
    
    d = deque([1,2,3,4,5])
    
    print(f"Lista deque originale: {d}")
    print(f"Lista deque con rotate: {d.rotate(-2)} ")
        
deque_exercise()

def itertools():
    
    import itertools as it
    
    a = [1, 2]
    b = ['x', 'y']
    p = list(it.product(a, b))
    
    print(f"Le due liste sono:{a} e {b}")
    print(f"Il prodotto cartesiano è: {p}")
    

itertools()

def esercizio_combinatoria_trading():
    
    import itertools as it
    
    print("\n---PRODICT: ASSET X EXCHANGE ---")
    
    assets = ["BTC", "ETH", "SOL"]
    exchange = ["Binance", "Kraken"]
    
    combos_iter = it.product(assets, exchange)
    print(f"Oggetto iteratore", combos_iter)
    
    combos_list = list(combos_iter)
    print(f"Combinazioni asset x exchange:", combos_list)
    
    print(f"\n---PERMUTATIONS: SEQUENZE DI AZIONI ---")
    
    actions = ["buy", "sell", "hold"]
    
    perm_2 = list(it.permutations(actions, 2))
    print(f" Permutazioni di 2 azioni", perm_2)
    
    print("\n---COMBINATIONS: COPPIE DI ASSETS ---")
    
    assets_pairs = list(it.combinations(assets, 2))
    print(" Coppie di asset (ordine non conta):", assets_pairs)
    
    print("\n---COMBINATIONS WITH REPLACEMENT: ASSET CON RIPETIZIONE ---")
    
    core_asset = ["BTC", "ETH"]
    portfolio_slots = list(it.combinations_with_replacement(core_asset, 3))
    print("Combinazione con ripetizione (3 slot BTC/ETH):", portfolio_slots)
    
    
esercizio_combinatoria_trading()

def allocation_combinations(n_asset: int, steps: int = 4, tol: float = 1e-6):
    import itertools as it
    weights = [i / steps for i in range(steps + 1)]
    for combo in it.product(weights, repeat = n_asset):
        if abs(sum(combo) -1.0) < tol:
            yield combo
            
assets = ['AAPL', 'GOOGL', 'MSFT']
allocations = list(allocation_combinations(n_asset=3, steps=4))

print(f"{len(allocations)} allocazioni valide per 3 asset:")
for alloc in allocations[:5]:
    print(" ", dict(zip(assets, alloc)))
print(f"   ...({len(allocations) - 5} altre)")




