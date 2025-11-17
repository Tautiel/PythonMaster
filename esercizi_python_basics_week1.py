"""
🎯 ESERCIZI PYTHON BASICS - WEEK 1
Esercizi fondamentali su Memory, Variables, e Data Types
Creati per colmare il gap nel materiale esistente
"""

# ============================================
# SEZIONE 1: VARIABLES & MEMORY (Giorni 1-2)
# ============================================

print("=" * 50)
print("ESERCIZI MEMORY & VARIABLES")
print("=" * 50)

# ESERCIZIO 1: Variable Inspector
print("\n📝 ESERCIZIO 1: Variable Inspector")
print("-" * 40)

def variable_inspector(var, name="variable"):
    """
    Analizza una variabile e mostra:
    - Nome, Valore, Tipo, ID memoria, Mutabilità
    """
    # Il tuo codice qui
    print(f"Nome: {name}")
    print(f"Valore: {var}")
    print(f"Tipo: {type(var).__name__}")
    print(f"ID Memoria: {id(var)}")
    
    # Check mutabilità
    immutable_types = (int, float, str, tuple, bool, frozenset)
    is_mutable = not isinstance(var, immutable_types)
    print(f"Mutabile: {'Sì' if is_mutable else 'No'}")
    print()
    return var

# Test dell'esercizio
if __name__ == "__main__":
    # Testa con diversi tipi
    x = 42
    variable_inspector(x, "x")
    
    y = "Python"
    variable_inspector(y, "y")
    
    z = [1, 2, 3]
    variable_inspector(z, "z")


# ESERCIZIO 2: Reference Counter
print("\n📝 ESERCIZIO 2: Reference Counter")
print("-" * 40)

def test_references():
    """
    Dimostra quando Python crea nuovi oggetti vs riusa riferimenti
    """
    # Integers (cached -5 to 256)
    a = 100
    b = 100
    print(f"a = 100, b = 100")
    print(f"a is b? {a is b}")  # True (stesso oggetto)
    print(f"ID a: {id(a)}, ID b: {id(b)}")
    
    # Large integers
    x = 1000
    y = 1000
    print(f"\nx = 1000, y = 1000")
    print(f"x is y? {x is y}")  # Potrebbe essere False
    print(f"ID x: {id(x)}, ID y: {id(y)}")
    
    # Lists (sempre nuovi oggetti)
    list1 = [1, 2, 3]
    list2 = [1, 2, 3]
    print(f"\nlist1 = [1,2,3], list2 = [1,2,3]")
    print(f"list1 is list2? {list1 is list2}")  # False
    print(f"list1 == list2? {list1 == list2}")  # True
    
    # TUO TURNO: Aggiungi test per stringhe
    # ...

test_references()


# ESERCIZIO 3: Mutable vs Immutable Operations
print("\n📝 ESERCIZIO 3: Mutable vs Immutable")
print("-" * 40)

def demonstrate_mutability():
    """
    Mostra la differenza tra operazioni su mutabili e immutabili
    """
    # Immutable (string)
    s = "Hello"
    print(f"String originale: {s}, ID: {id(s)}")
    s = s + " World"
    print(f"Dopo concatenazione: {s}, ID: {id(s)}")
    print("→ Nuovo oggetto creato!\n")
    
    # Mutable (list)
    lst = [1, 2, 3]
    print(f"Lista originale: {lst}, ID: {id(lst)}")
    lst.append(4)
    print(f"Dopo append: {lst}, ID: {id(lst)}")
    print("→ Stesso oggetto modificato!")
    
    # TUO TURNO: Dimostra con tuple vs list
    # ...

demonstrate_mutability()


# ESERCIZIO 4: Shallow vs Deep Copy
print("\n📝 ESERCIZIO 4: Shallow vs Deep Copy")
print("-" * 40)

import copy

def test_copy_behavior():
    """
    Comprendi la differenza tra shallow e deep copy
    """
    # Lista con sottoliste
    original = [[1, 2], [3, 4], [5, 6]]
    
    # Shallow copy
    shallow = original.copy()
    shallow[0].append(3)  # Modifica la sottolista
    
    print(f"Original dopo shallow copy: {original}")
    print(f"Shallow copy: {shallow}")
    print("→ Le sottoliste sono condivise!\n")
    
    # Deep copy
    original2 = [[1, 2], [3, 4], [5, 6]]
    deep = copy.deepcopy(original2)
    deep[0].append(3)
    
    print(f"Original dopo deep copy: {original2}")
    print(f"Deep copy: {deep}")
    print("→ Copie completamente indipendenti!")
    
    # TUO TURNO: Crea un esempio con dizionari annidati
    # ...

test_copy_behavior()


# ESERCIZIO 5: Variable Scope
print("\n📝 ESERCIZIO 5: Variable Scope")
print("-" * 40)

# Global variable
global_var = "Sono globale"

def test_scope():
    """
    Esplora local vs global scope
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
    print(f"Outer - local_var dopo: {local_var}")
    
    # TUO TURNO: Prova a modificare global_var
    # ...

test_scope()


# ============================================
# SEZIONE 2: DATA TYPES (Giorni 3-4)
# ============================================

print("\n" + "=" * 50)
print("ESERCIZI DATA TYPES")
print("=" * 50)

# ESERCIZIO 6: Type Converter
print("\n📝 ESERCIZIO 6: Universal Type Converter")
print("-" * 40)

def smart_convert(value, target_type):
    """
    Converte value nel target_type gestendo errori
    
    Args:
        value: valore da convertire
        target_type: 'int', 'float', 'str', 'bool', 'list'
    
    Returns:
        Valore convertito o messaggio di errore
    """
    try:
        if target_type == 'int':
            return int(value)
        elif target_type == 'float':
            return float(value)
        elif target_type == 'str':
            return str(value)
        elif target_type == 'bool':
            # Personalizza la conversione bool
            if isinstance(value, str):
                return value.lower() not in ['', 'false', '0', 'no']
            return bool(value)
        elif target_type == 'list':
            if isinstance(value, str):
                return list(value)
            return list(value)
        else:
            return f"Tipo non supportato: {target_type}"
    except (ValueError, TypeError) as e:
        return f"Errore conversione: {e}"

# Test
print(smart_convert("42", 'int'))      # 42
print(smart_convert("3.14", 'float'))  # 3.14
print(smart_convert(100, 'str'))       # "100"
print(smart_convert("yes", 'bool'))    # True
print(smart_convert("hello", 'list'))  # ['h','e','l','l','o']


# ESERCIZIO 7: Number Systems
print("\n📝 ESERCIZIO 7: Number Systems Converter")
print("-" * 40)

def number_systems(n):
    """
    Mostra un numero in diversi sistemi numerici
    """
    if isinstance(n, str):
        n = int(n)
    
    print(f"Decimale: {n}")
    print(f"Binario: {bin(n)}")
    print(f"Ottale: {oct(n)}")
    print(f"Esadecimale: {hex(n)}")
    
    # TUO TURNO: Aggiungi conversione da binario/hex a decimale
    # ...
    
    return {
        'decimal': n,
        'binary': bin(n),
        'octal': oct(n),
        'hex': hex(n)
    }

number_systems(42)


# ESERCIZIO 8: String Methods Explorer
print("\n📝 ESERCIZIO 8: String Methods")
print("-" * 40)

def explore_string_methods(text):
    """
    Applica vari metodi string e mostra risultati
    """
    print(f"Originale: '{text}'")
    print(f"Upper: '{text.upper()}'")
    print(f"Lower: '{text.lower()}'")
    print(f"Title: '{text.title()}'")
    print(f"Capitalize: '{text.capitalize()}'")
    print(f"Swapcase: '{text.swapcase()}'")
    print(f"Strip: '{text.strip()}'")
    print(f"Replace 'o' con '0': '{text.replace('o', '0')}'")
    print(f"Split: {text.split()}")
    print(f"Reversed: '{''.join(reversed(text))}'")
    
    # TUO TURNO: Aggiungi altri metodi utili
    # ...

explore_string_methods("  Hello Python World  ")


# ESERCIZIO 9: Float Precision
print("\n📝 ESERCIZIO 9: Float Precision Issues")
print("-" * 40)

def float_precision_demo():
    """
    Dimostra i problemi di precisione dei float
    """
    # Problema classico
    a = 0.1 + 0.1 + 0.1
    b = 0.3
    print(f"0.1 + 0.1 + 0.1 = {a}")
    print(f"0.3 = {b}")
    print(f"Sono uguali? {a == b}")
    print(f"Differenza: {abs(a - b)}")
    
    # Soluzione con decimal
    from decimal import Decimal
    x = Decimal('0.1') + Decimal('0.1') + Decimal('0.1')
    y = Decimal('0.3')
    print(f"\nCon Decimal:")
    print(f"Sono uguali? {x == y}")
    
    # TUO TURNO: Test con divisioni
    # ...

float_precision_demo()


# ESERCIZIO 10: Boolean Logic Puzzles
print("\n📝 ESERCIZIO 10: Boolean Logic")
print("-" * 40)

def boolean_puzzles():
    """
    Esercizi di logica booleana
    """
    # Valori da testare
    values = [0, 1, "", "text", [], [1,2], None, True, False]
    
    print("Truthiness test:")
    for val in values:
        print(f"{repr(val):10} → {bool(val)}")
    
    print("\nLogic operations:")
    print(f"True and False = {True and False}")
    print(f"True or False = {True or False}")
    print(f"not True = {not True}")
    print(f"1 and 2 = {1 and 2}")  # Ritorna 2!
    print(f"0 or 3 = {0 or 3}")    # Ritorna 3!
    
    # TUO TURNO: Crea truth table per AND/OR
    # ...

boolean_puzzles()


# ============================================
# SEZIONE 3: COLLECTIONS BASICS (Giorni 5-7)
# ============================================

print("\n" + "=" * 50)
print("ESERCIZI COLLECTIONS")
print("=" * 50)

# ESERCIZIO 11: List Operations
print("\n📝 ESERCIZIO 11: List Mastery")
print("-" * 40)

def list_operations():
    """
    Operazioni complete su liste
    """
    # Creazione
    lst = [1, 2, 3, 4, 5]
    print(f"Lista originale: {lst}")
    
    # Operazioni base
    lst.append(6)
    print(f"Dopo append(6): {lst}")
    
    lst.insert(0, 0)
    print(f"Dopo insert(0, 0): {lst}")
    
    removed = lst.pop()
    print(f"Dopo pop(): {lst}, rimosso: {removed}")
    
    lst.remove(3)
    print(f"Dopo remove(3): {lst}")
    
    # Slicing
    print(f"lst[2:5] = {lst[2:5]}")
    print(f"lst[::2] = {lst[::2]}")
    print(f"lst[::-1] = {lst[::-1]}")
    
    # TUO TURNO: List comprehension per quadrati
    squares = [x**2 for x in lst]
    print(f"Quadrati: {squares}")
    
    # Challenge: Rimuovi duplicati mantenendo ordine
    # ...

list_operations()


# ESERCIZIO 12: Dictionary Magic
print("\n📝 ESERCIZIO 12: Dictionary Operations")
print("-" * 40)

def dict_operations():
    """
    Operazioni avanzate con dizionari
    """
    # Creazione
    person = {
        'name': 'Marco',
        'age': 30,
        'skills': ['Python', 'Trading'],
        'location': 'Milano'
    }
    
    print(f"Dizionario: {person}")
    
    # Accesso sicuro
    print(f"Nome: {person.get('name', 'N/A')}")
    print(f"Email: {person.get('email', 'Non presente')}")
    
    # Update
    person.update({'email': 'marco@example.com', 'age': 31})
    print(f"Dopo update: {person}")
    
    # Keys, values, items
    print(f"Chiavi: {list(person.keys())}")
    print(f"Valori: {list(person.values())}")
    
    # Dictionary comprehension
    squared_numbers = {x: x**2 for x in range(5)}
    print(f"Quadrati: {squared_numbers}")
    
    # TUO TURNO: Inverti chiavi e valori
    # ...

dict_operations()


# ESERCIZIO 13: Set Operations
print("\n📝 ESERCIZIO 13: Set Theory")
print("-" * 40)

def set_operations():
    """
    Operazioni su insiemi
    """
    # Trading assets
    portfolio_a = {'BTC', 'ETH', 'ADA', 'DOT'}
    portfolio_b = {'BTC', 'BNB', 'ADA', 'SOL'}
    
    print(f"Portfolio A: {portfolio_a}")
    print(f"Portfolio B: {portfolio_b}")
    
    # Operazioni
    print(f"Unione (tutti): {portfolio_a | portfolio_b}")
    print(f"Intersezione (comuni): {portfolio_a & portfolio_b}")
    print(f"Differenza (solo in A): {portfolio_a - portfolio_b}")
    print(f"Differenza simmetrica: {portfolio_a ^ portfolio_b}")
    
    # Controlli
    print(f"'BTC' in A? {'BTC' in portfolio_a}")
    print(f"A è subset di B? {portfolio_a.issubset(portfolio_b)}")
    
    # TUO TURNO: Trova assets unici in 3 portfolio
    # ...

set_operations()


# ESERCIZIO 14: Tuple Unpacking
print("\n📝 ESERCIZIO 14: Tuple Magic")
print("-" * 40)

def tuple_operations():
    """
    Tuple e unpacking avanzato
    """
    # Coordinate
    point = (10, 20, 30)
    x, y, z = point
    print(f"Coordinate: x={x}, y={y}, z={z}")
    
    # Swap senza temp
    a, b = 5, 10
    print(f"Prima: a={a}, b={b}")
    a, b = b, a
    print(f"Dopo swap: a={a}, b={b}")
    
    # Extended unpacking
    numbers = (1, 2, 3, 4, 5, 6)
    first, *middle, last = numbers
    print(f"First: {first}, Middle: {middle}, Last: {last}")
    
    # Named tuples
    from collections import namedtuple
    Trade = namedtuple('Trade', ['symbol', 'price', 'quantity'])
    trade = Trade('BTC', 45000, 0.5)
    print(f"Trade: {trade.symbol} @ ${trade.price}")
    
    # TUO TURNO: Funzione che ritorna multiple values
    # ...

tuple_operations()


# ESERCIZIO 15: Nested Collections
print("\n📝 ESERCIZIO 15: Nested Structures")
print("-" * 40)

def nested_structures():
    """
    Lavora con strutture annidate
    """
    # Trading data structure
    trading_data = {
        'BTC': {
            'price': 45000,
            'volume': 1000000,
            'trades': [
                {'time': '10:00', 'price': 44900, 'qty': 0.1},
                {'time': '10:05', 'price': 45100, 'qty': 0.2}
            ]
        },
        'ETH': {
            'price': 3000,
            'volume': 500000,
            'trades': [
                {'time': '10:00', 'price': 2990, 'qty': 1.0},
                {'time': '10:05', 'price': 3010, 'qty': 0.5}
            ]
        }
    }
    
    # Accesso ai dati
    btc_price = trading_data['BTC']['price']
    print(f"BTC Price: ${btc_price}")
    
    # Primo trade ETH
    first_eth_trade = trading_data['ETH']['trades'][0]
    print(f"First ETH trade: {first_eth_trade}")
    
    # Calcola valore totale trades BTC
    total_btc_value = sum(
        trade['price'] * trade['qty'] 
        for trade in trading_data['BTC']['trades']
    )
    print(f"Total BTC traded value: ${total_btc_value}")
    
    # TUO TURNO: Trova il prezzo medio di tutti i trades
    # ...

nested_structures()


# ============================================
# MINI PROGETTO FINALE WEEK 1
# ============================================

print("\n" + "=" * 50)
print("MINI PROGETTO: Portfolio Tracker")
print("=" * 50)

class PortfolioTracker:
    """
    Sistema base per tracciare un portfolio crypto
    """
    def __init__(self):
        self.portfolio = {}  # {symbol: quantity}
        self.transactions = []  # Lista di transazioni
    
    def buy(self, symbol, quantity, price):
        """Registra un acquisto"""
        if symbol in self.portfolio:
            self.portfolio[symbol] += quantity
        else:
            self.portfolio[symbol] = quantity
        
        self.transactions.append({
            'type': 'BUY',
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'value': quantity * price
        })
        print(f"✅ Acquistato {quantity} {symbol} @ ${price}")
    
    def sell(self, symbol, quantity, price):
        """Registra una vendita"""
        if symbol not in self.portfolio:
            print(f"❌ Non possiedi {symbol}")
            return
        
        if self.portfolio[symbol] < quantity:
            print(f"❌ Non hai abbastanza {symbol}")
            return
        
        self.portfolio[symbol] -= quantity
        if self.portfolio[symbol] == 0:
            del self.portfolio[symbol]
        
        self.transactions.append({
            'type': 'SELL',
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'value': quantity * price
        })
        print(f"✅ Venduto {quantity} {symbol} @ ${price}")
    
    def show_portfolio(self):
        """Mostra il portfolio attuale"""
        print("\n📊 PORTFOLIO ATTUALE:")
        print("-" * 40)
        if not self.portfolio:
            print("Portfolio vuoto")
        else:
            for symbol, qty in self.portfolio.items():
                print(f"{symbol}: {qty}")
    
    def show_transactions(self):
        """Mostra tutte le transazioni"""
        print("\n📜 STORICO TRANSAZIONI:")
        print("-" * 40)
        for t in self.transactions:
            print(f"{t['type']}: {t['quantity']} {t['symbol']} @ ${t['price']} = ${t['value']}")
    
    def calculate_pnl(self, current_prices):
        """Calcola profit/loss con prezzi attuali"""
        print("\n💰 PROFIT/LOSS:")
        print("-" * 40)
        
        # Calcola valore investito
        total_invested = sum(
            t['value'] for t in self.transactions if t['type'] == 'BUY'
        )
        total_sold = sum(
            t['value'] for t in self.transactions if t['type'] == 'SELL'
        )
        
        # Calcola valore attuale
        current_value = sum(
            qty * current_prices.get(symbol, 0) 
            for symbol, qty in self.portfolio.items()
        )
        
        net_pnl = current_value + total_sold - total_invested
        
        print(f"Investito: ${total_invested:.2f}")
        print(f"Venduto: ${total_sold:.2f}")
        print(f"Valore attuale: ${current_value:.2f}")
        print(f"P&L: ${net_pnl:.2f}")
        
        return net_pnl


# Test del portfolio tracker
if __name__ == "__main__":
    # Crea tracker
    tracker = PortfolioTracker()
    
    # Simula trading
    tracker.buy('BTC', 0.1, 45000)
    tracker.buy('ETH', 2.0, 3000)
    tracker.buy('ADA', 1000, 1.5)
    
    tracker.sell('ADA', 500, 1.8)
    
    # Mostra stato
    tracker.show_portfolio()
    tracker.show_transactions()
    
    # Calcola P&L con prezzi attuali
    current_prices = {
        'BTC': 46000,
        'ETH': 3200,
        'ADA': 1.7
    }
    tracker.calculate_pnl(current_prices)
    
    print("\n" + "=" * 50)
    print("🎉 COMPLIMENTI! Hai completato gli esercizi Week 1!")
    print("=" * 50)


"""
📚 ISTRUZIONI PER L'USO:

1. ESEGUI ogni esercizio uno alla volta
2. MODIFICA il codice dove indicato "TUO TURNO"
3. SPERIMENTA con variazioni
4. USA il debugger (ipdb) per capire meglio
5. DOCUMENTA le difficoltà incontrate

🎯 OBIETTIVI WEEK 1:
- Padronanza memory model Python
- Comprensione profonda dei tipi
- Fluidità con collections base
- Primo mini progetto funzionante

💪 Buon lavoro Marco!
"""
