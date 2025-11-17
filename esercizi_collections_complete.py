"""
🎯 ESERCIZI COLLECTIONS COMPLETE - 20 ESERCIZI
Esercizi completi su Liste, Dizionari, Set e Tuple
MANCAVANO NEL MATERIALE - ORA COMPLETI!
"""

print("=" * 60)
print("COLLECTIONS: 20 ESERCIZI PROGRESSIVI")
print("=" * 60)

# ============================================
# SEZIONE 1: LIST OPERATIONS (Esercizi 1-5)
# ============================================

print("\n📚 LISTA EXERCISES")
print("-" * 40)

# ESERCIZIO 1: List Basics
print("\n📝 ESERCIZIO 1: Operazioni Base Liste")
def list_basic_operations():
    """
    Completa le operazioni base su liste
    """
    # Crea una lista di prezzi crypto
    prices = [45000, 3000, 1.5, 0.4, 250]
    
    # TODO: Aggiungi 35000 alla fine
    prices.append(35000)
    
    # TODO: Inserisci 2000 all'indice 2
    prices.insert(2, 2000)
    
    # TODO: Rimuovi il primo elemento
    first = prices.pop(0)
    
    # TODO: Trova l'indice di 3000
    index_3000 = prices.index(3000)
    
    # TODO: Conta quante volte appare 2000
    count_2000 = prices.count(2000)
    
    # TODO: Ordina la lista
    prices.sort()
    
    # TODO: Inverti la lista
    prices.reverse()
    
    print(f"Lista finale: {prices}")
    print(f"Primo rimosso: {first}")
    print(f"Index di 3000: {index_3000}")
    
    return prices

list_basic_operations()


# ESERCIZIO 2: List Slicing Master
print("\n📝 ESERCIZIO 2: Slicing Avanzato")
def advanced_slicing():
    """
    Padroneggia lo slicing delle liste
    """
    data = list(range(1, 21))  # [1, 2, ..., 20]
    
    # TODO: Primi 5 elementi
    first_5 = data[:5]
    
    # TODO: Ultimi 5 elementi
    last_5 = data[-5:]
    
    # TODO: Elementi dall'indice 5 al 15
    middle = data[5:15]
    
    # TODO: Ogni secondo elemento
    every_second = data[::2]
    
    # TODO: Ogni terzo elemento partendo dal secondo
    every_third_from_second = data[1::3]
    
    # TODO: Lista invertita
    reversed_list = data[::-1]
    
    # TODO: Ultimi 10 elementi in ordine inverso
    last_10_reversed = data[-10:][::-1]
    
    print(f"Primi 5: {first_5}")
    print(f"Ultimi 5: {last_5}")
    print(f"Dal 5 al 15: {middle}")
    print(f"Ogni secondo: {every_second}")
    print(f"Ogni terzo dal secondo: {every_third_from_second}")
    print(f"Invertita: {reversed_list[:5]}...")  # Solo primi 5 per brevità
    
    return True

advanced_slicing()


# ESERCIZIO 3: List Comprehensions
print("\n📝 ESERCIZIO 3: List Comprehensions")
def list_comprehension_practice():
    """
    Crea liste con comprehensions
    """
    # TODO: Quadrati dei numeri da 1 a 10
    squares = [x**2 for x in range(1, 11)]
    print(f"Quadrati: {squares}")
    
    # TODO: Solo numeri pari da 1 a 20
    evens = [x for x in range(1, 21) if x % 2 == 0]
    print(f"Pari: {evens}")
    
    # TODO: Prezzi con sconto 10% 
    prices = [100, 250, 75, 300, 150]
    discounted = [p * 0.9 for p in prices]
    print(f"Prezzi scontati: {discounted}")
    
    # TODO: Solo prezzi sopra 100 con sconto 20%
    premium_discounted = [p * 0.8 for p in prices if p > 100]
    print(f"Premium scontati: {premium_discounted}")
    
    # TODO: Lista di tuple (numero, quadrato) per 1-5
    number_squares = [(x, x**2) for x in range(1, 6)]
    print(f"Numero e quadrato: {number_squares}")
    
    # TODO: Nested comprehension - matrice 3x3
    matrix = [[i+j*3 for i in range(1, 4)] for j in range(3)]
    print(f"Matrice 3x3: {matrix}")
    
    return matrix

list_comprehension_practice()


# ESERCIZIO 4: List Methods Challenge
print("\n📝 ESERCIZIO 4: Metodi Liste Challenge")
def list_methods_challenge():
    """
    Usa tutti i metodi delle liste
    """
    crypto_portfolio = ['BTC', 'ETH', 'ADA']
    
    # TODO: Aggiungi 'DOT' e 'SOL' in una sola operazione
    crypto_portfolio.extend(['DOT', 'SOL'])
    
    # TODO: Copia la lista
    portfolio_backup = crypto_portfolio.copy()
    
    # TODO: Rimuovi 'ADA' per valore (non indice)
    crypto_portfolio.remove('ADA')
    
    # TODO: Svuota la copia
    portfolio_backup.clear()
    
    # TODO: Crea lista di 10 zeri
    zeros = [0] * 10
    
    # TODO: Concatena due liste
    list1 = [1, 2, 3]
    list2 = [4, 5, 6]
    combined = list1 + list2
    
    # TODO: Verifica se elemento in lista
    has_btc = 'BTC' in crypto_portfolio
    has_xrp = 'XRP' in crypto_portfolio
    
    print(f"Portfolio: {crypto_portfolio}")
    print(f"Backup (svuotato): {portfolio_backup}")
    print(f"Zeros: {zeros}")
    print(f"Combined: {combined}")
    print(f"Ha BTC? {has_btc}, Ha XRP? {has_xrp}")
    
    return crypto_portfolio

list_methods_challenge()


# ESERCIZIO 5: Nested Lists
print("\n📝 ESERCIZIO 5: Liste Annidate")
def nested_lists_practice():
    """
    Lavora con liste di liste
    """
    # Matrice di trading (symbol, price, quantity)
    trades = [
        ['BTC', 45000, 0.5],
        ['ETH', 3000, 2.0],
        ['ADA', 1.5, 1000],
        ['DOT', 25, 40]
    ]
    
    # TODO: Estrai tutti i simboli
    symbols = [trade[0] for trade in trades]
    print(f"Simboli: {symbols}")
    
    # TODO: Calcola valore totale di ogni trade
    values = [trade[1] * trade[2] for trade in trades]
    print(f"Valori: {values}")
    
    # TODO: Trova il trade con valore massimo
    max_value_index = values.index(max(values))
    max_trade = trades[max_value_index]
    print(f"Trade massimo: {max_trade[0]} = ${values[max_value_index]}")
    
    # TODO: Aggiungi colonna con valore
    for i, trade in enumerate(trades):
        trade.append(values[i])
    
    # TODO: Ordina per valore (ultima colonna)
    trades.sort(key=lambda x: x[3], reverse=True)
    
    print("\nTrades ordinati per valore:")
    for trade in trades:
        print(f"  {trade[0]}: ${trade[3]:.2f}")
    
    return trades

nested_lists_practice()


# ============================================
# SEZIONE 2: DICTIONARY OPERATIONS (Esercizi 6-10)
# ============================================

print("\n\n📚 DICTIONARY EXERCISES")
print("-" * 40)

# ESERCIZIO 6: Dictionary Basics
print("\n📝 ESERCIZIO 6: Operazioni Base Dizionari")
def dict_basic_operations():
    """
    Operazioni fondamentali sui dizionari
    """
    # Portfolio iniziale
    portfolio = {
        'BTC': 0.5,
        'ETH': 2.0,
        'ADA': 1000
    }
    
    # TODO: Aggiungi nuovo asset
    portfolio['DOT'] = 50
    
    # TODO: Modifica quantità ETH
    portfolio['ETH'] = 2.5
    
    # TODO: Rimuovi ADA
    removed = portfolio.pop('ADA')
    
    # TODO: Prova a rimuovere XRP (non esiste) con default
    xrp = portfolio.pop('XRP', 0)
    
    # TODO: Ottieni BTC con get (safe access)
    btc_qty = portfolio.get('BTC', 0)
    
    # TODO: Ottieni tutte le chiavi
    symbols = list(portfolio.keys())
    
    # TODO: Ottieni tutti i valori
    quantities = list(portfolio.values())
    
    # TODO: Ottieni coppie chiave-valore
    items = list(portfolio.items())
    
    print(f"Portfolio: {portfolio}")
    print(f"Rimosso ADA: {removed}")
    print(f"XRP (default): {xrp}")
    print(f"Simboli: {symbols}")
    print(f"Quantità: {quantities}")
    print(f"Items: {items}")
    
    return portfolio

dict_basic_operations()


# ESERCIZIO 7: Dictionary Comprehensions
print("\n📝 ESERCIZIO 7: Dictionary Comprehensions")
def dict_comprehension_practice():
    """
    Crea dizionari con comprehensions
    """
    # TODO: Dizionario quadrati {1: 1, 2: 4, ...}
    squares = {x: x**2 for x in range(1, 11)}
    print(f"Quadrati: {dict(list(squares.items())[:5])}...")  # Primi 5
    
    # TODO: Prezzi in EUR (converti da USD)
    prices_usd = {'BTC': 45000, 'ETH': 3000, 'ADA': 1.5}
    usd_to_eur = 0.85
    prices_eur = {coin: price * usd_to_eur for coin, price in prices_usd.items()}
    print(f"Prezzi EUR: {prices_eur}")
    
    # TODO: Filtra solo asset con prezzo > 1000
    expensive = {k: v for k, v in prices_usd.items() if v > 1000}
    print(f"Asset costosi: {expensive}")
    
    # TODO: Inverti chiavi e valori
    numbers = {'one': 1, 'two': 2, 'three': 3}
    inverted = {v: k for k, v in numbers.items()}
    print(f"Invertito: {inverted}")
    
    # TODO: Crea dict da due liste
    symbols = ['BTC', 'ETH', 'ADA']
    quantities = [0.5, 2.0, 1000]
    portfolio = dict(zip(symbols, quantities))
    print(f"Portfolio da zip: {portfolio}")
    
    return portfolio

dict_comprehension_practice()


# ESERCIZIO 8: Dictionary Methods
print("\n📝 ESERCIZIO 8: Metodi Dizionari")
def dict_methods_practice():
    """
    Tutti i metodi dei dizionari
    """
    config = {'api_key': 'xyz', 'timeout': 30}
    
    # TODO: Update con nuovo dict
    config.update({'timeout': 60, 'retry': 3})
    print(f"Config updated: {config}")
    
    # TODO: Setdefault - aggiungi se non esiste
    config.setdefault('max_connections', 10)
    config.setdefault('timeout', 100)  # Non cambia, esiste già
    print(f"Dopo setdefault: {config}")
    
    # TODO: Copy dictionary
    backup = config.copy()
    
    # TODO: Clear original
    config.clear()
    print(f"Config cleared: {config}")
    print(f"Backup: {backup}")
    
    # TODO: fromkeys - crea dict con chiavi e valore default
    symbols = ['BTC', 'ETH', 'ADA']
    zeros = dict.fromkeys(symbols, 0)
    print(f"Zeros portfolio: {zeros}")
    
    # TODO: Check if key exists
    has_btc = 'BTC' in zeros
    has_xrp = 'XRP' in zeros
    print(f"Ha BTC? {has_btc}, Ha XRP? {has_xrp}")
    
    return backup

dict_methods_practice()


# ESERCIZIO 9: Nested Dictionaries
print("\n📝 ESERCIZIO 9: Dizionari Annidati")
def nested_dict_practice():
    """
    Lavora con dizionari complessi
    """
    exchange_data = {
        'BTC': {
            'price': 45000,
            'volume_24h': 1000000,
            'change_24h': 2.5,
            'trades': [
                {'time': '10:00', 'price': 44900, 'amount': 0.1},
                {'time': '10:05', 'price': 45100, 'amount': 0.2}
            ]
        },
        'ETH': {
            'price': 3000,
            'volume_24h': 500000,
            'change_24h': -1.2,
            'trades': [
                {'time': '10:00', 'price': 2990, 'amount': 1.0}
            ]
        }
    }
    
    # TODO: Accedi al prezzo di BTC
    btc_price = exchange_data['BTC']['price']
    print(f"BTC Price: ${btc_price}")
    
    # TODO: Tutti i simboli con cambio positivo
    positive = [coin for coin, data in exchange_data.items() 
                if data['change_24h'] > 0]
    print(f"Coins in positivo: {positive}")
    
    # TODO: Somma tutti i volumi
    total_volume = sum(data['volume_24h'] for data in exchange_data.values())
    print(f"Volume totale: ${total_volume:,}")
    
    # TODO: Aggiungi nuovo campo 'market_cap' a ogni coin
    market_caps = {'BTC': 900000000000, 'ETH': 350000000000}
    for coin in exchange_data:
        exchange_data[coin]['market_cap'] = market_caps.get(coin, 0)
    
    # TODO: Trova il primo trade di BTC
    first_btc_trade = exchange_data['BTC']['trades'][0]
    print(f"Primo trade BTC: {first_btc_trade}")
    
    return exchange_data

nested_dict_practice()


# ESERCIZIO 10: Dictionary Performance
print("\n📝 ESERCIZIO 10: Performance Dizionari")
def dict_performance_practice():
    """
    Capire performance e uso ottimale
    """
    import time
    
    # Crea grande dizionario
    big_dict = {f'key_{i}': i for i in range(10000)}
    
    # TODO: Test lookup performance
    start = time.time()
    for _ in range(1000):
        value = big_dict.get('key_5000')
    dict_time = time.time() - start
    
    # Confronta con lista
    big_list = list(range(10000))
    start = time.time()
    for _ in range(1000):
        try:
            index = big_list.index(5000)
        except:
            pass
    list_time = time.time() - start
    
    print(f"Dict lookup (1000x): {dict_time:.4f}s")
    print(f"List search (1000x): {list_time:.4f}s")
    print(f"Dict è {list_time/dict_time:.1f}x più veloce!")
    
    # TODO: Memory usage
    import sys
    dict_size = sys.getsizeof(big_dict)
    list_size = sys.getsizeof(big_list)
    print(f"\nMemory - Dict: {dict_size:,} bytes")
    print(f"Memory - List: {list_size:,} bytes")
    
    # Best practices
    print("\n💡 Best Practices:")
    print("- Usa dict per lookup veloci O(1)")
    print("- Usa list per dati ordinati")
    print("- Dict usa più memoria ma è più veloce")
    print("- Keys devono essere immutabili")
    
    return True

dict_performance_practice()


# ============================================
# SEZIONE 3: SET OPERATIONS (Esercizi 11-15)
# ============================================

print("\n\n📚 SET EXERCISES")
print("-" * 40)

# ESERCIZIO 11: Set Basics
print("\n📝 ESERCIZIO 11: Operazioni Base Set")
def set_basic_operations():
    """
    Operazioni fondamentali sui set
    """
    # Portfolio assets
    my_coins = {'BTC', 'ETH', 'ADA', 'DOT'}
    friend_coins = {'BTC', 'XRP', 'ADA', 'SOL'}
    
    # TODO: Aggiungi elemento
    my_coins.add('LINK')
    
    # TODO: Rimuovi elemento (errore se non esiste)
    my_coins.remove('DOT')
    
    # TODO: Discard elemento (no errore se non esiste)
    my_coins.discard('XLM')  # Non esiste, ma OK
    
    # TODO: Pop elemento random
    popped = friend_coins.pop()
    print(f"Rimosso da friend: {popped}")
    
    # TODO: Lunghezza set
    size = len(my_coins)
    print(f"Ho {size} coins")
    
    # TODO: Check membership
    has_btc = 'BTC' in my_coins
    print(f"Ho BTC? {has_btc}")
    
    # TODO: Clear set
    empty_set = set()
    empty_set.clear()
    
    print(f"My coins: {my_coins}")
    print(f"Friend coins: {friend_coins}")
    
    return my_coins

set_basic_operations()


# ESERCIZIO 12: Set Operations
print("\n📝 ESERCIZIO 12: Operazioni Matematiche Set")
def set_math_operations():
    """
    Operazioni matematiche tra set
    """
    portfolio_a = {'BTC', 'ETH', 'ADA', 'DOT', 'LINK'}
    portfolio_b = {'BTC', 'ADA', 'XRP', 'SOL', 'AVAX'}
    watchlist = {'MATIC', 'ATOM', 'BTC'}
    
    # TODO: Unione (tutti gli asset)
    all_assets = portfolio_a | portfolio_b
    print(f"Tutti gli assets: {all_assets}")
    
    # TODO: Intersezione (asset comuni)
    common = portfolio_a & portfolio_b
    print(f"Asset comuni: {common}")
    
    # TODO: Differenza (solo in A)
    only_a = portfolio_a - portfolio_b
    print(f"Solo in A: {only_a}")
    
    # TODO: Differenza simmetrica (in A o B ma non entrambi)
    exclusive = portfolio_a ^ portfolio_b
    print(f"Esclusivi: {exclusive}")
    
    # TODO: Subset check
    small = {'BTC', 'ETH'}
    is_subset = small.issubset(portfolio_a)
    print(f"{small} è subset di A? {is_subset}")
    
    # TODO: Superset check
    is_superset = portfolio_a.issuperset(small)
    print(f"A è superset di {small}? {is_superset}")
    
    # TODO: Disjoint check (nessun elemento comune)
    are_disjoint = portfolio_a.isdisjoint(watchlist)
    print(f"A e watchlist disgiunti? {are_disjoint}")
    
    return common

set_math_operations()


# ESERCIZIO 13: Set Comprehensions
print("\n📝 ESERCIZIO 13: Set Comprehensions")
def set_comprehension_practice():
    """
    Crea set con comprehensions
    """
    # TODO: Set di quadrati pari
    even_squares = {x**2 for x in range(1, 11) if x % 2 == 0}
    print(f"Quadrati pari: {even_squares}")
    
    # TODO: Lettere uniche da stringa
    text = "cryptocurrency"
    unique_letters = {char for char in text}
    print(f"Lettere uniche: {unique_letters}")
    
    # TODO: Prezzi arrotondati
    prices = [45123.45, 3001.23, 1.567, 25.891]
    rounded = {round(p) for p in prices}
    print(f"Prezzi arrotondati: {rounded}")
    
    # TODO: Set da lista con duplicati
    numbers = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    unique = set(numbers)
    print(f"Unici da lista: {unique}")
    
    # TODO: Multipli di 3 o 5 fino a 20
    multiples = {x for x in range(1, 21) if x % 3 == 0 or x % 5 == 0}
    print(f"Multipli di 3 o 5: {multiples}")
    
    return unique_letters

set_comprehension_practice()


# ESERCIZIO 14: Frozen Sets
print("\n📝 ESERCIZIO 14: Frozen Sets (Immutabili)")
def frozen_set_practice():
    """
    Lavora con frozen sets
    """
    # TODO: Crea frozen set
    core_portfolio = frozenset(['BTC', 'ETH', 'ADA'])
    print(f"Core portfolio (frozen): {core_portfolio}")
    
    # TODO: Prova a modificare (genera errore)
    try:
        core_portfolio.add('DOT')
    except AttributeError as e:
        print(f"❌ Non posso modificare: {e}")
    
    # TODO: Operazioni permesse
    trading_portfolio = {'BTC', 'XRP', 'SOL'}
    common = core_portfolio & trading_portfolio
    print(f"Asset comuni: {common}")
    
    # TODO: Frozen set come chiave di dizionario
    portfolio_values = {
        frozenset(['BTC', 'ETH']): 50000,
        frozenset(['ADA', 'DOT']): 2000,
    }
    print(f"Portfolio values: {portfolio_values}")
    
    # TODO: Converti tra set e frozenset
    mutable = set(core_portfolio)
    mutable.add('LINK')
    immutable = frozenset(mutable)
    print(f"Convertito: {immutable}")
    
    return core_portfolio

frozen_set_practice()


# ESERCIZIO 15: Set Use Cases
print("\n📝 ESERCIZIO 15: Casi d'Uso Pratici")
def set_use_cases():
    """
    Casi pratici di uso dei set
    """
    # CASO 1: Rimuovi duplicati mantenendo ordine
    trades = ['BTC', 'ETH', 'BTC', 'ADA', 'ETH', 'DOT', 'BTC']
    
    # Metodo 1: set (perde ordine)
    unique_unordered = list(set(trades))
    print(f"Unici (no ordine): {unique_unordered}")
    
    # Metodo 2: dict.fromkeys (mantiene ordine)
    unique_ordered = list(dict.fromkeys(trades))
    print(f"Unici (con ordine): {unique_ordered}")
    
    # CASO 2: Trova elementi mancanti
    required = {'BTC', 'ETH', 'USDT', 'BNB'}
    current = {'BTC', 'ETH', 'ADA'}
    missing = required - current
    print(f"\nAsset mancanti: {missing}")
    
    # CASO 3: Validazione input
    valid_symbols = {'BTC', 'ETH', 'ADA', 'DOT', 'LINK'}
    user_input = ['BTC', 'XYZ', 'ETH', 'FAKE']
    
    valid_choices = [s for s in user_input if s in valid_symbols]
    invalid_choices = [s for s in user_input if s not in valid_symbols]
    
    print(f"\nScelte valide: {valid_choices}")
    print(f"Scelte invalide: {invalid_choices}")
    
    # CASO 4: Tag system
    article1_tags = {'python', 'trading', 'bot'}
    article2_tags = {'python', 'data', 'analysis'}
    article3_tags = {'javascript', 'web', 'frontend'}
    
    # Articoli con 'python'
    python_articles = []
    all_articles = [
        ('Article 1', article1_tags),
        ('Article 2', article2_tags),
        ('Article 3', article3_tags)
    ]
    
    for title, tags in all_articles:
        if 'python' in tags:
            python_articles.append(title)
    
    print(f"\nArticoli Python: {python_articles}")
    
    return True

set_use_cases()


# ============================================
# SEZIONE 4: TUPLE OPERATIONS (Esercizi 16-20)
# ============================================

print("\n\n📚 TUPLE EXERCISES")
print("-" * 40)

# ESERCIZIO 16: Tuple Basics
print("\n📝 ESERCIZIO 16: Operazioni Base Tuple")
def tuple_basic_operations():
    """
    Operazioni fondamentali sulle tuple
    """
    # TODO: Crea tuple
    coordinates = (10, 20, 30)
    single_element = (42,)  # Nota la virgola!
    empty_tuple = ()
    
    # TODO: Accesso elementi
    x = coordinates[0]
    y = coordinates[1]
    z = coordinates[2]
    print(f"Coordinate: x={x}, y={y}, z={z}")
    
    # TODO: Unpacking
    a, b, c = coordinates
    print(f"Unpacked: a={a}, b={b}, c={c}")
    
    # TODO: Slicing
    first_two = coordinates[:2]
    last_two = coordinates[-2:]
    print(f"Primi due: {first_two}, Ultimi due: {last_two}")
    
    # TODO: Concatenazione
    tuple1 = (1, 2, 3)
    tuple2 = (4, 5, 6)
    combined = tuple1 + tuple2
    print(f"Concatenate: {combined}")
    
    # TODO: Ripetizione
    repeated = (0, 1) * 3
    print(f"Ripetuta: {repeated}")
    
    # TODO: Membership test
    has_20 = 20 in coordinates
    print(f"Ha 20? {has_20}")
    
    # TODO: Count e Index
    numbers = (1, 2, 3, 2, 4, 2)
    count_2 = numbers.count(2)
    index_3 = numbers.index(3)
    print(f"Count di 2: {count_2}, Index di 3: {index_3}")
    
    return coordinates

tuple_basic_operations()


# ESERCIZIO 17: Tuple Unpacking Advanced
print("\n📝 ESERCIZIO 17: Unpacking Avanzato")
def advanced_unpacking():
    """
    Tecniche avanzate di unpacking
    """
    # TODO: Extended unpacking
    numbers = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    
    first, *middle, last = numbers
    print(f"First: {first}, Middle: {middle}, Last: {last}")
    
    first_two, *rest = numbers
    print(f"First two: {first_two}, Rest: {rest}")
    
    *beginning, last_two, last_one = numbers
    print(f"Beginning: {beginning}, Last two: {last_two}, Last: {last_one}")
    
    # TODO: Swap variables
    x, y = 10, 20
    print(f"Prima: x={x}, y={y}")
    x, y = y, x
    print(f"Dopo swap: x={x}, y={y}")
    
    # TODO: Multiple assignment
    a = b = c = 100
    print(f"Multiple: a={a}, b={b}, c={c}")
    
    # TODO: Unpacking in loops
    trades = [
        ('BTC', 45000, 0.5),
        ('ETH', 3000, 2.0),
        ('ADA', 1.5, 1000)
    ]
    
    for symbol, price, qty in trades:
        value = price * qty
        print(f"{symbol}: ${value:.2f}")
    
    # TODO: Ignora valori con _
    data = (10, 20, 30, 40, 50)
    first, _, third, *_ = data
    print(f"First: {first}, Third: {third} (altri ignorati)")
    
    return True

advanced_unpacking()


# ESERCIZIO 18: Named Tuples
print("\n📝 ESERCIZIO 18: Named Tuples")
def named_tuple_practice():
    """
    Usa namedtuple per strutture dati
    """
    from collections import namedtuple
    
    # TODO: Definisci namedtuple per Trade
    Trade = namedtuple('Trade', ['symbol', 'price', 'quantity', 'side'])
    
    # TODO: Crea istanze
    trade1 = Trade('BTC', 45000, 0.5, 'BUY')
    trade2 = Trade(symbol='ETH', price=3000, quantity=2.0, side='SELL')
    
    # TODO: Accesso con nome
    print(f"Trade 1: {trade1.symbol} {trade1.side} @ ${trade1.price}")
    
    # TODO: Accesso con indice (come tuple normale)
    print(f"Trade 2 primo campo: {trade2[0]}")
    
    # TODO: Convert a dictionary
    trade_dict = trade1._asdict()
    print(f"Come dict: {trade_dict}")
    
    # TODO: Replace values (crea nuova istanza)
    updated_trade = trade1._replace(price=46000)
    print(f"Original: ${trade1.price}, Updated: ${updated_trade.price}")
    
    # TODO: Crea da lista/iterabile
    trade_data = ['ADA', 1.5, 1000, 'BUY']
    trade3 = Trade._make(trade_data)
    print(f"Trade da lista: {trade3}")
    
    # TODO: Named tuple per Point
    Point = namedtuple('Point', 'x y z')
    origin = Point(0, 0, 0)
    position = Point(10, 20, 30)
    
    # Calcola distanza
    import math
    distance = math.sqrt(
        (position.x - origin.x)**2 + 
        (position.y - origin.y)**2 + 
        (position.z - origin.z)**2
    )
    print(f"\nDistanza dall'origine: {distance:.2f}")
    
    return Trade

named_tuple_practice()


# ESERCIZIO 19: Tuple Performance
print("\n📝 ESERCIZIO 19: Performance Tuple vs List")
def tuple_performance():
    """
    Confronta performance tuple vs list
    """
    import time
    import sys
    
    # TODO: Crea grande lista e tuple
    big_list = list(range(10000))
    big_tuple = tuple(range(10000))
    
    # TODO: Memory comparison
    list_size = sys.getsizeof(big_list)
    tuple_size = sys.getsizeof(big_tuple)
    
    print(f"Memory List: {list_size:,} bytes")
    print(f"Memory Tuple: {tuple_size:,} bytes")
    print(f"Tuple usa {(list_size-tuple_size)/list_size*100:.1f}% meno memoria")
    
    # TODO: Creation time
    start = time.time()
    for _ in range(10000):
        l = [1, 2, 3, 4, 5]
    list_time = time.time() - start
    
    start = time.time()
    for _ in range(10000):
        t = (1, 2, 3, 4, 5)
    tuple_time = time.time() - start
    
    print(f"\nCreation (10000x):")
    print(f"List: {list_time:.4f}s")
    print(f"Tuple: {tuple_time:.4f}s")
    print(f"Tuple è {list_time/tuple_time:.1f}x più veloce")
    
    # TODO: Iteration time
    start = time.time()
    for _ in range(1000):
        for item in big_list:
            pass
    list_iter_time = time.time() - start
    
    start = time.time()
    for _ in range(1000):
        for item in big_tuple:
            pass
    tuple_iter_time = time.time() - start
    
    print(f"\nIteration (1000x):")
    print(f"List: {list_iter_time:.4f}s")
    print(f"Tuple: {tuple_iter_time:.4f}s")
    
    print("\n💡 Conclusioni:")
    print("- Tuple usa meno memoria")
    print("- Tuple più veloce da creare")
    print("- Iteration simile")
    print("- Usa tuple per dati immutabili!")
    
    return True

tuple_performance()


# ESERCIZIO 20: Tuple Use Cases
print("\n📝 ESERCIZIO 20: Casi d'Uso Tuple")
def tuple_use_cases():
    """
    Quando e come usare le tuple
    """
    # CASO 1: Return multipli valori
    def calculate_stats(numbers):
        """Ritorna min, max, media"""
        return min(numbers), max(numbers), sum(numbers)/len(numbers)
    
    data = [10, 20, 30, 40, 50]
    minimum, maximum, average = calculate_stats(data)
    print(f"Stats: Min={minimum}, Max={maximum}, Avg={average:.1f}")
    
    # CASO 2: Chiavi dizionario (immutabili)
    # Coordinate come chiavi
    grid = {}
    grid[(0, 0)] = "origin"
    grid[(1, 0)] = "east"
    grid[(0, 1)] = "north"
    print(f"\nGrid map: {grid}")
    
    # CASO 3: Dati che non devono cambiare
    # Configurazione
    DB_CONFIG = ('localhost', 5432, 'trading_db', 'user123')
    host, port, database, user = DB_CONFIG
    print(f"\nDB Config: {host}:{port}/{database}")
    
    # CASO 4: Argomenti funzione con *args
    def process_trades(*trades):
        """Processa numero variabile di trades"""
        total = 0
        for symbol, price, qty in trades:
            value = price * qty
            total += value
            print(f"  {symbol}: ${value:.2f}")
        return total
    
    total = process_trades(
        ('BTC', 45000, 0.1),
        ('ETH', 3000, 0.5),
        ('ADA', 1.5, 100)
    )
    print(f"Totale trades: ${total:.2f}")
    
    # CASO 5: Enumerate con unpacking
    portfolio = ['BTC', 'ETH', 'ADA', 'DOT']
    print("\nPortfolio con indici:")
    for index, symbol in enumerate(portfolio, 1):
        print(f"  {index}. {symbol}")
    
    # CASO 6: Zip per combinare liste
    symbols = ['BTC', 'ETH', 'ADA']
    prices = [45000, 3000, 1.5]
    quantities = [0.1, 1.0, 1000]
    
    print("\nPortfolio completo:")
    for sym, price, qty in zip(symbols, prices, quantities):
        print(f"  {sym}: {qty} @ ${price} = ${price*qty:.2f}")
    
    return True

tuple_use_cases()


# ============================================
# PROGETTO FINALE: COLLECTIONS MASTER
# ============================================

print("\n\n" + "=" * 60)
print("PROGETTO FINALE: Trading Data Manager")
print("=" * 60)

class TradingDataManager:
    """
    Gestisce dati trading usando tutte le collections
    """
    def __init__(self):
        # Lista per ordini temporali
        self.trade_history = []
        
        # Dict per portfolio corrente
        self.portfolio = {}
        
        # Set per watchlist
        self.watchlist = set()
        
        # Named tuple per trades
        from collections import namedtuple
        self.Trade = namedtuple('Trade', ['timestamp', 'symbol', 'price', 'qty', 'side'])
        
    def add_trade(self, timestamp, symbol, price, qty, side):
        """Registra nuovo trade"""
        trade = self.Trade(timestamp, symbol, price, qty, side)
        self.trade_history.append(trade)
        
        # Aggiorna portfolio
        if side == 'BUY':
            self.portfolio[symbol] = self.portfolio.get(symbol, 0) + qty
        else:  # SELL
            self.portfolio[symbol] = self.portfolio.get(symbol, 0) - qty
            if self.portfolio[symbol] <= 0:
                del self.portfolio[symbol]
        
        print(f"✅ Trade registrato: {side} {qty} {symbol} @ ${price}")
        return trade
    
    def add_to_watchlist(self, symbols):
        """Aggiunge simboli alla watchlist"""
        if isinstance(symbols, str):
            symbols = [symbols]
        self.watchlist.update(symbols)
        print(f"✅ Aggiunti a watchlist: {symbols}")
    
    def get_portfolio_value(self, current_prices):
        """Calcola valore portfolio"""
        total = 0
        for symbol, qty in self.portfolio.items():
            price = current_prices.get(symbol, 0)
            value = qty * price
            total += value
            print(f"  {symbol}: {qty} @ ${price} = ${value:.2f}")
        return total
    
    def analyze_trades(self):
        """Analizza trading history"""
        if not self.trade_history:
            print("Nessun trade da analizzare")
            return
        
        # Trades per simbolo
        by_symbol = {}
        for trade in self.trade_history:
            if trade.symbol not in by_symbol:
                by_symbol[trade.symbol] = []
            by_symbol[trade.symbol].append(trade)
        
        print("\n📊 ANALISI TRADES:")
        for symbol, trades in by_symbol.items():
            buy_volume = sum(t.qty for t in trades if t.side == 'BUY')
            sell_volume = sum(t.qty for t in trades if t.side == 'SELL')
            avg_price = sum(t.price for t in trades) / len(trades)
            
            print(f"\n{symbol}:")
            print(f"  Trades totali: {len(trades)}")
            print(f"  Volume BUY: {buy_volume}")
            print(f"  Volume SELL: {sell_volume}")
            print(f"  Prezzo medio: ${avg_price:.2f}")
    
    def find_profitable_trades(self):
        """Trova trades profittevoli"""
        # Raggruppa buy e sell per simbolo
        buys = {}
        sells = {}
        
        for trade in self.trade_history:
            if trade.side == 'BUY':
                if trade.symbol not in buys:
                    buys[trade.symbol] = []
                buys[trade.symbol].append(trade.price)
            else:
                if trade.symbol not in sells:
                    sells[trade.symbol] = []
                sells[trade.symbol].append(trade.price)
        
        print("\n💰 ANALISI PROFITTI:")
        for symbol in set(buys.keys()) & set(sells.keys()):
            avg_buy = sum(buys[symbol]) / len(buys[symbol])
            avg_sell = sum(sells[symbol]) / len(sells[symbol])
            profit_pct = (avg_sell - avg_buy) / avg_buy * 100
            
            print(f"{symbol}: {profit_pct:+.2f}%")


# Test del Trading Data Manager
if __name__ == "__main__":
    manager = TradingDataManager()
    
    # Simula trading
    manager.add_trade('10:00', 'BTC', 45000, 0.1, 'BUY')
    manager.add_trade('10:30', 'ETH', 3000, 1.0, 'BUY')
    manager.add_trade('11:00', 'BTC', 46000, 0.05, 'SELL')
    manager.add_trade('11:30', 'ETH', 3100, 0.5, 'SELL')
    manager.add_trade('12:00', 'ADA', 1.5, 1000, 'BUY')
    
    # Aggiungi a watchlist
    manager.add_to_watchlist(['DOT', 'LINK', 'SOL'])
    
    # Calcola valore portfolio
    print("\n📊 VALORE PORTFOLIO:")
    current_prices = {
        'BTC': 46500,
        'ETH': 3150,
        'ADA': 1.6
    }
    total_value = manager.get_portfolio_value(current_prices)
    print(f"\nValore totale: ${total_value:.2f}")
    
    # Analizza trades
    manager.analyze_trades()
    manager.find_profitable_trades()
    
    print("\n" + "=" * 60)
    print("🎉 COMPLIMENTI! Hai completato tutti gli esercizi Collections!")
    print("=" * 60)


"""
📚 RIEPILOGO COLLECTIONS:

✅ LISTE:
- Ordinate, mutabili, duplicati OK
- Uso: quando l'ordine conta
- O(1) append/pop, O(n) search

✅ DIZIONARI:
- Chiave-valore, mutabili, no duplicati chiavi
- Uso: lookup veloci, mappature
- O(1) get/set/delete

✅ SET:
- Non ordinati, mutabili, no duplicati
- Uso: membership test, operazioni matematiche
- O(1) add/remove/check

✅ TUPLE:
- Ordinate, immutabili, duplicati OK
- Uso: dati che non cambiano, return multipli
- Meno memoria, più veloci da creare

🎯 SCEGLI LA COLLECTION GIUSTA:
- Servono duplicati? → List/Tuple
- Serve ordine? → List/Tuple/OrderedDict
- Serve immutabilità? → Tuple/Frozenset
- Serve lookup veloce? → Dict/Set
- Servono operazioni matematiche? → Set
"""
