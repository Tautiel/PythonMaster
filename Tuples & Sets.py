#===============Tulples e Sets========================

print("\n Operazioni base Set")
def set_basic_operations():
    '''Operazioni fondamentali sui set'''
    
    my_coins = {'BTC', 'ETH', 'ADA', 'DOT'}
    friend_coin = {'BTC', 'XRP', 'ADA', 'SOL'}
    
    # Aggiungi elemento
    my_coins.add('ALU')
    
    # Rimuovi elemento 
    my_coins.remove('ETH')
    
    # Rimuovi elemento senza errore
    my_coins.discard('DOT')
    
    # Rimozione casuale
    my_coins.pop()
    
    # Lunghezza dei dati
    print(f"Lunghezza : {len(my_coins)}")
    
    # Membership test
    print(f"BTC è nel mio set? : {'BTC' in my_coins}")
    
    # Clear
    empty = set()
    empty.clear()
    
    print(f"My coins = {my_coins}")
    print(f"Friens coin = {friend_coin}")


set_basic_operations()


def set_amth_operations():
    
    portfolio_a = {'BTC', 'ETH', 'ADA', 'DOT', 'LINK'}
    portfolio_b = {'BTC', 'ADA', 'XRP', 'SOL', 'AVAX'}
    matchlist = {'MATIC', 'ATOM', 'BTC'}
    
    all_assets = portfolio_a | portfolio_b
    print(f"Unione: {all_assets}")
    
    inters = portfolio_a & portfolio_b
    print(f"Intersezione: {inters}")
    
    diff = portfolio_a - portfolio_b
    print(f"Differenza: {diff}")
    
    diff_simm = portfolio_a ^ portfolio_b
    print(f"DIfferenza simmetrica: {diff_simm}")
    
    print(f"Subset: {portfolio_a.issubset(portfolio_b)}")
    
    print(F"Superset: {portfolio_a.issuperset(matchlist)}")
    
    print(f"Disjoint: {portfolio_a.isdisjoint(matchlist)}")
    
    
set_amth_operations()


def set_comprehension_practices():
    '''Quadrati dei par ida 1 a 10'''
    even_squares = {x**2 for x in range(1, 11) if x % 2 == 0}
    print(f"Quadrati dei pari: {even_squares}")
    
    text = "la mia vita è perfetta così com'é"
    unique_letters = {char for char in text}
    print(f"Caratteri unici: {unique_letters}")
    
    prezzi_lista = [11.2, 300.55, 225.31, 25.999, 84.667]
    rounded = {round(p) for p in prezzi_lista}
    print(f"Arrotondamenti prezzi: {rounded}")
    
    unici_list_duplicati = [2, 2, 2, 55, 55, 55, 9, 9, 9, 4, 4, 4, 66, 66 ,66]
    print(f"Unici da lista duplicati: {set(unici_list_duplicati)}")
    
    multipli = {x for x in range(1, 21)
                if x % 5 == 0 or x % 3 == 0}
    print(f"Multipli di 3 o 5 da 1 a 20: {multipli}")
    
    
set_comprehension_practices()

def frozen_set_practice():
    core_portfolio = frozenset(['BTC', 'ETH', 'ADA'])
    print(f"Core portfolio:", core_portfolio)
    
    print("Non puoi fare core_portfolio.add('DOT') perché sono immutabili")
    
    trading = {'BTC', 'XRP', 'SOL'}
    common = core_portfolio & trading
    print(f"Comuni", common)
    
    values = {
        frozenset(['BTC', 'ETH']): 50000,
        frozenset(['ADA', 'DOT']): 20000
    }
    
    print("valori", values)
    
    mutable = set(core_portfolio)
    mutable.add('LINK')
    immutable =frozenset(mutable)
    print("Convertito", immutable)
    

frozen_set_practice()


def set_use_cases():
    
    trades = ['BTC', 'ETH', 'BTC', 'ADA', 'ETH', 'DOT', 'BTC']
    u_unordered = list(set(trades))
    u_ordered = list(dict.fromkeys(trades))
    print(f"Senza ordine: {u_unordered}, con ordine: {u_ordered}")
    
    required = {'BTC', 'ETH', 'USDT', 'BNB'}
    current = {'BTC', 'ETH', 'ADA'}
    missing = required - current
    print("MAncanti", missing)
    
    valid_symbol = {'BTC', 'ETH', 'ADA', 'DOT', 'LINK'}
    user_imput = ['BTC', 'XZT', 'ETH', 'FAKE']
    valid =[s for s in user_imput if s in valid_symbol]
    invalid = [ s for s in user_imput if s not in valid_symbol]
    print(f"Validi: {valid}.\nNon validi: {invalid}")
   
    articolo1 = ('Articolo 1', {'python', 'trading', 'bot'})
    articolo2 = ('Articolo 2', {'python', 'data', 'analysis' })
    articolo3 = ('Articolo 3', {'javascript', 'web','frontend'})
    all_articles = [articolo1, articolo2, articolo3]
    
    python_articles = [title for title, tags in all_articles if 'python' in tags] 
    print("Articoli python", all_articles)
    

set_use_cases()

def tuple_basic_operations():
    
    coordinates = (10, 20, 30)
    single_tpl = (20,)
    empty = ()
    
    x = coordinates[0]
    y = coordinates[1]
    z = coordinates[2]
    print(f"Coordinate: x = {x}, y = {y}, z = {z}")
    
    a, b, c = coordinates
    print(f"Unpacking, a:{a}, b:{b}, c:{c}")
    
    first_two = coordinates[:2]
    last_two = coordinates[-2:]
    print(f"Slicing primi due:{first_two}, ultimi due:{last_two}")
    
    combined = (1, 2, 3) + (4, 5, 6)
    print(f"Concatenazione: {combined}")
    
    repeated = (2, 3) * 5
    print(f"Ripetizione: {repeated}")
    
    print(f"Contiene 20:", 20 in coordinates)
    
    numbers =  (1, 2, 3, 4, 3, 4, 5, 2)
    print(f"Count di 2:", numbers.count(2))
    print(f"Index di 3:", numbers.index(3))
    

tuple_basic_operations()

def advanced_unpacking():
    numbers = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    first, *middle, last = numbers
    print(f"First", first, "Middle", middle, "last", last)
    first_two, *rest = numbers
    print("First two", first_two, "Rest", rest)
    *beginning, last_two, last_one= numbers
    print("Beginning", beginning, "last two", last_two, "last one", last_one)
    x, y, z = 5, 6, 8
    print("Prima dello swap", x, y, z)
    x, y, z = z, x, y
    print("Dopo lo swap", x, y, z)
    a = b = c = 55
    print(f"Multiple: {a}, {b}, {c}")
    trades = [
        ('BTC', 92000, 0.5),
        ('ETH', 3100, 0.1),
        ('ADA', 1.70, 15)
    ]
    for symbol, price, qty in trades:
        print(symbol, price * qty)
   
    data = (10, 20, 30, 40, 50)
    first, _, third, *_ = data
    print("First", first, "Third", third)  


advanced_unpacking()


from collections import namedtuple

def namedtuple_practice():
    # 1) Definizione
    Trade = namedtuple('Trade', ['symbol', 'price', 'qty'])

    # 2) Creazione istanza
    t1 = Trade('BTC', 45000, 0.5)
    print("Trade 1:", t1)

    # 3) Accesso ai campi
    print("Simbolo:", t1.symbol)
    print("Prezzo:", t1.price)
    print("Quantità:", t1.qty)

    # 4) Unpacking
    s, p, q = t1
    print("Unpacked:", s, p, q)

    # 5) Lista di trade
    trades = [
        Trade('BTC', 45000, 0.5),
        Trade('ETH', 3000, 2),
        Trade('ADA', 1.5, 1000)
    ]

    # 6) Loop
    for trade in trades:
        value = trade.price * trade.qty
        print(f"{trade.symbol} → valore: {value}")

    return True
