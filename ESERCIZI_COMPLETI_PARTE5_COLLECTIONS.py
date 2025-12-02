"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    🐍 PYTHON MASTER - SCHEDA ESERCIZI COMPLETA               ║
║                                                                              ║
║                    PARTE 5: COLLECTIONS AVANZATE (itertools, collections)    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ==============================================================================
# SEZIONE 16: COLLECTIONS MODULE
# ==============================================================================

print("=" * 70)
print("SEZIONE 16: COLLECTIONS MODULE")
print("=" * 70)

# ------------------------------------------------------------------------------
# ESERCIZIO 16.1: Counter
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Usa Counter per:
1. Contare frequenze
2. Operazioni su Counter (somma, sottrazione, intersezione)
3. most_common() e elementi

💡 TEORIA:
Counter è un dict specializzato per conteggi.
Supporta operazioni matematiche tra Counter.

🎯 SKILLS: Counter, frequenze, operazioni set-like
⏱️ TEMPO: 10 minuti
🔢 LIVELLO: Intermedio
"""

# ✅ SOLUZIONE:
def esercizio_16_1():
    """Counter - Conteggio elementi"""
    
    from collections import Counter
    
    # 1. CREAZIONE E CONTEGGIO
    print("--- CONTEGGIO BASE ---")
    
    trades = ['BUY', 'SELL', 'BUY', 'BUY', 'SELL', 'BUY', 'HOLD', 'BUY', 'SELL']
    trade_counts = Counter(trades)
    
    print(f"  Trades: {trades}")
    print(f"  Counter: {trade_counts}")
    print(f"  BUY count: {trade_counts['BUY']}")
    print(f"  MISSING count: {trade_counts['MISSING']}")  # Ritorna 0, non KeyError!
    
    # Da stringa
    text = "AAPL GOOGL AAPL MSFT AAPL GOOGL"
    symbol_counts = Counter(text.split())
    print(f"\n  Symbols: {symbol_counts}")
    
    # 2. MOST_COMMON
    print("\n--- MOST COMMON ---")
    
    print(f"  Top 2: {trade_counts.most_common(2)}")
    print(f"  Least common: {trade_counts.most_common()[-1]}")
    
    # 3. OPERAZIONI
    print("\n--- OPERAZIONI ---")
    
    week1 = Counter({'BUY': 10, 'SELL': 5, 'HOLD': 2})
    week2 = Counter({'BUY': 8, 'SELL': 12, 'HOLD': 3})
    
    print(f"  Week 1: {week1}")
    print(f"  Week 2: {week2}")
    print(f"  Totale (week1 + week2): {week1 + week2}")
    print(f"  Diff (week1 - week2): {week1 - week2}")  # Solo positivi
    print(f"  Intersezione (week1 & week2): {week1 & week2}")  # Min
    print(f"  Unione (week1 | week2): {week1 | week2}")  # Max
    
    # 4. ELEMENTI E ITERAZIONE
    print("\n--- ELEMENTI ---")
    
    small_counter = Counter({'A': 3, 'B': 2})
    print(f"  elements(): {list(small_counter.elements())}")
    
    # 5. APPLICAZIONE TRADING: Analisi frequenze
    print("\n--- ANALISI PATTERN ---")
    
    # Simula pattern di candele
    candle_patterns = [
        'DOJI', 'HAMMER', 'ENGULFING', 'DOJI', 'HAMMER',
        'DOJI', 'SPINNING_TOP', 'HAMMER', 'DOJI', 'ENGULFING',
        'DOJI', 'HAMMER', 'DOJI', 'DOJI'
    ]
    
    pattern_freq = Counter(candle_patterns)
    total = sum(pattern_freq.values())
    
    print(f"  Pattern frequency:")
    for pattern, count in pattern_freq.most_common():
        pct = count / total * 100
        print(f"    {pattern}: {count} ({pct:.1f}%)")
    
    return Counter

# 🧪 TEST:
if __name__ == "__main__":
    Counter = esercizio_16_1()
    c = Counter(['a', 'b', 'a'])
    assert c['a'] == 2
    print("✅ Esercizio 16.1 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 16.2: defaultdict
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Usa defaultdict per:
1. Raggruppamento automatico
2. Contatori senza KeyError
3. Liste nested

💡 TEORIA:
defaultdict crea automaticamente valori mancanti usando una factory.
Evita il pattern "if key not in dict: dict[key] = default".

🎯 SKILLS: defaultdict, factory functions, grouping
⏱️ TEMPO: 10 minuti
🔢 LIVELLO: Intermedio
"""

# ✅ SOLUZIONE:
def esercizio_16_2():
    """defaultdict - Dict con valori default"""
    
    from collections import defaultdict
    
    # 1. RAGGRUPPAMENTO
    print("--- RAGGRUPPAMENTO ---")
    
    trades = [
        {'symbol': 'AAPL', 'side': 'BUY', 'qty': 100},
        {'symbol': 'GOOGL', 'side': 'SELL', 'qty': 50},
        {'symbol': 'AAPL', 'side': 'SELL', 'qty': 30},
        {'symbol': 'AAPL', 'side': 'BUY', 'qty': 70},
        {'symbol': 'MSFT', 'side': 'BUY', 'qty': 200},
    ]
    
    # Raggruppa per symbol
    by_symbol = defaultdict(list)
    for trade in trades:
        by_symbol[trade['symbol']].append(trade)
    
    print("  Trades per symbol:")
    for symbol, trades_list in by_symbol.items():
        print(f"    {symbol}: {len(trades_list)} trades")
    
    # 2. CONTATORI
    print("\n--- CONTATORI ---")
    
    volume_by_side = defaultdict(int)
    for trade in trades:
        volume_by_side[trade['side']] += trade['qty']
    
    print(f"  Volume per side: {dict(volume_by_side)}")
    
    # 3. NESTED DEFAULTDICT
    print("\n--- NESTED ---")
    
    # Per creare nested defaultdict
    def nested_dict():
        return defaultdict(nested_dict)
    
    # Più semplice: lambda
    portfolio = defaultdict(lambda: defaultdict(int))
    
    for trade in trades:
        portfolio[trade['symbol']][trade['side']] += trade['qty']
    
    print("  Portfolio (symbol -> side -> qty):")
    for symbol, sides in portfolio.items():
        print(f"    {symbol}: {dict(sides)}")
    
    # 4. SET DEFAULT
    print("\n--- SET DEFAULT ---")
    
    # Traccia simboli unici per giorno
    daily_symbols = defaultdict(set)
    
    activity = [
        ('2024-01-01', 'AAPL'),
        ('2024-01-01', 'GOOGL'),
        ('2024-01-01', 'AAPL'),  # Duplicato
        ('2024-01-02', 'MSFT'),
        ('2024-01-02', 'AAPL'),
    ]
    
    for date, symbol in activity:
        daily_symbols[date].add(symbol)
    
    print("  Simboli unici per giorno:")
    for date, symbols in daily_symbols.items():
        print(f"    {date}: {symbols}")
    
    # 5. APPLICAZIONE: Aggregazione P&L
    print("\n--- AGGREGAZIONE P&L ---")
    
    pnl_entries = [
        ('AAPL', 'realized', 500),
        ('AAPL', 'unrealized', 200),
        ('GOOGL', 'realized', -100),
        ('GOOGL', 'unrealized', 300),
        ('AAPL', 'realized', 150),
    ]
    
    pnl_summary = defaultdict(lambda: {'realized': 0, 'unrealized': 0})
    
    for symbol, pnl_type, amount in pnl_entries:
        pnl_summary[symbol][pnl_type] += amount
    
    print("  P&L Summary:")
    for symbol, pnl in pnl_summary.items():
        total = pnl['realized'] + pnl['unrealized']
        print(f"    {symbol}: R={pnl['realized']}, U={pnl['unrealized']}, T={total}")
    
    return defaultdict

# 🧪 TEST:
if __name__ == "__main__":
    esercizio_16_2()
    print("✅ Esercizio 16.2 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 16.3: deque
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Usa deque per:
1. Operazioni O(1) su entrambe le estremità
2. Rotating buffer
3. Sliding window

💡 TEORIA:
deque (double-ended queue) ha append/pop O(1) su entrambi i lati.
maxlen crea un buffer circolare automatico.

🎯 SKILLS: deque, maxlen, rotate, sliding window
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Intermedio
"""

# ✅ SOLUZIONE:
def esercizio_16_3():
    """deque - Coda double-ended"""
    
    from collections import deque
    
    # 1. OPERAZIONI BASE
    print("--- OPERAZIONI BASE ---")
    
    d = deque([1, 2, 3])
    print(f"  Iniziale: {d}")
    
    d.append(4)         # Aggiunge a destra
    d.appendleft(0)     # Aggiunge a sinistra
    print(f"  Dopo append: {d}")
    
    right = d.pop()     # Rimuove da destra
    left = d.popleft()  # Rimuove da sinistra
    print(f"  Dopo pop: {d} (rimossi {left} e {right})")
    
    # Extend
    d.extend([4, 5])
    d.extendleft([-1, -2])  # NOTA: ordine invertito!
    print(f"  Dopo extend: {d}")
    
    # 2. BUFFER CON MAXLEN
    print("\n--- BUFFER CIRCOLARE ---")
    
    # Ultimi 5 prezzi
    price_buffer = deque(maxlen=5)
    
    prices = [100, 101, 99, 102, 103, 104, 105, 106]
    
    for price in prices:
        price_buffer.append(price)
        print(f"  Add {price}: {list(price_buffer)}")
    
    # 3. ROTATE
    print("\n--- ROTATE ---")
    
    d = deque([1, 2, 3, 4, 5])
    print(f"  Iniziale: {d}")
    
    d.rotate(2)  # Ruota a destra
    print(f"  rotate(2): {d}")
    
    d.rotate(-3)  # Ruota a sinistra
    print(f"  rotate(-3): {d}")
    
    # 4. SLIDING WINDOW
    print("\n--- SLIDING WINDOW SMA ---")
    
    def moving_average(prices, window=3):
        """Calcola SMA con sliding window."""
        d = deque(maxlen=window)
        result = []
        
        for price in prices:
            d.append(price)
            if len(d) == window:
                avg = sum(d) / window
                result.append(round(avg, 2))
            else:
                result.append(None)
        
        return result
    
    prices = [100, 102, 101, 105, 103, 108, 107, 110]
    sma = moving_average(prices, window=3)
    
    print("  Price | SMA(3)")
    print("  ------+-------")
    for p, s in zip(prices, sma):
        print(f"  {p:>5} | {s}")
    
    # 5. APPLICAZIONE: Order Book Level
    print("\n--- ORDER BOOK (ultimi 5 ordini) ---")
    
    recent_orders = deque(maxlen=5)
    
    orders = [
        {'id': 1, 'side': 'BUY', 'price': 100},
        {'id': 2, 'side': 'SELL', 'price': 101},
        {'id': 3, 'side': 'BUY', 'price': 99},
        {'id': 4, 'side': 'BUY', 'price': 100},
        {'id': 5, 'side': 'SELL', 'price': 102},
        {'id': 6, 'side': 'BUY', 'price': 101},
        {'id': 7, 'side': 'SELL', 'price': 103},
    ]
    
    for order in orders:
        recent_orders.append(order)
    
    print(f"  Ultimi 5 ordini:")
    for o in recent_orders:
        print(f"    #{o['id']} {o['side']} @ {o['price']}")
    
    return deque

# 🧪 TEST:
if __name__ == "__main__":
    esercizio_16_3()
    print("✅ Esercizio 16.3 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 16.4: namedtuple e OrderedDict
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Usa namedtuple e OrderedDict:
1. namedtuple per dati strutturati leggeri
2. Conversioni namedtuple ↔ dict
3. OrderedDict per ordine garantito

💡 TEORIA:
namedtuple: tuple con nomi, immutabile, memory efficient
OrderedDict: dict che ricorda ordine inserimento (< Python 3.7 era necessario)

🎯 SKILLS: namedtuple, _asdict, _replace, OrderedDict
⏱️ TEMPO: 10 minuti
🔢 LIVELLO: Intermedio
"""

# ✅ SOLUZIONE:
def esercizio_16_4():
    """namedtuple e OrderedDict"""
    
    from collections import namedtuple, OrderedDict
    
    # 1. NAMEDTUPLE BASE
    print("--- NAMEDTUPLE ---")
    
    # Definizione
    OHLC = namedtuple('OHLC', ['open', 'high', 'low', 'close', 'volume'])
    
    # Creazione
    candle = OHLC(100.0, 105.0, 98.0, 103.0, 1000000)
    
    print(f"  Candle: {candle}")
    print(f"  Open: {candle.open}")
    print(f"  Close: {candle[3]}")  # Anche con indice
    
    # Unpacking
    o, h, l, c, v = candle
    print(f"  Unpacked: O={o}, H={h}, L={l}, C={c}, V={v}")
    
    # 2. METODI SPECIALI
    print("\n--- METODI NAMEDTUPLE ---")
    
    # _asdict: converte a dict
    as_dict = candle._asdict()
    print(f"  _asdict(): {as_dict}")
    
    # _replace: crea nuova istanza con modifiche
    updated = candle._replace(close=106.0, volume=1500000)
    print(f"  _replace: {updated}")
    print(f"  Originale invariato: {candle}")
    
    # _fields: nomi campi
    print(f"  _fields: {OHLC._fields}")
    
    # Creazione da dict
    data = {'open': 110, 'high': 115, 'low': 108, 'close': 113, 'volume': 500000}
    from_dict = OHLC(**data)
    print(f"  Da dict: {from_dict}")
    
    # 3. NAMEDTUPLE CON DEFAULTS
    print("\n--- CON DEFAULTS ---")
    
    Trade = namedtuple('Trade', ['symbol', 'side', 'qty', 'price', 'commission'], 
                       defaults=[0.0])  # Default per commission
    
    t1 = Trade('AAPL', 'BUY', 100, 150.0)
    t2 = Trade('GOOGL', 'SELL', 50, 140.0, 1.5)
    
    print(f"  t1: {t1}")
    print(f"  t2: {t2}")
    
    # 4. ORDEREDDICT
    print("\n--- ORDEREDDICT ---")
    
    # Nota: da Python 3.7+ i dict normali mantengono l'ordine
    # OrderedDict è ancora utile per:
    # - move_to_end()
    # - Comparazioni che considerano ordine
    
    portfolio = OrderedDict()
    portfolio['AAPL'] = 150
    portfolio['GOOGL'] = 140
    portfolio['MSFT'] = 380
    
    print(f"  Portfolio: {portfolio}")
    
    # move_to_end
    portfolio.move_to_end('AAPL')
    print(f"  Dopo move_to_end('AAPL'): {list(portfolio.keys())}")
    
    portfolio.move_to_end('MSFT', last=False)  # All'inizio
    print(f"  Dopo move_to_end('MSFT', last=False): {list(portfolio.keys())}")
    
    # Comparazione considera ordine
    od1 = OrderedDict([('a', 1), ('b', 2)])
    od2 = OrderedDict([('b', 2), ('a', 1)])
    print(f"\n  OrderedDict con ordine diverso uguali? {od1 == od2}")
    
    # Dict normale ignora ordine
    d1 = {'a': 1, 'b': 2}
    d2 = {'b': 2, 'a': 1}
    print(f"  Dict normale: {d1 == d2}")
    
    # 5. LRU CACHE MANUALE CON ORDEREDDICT
    print("\n--- LRU CACHE ---")
    
    class LRUCache:
        def __init__(self, capacity):
            self.cache = OrderedDict()
            self.capacity = capacity
        
        def get(self, key):
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            return self.cache[key]
        
        def put(self, key, value):
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
    
    cache = LRUCache(3)
    cache.put('AAPL', 150)
    cache.put('GOOGL', 140)
    cache.put('MSFT', 380)
    print(f"  Cache: {list(cache.cache.keys())}")
    
    cache.get('AAPL')  # AAPL diventa più recente
    cache.put('TSLA', 250)  # GOOGL viene rimosso
    print(f"  Dopo accesso AAPL e add TSLA: {list(cache.cache.keys())}")
    
    return namedtuple, OrderedDict

# 🧪 TEST:
if __name__ == "__main__":
    esercizio_16_4()
    print("✅ Esercizio 16.4 completato!\n")


# ==============================================================================
# SEZIONE 17: ITERTOOLS
# ==============================================================================

print("\n" + "=" * 70)
print("SEZIONE 17: ITERTOOLS")
print("=" * 70)

# ------------------------------------------------------------------------------
# ESERCIZIO 17.1: Iteratori Infiniti
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Usa itertools per iteratori infiniti:
1. count(): contatore infinito
2. cycle(): ciclo infinito
3. repeat(): ripetizione

💡 TEORIA:
Questi iteratori sono lazy (non creano liste in memoria).
Sempre usare con limite (islice, takewhile, break).

🎯 SKILLS: count, cycle, repeat, islice
⏱️ TEMPO: 10 minuti
🔢 LIVELLO: Intermedio
"""

# ✅ SOLUZIONE:
def esercizio_17_1():
    """Iteratori Infiniti"""
    
    import itertools as it
    
    # 1. COUNT
    print("--- COUNT ---")
    
    # count(start=0, step=1)
    counter = it.count(start=100, step=5)
    
    first_5 = [next(counter) for _ in range(5)]
    print(f"  count(100, 5): {first_5}")
    
    # Con enumerate alternativo
    items = ['A', 'B', 'C']
    for i, item in zip(it.count(1), items):
        print(f"    {i}. {item}")
    
    # 2. CYCLE
    print("\n--- CYCLE ---")
    
    # Cicla infinitamente
    signals = it.cycle(['BUY', 'HOLD', 'SELL'])
    
    first_7 = [next(signals) for _ in range(7)]
    print(f"  cycle(['BUY','HOLD','SELL']): {first_7}")
    
    # Applicazione: Round-robin su broker
    brokers = it.cycle(['Alpaca', 'IBKR', 'TD'])
    orders = ['Order1', 'Order2', 'Order3', 'Order4', 'Order5']
    
    assignments = [(order, next(brokers)) for order in orders]
    print(f"  Round-robin: {assignments}")
    
    # 3. REPEAT
    print("\n--- REPEAT ---")
    
    # repeat(value, times=infinite)
    fives = list(it.repeat(5, times=4))
    print(f"  repeat(5, 4): {fives}")
    
    # Utile con map
    squared = list(map(pow, range(5), it.repeat(2)))
    print(f"  map(pow, range(5), repeat(2)): {squared}")
    
    # 4. ISLICE
    print("\n--- ISLICE ---")
    
    # Prende slice da iteratore senza creare lista
    infinite = it.count()
    
    # islice(iterable, stop) or islice(iterable, start, stop, step)
    sliced = list(it.islice(infinite, 10, 20, 2))
    print(f"  islice(count(), 10, 20, 2): {sliced}")
    
    # 5. APPLICAZIONE: Generatore di Order ID
    print("\n--- ORDER ID GENERATOR ---")
    
    def order_id_generator(prefix='ORD'):
        """Genera ID ordini infiniti."""
        for n in it.count(1):
            yield f"{prefix}-{n:06d}"
    
    gen = order_id_generator()
    ids = [next(gen) for _ in range(5)]
    print(f"  Order IDs: {ids}")
    
    return it

# 🧪 TEST:
if __name__ == "__main__":
    esercizio_17_1()
    print("✅ Esercizio 17.1 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 17.2: Combinatoria
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Usa itertools per combinatoria:
1. product(): prodotto cartesiano
2. permutations(): permutazioni
3. combinations(): combinazioni
4. combinations_with_replacement()

💡 TEORIA:
- product: tutte le coppie possibili (n^k elementi)
- permutations: ordine conta, no ripetizioni
- combinations: ordine non conta, no ripetizioni
- combinations_with_replacement: ordine non conta, con ripetizioni

🎯 SKILLS: Combinatoria, parameter grid, pair analysis
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Intermedio
"""

# ✅ SOLUZIONE:
def esercizio_17_2():
    """Combinatoria con itertools"""
    
    import itertools as it
    
    # 1. PRODUCT
    print("--- PRODUCT (Prodotto Cartesiano) ---")
    
    # Tutte le combinazioni
    colors = ['red', 'blue']
    sizes = ['S', 'M', 'L']
    
    all_variants = list(it.product(colors, sizes))
    print(f"  product(colors, sizes): {all_variants}")
    
    # repeat per potenze
    binary = list(it.product([0, 1], repeat=3))
    print(f"  product([0,1], repeat=3): {binary}")
    
    # APPLICAZIONE: Grid di parametri per backtest
    print("\n  Parameter Grid per Backtest:")
    fast_periods = [5, 10]
    slow_periods = [20, 30]
    stop_loss = [0.02, 0.05]
    
    param_grid = list(it.product(fast_periods, slow_periods, stop_loss))
    print(f"  {len(param_grid)} combinazioni:")
    for fast, slow, sl in param_grid[:4]:
        print(f"    fast={fast}, slow={slow}, sl={sl}")
    print(f"    ...")
    
    # 2. PERMUTATIONS
    print("\n--- PERMUTATIONS ---")
    
    # Tutte le permutazioni (ordine conta)
    items = ['A', 'B', 'C']
    perms = list(it.permutations(items))
    print(f"  permutations(['A','B','C']): {perms}")
    
    # Con lunghezza specifica
    perms_2 = list(it.permutations(items, 2))
    print(f"  permutations(['A','B','C'], 2): {perms_2}")
    
    # 3. COMBINATIONS
    print("\n--- COMBINATIONS ---")
    
    # Combinazioni (ordine non conta)
    combs = list(it.combinations(items, 2))
    print(f"  combinations(['A','B','C'], 2): {combs}")
    
    # APPLICAZIONE: Pair trading analysis
    print("\n  Pairs per correlation:")
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    pairs = list(it.combinations(symbols, 2))
    print(f"  {len(pairs)} pairs: {pairs}")
    
    # 4. COMBINATIONS WITH REPLACEMENT
    print("\n--- COMBINATIONS WITH REPLACEMENT ---")
    
    # Può ripetere elementi
    cwr = list(it.combinations_with_replacement(['A', 'B'], 3))
    print(f"  combinations_with_replacement(['A','B'], 3): {cwr}")
    
    # 5. APPLICAZIONE: Portfolio Allocation
    print("\n--- PORTFOLIO ALLOCATION ---")
    
    def allocation_combinations(n_assets, steps=4):
        """Genera tutte le allocazioni possibili."""
        weights = [i/steps for i in range(steps + 1)]
        for combo in it.product(weights, repeat=n_assets):
            if abs(sum(combo) - 1.0) < 0.001:  # Somma = 100%
                yield combo
    
    assets = ['AAPL', 'GOOGL', 'MSFT']
    allocations = list(allocation_combinations(3, steps=4))
    
    print(f"  {len(allocations)} allocazioni valide per 3 asset:")
    for alloc in allocations[:5]:
        print(f"    {dict(zip(assets, alloc))}")
    print(f"    ... ({len(allocations) - 5} altre)")
    
    return it

# 🧪 TEST:
if __name__ == "__main__":
    esercizio_17_2()
    print("✅ Esercizio 17.2 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 17.3: Funzioni di Aggregazione
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Usa itertools per aggregazione:
1. accumulate(): somme cumulative
2. groupby(): raggruppamento
3. chain(): concatenazione
4. compress(), filterfalse()

💡 TEORIA:
Queste funzioni operano lazy su iterabili.
groupby richiede dati pre-ordinati!

🎯 SKILLS: accumulate, groupby, chain, filtering
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Intermedio-Avanzato
"""

# ✅ SOLUZIONE:
def esercizio_17_3():
    """Funzioni di Aggregazione"""
    
    import itertools as it
    import operator
    
    # 1. ACCUMULATE
    print("--- ACCUMULATE ---")
    
    # Default: somma cumulativa
    numbers = [1, 2, 3, 4, 5]
    cumsum = list(it.accumulate(numbers))
    print(f"  Cumulative sum: {cumsum}")
    
    # Con operatore custom
    cumprod = list(it.accumulate(numbers, operator.mul))
    print(f"  Cumulative product: {cumprod}")
    
    # Custom function
    cummax = list(it.accumulate(numbers, max))
    print(f"  Cumulative max: {cummax}")
    
    # APPLICAZIONE: Equity Curve
    print("\n  Equity Curve:")
    returns = [0.02, -0.01, 0.03, -0.02, 0.01, 0.04]
    initial = 10000
    
    # Calcola equity
    def compound(balance, ret):
        return balance * (1 + ret)
    
    equity = list(it.accumulate(returns, compound, initial=initial))
    
    for i, (ret, eq) in enumerate(zip(returns, equity)):
        print(f"    Day {i+1}: {ret:+.1%} → ${eq:.2f}")
    
    # 2. GROUPBY
    print("\n--- GROUPBY ---")
    
    # IMPORTANTE: dati devono essere ordinati per chiave!
    trades = [
        ('AAPL', 'BUY', 100),
        ('AAPL', 'SELL', 50),
        ('GOOGL', 'BUY', 200),
        ('GOOGL', 'BUY', 100),
        ('MSFT', 'SELL', 150),
    ]
    
    # Ordina prima di groupby!
    trades_sorted = sorted(trades, key=lambda x: x[0])
    
    print("  Trades per symbol:")
    for symbol, group in it.groupby(trades_sorted, key=lambda x: x[0]):
        trades_list = list(group)
        print(f"    {symbol}: {len(trades_list)} trades")
    
    # Groupby su side
    trades_by_side = sorted(trades, key=lambda x: x[1])
    
    print("\n  Volume per side:")
    for side, group in it.groupby(trades_by_side, key=lambda x: x[1]):
        total_qty = sum(t[2] for t in group)
        print(f"    {side}: {total_qty}")
    
    # 3. CHAIN
    print("\n--- CHAIN ---")
    
    list1 = [1, 2, 3]
    list2 = [4, 5, 6]
    list3 = [7, 8, 9]
    
    chained = list(it.chain(list1, list2, list3))
    print(f"  chain(list1, list2, list3): {chained}")
    
    # chain.from_iterable per liste di liste
    nested = [[1, 2], [3, 4], [5, 6]]
    flattened = list(it.chain.from_iterable(nested))
    print(f"  chain.from_iterable: {flattened}")
    
    # 4. COMPRESS e FILTERFALSE
    print("\n--- COMPRESS ---")
    
    data = ['A', 'B', 'C', 'D', 'E']
    selectors = [1, 0, 1, 0, 1]
    
    selected = list(it.compress(data, selectors))
    print(f"  compress(data, selectors): {selected}")
    
    # Applicazione: Filtra per segnali
    prices = [100, 102, 99, 105, 103]
    buy_signals = [True, False, True, False, True]
    
    buy_prices = list(it.compress(prices, buy_signals))
    print(f"  Prezzi con BUY signal: {buy_prices}")
    
    print("\n--- FILTERFALSE ---")
    
    # Opposto di filter
    numbers = range(10)
    evens = list(filter(lambda x: x % 2 == 0, numbers))
    odds = list(it.filterfalse(lambda x: x % 2 == 0, numbers))
    
    print(f"  filter (evens): {evens}")
    print(f"  filterfalse (odds): {odds}")
    
    # 5. TAKEWHILE e DROPWHILE
    print("\n--- TAKEWHILE / DROPWHILE ---")
    
    prices = [100, 102, 105, 103, 99, 95, 97, 101]
    
    # Prendi finché sopra 100
    above_100 = list(it.takewhile(lambda x: x >= 100, prices))
    print(f"  takewhile(>=100): {above_100}")
    
    # Scarta finché sopra 100
    after_drop = list(it.dropwhile(lambda x: x >= 100, prices))
    print(f"  dropwhile(>=100): {after_drop}")
    
    return it

# 🧪 TEST:
if __name__ == "__main__":
    esercizio_17_3()
    print("✅ Esercizio 17.3 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 17.4: Applicazioni Trading
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Combina itertools per casi reali di trading:
1. Sliding window per indicatori
2. Pairwise per calcolo rendimenti
3. Batching per ordini

💡 TEORIA:
itertools è perfetto per processare stream di dati finanziari
in modo memory-efficient.

🎯 SKILLS: Sliding window, pairwise, batching
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Avanzato
"""

# ✅ SOLUZIONE:
def esercizio_17_4():
    """Applicazioni Trading con itertools"""
    
    import itertools as it
    from collections import deque
    
    # 1. SLIDING WINDOW
    print("--- SLIDING WINDOW ---")
    
    def sliding_window(iterable, n):
        """Sliding window di dimensione n."""
        iterator = iter(iterable)
        window = deque(it.islice(iterator, n), maxlen=n)
        if len(window) == n:
            yield tuple(window)
        for item in iterator:
            window.append(item)
            yield tuple(window)
    
    prices = [100, 102, 101, 105, 103, 108, 107, 110]
    
    print(f"  Prices: {prices}")
    print("  Windows di 3:")
    for window in sliding_window(prices, 3):
        print(f"    {window} → SMA: {sum(window)/3:.2f}")
    
    # 2. PAIRWISE (Python 3.10+ o implementazione)
    print("\n--- PAIRWISE ---")
    
    def pairwise(iterable):
        """Return successive overlapping pairs."""
        a, b = it.tee(iterable)
        next(b, None)
        return zip(a, b)
    
    # Calcolo rendimenti
    print("  Daily Returns:")
    for prev, curr in pairwise(prices):
        ret = (curr - prev) / prev * 100
        print(f"    {prev} → {curr}: {ret:+.2f}%")
    
    # 3. BATCHING
    print("\n--- BATCHING ---")
    
    def batched(iterable, n):
        """Divide in batch di dimensione n."""
        iterator = iter(iterable)
        while True:
            batch = list(it.islice(iterator, n))
            if not batch:
                break
            yield batch
    
    orders = list(range(1, 12))  # 11 ordini
    
    print(f"  Orders: {orders}")
    print("  Batches di 3:")
    for i, batch in enumerate(batched(orders, 3)):
        print(f"    Batch {i+1}: {batch}")
    
    # 4. RUNNING STATISTICS
    print("\n--- RUNNING STATISTICS ---")
    
    def running_stats(prices):
        """Calcola statistiche running."""
        for i, window in enumerate(sliding_window(prices, 3)):
            yield {
                'index': i + 2,
                'price': window[-1],
                'sma': sum(window) / len(window),
                'high': max(window),
                'low': min(window),
            }
    
    print("  Running Stats (window=3):")
    for stats in running_stats(prices):
        print(f"    [{stats['index']}] Price={stats['price']}, "
              f"SMA={stats['sma']:.1f}, H={stats['high']}, L={stats['low']}")
    
    # 5. MULTI-TIMEFRAME
    print("\n--- MULTI-TIMEFRAME ---")
    
    def resample(prices, factor):
        """Resample prices (es: da 1min a 5min)."""
        for batch in batched(prices, factor):
            yield {
                'open': batch[0],
                'high': max(batch),
                'low': min(batch),
                'close': batch[-1],
            }
    
    minute_prices = [100, 101, 99, 102, 103, 101, 104, 106, 105, 108]
    
    print(f"  1-min prices: {minute_prices}")
    print("  5-min OHLC:")
    for candle in resample(minute_prices, 5):
        print(f"    {candle}")
    
    # 6. SIGNAL CONFIRMATION
    print("\n--- SIGNAL CONFIRMATION ---")
    
    def confirmed_signals(signals, confirm_periods=2):
        """Conferma segnale se persiste per N periodi."""
        for window in sliding_window(signals, confirm_periods):
            if all(s == window[0] for s in window):
                yield window[0]
            else:
                yield 'HOLD'
    
    raw_signals = ['BUY', 'BUY', 'SELL', 'BUY', 'BUY', 'BUY', 'SELL', 'SELL']
    
    print(f"  Raw: {raw_signals}")
    confirmed = list(confirmed_signals(raw_signals, 2))
    print(f"  Confirmed (2): {confirmed}")
    
    return sliding_window, pairwise, batched

# 🧪 TEST:
if __name__ == "__main__":
    esercizio_17_4()
    print("✅ Esercizio 17.4 completato!\n")


# ==============================================================================
# RIEPILOGO SEZIONE 16-17: COLLECTIONS AVANZATE
# ==============================================================================

print("\n" + "=" * 70)
print("RIEPILOGO: ESERCIZI COLLECTIONS COMPLETATI")
print("=" * 70)

print("""
ESERCIZI COMPLETATI IN QUESTA PARTE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SEZIONE 16 - Collections Module:
  ✅ 16.1 Counter
  ✅ 16.2 defaultdict
  ✅ 16.3 deque
  ✅ 16.4 namedtuple e OrderedDict

SEZIONE 17 - Itertools:
  ✅ 17.1 Iteratori Infiniti
  ✅ 17.2 Combinatoria
  ✅ 17.3 Funzioni di Aggregazione
  ✅ 17.4 Applicazioni Trading

TOTALE QUESTA PARTE: 8 esercizi
TOTALE CUMULATIVO: 53 esercizi
""")

if __name__ == "__main__":
    print("\n🎉 TUTTI GLI ESERCIZI DELLA PARTE 5 COMPLETATI!")
