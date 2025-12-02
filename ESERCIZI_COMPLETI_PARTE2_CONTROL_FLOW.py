"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    🐍 PYTHON MASTER - SCHEDA ESERCIZI COMPLETA               ║
║                                                                              ║
║                    PARTE 2: CONTROL FLOW (If, Loops, Comprehensions)         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ==============================================================================
# SEZIONE 5: CONDITIONAL STATEMENTS
# ==============================================================================

print("=" * 70)
print("SEZIONE 5: CONDITIONAL STATEMENTS")
print("=" * 70)

# ------------------------------------------------------------------------------
# ESERCIZIO 5.1: Basic Conditionals
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Scrivi una funzione che classifichi un numero come:
- "negative" se < 0
- "zero" se == 0  
- "positive" se > 0
E aggiungi se è pari o dispari (per numeri non-zero).

💡 TEORIA:
La struttura if-elif-else permette decisioni ramificate.
Importante: le condizioni sono valutate in ordine, la prima vera viene eseguita.

🎯 SKILLS: if/elif/else, operatore modulo, f-strings
⏱️ TEMPO: 10 minuti
🔢 LIVELLO: Principiante
"""

# ✅ SOLUZIONE:
def esercizio_5_1():
    """Basic Conditionals - Classificazione numeri"""
    
    def classify_number(n):
        """Classifica un numero"""
        if n < 0:
            parity = "even" if n % 2 == 0 else "odd"
            return f"negative {parity}"
        elif n == 0:
            return "zero"
        else:  # n > 0
            parity = "even" if n % 2 == 0 else "odd"
            return f"positive {parity}"
    
    # Test con vari numeri
    test_numbers = [-5, -4, 0, 1, 2, 100, -100]
    
    print("Classificazione numeri:")
    for num in test_numbers:
        result = classify_number(num)
        print(f"  {num:>4} → {result}")
    
    return classify_number

# 🧪 TEST:
if __name__ == "__main__":
    func = esercizio_5_1()
    assert func(-5) == "negative odd"
    assert func(-4) == "negative even"
    assert func(0) == "zero"
    assert func(1) == "positive odd"
    assert func(2) == "positive even"
    print("✅ Esercizio 5.1 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 5.2: Guard Clauses
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Riscrivi questa funzione usando guard clauses (early return):

def process_order_bad(order):
    if order is not None:
        if order.get('symbol'):
            if order.get('quantity', 0) > 0:
                if order.get('price', 0) > 0:
                    return f"Processing {order['symbol']}"
                else:
                    return "Invalid price"
            else:
                return "Invalid quantity"
        else:
            return "Missing symbol"
    else:
        return "No order"

💡 TEORIA:
Le guard clauses riducono il nesting e migliorano la leggibilità.
Pattern: valida e esci presto, processa alla fine.

🎯 SKILLS: Early return, flat code, readability
⏱️ TEMPO: 10 minuti
🔢 LIVELLO: Intermedio
"""

# ✅ SOLUZIONE:
def esercizio_5_2():
    """Guard Clauses - Early return pattern"""
    
    # VERSIONE CON GUARD CLAUSES (pulita)
    def process_order(order):
        """Processa un ordine trading con guard clauses"""
        
        # Guard 1: ordine esiste?
        if order is None:
            return "No order"
        
        # Guard 2: symbol presente?
        if not order.get('symbol'):
            return "Missing symbol"
        
        # Guard 3: quantity valida?
        if order.get('quantity', 0) <= 0:
            return "Invalid quantity"
        
        # Guard 4: price valido?
        if order.get('price', 0) <= 0:
            return "Invalid price"
        
        # Tutti i check passati, processa
        return f"Processing {order['symbol']}: {order['quantity']} @ ${order['price']}"
    
    # Test
    test_orders = [
        None,
        {},
        {'symbol': 'AAPL'},
        {'symbol': 'AAPL', 'quantity': -5},
        {'symbol': 'AAPL', 'quantity': 100},
        {'symbol': 'AAPL', 'quantity': 100, 'price': 156.78}
    ]
    
    print("Processing orders con Guard Clauses:")
    for order in test_orders:
        result = process_order(order)
        print(f"  {order} → {result}")
    
    return process_order

# 🧪 TEST:
if __name__ == "__main__":
    func = esercizio_5_2()
    assert func(None) == "No order"
    assert func({}) == "Missing symbol"
    assert "Processing" in func({'symbol': 'AAPL', 'quantity': 100, 'price': 150})
    print("✅ Esercizio 5.2 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 5.3: Ternary Operator
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Usa l'operatore ternario per:
1. Assegnare "BUY" o "SELL" basandosi sul segnale
2. Calcolare il massimo tra due numeri
3. Assegnare un default se un valore è None

💡 TEORIA:
Sintassi: value_if_true if condition else value_if_false
Usare solo per espressioni semplici, non per logica complessa.

🎯 SKILLS: Operatore ternario, one-liners, leggibilità
⏱️ TEMPO: 5 minuti
🔢 LIVELLO: Principiante
"""

# ✅ SOLUZIONE:
def esercizio_5_3():
    """Ternary Operator - Operatore condizionale"""
    
    # 1. Trading signal
    signal_value = 0.7  # Positivo = bullish
    
    action = "BUY" if signal_value > 0 else "SELL"
    print(f"Signal: {signal_value} → Action: {action}")
    
    # Con terza opzione (ternario annidato - usare con cautela!)
    def get_action(signal):
        return "BUY" if signal > 0.5 else "SELL" if signal < -0.5 else "HOLD"
    
    for s in [0.8, 0.2, -0.3, -0.7]:
        print(f"  Signal {s:>5} → {get_action(s)}")
    
    # 2. Max senza funzione max()
    a, b = 10, 25
    maximum = a if a > b else b
    print(f"\nmax({a}, {b}) = {maximum}")
    
    # 3. Default value
    user_config = None
    config = user_config if user_config is not None else {"default": True}
    print(f"Config: {config}")
    
    # Equivalente con or (più pythonic per questo caso)
    # Ma attenzione: or usa truthy/falsy, non solo None!
    name = "" or "Guest"  # "Guest" (stringa vuota è falsy)
    print(f"Name: {name}")
    
    return True

# 🧪 TEST:
if __name__ == "__main__":
    esercizio_5_3()
    print("✅ Esercizio 5.3 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 5.4: Match-Case (Python 3.10+)
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Usa match-case per gestire diversi tipi di ordini trading:
- MARKET order
- LIMIT order  
- STOP order
- STOP_LIMIT order

💡 TEORIA:
match-case è structural pattern matching (da Python 3.10).
Permette pattern complessi con guards e destructuring.

🎯 SKILLS: Pattern matching, destructuring, guards
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Intermedio-Avanzato
"""

# ✅ SOLUZIONE:
def esercizio_5_4():
    """Match-Case - Structural Pattern Matching"""
    
    def process_trading_order(order):
        """Processa diversi tipi di ordini con match-case"""
        
        match order:
            # Pattern con valore specifico
            case {"type": "MARKET", "symbol": symbol, "quantity": qty}:
                return f"MARKET: Buy/Sell {qty} {symbol} at current price"
            
            # Pattern con guard (condizione aggiuntiva)
            case {"type": "LIMIT", "symbol": symbol, "quantity": qty, "price": price} if price > 0:
                return f"LIMIT: Buy/Sell {qty} {symbol} at ${price:.2f}"
            
            # Pattern con più campi
            case {"type": "STOP", "symbol": symbol, "stop_price": stop}:
                return f"STOP: Trigger at ${stop:.2f} for {symbol}"
            
            # Pattern con valori opzionali
            case {"type": "STOP_LIMIT", "symbol": symbol, "stop_price": stop, "limit_price": limit}:
                return f"STOP_LIMIT: {symbol} stop=${stop:.2f}, limit=${limit:.2f}"
            
            # Pattern con tipo specifico
            case {"type": order_type} if order_type not in ["MARKET", "LIMIT", "STOP", "STOP_LIMIT"]:
                return f"Unknown order type: {order_type}"
            
            # Catch-all (sempre alla fine)
            case _:
                return "Invalid order format"
    
    # Test con vari ordini
    test_orders = [
        {"type": "MARKET", "symbol": "AAPL", "quantity": 100},
        {"type": "LIMIT", "symbol": "GOOGL", "quantity": 50, "price": 140.50},
        {"type": "LIMIT", "symbol": "BAD", "quantity": 50, "price": -10},  # Invalid
        {"type": "STOP", "symbol": "TSLA", "stop_price": 200.00},
        {"type": "STOP_LIMIT", "symbol": "MSFT", "stop_price": 380, "limit_price": 375},
        {"type": "UNKNOWN_TYPE", "symbol": "X"},
        {"invalid": "order"},
    ]
    
    print("Processing orders con match-case:")
    for order in test_orders:
        result = process_trading_order(order)
        print(f"  {order}")
        print(f"    → {result}\n")
    
    # Bonus: Pattern matching su tuple/sequenze
    def analyze_price_action(candle):
        """Analizza candlestick pattern"""
        match candle:
            case (open, high, low, close) if close > open and (high - close) < (close - open) * 0.1:
                return "Bullish (strong close)"
            case (open, high, low, close) if close < open and (close - low) < (open - close) * 0.1:
                return "Bearish (strong close)"
            case (open, high, low, close) if abs(close - open) < (high - low) * 0.1:
                return "Doji (indecision)"
            case (open, _, _, close):
                return "Bullish" if close > open else "Bearish"
            case _:
                return "Invalid candle"
    
    candles = [
        (100, 110, 99, 109),   # Bullish
        (100, 101, 90, 91),    # Bearish
        (100, 105, 95, 100.5), # Doji
    ]
    
    print("Candlestick analysis:")
    for candle in candles:
        print(f"  OHLC {candle} → {analyze_price_action(candle)}")
    
    return process_trading_order

# 🧪 TEST:
if __name__ == "__main__":
    import sys
    if sys.version_info >= (3, 10):
        func = esercizio_5_4()
        print("✅ Esercizio 5.4 completato!\n")
    else:
        print("⚠️ Esercizio 5.4 richiede Python 3.10+\n")


# ==============================================================================
# SEZIONE 6: LOOPS
# ==============================================================================

print("\n" + "=" * 70)
print("SEZIONE 6: LOOPS")
print("=" * 70)

# ------------------------------------------------------------------------------
# ESERCIZIO 6.1: For Loop Patterns
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Implementa questi pattern comuni di for loop:
1. Iterazione con indice (enumerate)
2. Iterazione parallela (zip)
3. Iterazione con contatore
4. Iterazione al contrario

💡 TEORIA:
In Python, for itera su "iterabili" (liste, stringhe, range, ecc).
enumerate() e zip() sono strumenti fondamentali.

🎯 SKILLS: enumerate, zip, range, reversed
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Principiante-Intermedio
"""

# ✅ SOLUZIONE:
def esercizio_6_1():
    """For Loop Patterns - Pattern di iterazione"""
    
    prices = [100.5, 102.3, 99.8, 103.2, 101.0]
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    
    # 1. ENUMERATE: indice + valore
    print("--- ENUMERATE ---")
    
    for i, price in enumerate(prices):
        print(f"  [{i}] Price: ${price}")
    
    # Con start diverso da 0
    print("\nCon start=1:")
    for day, price in enumerate(prices, start=1):
        print(f"  Day {day}: ${price}")
    
    # 2. ZIP: iterazione parallela
    print("\n--- ZIP ---")
    
    for symbol, price in zip(symbols, prices):
        print(f"  {symbol}: ${price}")
    
    # Zip con 3+ iterabili
    quantities = [100, 50, 75, 200, 150]
    for symbol, price, qty in zip(symbols, prices, quantities):
        value = price * qty
        print(f"  {symbol}: {qty} shares @ ${price} = ${value:.2f}")
    
    # Zip si ferma al più corto!
    short_list = [1, 2]
    long_list = [10, 20, 30, 40]
    print(f"\nzip si ferma al più corto:")
    for a, b in zip(short_list, long_list):
        print(f"  {a}, {b}")
    
    # 3. RANGE: contatore
    print("\n--- RANGE ---")
    
    # range(stop)
    print("range(5):", list(range(5)))
    
    # range(start, stop)
    print("range(2, 7):", list(range(2, 7)))
    
    # range(start, stop, step)
    print("range(0, 10, 2):", list(range(0, 10, 2)))
    print("range(10, 0, -1):", list(range(10, 0, -1)))
    
    # 4. REVERSED: al contrario
    print("\n--- REVERSED ---")
    
    print("Prices al contrario:")
    for price in reversed(prices):
        print(f"  ${price}")
    
    # Combinazione: enumerate + reversed
    print("\nEnumerate + Reversed:")
    for i, price in enumerate(reversed(prices)):
        print(f"  [{i}] ${price}")
    
    # 5. APPLICAZIONE TRADING: Calcolo rendimenti
    print("\n--- CALCOLO RENDIMENTI ---")
    
    returns = []
    for i in range(1, len(prices)):
        ret = (prices[i] - prices[i-1]) / prices[i-1] * 100
        returns.append(ret)
        print(f"  Day {i}: {ret:+.2f}%")
    
    print(f"  Rendimento totale: {sum(returns):.2f}%")
    
    return returns

# 🧪 TEST:
if __name__ == "__main__":
    result = esercizio_6_1()
    assert len(result) == 4  # 5 prezzi = 4 rendimenti
    print("✅ Esercizio 6.1 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 6.2: While Loop Patterns
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Implementa:
1. While con condizione semplice
2. While con break/continue
3. While-else (poco noto ma utile!)
4. Simulazione trading con stop loss

💡 TEORIA:
while continua finché la condizione è True.
break esce dal loop, continue salta alla prossima iterazione.
else viene eseguito se il loop termina normalmente (senza break).

🎯 SKILLS: while, break, continue, else clause
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Intermedio
"""

# ✅ SOLUZIONE:
def esercizio_6_2():
    """While Loop Patterns - Pattern di while"""
    
    # 1. WHILE BASE
    print("--- WHILE BASE ---")
    
    countdown = 5
    while countdown > 0:
        print(f"  {countdown}...")
        countdown -= 1
    print("  🚀 Launch!")
    
    # 2. WHILE CON BREAK/CONTINUE
    print("\n--- BREAK e CONTINUE ---")
    
    # Cerca il primo numero divisibile per 7
    n = 1
    while n < 100:
        if n % 7 == 0:
            print(f"  Primo divisibile per 7: {n}")
            break
        n += 1
    
    # Stampa solo numeri pari
    print("\n  Numeri pari da 1 a 10:")
    n = 0
    while n < 10:
        n += 1
        if n % 2 != 0:
            continue  # Salta i dispari
        print(f"    {n}")
    
    # 3. WHILE-ELSE (eseguito se NO break)
    print("\n--- WHILE-ELSE ---")
    
    def find_factor(n, max_check=100):
        """Trova un fattore di n, o None"""
        i = 2
        while i < min(n, max_check):
            if n % i == 0:
                print(f"  Trovato fattore di {n}: {i}")
                break
            i += 1
        else:
            # Eseguito solo se while termina senza break!
            print(f"  {n} è primo (o non trovato fattore)")
    
    find_factor(15)   # Trova 3
    find_factor(17)   # Primo
    find_factor(97)   # Primo
    
    # 4. SIMULAZIONE TRADING
    print("\n--- SIMULAZIONE STOP LOSS ---")
    
    import random
    random.seed(42)  # Per riproducibilità
    
    entry_price = 100.0
    stop_loss = 95.0      # -5%
    take_profit = 110.0   # +10%
    current_price = entry_price
    day = 0
    
    print(f"  Entry: ${entry_price}, SL: ${stop_loss}, TP: ${take_profit}")
    
    while stop_loss < current_price < take_profit:
        day += 1
        # Simula movimento prezzo (-3% a +3%)
        change = random.uniform(-0.03, 0.03)
        current_price *= (1 + change)
        print(f"  Day {day}: ${current_price:.2f} ({change*100:+.2f}%)")
        
        if day > 30:  # Safety exit
            print("  Timeout: 30 giorni senza trigger")
            break
    
    # Risultato
    if current_price <= stop_loss:
        print(f"  ❌ STOP LOSS HIT! Loss: {(current_price - entry_price):.2f}")
    elif current_price >= take_profit:
        print(f"  ✅ TAKE PROFIT HIT! Profit: {(current_price - entry_price):.2f}")
    
    return day

# 🧪 TEST:
if __name__ == "__main__":
    result = esercizio_6_2()
    print("✅ Esercizio 6.2 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 6.3: Nested Loops
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Implementa:
1. Tabella moltiplicazione
2. Pattern triangolare
3. Ricerca in matrice
4. Correlazione tra asset (esempio trading)

💡 TEORIA:
I loop annidati hanno complessità O(n*m).
Per break da loop annidati, usa flag o funzioni.

🎯 SKILLS: Loop annidati, matrici, complessità
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Intermedio
"""

# ✅ SOLUZIONE:
def esercizio_6_3():
    """Nested Loops - Loop annidati"""
    
    # 1. TABELLA MOLTIPLICAZIONE
    print("--- TABELLA MOLTIPLICAZIONE ---")
    
    print("   ", end="")
    for i in range(1, 6):
        print(f"{i:4}", end="")
    print("\n   " + "-" * 20)
    
    for i in range(1, 6):
        print(f"{i} |", end="")
        for j in range(1, 6):
            print(f"{i*j:4}", end="")
        print()
    
    # 2. PATTERN TRIANGOLARE
    print("\n--- PATTERN TRIANGOLARE ---")
    
    n = 5
    for i in range(1, n + 1):
        print("  " + "*" * i)
    
    print()
    for i in range(n, 0, -1):
        print("  " + " " * (n - i) + "*" * i)
    
    # 3. RICERCA IN MATRICE
    print("\n--- RICERCA IN MATRICE ---")
    
    matrix = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ]
    
    target = 7
    found = False
    
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            if value == target:
                print(f"  Trovato {target} in posizione [{row_idx}][{col_idx}]")
                found = True
                break
        if found:
            break
    
    # Versione con funzione (più pulita per break multiplo)
    def find_in_matrix(matrix, target):
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                if val == target:
                    return (i, j)
        return None
    
    pos = find_in_matrix(matrix, 10)
    print(f"  find_in_matrix(10) = {pos}")
    
    # 4. CORRELAZIONE ASSET (esempio trading)
    print("\n--- CORRELAZIONE ASSET ---")
    
    # Dati semplificati
    assets = {
        'AAPL': [1.2, -0.5, 0.8, 1.1, -0.3],
        'GOOGL': [1.0, -0.3, 0.9, 0.8, -0.1],
        'MSFT': [0.9, -0.4, 0.7, 1.0, -0.2],
    }
    
    def simple_correlation(x, y):
        """Correlazione semplificata (non production!)"""
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denom_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
        denom_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5
        
        if denom_x == 0 or denom_y == 0:
            return 0
        return numerator / (denom_x * denom_y)
    
    symbols = list(assets.keys())
    
    print("  Matrice correlazione:")
    print(f"       ", end="")
    for s in symbols:
        print(f"{s:>7}", end="")
    print()
    
    for s1 in symbols:
        print(f"  {s1}", end="")
        for s2 in symbols:
            corr = simple_correlation(assets[s1], assets[s2])
            print(f"{corr:>7.2f}", end="")
        print()
    
    return True

# 🧪 TEST:
if __name__ == "__main__":
    esercizio_6_3()
    print("✅ Esercizio 6.3 completato!\n")


# ==============================================================================
# SEZIONE 7: LIST COMPREHENSIONS
# ==============================================================================

print("\n" + "=" * 70)
print("SEZIONE 7: LIST COMPREHENSIONS")
print("=" * 70)

# ------------------------------------------------------------------------------
# ESERCIZIO 7.1: Basic Comprehensions
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Converti questi loop in list comprehensions:
1. Quadrati dei numeri da 1 a 10
2. Numeri pari da una lista
3. Prezzi maggiori di 100

💡 TEORIA:
Sintassi: [expression for item in iterable if condition]
Più concise e spesso più veloci dei loop equivalenti.

🎯 SKILLS: List comprehension, filtering, mapping
⏱️ TEMPO: 10 minuti
🔢 LIVELLO: Principiante-Intermedio
"""

# ✅ SOLUZIONE:
def esercizio_7_1():
    """Basic Comprehensions - Comprehension di base"""
    
    # 1. QUADRATI (mapping)
    print("--- QUADRATI ---")
    
    # Con loop
    squares_loop = []
    for x in range(1, 11):
        squares_loop.append(x ** 2)
    
    # Con comprehension
    squares_comp = [x ** 2 for x in range(1, 11)]
    
    print(f"  Loop: {squares_loop}")
    print(f"  Comp: {squares_comp}")
    assert squares_loop == squares_comp
    
    # 2. NUMERI PARI (filtering)
    print("\n--- NUMERI PARI ---")
    
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Con loop
    evens_loop = []
    for n in numbers:
        if n % 2 == 0:
            evens_loop.append(n)
    
    # Con comprehension
    evens_comp = [n for n in numbers if n % 2 == 0]
    
    print(f"  Loop: {evens_loop}")
    print(f"  Comp: {evens_comp}")
    
    # 3. PREZZI > 100 (filtering con condizione)
    print("\n--- PREZZI > 100 ---")
    
    prices = [95.5, 102.3, 99.8, 150.2, 88.0, 110.5]
    
    high_prices = [p for p in prices if p > 100]
    print(f"  Prezzi > $100: {high_prices}")
    
    # 4. TRASFORMAZIONE + FILTERING
    print("\n--- MAPPING + FILTERING ---")
    
    # Arrotonda solo i prezzi sopra 100
    rounded_high = [round(p, 1) for p in prices if p > 100]
    print(f"  Prezzi > $100 arrotondati: {rounded_high}")
    
    # 5. CON IF-ELSE (attenzione alla posizione!)
    print("\n--- IF-ELSE ---")
    
    # if-else PRIMA del for (mapping condizionale)
    labels = ["HIGH" if p > 100 else "LOW" for p in prices]
    print(f"  Labels: {labels}")
    
    # CONFRONTO:
    # [expr for x in iterable if cond]         → filtering
    # [expr_true if cond else expr_false for x in iterable]  → mapping con condizione
    
    return squares_comp

# 🧪 TEST:
if __name__ == "__main__":
    result = esercizio_7_1()
    assert result == [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
    print("✅ Esercizio 7.1 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 7.2: Dict and Set Comprehensions
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Crea:
1. Dict comprehension: symbol → price
2. Dict comprehension con filtering
3. Set comprehension per valori unici

💡 TEORIA:
- Dict: {key_expr: value_expr for item in iterable if cond}
- Set: {expr for item in iterable if cond}

🎯 SKILLS: Dict/Set comprehension, data transformation
⏱️ TEMPO: 10 minuti
🔢 LIVELLO: Intermedio
"""

# ✅ SOLUZIONE:
def esercizio_7_2():
    """Dict and Set Comprehensions"""
    
    # 1. DICT COMPREHENSION BASE
    print("--- DICT COMPREHENSION ---")
    
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    prices = [150.0, 140.0, 380.0, 250.0]
    
    # Da due liste a dict
    portfolio = {symbol: price for symbol, price in zip(symbols, prices)}
    print(f"  Portfolio: {portfolio}")
    
    # Con trasformazione
    portfolio_upper = {s.lower(): p for s, p in zip(symbols, prices)}
    print(f"  Portfolio lower: {portfolio_upper}")
    
    # 2. DICT CON FILTERING
    print("\n--- DICT CON FILTERING ---")
    
    # Solo azioni sopra $200
    expensive = {s: p for s, p in portfolio.items() if p > 200}
    print(f"  Azioni > $200: {expensive}")
    
    # Invertire key-value (se i valori sono unici)
    inverted = {v: k for k, v in portfolio.items()}
    print(f"  Inverted: {inverted}")
    
    # 3. DICT DA LISTA DI TUPLE
    print("\n--- DICT DA TUPLE ---")
    
    trades = [
        ('AAPL', 100, 150.0),
        ('GOOGL', 50, 140.0),
        ('AAPL', 200, 152.0),  # Secondo trade AAPL
    ]
    
    # Totale per symbol
    from collections import defaultdict
    totals_dd = defaultdict(float)
    for symbol, qty, price in trades:
        totals_dd[symbol] += qty * price
    print(f"  Totali (defaultdict): {dict(totals_dd)}")
    
    # 4. SET COMPREHENSION
    print("\n--- SET COMPREHENSION ---")
    
    # Valori unici
    numbers = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    unique_squares = {x ** 2 for x in numbers}
    print(f"  Quadrati unici: {unique_squares}")
    
    # Simboli unici dai trade
    traded_symbols = {t[0] for t in trades}
    print(f"  Simboli tradati: {traded_symbols}")
    
    # 5. APPLICAZIONE TRADING
    print("\n--- POSIZIONI PORTFOLIO ---")
    
    positions = [
        {'symbol': 'AAPL', 'qty': 100, 'price': 150},
        {'symbol': 'GOOGL', 'qty': 50, 'price': 140},
        {'symbol': 'MSFT', 'qty': 75, 'price': 380},
    ]
    
    # Dict con valore di posizione
    position_values = {p['symbol']: p['qty'] * p['price'] for p in positions}
    print(f"  Valori posizioni: {position_values}")
    
    total_value = sum(position_values.values())
    print(f"  Valore totale: ${total_value:,.2f}")
    
    # Pesi percentuali
    weights = {s: v / total_value * 100 for s, v in position_values.items()}
    print(f"  Pesi: {weights}")
    
    return portfolio

# 🧪 TEST:
if __name__ == "__main__":
    result = esercizio_7_2()
    assert 'AAPL' in result
    print("✅ Esercizio 7.2 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 7.3: Nested Comprehensions
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Implementa:
1. Flatten di lista nested
2. Matrice con comprehension
3. Prodotto cartesiano

💡 TEORIA:
[expr for sublist in nested for item in sublist]
L'ordine dei for è lo stesso dei loop annidati equivalenti.

🎯 SKILLS: Nested comprehension, flatten, matrices
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Intermedio-Avanzato
"""

# ✅ SOLUZIONE:
def esercizio_7_3():
    """Nested Comprehensions - Comprehension annidate"""
    
    # 1. FLATTEN
    print("--- FLATTEN ---")
    
    nested = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    
    # Equivalente loop:
    flat_loop = []
    for sublist in nested:
        for item in sublist:
            flat_loop.append(item)
    
    # Comprehension (stesso ordine dei for!)
    flat_comp = [item for sublist in nested for item in sublist]
    
    print(f"  Nested: {nested}")
    print(f"  Flat: {flat_comp}")
    
    # 2. CREARE MATRICE
    print("\n--- CREARE MATRICE ---")
    
    # Matrice 3x4 di zeri
    rows, cols = 3, 4
    matrix_zeros = [[0 for _ in range(cols)] for _ in range(rows)]
    print(f"  Matrice zeri:")
    for row in matrix_zeros:
        print(f"    {row}")
    
    # Matrice con valori
    matrix_values = [[i * cols + j for j in range(cols)] for i in range(rows)]
    print(f"\n  Matrice con valori:")
    for row in matrix_values:
        print(f"    {row}")
    
    # 3. PRODOTTO CARTESIANO
    print("\n--- PRODOTTO CARTESIANO ---")
    
    colors = ['red', 'blue']
    sizes = ['S', 'M', 'L']
    
    # Tutte le combinazioni
    combinations = [(c, s) for c in colors for s in sizes]
    print(f"  Combinazioni: {combinations}")
    
    # 4. APPLICAZIONE TRADING: Griglia parametri
    print("\n--- GRIGLIA PARAMETRI BACKTEST ---")
    
    fast_periods = [5, 10, 15]
    slow_periods = [20, 30, 50]
    
    # Solo combinazioni valide (fast < slow)
    param_grid = [
        {'fast': f, 'slow': s} 
        for f in fast_periods 
        for s in slow_periods 
        if f < s
    ]
    
    print(f"  Combinazioni parametri ({len(param_grid)}):")
    for params in param_grid:
        print(f"    {params}")
    
    # 5. TRASPOSIZIONE MATRICE
    print("\n--- TRASPOSIZIONE ---")
    
    matrix = [
        [1, 2, 3],
        [4, 5, 6],
    ]
    
    # Trasposta: righe diventano colonne
    transposed = [[row[i] for row in matrix] for i in range(len(matrix[0]))]
    
    print(f"  Originale:")
    for row in matrix:
        print(f"    {row}")
    print(f"  Trasposta:")
    for row in transposed:
        print(f"    {row}")
    
    return flat_comp

# 🧪 TEST:
if __name__ == "__main__":
    result = esercizio_7_3()
    assert result == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    print("✅ Esercizio 7.3 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 7.4: Generator Expressions
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Confronta list comprehension vs generator expression:
1. Memory usage
2. Lazy evaluation
3. Quando usare quale

💡 TEORIA:
Generator expression: (expr for item in iterable)
- Non crea la lista in memoria
- Valuta elementi uno alla volta (lazy)
- Usare quando non serve la lista intera

🎯 SKILLS: Generators, memory efficiency, lazy evaluation
⏱️ TEMPO: 10 minuti
🔢 LIVELLO: Intermedio-Avanzato
"""

# ✅ SOLUZIONE:
def esercizio_7_4():
    """Generator Expressions - Espressioni generatore"""
    
    import sys
    
    # 1. SINTASSI
    print("--- SINTASSI ---")
    
    # List comprehension: parentesi quadre
    list_comp = [x ** 2 for x in range(10)]
    print(f"  List: {list_comp}")
    print(f"  Tipo: {type(list_comp)}")
    
    # Generator expression: parentesi tonde
    gen_exp = (x ** 2 for x in range(10))
    print(f"  Generator: {gen_exp}")
    print(f"  Tipo: {type(gen_exp)}")
    
    # 2. MEMORY USAGE
    print("\n--- MEMORY USAGE ---")
    
    n = 1000000
    
    # List: alloca tutta la memoria subito
    # list_big = [x for x in range(n)]  # ~8MB!
    # print(f"  List size: {sys.getsizeof(list_big):,} bytes")
    
    # Generator: quasi niente
    gen_big = (x for x in range(n))
    print(f"  Generator size: {sys.getsizeof(gen_big):,} bytes")
    
    # Confronto con range piccolo
    list_small = [x ** 2 for x in range(100)]
    gen_small = (x ** 2 for x in range(100))
    print(f"\n  List[100] size: {sys.getsizeof(list_small):,} bytes")
    print(f"  Gen[100] size: {sys.getsizeof(gen_small):,} bytes")
    
    # 3. LAZY EVALUATION
    print("\n--- LAZY EVALUATION ---")
    
    def expensive_operation(x):
        print(f"    Computing {x}...")
        return x ** 2
    
    print("  List comprehension (eager):")
    list_eager = [expensive_operation(x) for x in range(3)]
    print(f"  Risultato: {list_eager}")
    
    print("\n  Generator expression (lazy):")
    gen_lazy = (expensive_operation(x) for x in range(3))
    print("  Generator creato, nessun calcolo ancora!")
    print("  Ora consumo:")
    for val in gen_lazy:
        print(f"    Got: {val}")
    
    # 4. QUANDO USARE GENERATOR
    print("\n--- CASI D'USO ---")
    
    # ✅ Generator: quando passi a funzioni che consumano iterabili
    numbers = range(1000)
    
    # sum(), max(), min(), any(), all() accettano iterabili
    total = sum(x ** 2 for x in numbers)  # No parentesi extra!
    print(f"  sum(x² for x in range(1000)) = {total}")
    
    # Verifica se esiste un numero
    has_big = any(x > 500 for x in numbers)  # Si ferma appena trova!
    print(f"  any(x > 500) = {has_big}")
    
    # ✅ Generator: file grandi
    # with open('big_file.txt') as f:
    #     total_lines = sum(1 for line in f)  # Non carica tutto in RAM!
    
    # ❌ NON usare generator se:
    # - Devi accedere più volte ai dati
    # - Devi usare len(), indexing, slicing
    # - I dati sono pochi (overhead non vale)
    
    # 5. APPLICAZIONE TRADING
    print("\n--- STREAMING PRICES ---")
    
    def price_stream(initial, n_ticks):
        """Genera stream di prezzi (generator)"""
        import random
        price = initial
        for _ in range(n_ticks):
            price *= (1 + random.gauss(0, 0.001))
            yield round(price, 2)
    
    # Consumo lazy - potrebbe essere infinito!
    stream = price_stream(100.0, 5)
    print("  Stream creato (no calcoli)")
    
    for tick, price in enumerate(stream):
        print(f"    Tick {tick}: ${price}")
    
    return True

# 🧪 TEST:
if __name__ == "__main__":
    esercizio_7_4()
    print("✅ Esercizio 7.4 completato!\n")


# ==============================================================================
# RIEPILOGO SEZIONE 5-7: CONTROL FLOW
# ==============================================================================

print("\n" + "=" * 70)
print("RIEPILOGO: ESERCIZI CONTROL FLOW COMPLETATI")
print("=" * 70)

print("""
ESERCIZI COMPLETATI IN QUESTA PARTE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SEZIONE 5 - Conditional Statements:
  ✅ 5.1 Basic Conditionals
  ✅ 5.2 Guard Clauses
  ✅ 5.3 Ternary Operator
  ✅ 5.4 Match-Case (Python 3.10+)

SEZIONE 6 - Loops:
  ✅ 6.1 For Loop Patterns
  ✅ 6.2 While Loop Patterns
  ✅ 6.3 Nested Loops

SEZIONE 7 - Comprehensions:
  ✅ 7.1 Basic Comprehensions
  ✅ 7.2 Dict and Set Comprehensions
  ✅ 7.3 Nested Comprehensions
  ✅ 7.4 Generator Expressions

TOTALE QUESTA PARTE: 11 esercizi
TOTALE CUMULATIVO: 24 esercizi

PROSSIMA PARTE: Functions (definizione, args, decorators, closures)
""")

# Esegui tutti gli esercizi
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ESECUZIONE TUTTI GLI ESERCIZI PARTE 2")
    print("=" * 70 + "\n")
    
    esercizio_5_1()
    esercizio_5_2()
    esercizio_5_3()
    
    import sys
    if sys.version_info >= (3, 10):
        esercizio_5_4()
    
    esercizio_6_1()
    esercizio_6_2()
    esercizio_6_3()
    esercizio_7_1()
    esercizio_7_2()
    esercizio_7_3()
    esercizio_7_4()
    
    print("\n🎉 TUTTI GLI ESERCIZI DELLA PARTE 2 COMPLETATI!")
