"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    🐍 PYTHON MASTER - SCHEDA ESERCIZI COMPLETA               ║
║                                                                              ║
║                    PARTE 3: FUNCTIONS (Args, Decorators, Closures)           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ==============================================================================
# SEZIONE 8: FUNCTION BASICS
# ==============================================================================

print("=" * 70)
print("SEZIONE 8: FUNCTION BASICS")
print("=" * 70)

# ------------------------------------------------------------------------------
# ESERCIZIO 8.1: Function Definition
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Scrivi funzioni che:
1. Calcolano l'area di un cerchio
2. Convertono temperatura (C ↔ F)
3. Hanno docstring complete

💡 TEORIA:
Le funzioni sono oggetti first-class in Python.
Ogni funzione dovrebbe fare UNA cosa e farla bene.
Le docstring documentano cosa fa la funzione.

🎯 SKILLS: def, return, docstrings, single responsibility
⏱️ TEMPO: 10 minuti
🔢 LIVELLO: Principiante
"""

# ✅ SOLUZIONE:
def esercizio_8_1():
    """Function Definition - Definizione funzioni"""
    
    import math
    
    # 1. FUNZIONE BASE CON DOCSTRING
    def circle_area(radius):
        """
        Calcola l'area di un cerchio.
        
        Args:
            radius: Il raggio del cerchio (deve essere >= 0)
            
        Returns:
            L'area del cerchio (π * r²)
            
        Raises:
            ValueError: Se radius è negativo
            
        Examples:
            >>> circle_area(1)
            3.141592653589793
            >>> circle_area(2)
            12.566370614359172
        """
        if radius < 0:
            raise ValueError("Il raggio non può essere negativo")
        return math.pi * radius ** 2
    
    # Test
    print("--- AREA CERCHIO ---")
    for r in [0, 1, 2, 5]:
        print(f"  circle_area({r}) = {circle_area(r):.4f}")
    
    # 2. FUNZIONI DI CONVERSIONE
    def celsius_to_fahrenheit(celsius):
        """Converte Celsius in Fahrenheit: F = C * 9/5 + 32"""
        return celsius * 9/5 + 32
    
    def fahrenheit_to_celsius(fahrenheit):
        """Converte Fahrenheit in Celsius: C = (F - 32) * 5/9"""
        return (fahrenheit - 32) * 5/9
    
    print("\n--- CONVERSIONE TEMPERATURE ---")
    temps_c = [0, 20, 37, 100]
    for c in temps_c:
        f = celsius_to_fahrenheit(c)
        back_to_c = fahrenheit_to_celsius(f)
        print(f"  {c}°C → {f}°F → {back_to_c}°C")
    
    # 3. RETURN MULTIPLI (tuple unpacking)
    def min_max(numbers):
        """Restituisce minimo e massimo di una lista."""
        if not numbers:
            return None, None
        return min(numbers), max(numbers)
    
    print("\n--- RETURN MULTIPLI ---")
    data = [5, 2, 8, 1, 9, 3]
    minimum, maximum = min_max(data)
    print(f"  min_max({data}) = ({minimum}, {maximum})")
    
    # 4. FUNZIONE CON SIDE EFFECTS (da evitare quando possibile)
    results = []
    
    def append_and_return(value, lst=None):
        """ATTENZIONE: ha side effect se lst è passato!"""
        if lst is None:
            lst = results
        lst.append(value)
        return lst
    
    print("\n--- SIDE EFFECTS ---")
    print(f"  results prima: {results}")
    append_and_return(1)
    append_and_return(2)
    print(f"  results dopo: {results}")
    
    return circle_area

# 🧪 TEST:
if __name__ == "__main__":
    func = esercizio_8_1()
    import math
    assert abs(func(1) - math.pi) < 0.0001
    print("✅ Esercizio 8.1 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 8.2: Arguments - *args e **kwargs
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Implementa funzioni che usano:
1. Argomenti posizionali e keyword
2. *args per argomenti variabili
3. **kwargs per keyword variabili
4. Combinazione di tutti

💡 TEORIA:
- *args: raccoglie argomenti posizionali extra in una tupla
- **kwargs: raccoglie keyword arguments extra in un dict
- Ordine: positional, *args, keyword, **kwargs

🎯 SKILLS: *args, **kwargs, argument unpacking
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Intermedio
"""

# ✅ SOLUZIONE:
def esercizio_8_2():
    """Arguments - *args e **kwargs"""
    
    # 1. ARGOMENTI POSIZIONALI E KEYWORD
    print("--- POSITIONAL E KEYWORD ---")
    
    def create_order(symbol, quantity, price, order_type="LIMIT"):
        """Crea un ordine con parametri misti."""
        return {
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'type': order_type
        }
    
    # Tutti posizionali
    order1 = create_order("AAPL", 100, 150.0)
    print(f"  Posizionali: {order1}")
    
    # Mix posizionali e keyword
    order2 = create_order("GOOGL", quantity=50, price=140.0, order_type="MARKET")
    print(f"  Mix: {order2}")
    
    # 2. *ARGS - Argomenti variabili
    print("\n--- *ARGS ---")
    
    def calculate_total(*prices):
        """Calcola il totale di N prezzi."""
        print(f"  prices è una tupla: {type(prices)}")
        print(f"  Contenuto: {prices}")
        return sum(prices)
    
    total = calculate_total(10.5, 20.3, 15.0, 8.99)
    print(f"  Totale: ${total:.2f}")
    
    # Passare lista esistente con *
    price_list = [10, 20, 30]
    total2 = calculate_total(*price_list)  # Spacchetta la lista!
    print(f"  Totale da lista: ${total2}")
    
    # 3. **KWARGS - Keyword variabili
    print("\n--- **KWARGS ---")
    
    def create_position(**details):
        """Crea una posizione con dettagli variabili."""
        print(f"  details è un dict: {type(details)}")
        print(f"  Contenuto: {details}")
        return details
    
    pos = create_position(symbol="AAPL", quantity=100, entry_price=150.0, 
                          stop_loss=145.0, take_profit=165.0)
    print(f"  Posizione: {pos}")
    
    # Passare dict esistente con **
    config = {'leverage': 10, 'margin_type': 'cross'}
    pos2 = create_position(symbol="BTC", **config)  # Spacchetta il dict!
    print(f"  Posizione con config: {pos2}")
    
    # 4. COMBINAZIONE COMPLETA
    print("\n--- COMBINAZIONE ---")
    
    def advanced_order(symbol, quantity, *tags, price=None, **extra):
        """
        symbol: obbligatorio
        quantity: obbligatorio
        *tags: argomenti extra posizionali
        price: keyword con default
        **extra: keyword extra
        """
        order = {
            'symbol': symbol,
            'quantity': quantity,
            'tags': tags,
            'price': price,
            'extra': extra
        }
        return order
    
    order = advanced_order(
        "TSLA",           # symbol (positional)
        100,              # quantity (positional)
        "urgent",         # *tags
        "vip",            # *tags
        price=250.0,      # price (keyword)
        broker="Alpaca",  # **extra
        strategy="momentum"  # **extra
    )
    
    print(f"  Ordine avanzato:")
    for key, value in order.items():
        print(f"    {key}: {value}")
    
    # 5. ARGUMENT FORWARDING
    print("\n--- ARGUMENT FORWARDING ---")
    
    def wrapper_function(*args, **kwargs):
        """Inoltra tutti gli argomenti a un'altra funzione."""
        print(f"  Ricevuto args: {args}")
        print(f"  Ricevuto kwargs: {kwargs}")
        return create_order(*args, **kwargs)
    
    forwarded = wrapper_function("MSFT", 200, 380.0, order_type="STOP")
    print(f"  Forwarded: {forwarded}")
    
    return True

# 🧪 TEST:
if __name__ == "__main__":
    esercizio_8_2()
    print("✅ Esercizio 8.2 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 8.3: Positional-Only e Keyword-Only
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Implementa funzioni con:
1. Parametri positional-only (/)
2. Parametri keyword-only (*)
3. Combinazione di entrambi

💡 TEORIA:
- / separa positional-only a sinistra
- * separa keyword-only a destra
- Utile per API chiare e retrocompatibilità

🎯 SKILLS: Positional-only, keyword-only, API design
⏱️ TEMPO: 10 minuti
🔢 LIVELLO: Intermedio-Avanzato
"""

# ✅ SOLUZIONE:
def esercizio_8_3():
    """Positional-Only e Keyword-Only"""
    
    # 1. POSITIONAL-ONLY (Python 3.8+)
    print("--- POSITIONAL-ONLY ---")
    
    def divide(x, y, /):
        """x e y DEVONO essere passati come posizionali."""
        return x / y
    
    print(f"  divide(10, 3) = {divide(10, 3)}")
    # divide(x=10, y=3)  # ERRORE! Non puoi usare keyword
    
    # Perché usarlo?
    # - Permette di cambiare i nomi dei parametri senza breaking change
    # - Rende l'API più chiara
    
    # 2. KEYWORD-ONLY
    print("\n--- KEYWORD-ONLY ---")
    
    def place_order(symbol, quantity, *, price, order_type="LIMIT"):
        """price e order_type DEVONO essere keyword."""
        return f"{order_type} {symbol}: {quantity} @ ${price}"
    
    # OK
    print(f"  {place_order('AAPL', 100, price=150.0)}")
    print(f"  {place_order('GOOGL', 50, price=140.0, order_type='MARKET')}")
    
    # place_order('AAPL', 100, 150.0)  # ERRORE! price deve essere keyword
    
    # 3. COMBINAZIONE
    print("\n--- COMBINAZIONE ---")
    
    def complex_function(pos_only, /, standard, *, kw_only):
        """
        pos_only: solo posizionale (prima di /)
        standard: posizionale O keyword (tra / e *)
        kw_only: solo keyword (dopo *)
        """
        return f"pos={pos_only}, std={standard}, kw={kw_only}"
    
    # Varie combinazioni valide
    print(f"  {complex_function(1, 2, kw_only=3)}")
    print(f"  {complex_function(1, standard=2, kw_only=3)}")
    
    # 4. APPLICAZIONE TRADING
    print("\n--- API TRADING ---")
    
    def execute_trade(symbol, /, *, side, quantity, price=None, 
                      time_in_force="GTC", client_order_id=None):
        """
        API trade chiara e safe:
        - symbol: sempre primo, no ambiguità
        - side, quantity: espliciti (evita errori)
        - altri: opzionali con default
        """
        trade = {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'time_in_force': time_in_force,
        }
        if client_order_id:
            trade['client_order_id'] = client_order_id
        return trade
    
    trade1 = execute_trade("AAPL", side="BUY", quantity=100, price=150.0)
    trade2 = execute_trade("BTC-USD", side="SELL", quantity=0.5)
    
    print(f"  Trade 1: {trade1}")
    print(f"  Trade 2: {trade2}")
    
    return True

# 🧪 TEST:
if __name__ == "__main__":
    esercizio_8_3()
    print("✅ Esercizio 8.3 completato!\n")


# ==============================================================================
# SEZIONE 9: CLOSURES E SCOPE
# ==============================================================================

print("\n" + "=" * 70)
print("SEZIONE 9: CLOSURES E SCOPE")
print("=" * 70)

# ------------------------------------------------------------------------------
# ESERCIZIO 9.1: Variable Scope (LEGB)
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Dimostra le regole di scope LEGB:
- Local: variabili nella funzione
- Enclosing: variabili in funzioni contenitore
- Global: variabili a livello modulo
- Built-in: funzioni/nomi predefiniti

💡 TEORIA:
Python cerca i nomi nell'ordine L → E → G → B.
global e nonlocal permettono di modificare scope esterni.

🎯 SKILLS: LEGB rule, global, nonlocal
⏱️ TEMPO: 10 minuti
🔢 LIVELLO: Intermedio
"""

# Variabile Global
GLOBAL_VAR = "I'm global"

# ✅ SOLUZIONE:
def esercizio_9_1():
    """Variable Scope - Regole LEGB"""
    
    # 1. SCOPE LOCALI E GLOBALI
    print("--- LOCAL vs GLOBAL ---")
    
    x = "global x"
    
    def show_scope():
        x = "local x"  # Crea nuova variabile locale!
        print(f"  Dentro funzione: {x}")
    
    show_scope()
    print(f"  Fuori funzione: {x}")  # Global non modificato
    
    # 2. GLOBAL KEYWORD
    print("\n--- GLOBAL KEYWORD ---")
    
    counter = 0
    
    def increment():
        global counter  # Modifica la variabile globale
        counter += 1
        print(f"  Counter dopo increment: {counter}")
    
    print(f"  Counter iniziale: {counter}")
    increment()
    increment()
    print(f"  Counter finale: {counter}")
    
    # 3. ENCLOSING SCOPE (closure)
    print("\n--- ENCLOSING SCOPE ---")
    
    def outer():
        outer_var = "outer"
        
        def inner():
            # Può leggere outer_var (enclosing scope)
            print(f"  Inner vede outer_var: {outer_var}")
        
        inner()
    
    outer()
    
    # 4. NONLOCAL KEYWORD
    print("\n--- NONLOCAL KEYWORD ---")
    
    def counter_factory():
        count = 0  # Enclosing scope
        
        def increment():
            nonlocal count  # Modifica variabile enclosing
            count += 1
            return count
        
        def get():
            return count
        
        return increment, get
    
    inc, get = counter_factory()
    print(f"  get() = {get()}")
    print(f"  inc() = {inc()}")
    print(f"  inc() = {inc()}")
    print(f"  get() = {get()}")
    
    # 5. SHADOWING (da evitare!)
    print("\n--- SHADOWING (BAD!) ---")
    
    # Evita di usare nomi builtin!
    # list = [1, 2, 3]  # MALE! Ora list() non funziona più
    
    # Invece usa nomi descrittivi
    price_list = [100, 200, 300]
    print(f"  price_list: {price_list}")
    print(f"  list builtin funziona ancora: {list(range(3))}")
    
    return True

# 🧪 TEST:
if __name__ == "__main__":
    esercizio_9_1()
    print("✅ Esercizio 9.1 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 9.2: Closures
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Implementa closures per:
1. Contatore con stato
2. Moltiplicatore configurabile
3. Logger con prefisso

💡 TEORIA:
Una closure "cattura" variabili dallo scope esterno.
La funzione interna mantiene accesso a quelle variabili anche dopo
che la funzione esterna è terminata.

🎯 SKILLS: Closures, function factories, state encapsulation
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Intermedio-Avanzato
"""

# ✅ SOLUZIONE:
def esercizio_9_2():
    """Closures - Funzioni che catturano stato"""
    
    # 1. CONTATORE CON STATO
    print("--- CONTATORE CLOSURE ---")
    
    def make_counter(start=0):
        """Factory che crea contatori."""
        count = start
        
        def counter():
            nonlocal count
            count += 1
            return count
        
        return counter
    
    counter1 = make_counter()
    counter2 = make_counter(100)
    
    print(f"  counter1: {counter1()}, {counter1()}, {counter1()}")
    print(f"  counter2: {counter2()}, {counter2()}")
    print(f"  counter1 ancora: {counter1()}")  # Indipendente!
    
    # 2. MOLTIPLICATORE CONFIGURABILE
    print("\n--- MOLTIPLICATORE ---")
    
    def make_multiplier(factor):
        """Crea una funzione che moltiplica per factor."""
        def multiply(x):
            return x * factor
        return multiply
    
    double = make_multiplier(2)
    triple = make_multiplier(3)
    
    print(f"  double(5) = {double(5)}")
    print(f"  triple(5) = {triple(5)}")
    print(f"  double(triple(4)) = {double(triple(4))}")
    
    # 3. LOGGER CON PREFISSO
    print("\n--- LOGGER CLOSURE ---")
    
    def make_logger(prefix, include_timestamp=False):
        """Crea un logger configurato."""
        from datetime import datetime
        
        def log(message):
            if include_timestamp:
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"[{ts}] [{prefix}] {message}")
            else:
                print(f"[{prefix}] {message}")
        
        return log
    
    debug = make_logger("DEBUG", include_timestamp=True)
    error = make_logger("ERROR")
    
    debug("Starting process")
    debug("Loading data")
    error("Something went wrong!")
    
    # 4. APPLICAZIONE TRADING: Position Sizer
    print("\n--- POSITION SIZER ---")
    
    def make_position_sizer(max_risk_percent, account_balance):
        """
        Crea un position sizer con risk management.
        Cattura i parametri di rischio.
        """
        def calculate_size(entry_price, stop_loss_price):
            risk_per_share = abs(entry_price - stop_loss_price)
            if risk_per_share == 0:
                return 0
            
            max_risk_amount = account_balance * (max_risk_percent / 100)
            position_size = max_risk_amount / risk_per_share
            
            return int(position_size)
        
        return calculate_size
    
    # Configurazione rischio: 1% del conto da $10,000
    sizer = make_position_sizer(max_risk_percent=1.0, account_balance=10000)
    
    # Calcola size per diversi trade
    trades = [
        (100.0, 95.0),   # Entry 100, SL 95 (risk $5/share)
        (50.0, 48.0),    # Entry 50, SL 48 (risk $2/share)
        (200.0, 190.0),  # Entry 200, SL 190 (risk $10/share)
    ]
    
    for entry, sl in trades:
        size = sizer(entry, sl)
        risk_total = size * abs(entry - sl)
        print(f"  Entry ${entry}, SL ${sl} → Size: {size} shares (Risk: ${risk_total:.2f})")
    
    # 5. LATE BINDING PROBLEM
    print("\n--- LATE BINDING (GOTCHA!) ---")
    
    # Problema: closure cattura la variabile, non il valore!
    funcs_bad = []
    for i in range(3):
        funcs_bad.append(lambda: i)  # Tutte vedono l'ultimo i!
    
    print("  Late binding (sbagliato):")
    for f in funcs_bad:
        print(f"    {f()}", end=" ")  # 2, 2, 2!
    print()
    
    # Soluzione: cattura il valore con default argument
    funcs_good = []
    for i in range(3):
        funcs_good.append(lambda i=i: i)  # Cattura il valore!
    
    print("  Con default argument (corretto):")
    for f in funcs_good:
        print(f"    {f()}", end=" ")  # 0, 1, 2!
    print()
    
    return make_counter

# 🧪 TEST:
if __name__ == "__main__":
    factory = esercizio_9_2()
    counter = factory(10)
    assert counter() == 11
    assert counter() == 12
    print("✅ Esercizio 9.2 completato!\n")


# ==============================================================================
# SEZIONE 10: DECORATORS
# ==============================================================================

print("\n" + "=" * 70)
print("SEZIONE 10: DECORATORS")
print("=" * 70)

# ------------------------------------------------------------------------------
# ESERCIZIO 10.1: Basic Decorators
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Implementa decoratori:
1. Timer decorator
2. Logger decorator
3. Decorator con @wraps

💡 TEORIA:
Un decorator è una funzione che prende una funzione e restituisce
una funzione modificata. La sintassi @decorator è zucchero sintattico.

🎯 SKILLS: Decorators, @wraps, function metadata
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Intermedio-Avanzato
"""

# ✅ SOLUZIONE:
def esercizio_10_1():
    """Basic Decorators - Decoratori di base"""
    
    import time
    from functools import wraps
    
    # 1. DECORATOR SEMPLICE
    print("--- DECORATOR SEMPLICE ---")
    
    def simple_decorator(func):
        def wrapper(*args, **kwargs):
            print(f"  Prima di {func.__name__}")
            result = func(*args, **kwargs)
            print(f"  Dopo di {func.__name__}")
            return result
        return wrapper
    
    @simple_decorator
    def say_hello(name):
        print(f"  Hello, {name}!")
        return f"Greeted {name}"
    
    result = say_hello("Marco")
    print(f"  Risultato: {result}")
    
    # 2. TIMER DECORATOR
    print("\n--- TIMER DECORATOR ---")
    
    def timer(func):
        """Misura il tempo di esecuzione."""
        @wraps(func)  # Preserva metadata della funzione!
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            print(f"  {func.__name__} eseguita in {end - start:.4f}s")
            return result
        return wrapper
    
    @timer
    def slow_function():
        """Funzione lenta per test."""
        time.sleep(0.1)
        return "done"
    
    slow_function()
    
    # Verifica che @wraps preserva i metadati
    print(f"  Nome funzione: {slow_function.__name__}")
    print(f"  Docstring: {slow_function.__doc__}")
    
    # 3. LOGGER DECORATOR
    print("\n--- LOGGER DECORATOR ---")
    
    def logger(func):
        """Logga chiamate e risultati."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)
            
            print(f"  → Chiamata: {func.__name__}({signature})")
            
            try:
                result = func(*args, **kwargs)
                print(f"  ← Ritorno: {result!r}")
                return result
            except Exception as e:
                print(f"  ✗ Eccezione: {e}")
                raise
        
        return wrapper
    
    @logger
    def add(a, b):
        return a + b
    
    @logger
    def divide(a, b):
        return a / b
    
    add(2, 3)
    add(10, b=20)
    divide(10, 2)
    
    try:
        divide(10, 0)
    except ZeroDivisionError:
        pass
    
    # 4. MULTIPLI DECORATORI
    print("\n--- MULTIPLI DECORATORI ---")
    
    @timer
    @logger
    def complex_operation(n):
        """Operazione complessa."""
        return sum(range(n))
    
    # Ordine: timer(logger(complex_operation))
    # Il più interno (logger) viene eseguito prima
    result = complex_operation(10000)
    
    return timer

# 🧪 TEST:
if __name__ == "__main__":
    decorator = esercizio_10_1()
    print("✅ Esercizio 10.1 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 10.2: Decorators with Arguments
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Implementa decoratori parametrici:
1. Retry decorator
2. Rate limiter
3. Validator decorator

💡 TEORIA:
Per decoratori con argomenti, serve un livello di nesting in più:
decorator_factory(args) → decorator(func) → wrapper(*args, **kwargs)

🎯 SKILLS: Decorator factories, parametric decorators
⏱️ TEMPO: 20 minuti
🔢 LIVELLO: Avanzato
"""

# ✅ SOLUZIONE:
def esercizio_10_2():
    """Decorators with Arguments - Decoratori parametrici"""
    
    import time
    from functools import wraps
    import random
    
    # 1. RETRY DECORATOR
    print("--- RETRY DECORATOR ---")
    
    def retry(max_attempts=3, delay=1.0, exceptions=(Exception,)):
        """
        Riprova la funzione se fallisce.
        
        Args:
            max_attempts: Numero massimo di tentativi
            delay: Secondi tra tentativi
            exceptions: Tuple di eccezioni da catchare
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(1, max_attempts + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        print(f"  Tentativo {attempt}/{max_attempts} fallito: {e}")
                        if attempt < max_attempts:
                            time.sleep(delay)
                
                raise last_exception
            return wrapper
        return decorator
    
    # Simula funzione che fallisce random
    @retry(max_attempts=3, delay=0.1)
    def unreliable_api():
        """API inaffidabile per test."""
        if random.random() < 0.7:  # 70% fallimento
            raise ConnectionError("API non disponibile")
        return "Successo!"
    
    try:
        result = unreliable_api()
        print(f"  Risultato: {result}")
    except ConnectionError:
        print("  Fallito dopo tutti i tentativi")
    
    # 2. RATE LIMITER
    print("\n--- RATE LIMITER ---")
    
    def rate_limit(calls_per_second):
        """Limita le chiamate per secondo."""
        min_interval = 1.0 / calls_per_second
        last_call = [0.0]  # Lista per mutabilità in closure
        
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                elapsed = time.time() - last_call[0]
                if elapsed < min_interval:
                    sleep_time = min_interval - elapsed
                    print(f"  Rate limiting: sleep {sleep_time:.3f}s")
                    time.sleep(sleep_time)
                
                last_call[0] = time.time()
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    @rate_limit(calls_per_second=2)  # Max 2 chiamate al secondo
    def api_call(endpoint):
        return f"Called {endpoint}"
    
    print("  Chiamate rapide:")
    for i in range(3):
        result = api_call(f"/endpoint/{i}")
        print(f"  {result}")
    
    # 3. VALIDATOR DECORATOR
    print("\n--- VALIDATOR DECORATOR ---")
    
    def validate_types(**type_hints):
        """Valida i tipi degli argomenti."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Ottieni i nomi dei parametri
                import inspect
                sig = inspect.signature(func)
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                
                # Valida i tipi
                for param_name, expected_type in type_hints.items():
                    if param_name in bound.arguments:
                        value = bound.arguments[param_name]
                        if not isinstance(value, expected_type):
                            raise TypeError(
                                f"{param_name} deve essere {expected_type.__name__}, "
                                f"ricevuto {type(value).__name__}"
                            )
                
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    @validate_types(symbol=str, quantity=(int, float), price=(int, float))
    def create_order(symbol, quantity, price):
        return {'symbol': symbol, 'quantity': quantity, 'price': price}
    
    # OK
    order = create_order("AAPL", 100, 150.0)
    print(f"  Ordine valido: {order}")
    
    # Errore
    try:
        create_order(123, "cento", 150.0)
    except TypeError as e:
        print(f"  Errore validazione: {e}")
    
    # 4. DECORATOR CON OPZIONI
    print("\n--- DECORATOR CON OPZIONI ---")
    
    def debug(enabled=True, prefix="DEBUG"):
        """Decorator che può essere abilitato/disabilitato."""
        def decorator(func):
            if not enabled:
                return func  # Nessun wrapping!
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                print(f"  [{prefix}] Calling {func.__name__}")
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    @debug(enabled=True, prefix="TRADE")
    def buy_stock(symbol):
        return f"Bought {symbol}"
    
    @debug(enabled=False)  # Disabilitato in produzione
    def internal_function():
        return "internal"
    
    buy_stock("AAPL")
    internal_function()  # Nessun output debug
    
    return retry

# 🧪 TEST:
if __name__ == "__main__":
    esercizio_10_2()
    print("✅ Esercizio 10.2 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 10.3: Class Decorators
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Implementa:
1. Decorator che è una classe
2. Decorator che decora una classe
3. Cache decorator con classe

💡 TEORIA:
I decorator possono essere anche classi (con __call__).
Si può decorare una classe per modificarne il comportamento.

🎯 SKILLS: Class decorators, __call__, singleton pattern
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Avanzato
"""

# ✅ SOLUZIONE:
def esercizio_10_3():
    """Class Decorators - Decoratori con classi"""
    
    from functools import wraps
    
    # 1. DECORATOR COME CLASSE
    print("--- DECORATOR COME CLASSE ---")
    
    class CountCalls:
        """Conta quante volte una funzione viene chiamata."""
        
        def __init__(self, func):
            wraps(func)(self)  # Copia metadata
            self.func = func
            self.count = 0
        
        def __call__(self, *args, **kwargs):
            self.count += 1
            print(f"  Chiamata #{self.count} di {self.func.__name__}")
            return self.func(*args, **kwargs)
    
    @CountCalls
    def say_hello():
        return "Hello!"
    
    say_hello()
    say_hello()
    say_hello()
    print(f"  Totale chiamate: {say_hello.count}")
    
    # 2. CACHE CON CLASSE
    print("\n--- CACHE DECORATOR ---")
    
    class Memoize:
        """Cache risultati di funzioni pure."""
        
        def __init__(self, func):
            wraps(func)(self)
            self.func = func
            self.cache = {}
        
        def __call__(self, *args):
            if args in self.cache:
                print(f"  Cache hit per {args}")
                return self.cache[args]
            
            print(f"  Computing {self.func.__name__}{args}")
            result = self.func(*args)
            self.cache[args] = result
            return result
        
        def clear_cache(self):
            self.cache.clear()
    
    @Memoize
    def fibonacci(n):
        if n < 2:
            return n
        return fibonacci(n - 1) + fibonacci(n - 2)
    
    print(f"  fib(10) = {fibonacci(10)}")
    print(f"  fib(10) again = {fibonacci(10)}")  # Cache hit
    print(f"  Cache size: {len(fibonacci.cache)}")
    
    # 3. DECORATOR DI CLASSE (decora una classe)
    print("\n--- DECORATOR DI CLASSE ---")
    
    def singleton(cls):
        """Assicura che esista solo un'istanza della classe."""
        instances = {}
        
        @wraps(cls, updated=[])
        def get_instance(*args, **kwargs):
            if cls not in instances:
                instances[cls] = cls(*args, **kwargs)
            return instances[cls]
        
        return get_instance
    
    @singleton
    class DatabaseConnection:
        def __init__(self, host="localhost"):
            print(f"  Creando connessione a {host}")
            self.host = host
    
    db1 = DatabaseConnection("server1")
    db2 = DatabaseConnection("server2")  # Non crea nuova istanza!
    
    print(f"  db1 is db2: {db1 is db2}")
    print(f"  Host: {db1.host}")  # Sempre server1
    
    # 4. ADD METHODS TO CLASS
    print("\n--- AGGIUNGI METODI ---")
    
    def add_comparison(cls):
        """Aggiunge metodi di confronto basati su un attributo 'value'."""
        
        def __lt__(self, other):
            return self.value < other.value
        
        def __le__(self, other):
            return self.value <= other.value
        
        def __gt__(self, other):
            return self.value > other.value
        
        def __ge__(self, other):
            return self.value >= other.value
        
        cls.__lt__ = __lt__
        cls.__le__ = __le__
        cls.__gt__ = __gt__
        cls.__ge__ = __ge__
        
        return cls
    
    @add_comparison
    class Price:
        def __init__(self, value):
            self.value = value
        
        def __repr__(self):
            return f"Price({self.value})"
    
    p1 = Price(100)
    p2 = Price(150)
    
    print(f"  {p1} < {p2}: {p1 < p2}")
    print(f"  {p1} > {p2}: {p1 > p2}")
    
    return CountCalls

# 🧪 TEST:
if __name__ == "__main__":
    esercizio_10_3()
    print("✅ Esercizio 10.3 completato!\n")


# ==============================================================================
# SEZIONE 11: GENERATORS
# ==============================================================================

print("\n" + "=" * 70)
print("SEZIONE 11: GENERATORS")
print("=" * 70)

# ------------------------------------------------------------------------------
# ESERCIZIO 11.1: Generator Functions
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Implementa generators per:
1. Range personalizzato
2. Fibonacci infinito
3. File reader lazy

💡 TEORIA:
Un generator usa yield invece di return.
Ogni yield sospende l'esecuzione e restituisce un valore.
La funzione riprende dalla prossima chiamata a next().

🎯 SKILLS: yield, generator protocol, lazy evaluation
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Intermedio-Avanzato
"""

# ✅ SOLUZIONE:
def esercizio_11_1():
    """Generator Functions - Funzioni generatore"""
    
    # 1. GENERATOR SEMPLICE
    print("--- GENERATOR SEMPLICE ---")
    
    def countdown(n):
        """Genera countdown da n a 1."""
        while n > 0:
            yield n
            n -= 1
    
    print("  Countdown:")
    for num in countdown(5):
        print(f"    {num}...")
    print("    🚀 Launch!")
    
    # 2. CUSTOM RANGE
    print("\n--- CUSTOM RANGE ---")
    
    def my_range(start, stop=None, step=1):
        """Implementazione di range come generator."""
        if stop is None:
            start, stop = 0, start
        
        current = start
        while (step > 0 and current < stop) or (step < 0 and current > stop):
            yield current
            current += step
    
    print(f"  my_range(5): {list(my_range(5))}")
    print(f"  my_range(2, 8): {list(my_range(2, 8))}")
    print(f"  my_range(0, 10, 2): {list(my_range(0, 10, 2))}")
    print(f"  my_range(10, 0, -2): {list(my_range(10, 0, -2))}")
    
    # 3. FIBONACCI INFINITO
    print("\n--- FIBONACCI INFINITO ---")
    
    def fibonacci():
        """Genera la sequenza di Fibonacci all'infinito."""
        a, b = 0, 1
        while True:
            yield a
            a, b = b, a + b
    
    fib = fibonacci()
    print("  Primi 10 Fibonacci:")
    for i, num in enumerate(fib):
        if i >= 10:
            break
        print(f"    F({i}) = {num}")
    
    # 4. GENERATOR CON RETURN
    print("\n--- GENERATOR CON RETURN ---")
    
    def limited_counter(limit):
        """Conta fino a limit e ritorna il totale."""
        count = 0
        while count < limit:
            yield count
            count += 1
        return f"Contato fino a {count}"  # Accessibile via StopIteration
    
    gen = limited_counter(3)
    try:
        while True:
            print(f"    {next(gen)}")
    except StopIteration as e:
        print(f"  Return value: {e.value}")
    
    # 5. YIELD FROM
    print("\n--- YIELD FROM ---")
    
    def flatten(nested):
        """Flatten lista nested usando yield from."""
        for item in nested:
            if isinstance(item, list):
                yield from flatten(item)  # Delega a sub-generator
            else:
                yield item
    
    nested = [1, [2, 3, [4, 5]], 6, [7, [8, 9]]]
    flat = list(flatten(nested))
    print(f"  Nested: {nested}")
    print(f"  Flat: {flat}")
    
    # 6. APPLICAZIONE TRADING: Price Stream
    print("\n--- PRICE STREAM ---")
    
    import random
    
    def price_stream(symbol, initial_price, volatility=0.01):
        """Genera stream di prezzi simulati."""
        price = initial_price
        tick = 0
        while True:
            yield {
                'symbol': symbol,
                'tick': tick,
                'price': round(price, 2),
                'timestamp': tick
            }
            # Random walk
            price *= (1 + random.gauss(0, volatility))
            tick += 1
    
    stream = price_stream("AAPL", 150.0)
    
    print("  Price stream (primi 5 tick):")
    for _ in range(5):
        data = next(stream)
        print(f"    {data}")
    
    return fibonacci

# 🧪 TEST:
if __name__ == "__main__":
    gen_factory = esercizio_11_1()
    fib = gen_factory()
    assert next(fib) == 0
    assert next(fib) == 1
    assert next(fib) == 1
    assert next(fib) == 2
    print("✅ Esercizio 11.1 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 11.2: Generator send() and throw()
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Implementa generators avanzati con:
1. Comunicazione bidirezionale con send()
2. Gestione errori con throw()
3. Coroutine-style generator

💡 TEORIA:
send(value): invia un valore AL generator (ricevuto da yield)
throw(exception): solleva un'eccezione NEL generator
close(): termina il generator

🎯 SKILLS: send, throw, close, bidirectional generators
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Avanzato
"""

# ✅ SOLUZIONE:
def esercizio_11_2():
    """Generator send() and throw() - Generators bidirezionali"""
    
    # 1. SEND BASIC
    print("--- SEND BASIC ---")
    
    def echo():
        """Echo ciò che viene inviato."""
        while True:
            received = yield
            print(f"    Ricevuto: {received}")
    
    gen = echo()
    next(gen)  # Avvia il generator (necessario!)
    gen.send("Hello")
    gen.send("World")
    gen.close()
    
    # 2. ACCUMULATOR CON SEND
    print("\n--- ACCUMULATOR ---")
    
    def running_average():
        """Calcola media mobile."""
        total = 0.0
        count = 0
        average = None
        
        while True:
            value = yield average
            if value is not None:
                total += value
                count += 1
                average = total / count
    
    avg = running_average()
    next(avg)  # Avvia
    
    values = [10, 20, 30, 40, 50]
    for v in values:
        result = avg.send(v)
        print(f"    Aggiunto {v}, media: {result}")
    
    # 3. THROW PER ERROR HANDLING
    print("\n--- THROW ---")
    
    def careful_counter():
        """Counter che gestisce errori."""
        count = 0
        while True:
            try:
                yield count
                count += 1
            except ValueError as e:
                print(f"    Errore gestito: {e}")
                count = 0  # Reset!
    
    counter = careful_counter()
    print(f"    {next(counter)}")
    print(f"    {next(counter)}")
    print(f"    {next(counter)}")
    counter.throw(ValueError, "Reset richiesto!")
    print(f"    Dopo reset: {next(counter)}")
    
    # 4. APPLICAZIONE: Trading Signal Processor
    print("\n--- SIGNAL PROCESSOR ---")
    
    def signal_processor():
        """
        Processa segnali trading con stato.
        Riceve prezzi, emette segnali.
        """
        position = None  # 'LONG', 'SHORT', None
        entry_price = 0.0
        
        price = yield "READY"  # Primo yield per avviare
        
        while True:
            signal = "HOLD"
            
            if price is None:
                # Reset richiesto
                if position:
                    signal = f"CLOSE {position} (forced)"
                position = None
                entry_price = 0.0
            elif position is None:
                # Logica entry semplificata
                signal = "BUY"
                position = "LONG"
                entry_price = price
            elif position == "LONG":
                pnl = (price - entry_price) / entry_price * 100
                if pnl >= 5:  # Take profit
                    signal = f"SELL (TP: {pnl:.2f}%)"
                    position = None
                elif pnl <= -2:  # Stop loss
                    signal = f"SELL (SL: {pnl:.2f}%)"
                    position = None
            
            price = yield signal
    
    processor = signal_processor()
    print(f"    Init: {next(processor)}")
    
    prices = [100, 102, 103, 104, 105, 106]
    for p in prices:
        result = processor.send(p)
        print(f"    Price ${p} → {result}")
    
    # Reset
    processor.send(None)
    print("    Reset done")
    
    return running_average

# 🧪 TEST:
if __name__ == "__main__":
    esercizio_11_2()
    print("✅ Esercizio 11.2 completato!\n")


# ==============================================================================
# RIEPILOGO SEZIONE 8-11: FUNCTIONS
# ==============================================================================

print("\n" + "=" * 70)
print("RIEPILOGO: ESERCIZI FUNCTIONS COMPLETATI")
print("=" * 70)

print("""
ESERCIZI COMPLETATI IN QUESTA PARTE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SEZIONE 8 - Function Basics:
  ✅ 8.1 Function Definition
  ✅ 8.2 Arguments (*args, **kwargs)
  ✅ 8.3 Positional-Only e Keyword-Only

SEZIONE 9 - Closures e Scope:
  ✅ 9.1 Variable Scope (LEGB)
  ✅ 9.2 Closures

SEZIONE 10 - Decorators:
  ✅ 10.1 Basic Decorators
  ✅ 10.2 Decorators with Arguments
  ✅ 10.3 Class Decorators

SEZIONE 11 - Generators:
  ✅ 11.1 Generator Functions
  ✅ 11.2 Generator send() and throw()

TOTALE QUESTA PARTE: 10 esercizi
TOTALE CUMULATIVO: 34 esercizi

PROSSIMA PARTE: OOP (classi, ereditarietà, design patterns)
""")

# Esegui tutti gli esercizi
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ESECUZIONE TUTTI GLI ESERCIZI PARTE 3")
    print("=" * 70 + "\n")
    
    esercizio_8_1()
    esercizio_8_2()
    esercizio_8_3()
    esercizio_9_1()
    esercizio_9_2()
    esercizio_10_1()
    esercizio_10_2()
    esercizio_10_3()
    esercizio_11_1()
    esercizio_11_2()
    
    print("\n🎉 TUTTI GLI ESERCIZI DELLA PARTE 3 COMPLETATI!")
