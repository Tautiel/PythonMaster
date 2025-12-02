"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    🐍 PYTHON MASTER - SCHEDA ESERCIZI COMPLETA               ║
║                                                                              ║
║                    PARTE 6: ERROR HANDLING & FILE I/O                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ==============================================================================
# SEZIONE 18: EXCEPTION HANDLING
# ==============================================================================

print("=" * 70)
print("SEZIONE 18: EXCEPTION HANDLING")
print("=" * 70)

# ------------------------------------------------------------------------------
# ESERCIZIO 18.1: Try-Except Basics
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Implementa gestione errori:
1. Try-except base
2. Cattura eccezioni multiple
3. else e finally
4. Accesso all'oggetto eccezione

💡 TEORIA:
try: codice che potrebbe fallire
except: gestisce l'errore
else: eseguito se nessun errore
finally: eseguito sempre (cleanup)

🎯 SKILLS: try/except/else/finally, exception types
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Intermedio
"""

# ✅ SOLUZIONE:
def esercizio_18_1():
    """Try-Except Basics"""
    
    # 1. BASE
    print("--- TRY-EXCEPT BASE ---")
    
    def safe_divide(a, b):
        try:
            result = a / b
        except ZeroDivisionError:
            print(f"  ❌ Divisione per zero!")
            return None
        return result
    
    print(f"  10 / 2 = {safe_divide(10, 2)}")
    print(f"  10 / 0 = {safe_divide(10, 0)}")
    
    # 2. ECCEZIONI MULTIPLE
    print("\n--- ECCEZIONI MULTIPLE ---")
    
    def parse_price(value):
        try:
            # Potrebbe essere string, None, o invalido
            price = float(value)
            if price < 0:
                raise ValueError("Prezzo negativo")
            return price
        except TypeError:
            print(f"  ❌ TypeError: {value} non è convertibile")
            return None
        except ValueError as e:
            print(f"  ❌ ValueError: {e}")
            return None
    
    test_values = ["100.50", "invalid", None, "-50", 200]
    for val in test_values:
        result = parse_price(val)
        print(f"  parse_price({val!r}) = {result}")
    
    # Cattura multipla in una riga
    def compact_parse(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
    
    # 3. ELSE e FINALLY
    print("\n--- ELSE e FINALLY ---")
    
    def process_order(order_id):
        print(f"  Processing order {order_id}...")
        try:
            if order_id < 0:
                raise ValueError("Invalid order ID")
            if order_id == 0:
                raise ConnectionError("Database unavailable")
            result = f"Order {order_id} processed"
        except ValueError as e:
            print(f"    ❌ Validation error: {e}")
            return None
        except ConnectionError as e:
            print(f"    ❌ Connection error: {e}")
            return None
        else:
            # Eseguito SOLO se nessuna eccezione
            print(f"    ✅ Success!")
            return result
        finally:
            # Eseguito SEMPRE
            print(f"    🔚 Cleanup for order {order_id}")
    
    for oid in [123, -1, 0]:
        result = process_order(oid)
        print(f"  Result: {result}\n")
    
    # 4. ACCESSO ECCEZIONE
    print("--- ACCESSO ECCEZIONE ---")
    
    def detailed_error_info():
        try:
            data = {'price': 100}
            value = data['volume']  # KeyError
        except KeyError as e:
            print(f"  Exception type: {type(e).__name__}")
            print(f"  Exception args: {e.args}")
            print(f"  Missing key: {e}")
    
    detailed_error_info()
    
    return safe_divide

# 🧪 TEST:
if __name__ == "__main__":
    func = esercizio_18_1()
    assert func(10, 2) == 5
    assert func(10, 0) is None
    print("✅ Esercizio 18.1 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 18.2: Custom Exceptions
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Crea eccezioni custom per trading:
1. Gerarchia di eccezioni
2. Eccezioni con dati aggiuntivi
3. Uso appropriato

💡 TEORIA:
Le eccezioni custom permettono gestione errori specifica per dominio.
Ereditano da Exception o sottoclassi.

🎯 SKILLS: Custom exceptions, exception hierarchy
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Intermedio-Avanzato
"""

# ✅ SOLUZIONE:
def esercizio_18_2():
    """Custom Exceptions"""
    
    # 1. GERARCHIA ECCEZIONI TRADING
    print("--- GERARCHIA ECCEZIONI ---")
    
    class TradingError(Exception):
        """Base exception per errori trading."""
        pass
    
    class OrderError(TradingError):
        """Errori relativi agli ordini."""
        pass
    
    class InsufficientFundsError(OrderError):
        """Fondi insufficienti."""
        def __init__(self, required, available):
            self.required = required
            self.available = available
            super().__init__(
                f"Richiesti ${required:.2f}, disponibili ${available:.2f}"
            )
    
    class InvalidSymbolError(OrderError):
        """Simbolo non valido."""
        def __init__(self, symbol):
            self.symbol = symbol
            super().__init__(f"Simbolo non valido: {symbol}")
    
    class PositionError(TradingError):
        """Errori relativi alle posizioni."""
        pass
    
    class PositionNotFoundError(PositionError):
        """Posizione non trovata."""
        pass
    
    class InsufficientSharesError(PositionError):
        """Azioni insufficienti per vendita."""
        def __init__(self, symbol, requested, available):
            self.symbol = symbol
            self.requested = requested
            self.available = available
            super().__init__(
                f"Cannot sell {requested} {symbol}, only {available} available"
            )
    
    # 2. USO DELLE ECCEZIONI
    print("--- USO ECCEZIONI CUSTOM ---")
    
    class TradingAccount:
        def __init__(self, balance):
            self.balance = balance
            self.positions = {}
        
        def buy(self, symbol, quantity, price):
            if not symbol or len(symbol) > 10:
                raise InvalidSymbolError(symbol)
            
            cost = quantity * price
            if cost > self.balance:
                raise InsufficientFundsError(cost, self.balance)
            
            self.balance -= cost
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity
            return f"Bought {quantity} {symbol} @ ${price}"
        
        def sell(self, symbol, quantity, price):
            if symbol not in self.positions:
                raise PositionNotFoundError(f"No position in {symbol}")
            
            if self.positions[symbol] < quantity:
                raise InsufficientSharesError(
                    symbol, quantity, self.positions[symbol]
                )
            
            self.positions[symbol] -= quantity
            self.balance += quantity * price
            return f"Sold {quantity} {symbol} @ ${price}"
    
    # Test
    account = TradingAccount(1000)
    
    operations = [
        ('buy', 'AAPL', 5, 150),      # OK
        ('buy', 'GOOGL', 10, 150),    # InsufficientFunds
        ('sell', 'MSFT', 10, 380),    # PositionNotFound
        ('sell', 'AAPL', 10, 155),    # InsufficientShares
        ('buy', '', 10, 100),         # InvalidSymbol
    ]
    
    for op, symbol, qty, price in operations:
        try:
            if op == 'buy':
                result = account.buy(symbol, qty, price)
            else:
                result = account.sell(symbol, qty, price)
            print(f"  ✅ {result}")
        except InsufficientFundsError as e:
            print(f"  💰 {e}")
        except InvalidSymbolError as e:
            print(f"  ❓ {e}")
        except PositionNotFoundError as e:
            print(f"  📭 {e}")
        except InsufficientSharesError as e:
            print(f"  📉 {e}")
        except TradingError as e:
            # Catch-all per errori trading
            print(f"  ⚠️ Trading error: {e}")
    
    # 3. EXCEPTION CHAINING
    print("\n--- EXCEPTION CHAINING ---")
    
    def fetch_market_data(symbol):
        try:
            # Simula errore API
            raise ConnectionError("API timeout")
        except ConnectionError as e:
            raise TradingError(f"Cannot fetch data for {symbol}") from e
    
    try:
        fetch_market_data("AAPL")
    except TradingError as e:
        print(f"  Error: {e}")
        print(f"  Caused by: {e.__cause__}")
    
    return TradingError

# 🧪 TEST:
if __name__ == "__main__":
    esercizio_18_2()
    print("✅ Esercizio 18.2 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 18.3: Context Managers
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Implementa context managers:
1. Usa with per risorse
2. Crea context manager con classe
3. Crea context manager con @contextmanager
4. Context manager per trading

💡 TEORIA:
Context managers gestiscono setup/cleanup automaticamente.
__enter__ per setup, __exit__ per cleanup.
@contextmanager semplifica la creazione.

🎯 SKILLS: with, __enter__, __exit__, @contextmanager
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Intermedio-Avanzato
"""

# ✅ SOLUZIONE:
def esercizio_18_3():
    """Context Managers"""
    
    from contextlib import contextmanager
    import time
    
    # 1. WITH PER FILE (built-in)
    print("--- WITH PER FILE ---")
    
    # Il file viene chiuso automaticamente
    # with open('file.txt', 'w') as f:
    #     f.write('Hello')
    # # f è chiuso qui
    
    print("  with open() gestisce apertura/chiusura automaticamente")
    
    # 2. CONTEXT MANAGER CON CLASSE
    print("\n--- CONTEXT MANAGER CLASSE ---")
    
    class Timer:
        """Misura tempo di esecuzione."""
        
        def __init__(self, name="Timer"):
            self.name = name
            self.elapsed = None
        
        def __enter__(self):
            self.start = time.perf_counter()
            print(f"  ⏱️ {self.name} started")
            return self  # Ritorna oggetto usabile in 'as'
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.elapsed = time.perf_counter() - self.start
            print(f"  ⏱️ {self.name} finished: {self.elapsed:.4f}s")
            
            # Gestione eccezione
            if exc_type is not None:
                print(f"  ❌ Exception occurred: {exc_val}")
            
            # Return False per propagare eccezione, True per sopprimerla
            return False
    
    with Timer("Calculation"):
        total = sum(range(1000000))
    
    # Con eccezione
    print("\n  Con eccezione:")
    try:
        with Timer("Failing"):
            raise ValueError("Test error")
    except ValueError:
        print("  Exception propagata correttamente")
    
    # 3. CON @CONTEXTMANAGER
    print("\n--- @CONTEXTMANAGER ---")
    
    @contextmanager
    def trading_session(session_name):
        """Context manager per sessione di trading."""
        print(f"  📈 Opening session: {session_name}")
        session = {'name': session_name, 'trades': []}
        
        try:
            yield session  # Punto dove viene eseguito il blocco with
        except Exception as e:
            print(f"  ❌ Session error: {e}")
            session['status'] = 'error'
            raise
        else:
            session['status'] = 'success'
        finally:
            print(f"  📉 Closing session: {session_name}")
            print(f"     Trades executed: {len(session['trades'])}")
    
    with trading_session("Morning") as session:
        session['trades'].append({'symbol': 'AAPL', 'side': 'BUY'})
        session['trades'].append({'symbol': 'GOOGL', 'side': 'SELL'})
    
    # 4. CONTEXT MANAGER PER RISORSE
    print("\n--- RESOURCE MANAGEMENT ---")
    
    @contextmanager
    def database_connection(host):
        """Simula connessione database."""
        print(f"  🔌 Connecting to {host}...")
        connection = {'host': host, 'connected': True}
        
        try:
            yield connection
        finally:
            print(f"  🔌 Disconnecting from {host}")
            connection['connected'] = False
    
    with database_connection("localhost:5432") as db:
        print(f"    Connected: {db['connected']}")
        print(f"    Executing queries...")
    
    print(f"  After with: connected={db['connected']}")
    
    # 5. NESTED CONTEXT MANAGERS
    print("\n--- NESTED ---")
    
    @contextmanager
    def locked_position(symbol):
        print(f"    🔒 Locking {symbol}")
        yield
        print(f"    🔓 Unlocking {symbol}")
    
    with trading_session("Afternoon") as session:
        with locked_position("AAPL"):
            session['trades'].append({'symbol': 'AAPL', 'side': 'BUY'})
    
    # 6. SUPPRESS EXCEPTIONS
    print("\n--- SUPPRESS ---")
    
    from contextlib import suppress
    
    # Ignora eccezioni specifiche
    with suppress(FileNotFoundError):
        # Non solleva errore se file non esiste
        # open('nonexistent.txt').read()
        pass
    
    print("  suppress() ignora eccezioni specifiche")
    
    return Timer

# 🧪 TEST:
if __name__ == "__main__":
    Timer = esercizio_18_3()
    with Timer("Test") as t:
        pass
    assert t.elapsed is not None
    print("✅ Esercizio 18.3 completato!\n")


# ==============================================================================
# SEZIONE 19: FILE I/O
# ==============================================================================

print("\n" + "=" * 70)
print("SEZIONE 19: FILE I/O")
print("=" * 70)

# ------------------------------------------------------------------------------
# ESERCIZIO 19.1: Text Files
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Lavora con file di testo:
1. Lettura/scrittura base
2. Modalità append
3. Iterazione su righe
4. Encoding

💡 TEORIA:
Usa sempre with per gestire file.
Modalità: r (read), w (write), a (append), r+ (read/write)
Specifica encoding per caratteri speciali.

🎯 SKILLS: open, read, write, encoding
⏱️ TEMPO: 10 minuti
🔢 LIVELLO: Principiante-Intermedio
"""

# ✅ SOLUZIONE:
def esercizio_19_1():
    """Text Files"""
    
    import os
    
    # 1. SCRITTURA
    print("--- SCRITTURA ---")
    
    # Crea file di test
    test_file = "/tmp/trades.txt"
    
    trades_data = [
        "AAPL,BUY,100,150.00",
        "GOOGL,SELL,50,140.00",
        "MSFT,BUY,75,380.00",
    ]
    
    with open(test_file, 'w', encoding='utf-8') as f:
        for trade in trades_data:
            f.write(trade + '\n')
    
    print(f"  Scritto {test_file}")
    
    # 2. LETTURA
    print("\n--- LETTURA ---")
    
    # Leggi tutto
    with open(test_file, 'r', encoding='utf-8') as f:
        content = f.read()
    print(f"  Contenuto completo:\n{content}")
    
    # Leggi righe come lista
    with open(test_file, 'r') as f:
        lines = f.readlines()
    print(f"  Righe: {lines}")
    
    # Itera su righe (memory efficient)
    print("\n  Iterazione su righe:")
    with open(test_file, 'r') as f:
        for i, line in enumerate(f):
            print(f"    {i}: {line.strip()}")
    
    # 3. APPEND
    print("\n--- APPEND ---")
    
    with open(test_file, 'a') as f:
        f.write("TSLA,BUY,25,250.00\n")
    
    with open(test_file, 'r') as f:
        print(f"  Dopo append: {len(f.readlines())} righe")
    
    # 4. LETTURA CON PARSING
    print("\n--- PARSING ---")
    
    def parse_trades(filename):
        trades = []
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 4:
                    trades.append({
                        'symbol': parts[0],
                        'side': parts[1],
                        'quantity': int(parts[2]),
                        'price': float(parts[3])
                    })
        return trades
    
    parsed = parse_trades(test_file)
    for trade in parsed:
        print(f"    {trade}")
    
    # Cleanup
    os.remove(test_file)
    
    return parse_trades

# 🧪 TEST:
if __name__ == "__main__":
    esercizio_19_1()
    print("✅ Esercizio 19.1 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 19.2: CSV Files
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Lavora con file CSV:
1. csv.reader e csv.writer
2. csv.DictReader e csv.DictWriter
3. Gestione header e quoting

💡 TEORIA:
Il modulo csv gestisce automaticamente escape, quote, delimiters.
DictReader/DictWriter usano header come chiavi.

🎯 SKILLS: csv module, DictReader, DictWriter
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Intermedio
"""

# ✅ SOLUZIONE:
def esercizio_19_2():
    """CSV Files"""
    
    import csv
    import os
    
    test_file = "/tmp/portfolio.csv"
    
    # 1. CSV WRITER
    print("--- CSV WRITER ---")
    
    with open(test_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['Symbol', 'Quantity', 'Price', 'Value'])
        
        # Rows
        writer.writerows([
            ['AAPL', 100, 150.00, 15000.00],
            ['GOOGL', 50, 140.00, 7000.00],
            ['MSFT', 75, 380.00, 28500.00],
        ])
    
    print(f"  Scritto {test_file}")
    
    # 2. CSV READER
    print("\n--- CSV READER ---")
    
    with open(test_file, 'r', newline='') as f:
        reader = csv.reader(f)
        
        header = next(reader)  # Prima riga = header
        print(f"  Header: {header}")
        
        for row in reader:
            print(f"  Row: {row}")
    
    # 3. DICT WRITER/READER
    print("\n--- DICT WRITER ---")
    
    trades_file = "/tmp/trades.csv"
    fieldnames = ['date', 'symbol', 'side', 'quantity', 'price']
    
    trades = [
        {'date': '2024-01-15', 'symbol': 'AAPL', 'side': 'BUY', 'quantity': 100, 'price': 150.00},
        {'date': '2024-01-15', 'symbol': 'GOOGL', 'side': 'SELL', 'quantity': 50, 'price': 140.00},
        {'date': '2024-01-16', 'symbol': 'MSFT', 'side': 'BUY', 'quantity': 75, 'price': 380.00},
    ]
    
    with open(trades_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trades)
    
    print(f"  Scritto {trades_file}")
    
    print("\n--- DICT READER ---")
    
    with open(trades_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            value = int(row['quantity']) * float(row['price'])
            print(f"  {row['date']}: {row['side']} {row['quantity']} {row['symbol']} "
                  f"@ ${float(row['price']):.2f} = ${value:.2f}")
    
    # 4. QUOTING
    print("\n--- QUOTING ---")
    
    # Per valori con virgole o caratteri speciali
    data_with_commas = [
        ['Name', 'Description'],
        ['Apple Inc.', 'Technology company, Cupertino'],
        ['Google', 'Search engine, ads platform'],
    ]
    
    quoted_file = "/tmp/quoted.csv"
    with open(quoted_file, 'w', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerows(data_with_commas)
    
    with open(quoted_file, 'r') as f:
        print(f"  Quoted CSV:\n{f.read()}")
    
    # Cleanup
    for f in [test_file, trades_file, quoted_file]:
        os.remove(f)
    
    return csv

# 🧪 TEST:
if __name__ == "__main__":
    esercizio_19_2()
    print("✅ Esercizio 19.2 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 19.3: JSON Files
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Lavora con JSON:
1. json.dump/load per file
2. json.dumps/loads per stringhe
3. Custom encoders per oggetti complessi

💡 TEORIA:
JSON è standard per API e config.
dump/load per file, dumps/loads per stringhe.
Custom encoder per datetime, Decimal, dataclass.

🎯 SKILLS: json module, custom encoders
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Intermedio
"""

# ✅ SOLUZIONE:
def esercizio_19_3():
    """JSON Files"""
    
    import json
    import os
    from datetime import datetime
    from decimal import Decimal
    from dataclasses import dataclass, asdict
    
    test_file = "/tmp/config.json"
    
    # 1. SCRITTURA/LETTURA BASE
    print("--- JSON BASE ---")
    
    config = {
        'api_key': 'xxx-xxx-xxx',
        'symbols': ['AAPL', 'GOOGL', 'MSFT'],
        'settings': {
            'max_position_size': 1000,
            'stop_loss_percent': 2.0,
            'take_profit_percent': 5.0,
        }
    }
    
    # Scrivi con indentazione
    with open(test_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"  Scritto {test_file}")
    
    # Leggi
    with open(test_file, 'r') as f:
        loaded = json.load(f)
    
    print(f"  Loaded: {loaded['settings']}")
    
    # 2. STRINGHE JSON
    print("\n--- JSON STRINGS ---")
    
    data = {'symbol': 'AAPL', 'price': 150.0}
    
    json_string = json.dumps(data)
    print(f"  dumps: {json_string}")
    
    parsed = json.loads(json_string)
    print(f"  loads: {parsed}")
    
    # 3. CUSTOM ENCODER
    print("\n--- CUSTOM ENCODER ---")
    
    class TradingEncoder(json.JSONEncoder):
        """Encoder per oggetti trading."""
        
        def default(self, obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, Decimal):
                return float(obj)
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            return super().default(obj)
    
    @dataclass
    class Trade:
        symbol: str
        side: str
        quantity: int
        price: Decimal
        timestamp: datetime
    
    trade = Trade(
        symbol='AAPL',
        side='BUY',
        quantity=100,
        price=Decimal('150.50'),
        timestamp=datetime.now()
    )
    
    # Con encoder custom
    trade_json = json.dumps(asdict(trade), cls=TradingEncoder, indent=2)
    print(f"  Trade JSON:\n{trade_json}")
    
    # 4. OBJECT HOOK PER DESERIALIZZAZIONE
    print("\n--- OBJECT HOOK ---")
    
    def trading_decoder(obj):
        """Converte date ISO in datetime."""
        for key, value in obj.items():
            if isinstance(value, str):
                try:
                    obj[key] = datetime.fromisoformat(value)
                except ValueError:
                    pass
        return obj
    
    loaded_trade = json.loads(trade_json, object_hook=trading_decoder)
    print(f"  Loaded trade: {loaded_trade}")
    print(f"  Timestamp type: {type(loaded_trade['timestamp'])}")
    
    # Cleanup
    os.remove(test_file)
    
    return json

# 🧪 TEST:
if __name__ == "__main__":
    esercizio_19_3()
    print("✅ Esercizio 19.3 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 19.4: Path e Directory Operations
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Usa pathlib per operazioni su path:
1. Creazione e navigazione path
2. Operazioni su directory
3. Globbing e ricerca file

💡 TEORIA:
pathlib è l'approccio moderno (Python 3.4+).
Path oggetti sono più intuitivi di os.path.

🎯 SKILLS: pathlib, Path operations, globbing
⏱️ TEMPO: 10 minuti
🔢 LIVELLO: Intermedio
"""

# ✅ SOLUZIONE:
def esercizio_19_4():
    """Path e Directory Operations"""
    
    from pathlib import Path
    import os
    
    # 1. CREAZIONE PATH
    print("--- PATH BASICS ---")
    
    # Vari modi di creare Path
    p1 = Path('/home/user/data')
    p2 = Path.home() / 'data'
    p3 = Path.cwd()
    
    print(f"  Path assoluto: {p1}")
    print(f"  Home + data: {p2}")
    print(f"  Current dir: {p3}")
    
    # 2. COMPONENTI PATH
    print("\n--- COMPONENTI ---")
    
    file_path = Path('/home/user/data/trades/2024/january.csv')
    
    print(f"  Path: {file_path}")
    print(f"  name: {file_path.name}")
    print(f"  stem: {file_path.stem}")
    print(f"  suffix: {file_path.suffix}")
    print(f"  parent: {file_path.parent}")
    print(f"  parents: {list(file_path.parents)}")
    print(f"  parts: {file_path.parts}")
    
    # 3. OPERAZIONI
    print("\n--- OPERAZIONI ---")
    
    # Creare directory temporanea per test
    test_dir = Path('/tmp/trading_test')
    test_dir.mkdir(exist_ok=True)
    
    # Sottodirectory
    (test_dir / 'data').mkdir(exist_ok=True)
    (test_dir / 'logs').mkdir(exist_ok=True)
    
    # Creare file
    (test_dir / 'config.json').write_text('{"key": "value"}')
    (test_dir / 'data' / 'trades.csv').write_text('symbol,price\nAAPL,150')
    (test_dir / 'data' / 'positions.csv').write_text('symbol,qty\nAAPL,100')
    (test_dir / 'logs' / 'app.log').write_text('Started...')
    
    print(f"  Creata struttura in {test_dir}")
    
    # Check esistenza
    print(f"\n  {test_dir} exists: {test_dir.exists()}")
    print(f"  {test_dir} is_dir: {test_dir.is_dir()}")
    print(f"  config.json is_file: {(test_dir / 'config.json').is_file()}")
    
    # 4. GLOBBING
    print("\n--- GLOBBING ---")
    
    # Trova tutti i CSV
    csv_files = list(test_dir.glob('**/*.csv'))
    print(f"  CSV files: {[f.name for f in csv_files]}")
    
    # Trova tutto nella dir corrente
    all_files = list(test_dir.glob('*'))
    print(f"  Items in root: {[f.name for f in all_files]}")
    
    # Ricorsivo
    all_recursive = list(test_dir.rglob('*'))
    print(f"  All recursive: {[str(f.relative_to(test_dir)) for f in all_recursive]}")
    
    # 5. ITERAZIONE DIRECTORY
    print("\n--- ITERDIR ---")
    
    for item in test_dir.iterdir():
        item_type = 'DIR' if item.is_dir() else 'FILE'
        print(f"    [{item_type}] {item.name}")
    
    # 6. LETTURA/SCRITTURA
    print("\n--- READ/WRITE ---")
    
    config_path = test_dir / 'config.json'
    content = config_path.read_text()
    print(f"  Read: {content}")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    print(f"\n  Cleanup: rimossa {test_dir}")
    
    return Path

# 🧪 TEST:
if __name__ == "__main__":
    esercizio_19_4()
    print("✅ Esercizio 19.4 completato!\n")


# ==============================================================================
# RIEPILOGO SEZIONE 18-19: ERROR HANDLING & FILE I/O
# ==============================================================================

print("\n" + "=" * 70)
print("RIEPILOGO: ESERCIZI ERROR HANDLING & FILE I/O")
print("=" * 70)

print("""
ESERCIZI COMPLETATI IN QUESTA PARTE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SEZIONE 18 - Exception Handling:
  ✅ 18.1 Try-Except Basics
  ✅ 18.2 Custom Exceptions
  ✅ 18.3 Context Managers

SEZIONE 19 - File I/O:
  ✅ 19.1 Text Files
  ✅ 19.2 CSV Files
  ✅ 19.3 JSON Files
  ✅ 19.4 Path e Directory Operations

TOTALE QUESTA PARTE: 7 esercizi
TOTALE CUMULATIVO: 60 esercizi
""")

if __name__ == "__main__":
    print("\n🎉 TUTTI GLI ESERCIZI DELLA PARTE 6 COMPLETATI!")
