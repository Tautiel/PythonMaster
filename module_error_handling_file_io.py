"""
🎯 ERROR HANDLING & FILE I/O MASTERY
Modulo essenziale per gestione errori e file operations
MANCAVA NEL MATERIALE - ESSENZIALE PER PRODUZIONE!
"""

import os
import sys
import json
import csv
import pickle
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Any, Dict, List
import traceback

print("=" * 60)
print("ERROR HANDLING & FILE I/O - MODULO COMPLETO")
print("=" * 60)

# ============================================
# PARTE 1: ERROR HANDLING FUNDAMENTALS
# ============================================

print("\n📚 PARTE 1: ERROR HANDLING")
print("-" * 40)

# 1. Try/Except Base
print("\n1. TRY/EXCEPT BASICS:")

def divide_safe(a: float, b: float) -> Optional[float]:
    """Divisione con gestione errori"""
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print(f"❌ Errore: divisione per zero!")
        return None
    except TypeError as e:
        print(f"❌ Errore tipo: {e}")
        return None
    finally:
        print(f"  Tentata divisione {a}/{b}")

# Test
print(f"10/2 = {divide_safe(10, 2)}")
print(f"10/0 = {divide_safe(10, 0)}")
print(f"10/'a' = {divide_safe(10, 'a')}")


# 2. Multiple Exceptions
print("\n\n2. MULTIPLE EXCEPTION HANDLING:")

def process_data(data):
    """Gestisce diversi tipi di errori"""
    try:
        # Potenziali errori multipli
        value = data['key']
        number = int(value)
        result = 100 / number
        items = data['items']
        first_item = items[0]
        return result, first_item
        
    except KeyError as e:
        print(f"❌ Chiave mancante: {e}")
        return None, None
    except ValueError as e:
        print(f"❌ Valore non valido: {e}")
        return None, None
    except ZeroDivisionError:
        print(f"❌ Divisione per zero")
        return None, None
    except IndexError:
        print(f"❌ Indice fuori range")
        return None, None
    except Exception as e:
        # Catch-all per errori non previsti
        print(f"❌ Errore generico: {type(e).__name__}: {e}")
        return None, None

# Test vari errori
test_cases = [
    {'key': '10', 'items': [1, 2, 3]},  # OK
    {'wrong_key': '10'},                 # KeyError
    {'key': 'abc'},                       # ValueError
    {'key': '0'},                         # ZeroDivisionError
    {'key': '10', 'items': []},          # IndexError
]

for i, case in enumerate(test_cases, 1):
    print(f"\nTest {i}: {case}")
    result, item = process_data(case)
    print(f"  Result: {result}, Item: {item}")


# 3. Raising Exceptions
print("\n\n3. RAISING CUSTOM EXCEPTIONS:")

class ValidationError(Exception):
    """Custom exception per validazione"""
    pass

class InsufficientFundsError(Exception):
    """Errore per fondi insufficienti"""
    def __init__(self, balance, amount):
        self.balance = balance
        self.amount = amount
        super().__init__(f"Fondi insufficienti: richiesti {amount}, disponibili {balance}")

def validate_email(email: str) -> str:
    """Valida email con custom exception"""
    if not email:
        raise ValidationError("Email vuota")
    if '@' not in email:
        raise ValidationError("Email deve contenere @")
    if '.' not in email.split('@')[1]:
        raise ValidationError("Dominio email non valido")
    return email

def withdraw(balance: float, amount: float) -> float:
    """Preleva con controllo fondi"""
    if amount <= 0:
        raise ValueError("Importo deve essere positivo")
    if amount > balance:
        raise InsufficientFundsError(balance, amount)
    return balance - amount

# Test custom exceptions
emails = ['test@example.com', 'invalid', '', 'test@domain']

for email in emails:
    try:
        validated = validate_email(email)
        print(f"✅ Email valida: {validated}")
    except ValidationError as e:
        print(f"❌ Validazione fallita: {e}")

# Test withdrawal
try:
    balance = 1000
    new_balance = withdraw(balance, 500)
    print(f"\n✅ Prelievo OK: {balance} → {new_balance}")
    
    new_balance = withdraw(balance, 1500)  # Fallirà
except InsufficientFundsError as e:
    print(f"❌ {e}")
    print(f"   Balance: {e.balance}, Richiesti: {e.amount}")


# 4. Context Managers per Error Handling
print("\n\n4. CONTEXT MANAGERS:")

@contextmanager
def managed_resource(name):
    """Context manager con error handling"""
    print(f"📂 Apertura risorsa: {name}")
    resource = {'name': name, 'data': []}
    try:
        yield resource
    except Exception as e:
        print(f"❌ Errore durante uso risorsa: {e}")
        # Cleanup in caso di errore
        resource['data'].clear()
        raise
    finally:
        print(f"📁 Chiusura risorsa: {name}")
        # Cleanup sempre eseguito

# Uso del context manager
with managed_resource("database") as db:
    db['data'].append("record1")
    print(f"  Usando: {db}")
    # raise ValueError("Simulated error")  # Uncommenta per test


# 5. Logging degli Errori
print("\n\n5. LOGGING SYSTEM:")

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        # logging.FileHandler('app.log')  # Uncommenta per log su file
    ]
)

logger = logging.getLogger(__name__)

def risky_operation(value):
    """Operazione con logging"""
    logger.info(f"Inizio operazione con valore: {value}")
    
    try:
        if value < 0:
            raise ValueError("Valore negativo non permesso")
        
        result = 100 / value
        logger.info(f"Operazione riuscita: {result}")
        return result
        
    except ValueError as e:
        logger.error(f"Errore di validazione: {e}")
        raise
    except ZeroDivisionError:
        logger.critical("Tentata divisione per zero!")
        return float('inf')
    except Exception as e:
        logger.exception("Errore inaspettato:")
        raise

# Test con logging
for val in [10, 0, -5]:
    try:
        result = risky_operation(val)
        print(f"Result for {val}: {result}")
    except Exception as e:
        print(f"Failed for {val}: {e}")


# ============================================
# PARTE 2: FILE I/O OPERATIONS
# ============================================

print("\n\n📚 PARTE 2: FILE I/O OPERATIONS")
print("-" * 40)

# 1. Text Files
print("\n1. TEXT FILE OPERATIONS:")

def write_text_file(filepath: str, content: str, encoding='utf-8'):
    """Scrive file di testo con error handling"""
    try:
        with open(filepath, 'w', encoding=encoding) as f:
            f.write(content)
        print(f"✅ Scritto: {filepath}")
    except IOError as e:
        print(f"❌ Errore scrittura: {e}")

def read_text_file(filepath: str, encoding='utf-8') -> Optional[str]:
    """Legge file di testo con error handling"""
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            content = f.read()
        return content
    except FileNotFoundError:
        print(f"❌ File non trovato: {filepath}")
        return None
    except IOError as e:
        print(f"❌ Errore lettura: {e}")
        return None

# Test text files
test_file = "test_data.txt"
test_content = """Python File I/O Test
Line 1: Hello World
Line 2: 🐍 Python Master
Line 3: Error Handling"""

write_text_file(test_file, test_content)
content = read_text_file(test_file)
if content:
    print(f"📖 Contenuto letto:\n{content[:50]}...")


# 2. JSON Files
print("\n\n2. JSON FILE OPERATIONS:")

def save_json(filepath: str, data: Any, indent: int = 2):
    """Salva dati in JSON"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        print(f"✅ JSON salvato: {filepath}")
    except (IOError, TypeError) as e:
        print(f"❌ Errore JSON save: {e}")

def load_json(filepath: str) -> Optional[Any]:
    """Carica dati da JSON"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"❌ JSON non trovato: {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ JSON non valido: {e}")
        return None

# Test JSON
test_data = {
    'user': 'Marco',
    'scores': [95, 87, 92],
    'settings': {
        'theme': 'dark',
        'language': 'it'
    }
}

json_file = "test_data.json"
save_json(json_file, test_data)
loaded_data = load_json(json_file)
if loaded_data:
    print(f"📊 JSON caricato: {loaded_data}")


# 3. CSV Files
print("\n\n3. CSV FILE OPERATIONS:")

def write_csv(filepath: str, data: List[Dict], fieldnames: List[str] = None):
    """Scrive CSV da lista di dizionari"""
    try:
        if not data:
            raise ValueError("Dati vuoti")
        
        if not fieldnames:
            fieldnames = list(data[0].keys())
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        print(f"✅ CSV scritto: {filepath}")
    except Exception as e:
        print(f"❌ Errore CSV write: {e}")

def read_csv(filepath: str) -> Optional[List[Dict]]:
    """Legge CSV in lista di dizionari"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        return data
    except FileNotFoundError:
        print(f"❌ CSV non trovato: {filepath}")
        return None
    except Exception as e:
        print(f"❌ Errore CSV read: {e}")
        return None

# Test CSV
csv_data = [
    {'symbol': 'BTC', 'price': 45000, 'volume': 1000000},
    {'symbol': 'ETH', 'price': 3000, 'volume': 500000},
    {'symbol': 'ADA', 'price': 1.5, 'volume': 100000}
]

csv_file = "crypto_data.csv"
write_csv(csv_file, csv_data)
loaded_csv = read_csv(csv_file)
if loaded_csv:
    print(f"📈 CSV caricato: {loaded_csv[0]}")


# 4. Binary Files (Pickle)
print("\n\n4. BINARY FILES (PICKLE):")

def save_pickle(filepath: str, data: Any):
    """Salva oggetto Python con pickle"""
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"✅ Pickle salvato: {filepath}")
    except Exception as e:
        print(f"❌ Errore pickle save: {e}")

def load_pickle(filepath: str) -> Optional[Any]:
    """Carica oggetto Python da pickle"""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        print(f"❌ Pickle non trovato: {filepath}")
        return None
    except Exception as e:
        print(f"❌ Errore pickle load: {e}")
        return None

# Test pickle con oggetto complesso
class TradingData:
    def __init__(self):
        self.prices = [100, 102, 98, 105]
        self.strategy = "momentum"
        self.profit = 250.50

trading = TradingData()
pickle_file = "trading_data.pkl"

save_pickle(pickle_file, trading)
loaded_trading = load_pickle(pickle_file)
if loaded_trading:
    print(f"📦 Pickle caricato: strategy={loaded_trading.strategy}, profit={loaded_trading.profit}")


# 5. Path Operations
print("\n\n5. PATH OPERATIONS WITH PATHLIB:")

def safe_file_operations():
    """Operazioni file sicure con pathlib"""
    from pathlib import Path
    
    # Crea directory se non esiste
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    print(f"📁 Directory: {data_dir.absolute()}")
    
    # File paths
    file_path = data_dir / "example.txt"
    
    # Scrivi file
    try:
        file_path.write_text("Test content")
        print(f"✅ Scritto: {file_path}")
    except IOError as e:
        print(f"❌ Errore: {e}")
    
    # Info file
    if file_path.exists():
        print(f"📊 File info:")
        print(f"  - Dimensione: {file_path.stat().st_size} bytes")
        print(f"  - È file: {file_path.is_file()}")
        print(f"  - È directory: {file_path.is_dir()}")
        print(f"  - Estensione: {file_path.suffix}")
    
    # Lista files in directory
    print(f"\n📂 Files in {data_dir}:")
    for file in data_dir.iterdir():
        if file.is_file():
            print(f"  - {file.name}")
    
    # Cleanup
    # file_path.unlink()  # Elimina file
    # data_dir.rmdir()    # Elimina directory (deve essere vuota)

safe_file_operations()


# ============================================
# PARTE 3: ADVANCED ERROR PATTERNS
# ============================================

print("\n\n📚 PARTE 3: ADVANCED PATTERNS")
print("-" * 40)

# 1. Retry Decorator
print("\n1. RETRY DECORATOR:")

import time
from functools import wraps

def retry(max_attempts=3, delay=1, exceptions=(Exception,)):
    """Decorator per retry automatico"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 1:
                        print(f"  ✅ Successo al tentativo {attempt}")
                    return result
                except exceptions as e:
                    if attempt == max_attempts:
                        print(f"  ❌ Fallito dopo {max_attempts} tentativi")
                        raise
                    print(f"  ⚠️ Tentativo {attempt} fallito: {e}")
                    time.sleep(delay)
            
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.5)
def unreliable_api_call(success_rate=0.3):
    """Simula chiamata API instabile"""
    import random
    if random.random() > success_rate:
        raise ConnectionError("API timeout")
    return "Data retrieved successfully"

# Test retry
print("Testing retry decorator:")
try:
    result = unreliable_api_call()
    print(f"Result: {result}")
except ConnectionError as e:
    print(f"Final failure: {e}")


# 2. Circuit Breaker Pattern
print("\n\n2. CIRCUIT BREAKER PATTERN:")

class CircuitBreaker:
    """Circuit breaker per prevenire cascading failures"""
    def __init__(self, failure_threshold=3, recovery_timeout=5):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        """Esegue funzione con circuit breaker"""
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'HALF_OPEN'
                print("  🔄 Circuit breaker: HALF_OPEN (testing)")
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            if self.state == 'HALF_OPEN':
                self.state = 'CLOSED'
                self.failure_count = 0
                print("  ✅ Circuit breaker: CLOSED (recovered)")
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'
                print(f"  🔴 Circuit breaker: OPEN (failures: {self.failure_count})")
            
            raise e

# Test circuit breaker
breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=2)

def flaky_service(fail=True):
    """Servizio che può fallire"""
    if fail:
        raise Exception("Service error")
    return "Service OK"

print("Testing circuit breaker:")
for i in range(5):
    try:
        result = breaker.call(flaky_service, fail=(i < 3))
        print(f"  Call {i+1}: {result}")
    except Exception as e:
        print(f"  Call {i+1}: {e}")
    
    if i == 2:
        print("  Waiting for recovery...")
        time.sleep(2.5)


# 3. Error Aggregator
print("\n\n3. ERROR AGGREGATOR:")

class ErrorCollector:
    """Colleziona errori per report finale"""
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def add_error(self, error, context=None):
        """Aggiunge errore alla collezione"""
        self.errors.append({
            'error': error,
            'type': type(error).__name__,
            'message': str(error),
            'context': context,
            'traceback': traceback.format_exc()
        })
    
    def add_warning(self, message):
        """Aggiunge warning"""
        self.warnings.append(message)
    
    def has_errors(self):
        """Controlla se ci sono errori"""
        return len(self.errors) > 0
    
    def get_report(self):
        """Genera report degli errori"""
        report = []
        report.append(f"ERRORS: {len(self.errors)}, WARNINGS: {len(self.warnings)}")
        
        if self.errors:
            report.append("\nERRORS:")
            for i, err in enumerate(self.errors, 1):
                report.append(f"  {i}. [{err['type']}] {err['message']}")
                if err['context']:
                    report.append(f"     Context: {err['context']}")
        
        if self.warnings:
            report.append("\nWARNINGS:")
            for i, warn in enumerate(self.warnings, 1):
                report.append(f"  {i}. {warn}")
        
        return "\n".join(report)

# Test error collector
collector = ErrorCollector()

def process_batch(items):
    """Processa batch con error collection"""
    results = []
    
    for item in items:
        try:
            if item < 0:
                raise ValueError(f"Negative value: {item}")
            if item == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            
            result = 100 / item
            results.append(result)
            
        except Exception as e:
            collector.add_error(e, context=f"Processing item: {item}")
            results.append(None)
    
    return results

# Process con alcuni errori
items = [5, -2, 10, 0, 20]
results = process_batch(items)

print("Batch processing results:")
for item, result in zip(items, results):
    print(f"  {item} → {result}")

print(f"\n{collector.get_report()}")


# ============================================
# PARTE 4: FILE SYSTEM MONITORING
# ============================================

print("\n\n📚 PARTE 4: FILE SYSTEM MONITORING")
print("-" * 40)

class FileMonitor:
    """Monitora modifiche ai file"""
    def __init__(self, directory="."):
        self.directory = Path(directory)
        self.file_states = {}
    
    def scan_directory(self):
        """Scansiona directory per modifiche"""
        current_files = {}
        
        for file_path in self.directory.glob("*"):
            if file_path.is_file():
                stat = file_path.stat()
                current_files[str(file_path)] = {
                    'size': stat.st_size,
                    'modified': stat.st_mtime
                }
        
        # Confronta con stato precedente
        changes = []
        
        # File nuovi
        for file, info in current_files.items():
            if file not in self.file_states:
                changes.append(f"NEW: {file}")
        
        # File modificati
        for file, info in current_files.items():
            if file in self.file_states:
                if info['modified'] > self.file_states[file]['modified']:
                    changes.append(f"MODIFIED: {file}")
        
        # File eliminati
        for file in self.file_states:
            if file not in current_files:
                changes.append(f"DELETED: {file}")
        
        self.file_states = current_files
        return changes

# Test file monitor
monitor = FileMonitor(".")
monitor.scan_directory()  # Prima scansione

# Crea un file per test
test_file = Path("monitor_test.txt")
test_file.write_text("Test")

changes = monitor.scan_directory()
if changes:
    print("File system changes detected:")
    for change in changes:
        print(f"  - {change}")

# Cleanup
test_file.unlink()


# ============================================
# PARTE 5: PROGETTO FINALE - ERROR HANDLER
# ============================================

print("\n\n📚 PROGETTO FINALE: ROBUST FILE PROCESSOR")
print("-" * 40)

class RobustFileProcessor:
    """Sistema robusto per processare file con error handling completo"""
    
    def __init__(self, input_dir="input", output_dir="output", error_dir="errors"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.error_dir = Path(error_dir)
        
        # Crea directories
        for dir_path in [self.input_dir, self.output_dir, self.error_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.error_collector = ErrorCollector()
    
    def process_file(self, file_path: Path) -> bool:
        """Processa un singolo file con error handling"""
        self.logger.info(f"Processing: {file_path.name}")
        
        try:
            # Leggi file
            content = file_path.read_text(encoding='utf-8')
            
            # Processa contenuto (esempio: uppercase)
            processed_content = self.transform_content(content)
            
            # Salva output
            output_path = self.output_dir / f"processed_{file_path.name}"
            output_path.write_text(processed_content)
            
            # Archivia originale
            archive_path = self.output_dir / "archive" / file_path.name
            archive_path.parent.mkdir(exist_ok=True)
            file_path.rename(archive_path)
            
            self.logger.info(f"✅ Processed successfully: {file_path.name}")
            return True
            
        except UnicodeDecodeError as e:
            self.handle_encoding_error(file_path, e)
            return False
        except Exception as e:
            self.handle_general_error(file_path, e)
            return False
    
    def transform_content(self, content: str) -> str:
        """Trasforma contenuto (override per logica custom)"""
        # Esempio: uppercase e aggiungi timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[PROCESSED: {timestamp}]\n{content.upper()}"
    
    def handle_encoding_error(self, file_path: Path, error: Exception):
        """Gestisce errori di encoding"""
        self.logger.error(f"Encoding error in {file_path.name}: {error}")
        self.error_collector.add_error(error, context=str(file_path))
        
        # Sposta in error directory
        error_path = self.error_dir / f"encoding_error_{file_path.name}"
        file_path.rename(error_path)
    
    def handle_general_error(self, file_path: Path, error: Exception):
        """Gestisce errori generici"""
        self.logger.error(f"Error processing {file_path.name}: {error}")
        self.error_collector.add_error(error, context=str(file_path))
        
        # Sposta in error directory
        error_path = self.error_dir / f"error_{file_path.name}"
        file_path.rename(error_path)
    
    def process_all(self):
        """Processa tutti i file nella directory input"""
        files = list(self.input_dir.glob("*.txt"))
        
        if not files:
            self.logger.info("No files to process")
            return
        
        self.logger.info(f"Found {len(files)} files to process")
        
        success_count = 0
        for file_path in files:
            if self.process_file(file_path):
                success_count += 1
        
        # Report finale
        self.logger.info(f"\nProcessing complete:")
        self.logger.info(f"  Success: {success_count}/{len(files)}")
        
        if self.error_collector.has_errors():
            print(f"\n{self.error_collector.get_report()}")

# Test del sistema completo
def test_robust_processor():
    """Test del file processor robusto"""
    
    # Crea file di test
    input_dir = Path("input")
    input_dir.mkdir(exist_ok=True)
    
    # File valido
    (input_dir / "valid.txt").write_text("Hello World")
    
    # File con caratteri speciali
    (input_dir / "special.txt").write_text("Python 🐍 Master")
    
    # Processa
    processor = RobustFileProcessor()
    processor.process_all()
    
    print("\n✅ File processor test completato!")

# Esegui test
test_robust_processor()


# ============================================
# RIEPILOGO E BEST PRACTICES
# ============================================

print("\n\n" + "=" * 60)
print("RIEPILOGO ERROR HANDLING & FILE I/O")
print("=" * 60)

best_practices = """
✅ ERROR HANDLING BEST PRACTICES:

1. SPECIFICITY: Cattura eccezioni specifiche, non generiche
   - ❌ except: pass
   - ✅ except ValueError as e: handle_error(e)

2. FINALLY: Usa finally per cleanup garantito
   - Files sempre chiusi
   - Risorse sempre rilasciate

3. LOGGING: Logga sempre gli errori
   - Non silenziare errori importanti
   - Usa livelli appropriati (DEBUG, INFO, WARNING, ERROR, CRITICAL)

4. CUSTOM EXCEPTIONS: Crea eccezioni custom per domini specifici
   - Più espressive
   - Più facili da gestire

5. FAIL FAST: Fallisci presto e chiaramente
   - Valida input all'inizio
   - Non nascondere problemi

✅ FILE I/O BEST PRACTICES:

1. CONTEXT MANAGERS: Sempre usa 'with' statement
   - ✅ with open(file) as f:
   - ❌ f = open(file)

2. ENCODING: Specifica sempre encoding
   - encoding='utf-8' per testo

3. PATH OPERATIONS: Usa pathlib invece di os.path
   - Più moderno e sicuro
   - Cross-platform

4. ERROR HANDLING: Gestisci FileNotFoundError e IOError
   - Controlla esistenza file
   - Gestisci permessi

5. ATOMIC OPERATIONS: Scrivi su file temporaneo poi rinomina
   - Previene corruzioni
   - Garantisce atomicità

✅ PATTERNS AVANZATI:

1. RETRY: Per operazioni instabili
2. CIRCUIT BREAKER: Per prevenire cascading failures
3. ERROR COLLECTOR: Per batch processing
4. MONITORING: Per tracking modifiche
5. ROBUST PROCESSING: Combina tutto
"""

print(best_practices)

# Cleanup file di test
for file in ["test_data.txt", "test_data.json", "crypto_data.csv", 
             "trading_data.pkl", "monitor_test.txt"]:
    try:
        Path(file).unlink()
    except:
        pass

print("\n🎉 MODULO COMPLETATO!")
print("Ora sei pronto per gestire errori e file operations in produzione!")
