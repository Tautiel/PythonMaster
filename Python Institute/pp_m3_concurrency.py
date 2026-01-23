"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║            PROFESSIONAL PYTHON PROGRAMMER - MODULE 3                         ║
║                     Concurrency & Parallelism                                ║
║                                                                              ║
║                     Allineato al Syllabus PCPP2                              ║
║                         OpenEDG Python Institute                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

PCPP2 Exam Section: Concurrency (~25%)

STRUTTURA MODULO:
├── Section 3.1: Threading (concurrent I/O)
├── Section 3.2: Multiprocessing (CPU parallelism)
├── Section 3.3: AsyncIO (async/await)
├── Section 3.4: Synchronization & Thread Safety
└── Module 3 Test (20 domande)

TEMPO STIMATO: 5-6 ore

═══════════════════════════════════════════════════════════════════════════════
"""

import threading
import multiprocessing
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from queue import Queue
from typing import List

# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 3.1: THREADING
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    3.1 TEORIA: THREADING                                     │
└──────────────────────────────────────────────────────────────────────────────┘

THREAD: Unità di esecuzione all'interno di un processo.
- Condividono la stessa memoria
- Leggeri (creazione veloce)
- Ideali per I/O-bound tasks (network, file)

GIL (Global Interpreter Lock):
- CPython ha un lock globale
- Solo un thread esegue bytecode Python alla volta
- Limita il parallelismo CPU, MA:
- I/O operations rilasciano il GIL!
- Threading è comunque utile per I/O-bound tasks
"""

# ═══════════════════════════════════════════════════════════════════════════
# CREARE THREAD
# ═══════════════════════════════════════════════════════════════════════════

def worker(name: str, delay: float):
    """Funzione worker per il thread."""
    print(f"Thread {name} started")
    time.sleep(delay)  # Simula I/O
    print(f"Thread {name} finished")
    return f"Result from {name}"


def threading_basic_example():
    """Creare e gestire thread."""
    
    # Metodo 1: threading.Thread con target
    t1 = threading.Thread(target=worker, args=("A", 1))
    t2 = threading.Thread(target=worker, args=("B", 2))
    
    # Avviare i thread
    t1.start()
    t2.start()
    
    # Attendere che finiscano
    t1.join()
    t2.join()
    
    print("All threads completed")


# Metodo 2: Sottoclasse di Thread
class WorkerThread(threading.Thread):
    """Thread come classe."""
    
    def __init__(self, name: str, delay: float):
        super().__init__()
        self.thread_name = name
        self.delay = delay
        self.result = None
    
    def run(self):
        """Metodo eseguito nel thread."""
        print(f"Thread {self.thread_name} running")
        time.sleep(self.delay)
        self.result = f"Result from {self.thread_name}"
        print(f"Thread {self.thread_name} done")


# ═══════════════════════════════════════════════════════════════════════════
# THREAD POOL
# ═══════════════════════════════════════════════════════════════════════════

def thread_pool_example():
    """ThreadPoolExecutor per gestire pool di thread."""
    
    def fetch_url(url: str) -> str:
        """Simula fetch di una URL."""
        time.sleep(1)  # Simula latenza rete
        return f"Content from {url}"
    
    urls = [
        "http://example.com/1",
        "http://example.com/2",
        "http://example.com/3",
        "http://example.com/4",
    ]
    
    # Context manager gestisce shutdown automatico
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Metodo 1: submit() - restituisce Future
        futures = [executor.submit(fetch_url, url) for url in urls]
        
        for future in futures:
            result = future.result()  # Blocking, attende completamento
            print(result)
        
        # Metodo 2: map() - più semplice per stessa funzione
        results = list(executor.map(fetch_url, urls))
        print(results)


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 3.2: MULTIPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    3.2 TEORIA: MULTIPROCESSING                               │
└──────────────────────────────────────────────────────────────────────────────┘

PROCESS: Unità di esecuzione indipendente.
- Memoria separata (isolamento)
- Aggira il GIL (vero parallelismo)
- Overhead maggiore di threading
- Ideale per CPU-bound tasks

QUANDO USARE:
- Threading: I/O-bound (network, file, database)
- Multiprocessing: CPU-bound (calcoli, elaborazione dati)
"""

def cpu_intensive_task(n: int) -> int:
    """Task CPU-intensive."""
    total = 0
    for i in range(n):
        total += i * i
    return total


def multiprocessing_example():
    """Usare multiprocessing per parallelismo CPU."""
    
    # Creare processo
    p = multiprocessing.Process(target=cpu_intensive_task, args=(1000000,))
    p.start()
    p.join()
    
    # Pool di processi
    with multiprocessing.Pool(processes=4) as pool:
        # map - applica funzione a lista di input
        numbers = [1000000, 2000000, 3000000, 4000000]
        results = pool.map(cpu_intensive_task, numbers)
        print(results)
        
        # apply_async - singola chiamata asincrona
        result = pool.apply_async(cpu_intensive_task, (1000000,))
        print(result.get())  # Blocking


def process_pool_executor_example():
    """ProcessPoolExecutor - API simile a ThreadPoolExecutor."""
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        numbers = [1000000, 2000000, 3000000, 4000000]
        results = list(executor.map(cpu_intensive_task, numbers))
        print(results)


# ═══════════════════════════════════════════════════════════════════════════
# COMUNICAZIONE TRA PROCESSI
# ═══════════════════════════════════════════════════════════════════════════

def producer(queue: multiprocessing.Queue):
    """Processo produttore."""
    for i in range(5):
        queue.put(f"Item {i}")
        time.sleep(0.1)
    queue.put(None)  # Segnale di fine


def consumer(queue: multiprocessing.Queue):
    """Processo consumatore."""
    while True:
        item = queue.get()
        if item is None:
            break
        print(f"Consumed: {item}")


def ipc_example():
    """Inter-Process Communication con Queue."""
    queue = multiprocessing.Queue()
    
    p1 = multiprocessing.Process(target=producer, args=(queue,))
    p2 = multiprocessing.Process(target=consumer, args=(queue,))
    
    p1.start()
    p2.start()
    
    p1.join()
    p2.join()


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 3.3: ASYNCIO
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    3.3 TEORIA: ASYNCIO                                       │
└──────────────────────────────────────────────────────────────────────────────┘

ASYNCIO: Programmazione asincrona single-threaded.
- Cooperative multitasking (non preemptive)
- Event loop gestisce le coroutine
- await sospende esecuzione senza bloccare
- Ideale per I/O-bound con molte connessioni

VANTAGGI:
- Meno overhead di threading
- Nessun problema di sincronizzazione
- Scalabile (migliaia di connessioni)
"""

# ═══════════════════════════════════════════════════════════════════════════
# COROUTINE BASE
# ═══════════════════════════════════════════════════════════════════════════

async def async_task(name: str, delay: float) -> str:
    """Coroutine base."""
    print(f"Task {name} started")
    await asyncio.sleep(delay)  # await rilascia controllo all'event loop
    print(f"Task {name} completed")
    return f"Result from {name}"


async def main_sequential():
    """Esecuzione sequenziale (per confronto)."""
    result1 = await async_task("A", 1)
    result2 = await async_task("B", 1)
    # Totale: ~2 secondi
    return [result1, result2]


async def main_concurrent():
    """Esecuzione concorrente."""
    # gather() esegue coroutine concorrentemente
    results = await asyncio.gather(
        async_task("A", 1),
        async_task("B", 1),
        async_task("C", 1)
    )
    # Totale: ~1 secondo (parallelo)
    return results


async def main_with_tasks():
    """Creare Task esplicitamente."""
    # create_task() schedula la coroutine
    task1 = asyncio.create_task(async_task("A", 1))
    task2 = asyncio.create_task(async_task("B", 1))
    
    # Fare altro mentre i task eseguono...
    print("Doing other work...")
    
    # Attendere risultati
    result1 = await task1
    result2 = await task2
    
    return [result1, result2]


# Eseguire
# asyncio.run(main_concurrent())


# ═══════════════════════════════════════════════════════════════════════════
# TIMEOUT E ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════════════

async def async_with_timeout():
    """Gestire timeout."""
    try:
        # wait_for con timeout
        result = await asyncio.wait_for(
            async_task("Slow", 10),
            timeout=2.0
        )
    except asyncio.TimeoutError:
        print("Task timed out!")
        result = None
    
    return result


async def async_error_handling():
    """Gestire errori in gather."""
    
    async def failing_task():
        raise ValueError("Task failed!")
    
    # return_exceptions=True: restituisce eccezioni invece di sollevarle
    results = await asyncio.gather(
        async_task("A", 1),
        failing_task(),
        return_exceptions=True
    )
    
    for result in results:
        if isinstance(result, Exception):
            print(f"Error: {result}")
        else:
            print(f"Success: {result}")


# ═══════════════════════════════════════════════════════════════════════════
# ESEMPIO PRATICO: FETCH MULTIPLE URLs
# ═══════════════════════════════════════════════════════════════════════════

async def fetch_url_async(url: str) -> dict:
    """Simula fetch asincrono (userebbe aiohttp in produzione)."""
    await asyncio.sleep(0.5)  # Simula latenza rete
    return {"url": url, "status": 200}


async def fetch_all_urls(urls: List[str]) -> List[dict]:
    """Fetch multiple URLs concurrently."""
    tasks = [fetch_url_async(url) for url in urls]
    return await asyncio.gather(*tasks)


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 3.4: SYNCHRONIZATION
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    3.4 TEORIA: SINCRONIZZAZIONE                              │
└──────────────────────────────────────────────────────────────────────────────┘

PROBLEMI DI CONCORRENZA:
- Race conditions: risultato dipende dal timing
- Deadlock: thread bloccati in attesa reciproca
- Starvation: thread non ottiene mai accesso

PRIMITIVE DI SINCRONIZZAZIONE:
- Lock: mutua esclusione
- RLock: lock rientrante
- Semaphore: limite accesso concorrente
- Event: segnalazione tra thread
- Condition: sincronizzazione complessa
"""

# ═══════════════════════════════════════════════════════════════════════════
# LOCK
# ═══════════════════════════════════════════════════════════════════════════

class BankAccount:
    """Esempio di race condition e soluzione con Lock."""
    
    def __init__(self, balance: float):
        self.balance = balance
        self.lock = threading.Lock()
    
    def withdraw_unsafe(self, amount: float) -> bool:
        """UNSAFE: Race condition!"""
        if self.balance >= amount:
            # Thread potrebbe essere interrotto qui!
            time.sleep(0.001)  # Simula elaborazione
            self.balance -= amount
            return True
        return False
    
    def withdraw_safe(self, amount: float) -> bool:
        """SAFE: Usa lock."""
        with self.lock:  # Acquisisce lock automaticamente
            if self.balance >= amount:
                time.sleep(0.001)
                self.balance -= amount
                return True
            return False


# ═══════════════════════════════════════════════════════════════════════════
# SEMAPHORE
# ═══════════════════════════════════════════════════════════════════════════

class ConnectionPool:
    """Pool di connessioni con Semaphore."""
    
    def __init__(self, max_connections: int):
        self.semaphore = threading.Semaphore(max_connections)
        self.connections_in_use = 0
    
    def get_connection(self):
        """Ottieni connessione (blocca se pool pieno)."""
        self.semaphore.acquire()
        self.connections_in_use += 1
        print(f"Connection acquired. In use: {self.connections_in_use}")
        return f"Connection-{self.connections_in_use}"
    
    def release_connection(self):
        """Rilascia connessione."""
        self.connections_in_use -= 1
        self.semaphore.release()
        print(f"Connection released. In use: {self.connections_in_use}")


# ═══════════════════════════════════════════════════════════════════════════
# EVENT
# ═══════════════════════════════════════════════════════════════════════════

def event_example():
    """Event per segnalazione tra thread."""
    
    event = threading.Event()
    
    def waiter():
        print("Waiter: waiting for event...")
        event.wait()  # Blocca finché event non è set
        print("Waiter: event received!")
    
    def setter():
        time.sleep(2)
        print("Setter: setting event")
        event.set()  # Segnala a tutti i waiter
    
    t1 = threading.Thread(target=waiter)
    t2 = threading.Thread(target=setter)
    
    t1.start()
    t2.start()
    
    t1.join()
    t2.join()


# ═══════════════════════════════════════════════════════════════════════════
# THREAD-SAFE QUEUE
# ═══════════════════════════════════════════════════════════════════════════

def producer_consumer_queue():
    """Pattern Producer-Consumer con Queue thread-safe."""
    
    queue = Queue(maxsize=10)
    
    def producer():
        for i in range(20):
            item = f"Item-{i}"
            queue.put(item)  # Blocca se piena
            print(f"Produced: {item}")
            time.sleep(0.1)
        queue.put(None)  # Segnale di stop
    
    def consumer():
        while True:
            item = queue.get()  # Blocca se vuota
            if item is None:
                break
            print(f"Consumed: {item}")
            queue.task_done()
    
    t1 = threading.Thread(target=producer)
    t2 = threading.Thread(target=consumer)
    
    t1.start()
    t2.start()
    
    t1.join()
    t2.join()


# ══════════════════════════════════════════════════════════════════════════════
#                    MODULE 3 TEST
# ══════════════════════════════════════════════════════════════════════════════

MODULE_3_TEST = """
══════════════════════════════════════════════════════════════════════════════
                    PP MODULE 3 - TEST FINALE
                    20 domande - Target: 70%
══════════════════════════════════════════════════════════════════════════════

Q1. Threading è ideale per task:
    A) CPU-bound    B) I/O-bound    C) Memory-bound    D) Tutti

Q2. Il GIL (Global Interpreter Lock) è in:
    A) PyPy    B) Jython    C) CPython    D) Tutti

Q3. Multiprocessing aggira il GIL perché:
    A) Lo disabilita    B) Processi separati    C) È più veloce    D) Non esiste

Q4. join() su un thread:
    A) Unisce thread    B) Attende completamento    C) Ferma thread    D) Crea thread

Q5. ThreadPoolExecutor si importa da:
    A) threading    B) multiprocessing    C) concurrent.futures    D) asyncio

Q6. asyncio è basato su:
    A) Threading    B) Multiprocessing    C) Event loop    D) GIL

Q7. 'await' può essere usato:
    A) Ovunque    B) Solo in async def    C) Solo con Thread    D) Mai

Q8. asyncio.gather() esegue coroutine:
    A) Sequenzialmente    B) Concorrentemente    C) In parallelo    D) Mai

Q9. Lock serve per:
    A) Velocità    B) Mutua esclusione    C) Parallelismo    D) I/O

Q10. Semaphore limita:
     A) Memoria    B) Accessi concorrenti    C) Thread totali    D) CPU

Q11. Event serve per:
     A) Logging    B) Segnalazione    C) Lock    D) Pool

Q12. Queue è thread-safe?
     A) Sì    B) No    C) Dipende    D) Solo con lock

Q13. Race condition è:
     A) Competizione veloce    B) Bug di timing    C) Ottimizzazione    D) Pattern

Q14. Deadlock è quando:
     A) Thread veloci    B) Thread bloccati a vicenda    C) Thread terminati    D) Semaphore

Q15. asyncio.sleep() è:
     A) Blocking    B) Non-blocking    C) Sincrono    D) Multiprocess

Q16. time.sleep() in asyncio:
     A) È corretto    B) Blocca event loop    C) È asincrono    D) Errore

Q17. create_task() restituisce:
     A) Coroutine    B) Task    C) Future    D) Thread

Q18. ProcessPoolExecutor usa:
     A) Thread    B) Processi    C) Coroutine    D) Event loop

Q19. with lock: è equivalente a:
     A) lock.start()    B) acquire/release    C) lock.join()    D) Niente

Q20. asyncio.run() può essere chiamato:
     A) Sempre    B) Solo dal main    C) Da coroutine    D) Mai


══════════════════════════════════════════════════════════════════════════════
"""

MODULE_3_TEST_ANSWERS = """
══════════════════════════════════════════════════════════════════════════════
                         RISPOSTE TEST MODULE 3
══════════════════════════════════════════════════════════════════════════════

Q1:  B) I/O-bound (GIL rilasciato durante I/O)
Q2:  C) CPython (implementazione standard)
Q3:  B) Processi separati (ognuno ha il suo GIL)
Q4:  B) Attende completamento del thread
Q5:  C) concurrent.futures
Q6:  C) Event loop (single-threaded)
Q7:  B) Solo in async def
Q8:  B) Concorrentemente (non parallelo, single thread)
Q9:  B) Mutua esclusione
Q10: B) Accessi concorrenti (max N)
Q11: B) Segnalazione tra thread
Q12: A) Sì (put/get sono thread-safe)
Q13: B) Bug di timing (risultato dipende da ordine esecuzione)
Q14: B) Thread bloccati in attesa reciproca
Q15: B) Non-blocking (rilascia controllo)
Q16: B) Blocca event loop (usare asyncio.sleep!)
Q17: B) Task
Q18: B) Processi
Q19: B) acquire/release automatico
Q20: B) Solo dal main (non da coroutine già in esecuzione)

══════════════════════════════════════════════════════════════════════════════
"""


if __name__ == "__main__":
    print("=" * 70)
    print("PP MODULE 3: Concurrency & Parallelism")
    print("=" * 70)
    print("""
    Comandi:
    print(MODULE_3_TEST)         # Test finale
    print(MODULE_3_TEST_ANSWERS) # Risposte
    """)
