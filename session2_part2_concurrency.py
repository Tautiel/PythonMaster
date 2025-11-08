"""
🚀 SESSIONE 2 - PARTE 2: CONCURRENCY MASTERY
============================================
Threading, Async/Await, Multiprocessing
Durata: 90 minuti di programmazione concorrente
"""

import threading
import asyncio
import multiprocessing
import concurrent.futures
import time
import queue
import random
from typing import List, Dict, Any, Callable, Optional
from functools import wraps
import aiohttp
import requests
from dataclasses import dataclass
from datetime import datetime

print("="*80)
print("⚡ SESSIONE 2 PARTE 2: CONCURRENCY MASTERY")
print("="*80)

# ==============================================================================
# SEZIONE 1: THREADING
# ==============================================================================

def section1_threading():
    """Threading: concorrenza con threads"""
    
    print("\n" + "="*60)
    print("🧵 SEZIONE 1: THREADING")
    print("="*60)
    
    # 1.1 BASIC THREADING
    print("\n📌 1.1 BASIC THREADING")
    print("-"*40)
    
    def worker(name: str, delay: float):
        """Worker function per thread"""
        print(f"  Thread {name} starting...")
        time.sleep(delay)
        print(f"  Thread {name} finished after {delay}s")
    
    # Crea e avvia threads
    threads = []
    for i in range(3):
        t = threading.Thread(target=worker, args=(f"T{i}", 1))
        threads.append(t)
        t.start()
    
    # Aspetta che finiscano
    for t in threads:
        t.join()
    
    print("All threads completed!")
    
    # 1.2 THREAD SYNCHRONIZATION
    print("\n🔒 1.2 THREAD SYNCHRONIZATION")
    print("-"*40)
    
    class Counter:
        """Counter thread-safe con Lock"""
        def __init__(self):
            self.value = 0
            self.lock = threading.Lock()
        
        def increment(self):
            with self.lock:
                current = self.value
                time.sleep(0.001)  # Simula operazione
                self.value = current + 1
    
    counter = Counter()
    
    def increment_many(counter: Counter, n: int):
        """Incrementa counter n volte"""
        for _ in range(n):
            counter.increment()
    
    # Test con threads
    threads = []
    for i in range(5):
        t = threading.Thread(target=increment_many, args=(counter, 100))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    print(f"Counter value: {counter.value} (expected: 500)")
    
    # 1.3 THREAD POOL EXECUTOR
    print("\n🏊 1.3 THREAD POOL EXECUTOR")
    print("-"*40)
    
    def fetch_url(url: str) -> str:
        """Simula fetch URL"""
        time.sleep(random.uniform(0.5, 1))
        return f"Content from {url}"
    
    urls = [f"http://example.com/page{i}" for i in range(5)]
    
    # Con ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit tasks
        futures = {executor.submit(fetch_url, url): url for url in urls}
        
        # Get results as completed
        for future in concurrent.futures.as_completed(futures):
            url = futures[future]
            try:
                result = future.result()
                print(f"  {url}: {result}")
            except Exception as e:
                print(f"  {url} failed: {e}")
    
    # 1.4 PRODUCER-CONSUMER con QUEUE
    print("\n📦 1.4 PRODUCER-CONSUMER PATTERN")
    print("-"*40)
    
    def producer(q: queue.Queue, n_items: int):
        """Produce items"""
        for i in range(n_items):
            item = f"Item-{i}"
            q.put(item)
            print(f"  Produced: {item}")
            time.sleep(0.1)
        q.put(None)  # Sentinel
    
    def consumer(q: queue.Queue, name: str):
        """Consume items"""
        while True:
            item = q.get()
            if item is None:
                q.put(None)  # Per altri consumer
                break
            print(f"  {name} consumed: {item}")
            time.sleep(0.2)
            q.task_done()
    
    # Test producer-consumer
    q = queue.Queue()
    
    # Start producer
    prod_thread = threading.Thread(target=producer, args=(q, 5))
    prod_thread.start()
    
    # Start consumers
    consumers = []
    for i in range(2):
        t = threading.Thread(target=consumer, args=(q, f"Consumer-{i}"))
        consumers.append(t)
        t.start()
    
    # Wait
    prod_thread.join()
    for t in consumers:
        t.join()
    
    print("Producer-Consumer completed!")

# ==============================================================================
# SEZIONE 2: ASYNC/AWAIT
# ==============================================================================

def section2_async_await():
    """Async/Await: programmazione asincrona"""
    
    print("\n" + "="*60)
    print("⚡ SEZIONE 2: ASYNC/AWAIT")
    print("="*60)
    
    # 2.1 BASIC ASYNC
    print("\n🚀 2.1 BASIC ASYNC/AWAIT")
    print("-"*40)
    
    async def fetch_data(id: int) -> str:
        """Simula fetch asincrono"""
        print(f"  Fetching data {id}...")
        await asyncio.sleep(1)  # Simula I/O
        return f"Data-{id}"
    
    async def main_basic():
        """Main function asincrona"""
        # Esegui sequenziale
        print("Sequential execution:")
        start = time.time()
        
        result1 = await fetch_data(1)
        result2 = await fetch_data(2)
        
        print(f"  Results: {result1}, {result2}")
        print(f"  Time: {time.time() - start:.2f}s")
        
        # Esegui parallelo
        print("\nParallel execution:")
        start = time.time()
        
        results = await asyncio.gather(
            fetch_data(3),
            fetch_data(4)
        )
        
        print(f"  Results: {results}")
        print(f"  Time: {time.time() - start:.2f}s")
    
    # Run async function
    asyncio.run(main_basic())
    
    # 2.2 ASYNC CONTEXT MANAGERS
    print("\n🔒 2.2 ASYNC CONTEXT MANAGERS")
    print("-"*40)
    
    class AsyncConnection:
        """Async context manager per connessioni"""
        def __init__(self, name: str):
            self.name = name
            self.connected = False
        
        async def __aenter__(self):
            print(f"  Opening async connection: {self.name}")
            await asyncio.sleep(0.5)  # Simula connessione
            self.connected = True
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            print(f"  Closing async connection: {self.name}")
            await asyncio.sleep(0.5)  # Simula chiusura
            self.connected = False
        
        async def query(self, sql: str):
            if not self.connected:
                raise RuntimeError("Not connected")
            await asyncio.sleep(0.2)
            return f"Result of: {sql}"
    
    async def test_async_context():
        async with AsyncConnection("DB") as conn:
            result = await conn.query("SELECT * FROM users")
            print(f"  Query result: {result}")
    
    asyncio.run(test_async_context())
    
    # 2.3 ASYNC ITERATORS
    print("\n🔄 2.3 ASYNC ITERATORS")
    print("-"*40)
    
    class AsyncRange:
        """Async iterator simile a range()"""
        def __init__(self, start: int, stop: int):
            self.start = start
            self.stop = stop
        
        def __aiter__(self):
            self.current = self.start
            return self
        
        async def __anext__(self):
            if self.current >= self.stop:
                raise StopAsyncIteration
            
            await asyncio.sleep(0.1)  # Simula async operation
            value = self.current
            self.current += 1
            return value
    
    async def test_async_iterator():
        print("Async iteration:")
        async for i in AsyncRange(0, 5):
            print(f"  Value: {i}")
    
    asyncio.run(test_async_iterator())
    
    # 2.4 ASYNC TASKS & CANCELLATION
    print("\n⏹️ 2.4 ASYNC TASKS & CANCELLATION")
    print("-"*40)
    
    async def long_running_task(name: str):
        """Task che può essere cancellato"""
        try:
            for i in range(10):
                print(f"  {name}: Step {i}")
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            print(f"  {name} was cancelled!")
            raise
    
    async def main_with_cancellation():
        """Test cancellazione task"""
        # Crea task
        task = asyncio.create_task(long_running_task("Task1"))
        
        # Aspetta un po'
        await asyncio.sleep(1.5)
        
        # Cancella task
        task.cancel()
        
        # Aspetta che finisca
        try:
            await task
        except asyncio.CancelledError:
            print("  Task cancelled successfully")
    
    asyncio.run(main_with_cancellation())
    
    # 2.5 ASYNC QUEUE
    print("\n📦 2.5 ASYNC QUEUE")
    print("-"*40)
    
    async def async_producer(queue: asyncio.Queue):
        """Producer asincrono"""
        for i in range(5):
            await asyncio.sleep(0.3)
            await queue.put(f"Item-{i}")
            print(f"  Produced: Item-{i}")
    
    async def async_consumer(queue: asyncio.Queue, name: str):
        """Consumer asincrono"""
        while True:
            item = await queue.get()
            if item is None:
                break
            print(f"  {name} consumed: {item}")
            await asyncio.sleep(0.5)
            queue.task_done()
    
    async def test_async_queue():
        """Test async queue"""
        queue = asyncio.Queue()
        
        # Create tasks
        producer = asyncio.create_task(async_producer(queue))
        consumers = [
            asyncio.create_task(async_consumer(queue, f"Consumer-{i}"))
            for i in range(2)
        ]
        
        # Wait for producer
        await producer
        
        # Send stop signal
        for _ in consumers:
            await queue.put(None)
        
        # Wait for consumers
        await asyncio.gather(*consumers)
    
    asyncio.run(test_async_queue())

# ==============================================================================
# SEZIONE 3: MULTIPROCESSING
# ==============================================================================

def section3_multiprocessing():
    """Multiprocessing: parallelismo reale"""
    
    print("\n" + "="*60)
    print("🔧 SEZIONE 3: MULTIPROCESSING")
    print("="*60)
    
    # 3.1 BASIC MULTIPROCESSING
    print("\n🎯 3.1 BASIC MULTIPROCESSING")
    print("-"*40)
    
    def cpu_bound_task(n: int) -> int:
        """Task CPU-intensive"""
        total = 0
        for i in range(n):
            total += i ** 2
        return total
    
    # Sequential execution
    start = time.time()
    results = [cpu_bound_task(1000000) for _ in range(4)]
    sequential_time = time.time() - start
    print(f"Sequential time: {sequential_time:.2f}s")
    
    # Parallel execution with multiprocessing
    start = time.time()
    with multiprocessing.Pool() as pool:
        results = pool.map(cpu_bound_task, [1000000] * 4)
    parallel_time = time.time() - start
    print(f"Parallel time: {parallel_time:.2f}s")
    print(f"Speedup: {sequential_time / parallel_time:.2f}x")
    
    # 3.2 PROCESS POOL EXECUTOR
    print("\n🏊 3.2 PROCESS POOL EXECUTOR")
    print("-"*40)
    
    def process_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Processa dati in processo separato"""
        import os
        result = {
            "input": data,
            "pid": os.getpid(),
            "processed": data["value"] ** 2
        }
        time.sleep(0.5)
        return result
    
    # Prepare data
    data_items = [{"id": i, "value": i * 10} for i in range(5)]
    
    # Process with ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(process_data, item) for item in data_items]
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(f"  Processed in PID {result['pid']}: {result['processed']}")
    
    # 3.3 SHARED MEMORY
    print("\n💾 3.3 SHARED MEMORY")
    print("-"*40)
    
    def worker_with_shared_memory(shared_array, index, value):
        """Worker che modifica memoria condivisa"""
        shared_array[index] = value * 2
        print(f"  Worker {index}: Set array[{index}] = {value * 2}")
    
    # Create shared array
    shared_array = multiprocessing.Array('i', 5)  # 'i' = integer
    
    # Create processes
    processes = []
    for i in range(5):
        p = multiprocessing.Process(
            target=worker_with_shared_memory,
            args=(shared_array, i, i * 10)
        )
        processes.append(p)
        p.start()
    
    # Wait for completion
    for p in processes:
        p.join()
    
    print(f"Final array: {list(shared_array)}")
    
    # 3.4 QUEUE E PIPE
    print("\n📡 3.4 QUEUE E PIPE")
    print("-"*40)
    
    def sender(conn):
        """Invia dati attraverso pipe"""
        for i in range(3):
            msg = f"Message-{i}"
            conn.send(msg)
            print(f"  Sent: {msg}")
        conn.close()
    
    def receiver(conn):
        """Riceve dati attraverso pipe"""
        while True:
            try:
                msg = conn.recv()
                print(f"  Received: {msg}")
            except EOFError:
                break
    
    # Create pipe
    parent_conn, child_conn = multiprocessing.Pipe()
    
    # Create processes
    p1 = multiprocessing.Process(target=sender, args=(child_conn,))
    p2 = multiprocessing.Process(target=receiver, args=(parent_conn,))
    
    p1.start()
    p2.start()
    
    p1.join()
    p2.join()

# ==============================================================================
# SEZIONE 4: CONFRONTO E BEST PRACTICES
# ==============================================================================

def section4_comparison():
    """Confronto tra threading, async e multiprocessing"""
    
    print("\n" + "="*60)
    print("📊 SEZIONE 4: CONFRONTO E BEST PRACTICES")
    print("="*60)
    
    # 4.1 QUANDO USARE COSA
    print("\n🎯 4.1 QUANDO USARE COSA")
    print("-"*40)
    
    print("""
    THREADING:
    ✅ I/O-bound tasks (file, network)
    ✅ Quando serve condividere stato
    ❌ CPU-intensive tasks (GIL!)
    
    ASYNC/AWAIT:
    ✅ Molte connessioni I/O simultanee
    ✅ Web servers, API clients
    ✅ Quando controlli tutto il codice
    ❌ CPU-intensive tasks
    ❌ Librerie non async
    
    MULTIPROCESSING:
    ✅ CPU-intensive tasks
    ✅ Parallelismo reale
    ❌ Overhead di creazione processi
    ❌ Condivisione stato complessa
    """)
    
    # 4.2 ESEMPIO COMPARATIVO
    print("\n⚡ 4.2 ESEMPIO COMPARATIVO")
    print("-"*40)
    
    # I/O-bound task
    def io_task(name: str):
        """Simula I/O task"""
        time.sleep(1)
        return f"IO-{name}"
    
    # CPU-bound task
    def cpu_task(n: int):
        """Simula CPU task"""
        total = sum(i**2 for i in range(n))
        return total
    
    # Test I/O-bound con threading
    print("I/O-bound task comparison:")
    
    # Sequential
    start = time.time()
    results = [io_task(str(i)) for i in range(3)]
    print(f"  Sequential: {time.time() - start:.2f}s")
    
    # Threading
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(io_task, ["1", "2", "3"]))
    print(f"  Threading: {time.time() - start:.2f}s ✅")
    
    # CPU-bound comparison
    print("\nCPU-bound task comparison:")
    
    # Sequential
    start = time.time()
    results = [cpu_task(1000000) for _ in range(3)]
    seq_time = time.time() - start
    print(f"  Sequential: {seq_time:.2f}s")
    
    # Multiprocessing
    start = time.time()
    with multiprocessing.Pool(3) as pool:
        results = pool.map(cpu_task, [1000000] * 3)
    mp_time = time.time() - start
    print(f"  Multiprocessing: {mp_time:.2f}s ✅")
    
    # 4.3 ASYNC HTTP REQUESTS
    print("\n🌐 4.3 ASYNC HTTP REQUESTS (Esempio)")
    print("-"*40)
    
    async def fetch_url_async(session, url: str):
        """Fetch URL asincrono"""
        # Simulato - in realtà useresti aiohttp
        await asyncio.sleep(0.5)
        return f"Content from {url}"
    
    async def fetch_all_async(urls: List[str]):
        """Fetch multipli URL in parallelo"""
        # async with aiohttp.ClientSession() as session:
        tasks = []
        for url in urls:
            # In realtà: session.get(url)
            tasks.append(fetch_url_async(None, url))
        
        results = await asyncio.gather(*tasks)
        return results
    
    # Test async fetching
    urls = [f"http://example.com/page{i}" for i in range(5)]
    
    async def test_async_fetch():
        start = time.time()
        results = await fetch_all_async(urls)
        print(f"  Fetched {len(results)} URLs in {time.time() - start:.2f}s")
    
    asyncio.run(test_async_fetch())

# ==============================================================================
# SEZIONE 5: PATTERNS AVANZATI
# ==============================================================================

def section5_advanced_patterns():
    """Pattern avanzati di concurrency"""
    
    print("\n" + "="*60)
    print("🎨 SEZIONE 5: PATTERNS AVANZATI")
    print("="*60)
    
    # 5.1 SEMAPHORE PATTERN
    print("\n🚦 5.1 SEMAPHORE PATTERN")
    print("-"*40)
    
    async def limited_resource(semaphore: asyncio.Semaphore, id: int):
        """Accesso a risorsa limitata"""
        async with semaphore:
            print(f"  Worker {id} accessing resource")
            await asyncio.sleep(1)
            print(f"  Worker {id} releasing resource")
    
    async def test_semaphore():
        """Test semaphore con max 2 concurrent"""
        semaphore = asyncio.Semaphore(2)
        
        tasks = [
            limited_resource(semaphore, i)
            for i in range(5)
        ]
        
        await asyncio.gather(*tasks)
    
    asyncio.run(test_semaphore())
    
    # 5.2 FAN-OUT/FAN-IN PATTERN
    print("\n🌟 5.2 FAN-OUT/FAN-IN PATTERN")
    print("-"*40)
    
    async def process_chunk(chunk: List[int]) -> int:
        """Processa chunk di dati"""
        await asyncio.sleep(0.5)
        return sum(chunk)
    
    async def fan_out_fan_in(data: List[int], chunk_size: int = 3):
        """Divide lavoro in chunks, processa, e combina"""
        # Fan-out: divide in chunks
        chunks = [
            data[i:i+chunk_size]
            for i in range(0, len(data), chunk_size)
        ]
        
        print(f"  Fan-out: {len(chunks)} chunks")
        
        # Process in parallel
        results = await asyncio.gather(
            *[process_chunk(chunk) for chunk in chunks]
        )
        
        # Fan-in: combine results
        total = sum(results)
        print(f"  Fan-in: Total = {total}")
        
        return total
    
    # Test fan-out/fan-in
    data = list(range(10))
    asyncio.run(fan_out_fan_in(data))
    
    # 5.3 CIRCUIT BREAKER PATTERN
    print("\n⚡ 5.3 CIRCUIT BREAKER PATTERN")
    print("-"*40)
    
    class CircuitBreaker:
        """Circuit breaker per gestire failures"""
        def __init__(self, failure_threshold: int = 3, timeout: float = 5.0):
            self.failure_threshold = failure_threshold
            self.timeout = timeout
            self.failures = 0
            self.last_failure_time = None
            self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
        async def call(self, func: Callable, *args, **kwargs):
            """Chiama funzione con circuit breaker"""
            # Check state
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = "HALF_OPEN"
                    print("  Circuit: HALF_OPEN (trying...)")
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = await func(*args, **kwargs)
                
                # Success - reset
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failures = 0
                    print("  Circuit: CLOSED (recovered)")
                
                return result
                
            except Exception as e:
                self.failures += 1
                self.last_failure_time = time.time()
                
                if self.failures >= self.failure_threshold:
                    self.state = "OPEN"
                    print(f"  Circuit: OPEN (failures: {self.failures})")
                
                raise e
    
    # Test circuit breaker
    async def unreliable_service(fail: bool):
        """Servizio che può fallire"""
        if fail:
            raise Exception("Service failed")
        return "Success"
    
    async def test_circuit_breaker():
        breaker = CircuitBreaker(failure_threshold=2)
        
        # Successo
        try:
            result = await breaker.call(unreliable_service, False)
            print(f"  Result: {result}")
        except Exception as e:
            print(f"  Error: {e}")
        
        # Failures
        for i in range(3):
            try:
                result = await breaker.call(unreliable_service, True)
            except Exception as e:
                print(f"  Attempt {i+1} failed: {e}")
    
    asyncio.run(test_circuit_breaker())

# ==============================================================================
# MAIN - Menu per le sezioni
# ==============================================================================

def main():
    """Menu principale per Concurrency"""
    
    print("\n" + "="*60)
    print("⚡ CONCURRENCY - SCEGLI SEZIONE")
    print("="*60)
    
    sections = [
        ("Threading", section1_threading),
        ("Async/Await", section2_async_await),
        ("Multiprocessing", section3_multiprocessing),
        ("Comparison", section4_comparison),
        ("Advanced Patterns", section5_advanced_patterns)
    ]
    
    print("\n0. Esegui TUTTO")
    for i, (name, _) in enumerate(sections, 1):
        print(f"{i}. {name}")
    
    choice = input("\nScegli (0-5): ")
    
    try:
        choice = int(choice)
        if choice == 0:
            for name, func in sections:
                input(f"\n➡️ Press ENTER for: {name}")
                func()
        elif 1 <= choice <= len(sections):
            sections[choice-1][1]()
        else:
            print("Scelta non valida")
    except (ValueError, IndexError):
        print("Scelta non valida")
    
    print("\n" + "="*60)
    print("✅ PARTE 2 COMPLETATA!")
    print("Prossimo: session2_part3_projects.py")
    print("="*60)

if __name__ == "__main__":
    main()
