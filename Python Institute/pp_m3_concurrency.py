#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PYTHON PROFESSIONAL 2 - MODULE 3                          ║
║                    CONCURRENCY (threading, multiprocessing, asyncio)         ║
║                    PCPP2 Prep - Concurrency expected ~25% of exam            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import threading
import multiprocessing
import asyncio
import time
import queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# ══════════════════════════════════════════════════════════════════════════════
# 3.1 THREADING BASICS
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("3.1 THREADING BASICS")
print("=" * 70)

print("""
📋 THREADING vs MULTIPROCESSING:

THREADING:
  - Shared memory
  - GIL limits CPU-bound tasks
  - Good for I/O-bound tasks
  - Lighter than processes

MULTIPROCESSING:
  - Separate memory
  - True parallelism
  - Good for CPU-bound tasks
  - Heavier than threads
""")

# Basic thread
def worker(name, delay):
    print(f"  Thread {name} starting")
    time.sleep(delay)
    print(f"  Thread {name} done")

# Note: Not running these to avoid output issues
print("""
# Create and start thread
t = threading.Thread(target=worker, args=("A", 1))
t.start()
t.join()  # Wait for completion

# Multiple threads
threads = []
for i in range(3):
    t = threading.Thread(target=worker, args=(i, 0.5))
    threads.append(t)
    t.start()

for t in threads:
    t.join()
""")

# ══════════════════════════════════════════════════════════════════════════════
# 3.2 THREAD SYNCHRONIZATION
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("3.2 THREAD SYNCHRONIZATION")
print("=" * 70)

print("""
📋 SYNCHRONIZATION PRIMITIVES:

LOCK (Mutex):
  lock = threading.Lock()
  with lock:
      # Critical section
      shared_data += 1

RLOCK (Reentrant Lock):
  rlock = threading.RLock()
  # Can be acquired multiple times by same thread

SEMAPHORE:
  sem = threading.Semaphore(3)  # Max 3 threads
  with sem:
      # Limited access section

EVENT:
  event = threading.Event()
  event.set()      # Signal
  event.clear()    # Reset
  event.wait()     # Block until set

CONDITION:
  cond = threading.Condition()
  with cond:
      cond.wait()     # Wait for notification
      cond.notify()   # Wake one waiting thread
      cond.notify_all()  # Wake all
""")

# Lock example
lock = threading.Lock()
counter = 0

def increment_safe():
    global counter
    with lock:
        temp = counter
        temp += 1
        counter = temp

print("Lock example: Protects shared counter from race conditions")

# ══════════════════════════════════════════════════════════════════════════════
# 3.3 THREAD COMMUNICATION
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("3.3 THREAD COMMUNICATION")
print("=" * 70)

print("""
📋 QUEUE FOR THREAD-SAFE COMMUNICATION:

import queue

q = queue.Queue()    # FIFO
q = queue.LifoQueue()  # LIFO (stack)
q = queue.PriorityQueue()  # Sorted

# Producer
q.put(item)
q.put(item, block=False)  # Raises queue.Full if full

# Consumer
item = q.get()
item = q.get(timeout=5)  # Wait max 5 seconds
q.task_done()  # Signal item processed

# Wait for all items processed
q.join()
""")

# Producer-Consumer example
def producer(q):
    for i in range(5):
        q.put(i)
        print(f"  Produced: {i}")

def consumer(q):
    while True:
        item = q.get()
        if item is None:
            break
        print(f"  Consumed: {item}")
        q.task_done()

# ══════════════════════════════════════════════════════════════════════════════
# 3.4 MULTIPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("3.4 MULTIPROCESSING")
print("=" * 70)

print("""
📋 MULTIPROCESSING:

from multiprocessing import Process, Pool, Queue, Value, Array

# Basic process
def worker(x):
    return x ** 2

p = Process(target=worker, args=(5,))
p.start()
p.join()

# Process Pool
with Pool(4) as pool:
    results = pool.map(worker, range(10))
    # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# Shared memory
counter = Value('i', 0)  # 'i' = integer
with counter.get_lock():
    counter.value += 1

# Shared array
arr = Array('d', [1.0, 2.0, 3.0])  # 'd' = double
""")

# ══════════════════════════════════════════════════════════════════════════════
# 3.5 CONCURRENT.FUTURES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("3.5 CONCURRENT.FUTURES")
print("=" * 70)

print("""
📋 HIGH-LEVEL CONCURRENCY:

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=4) as executor:
    # Submit single task
    future = executor.submit(func, arg1, arg2)
    result = future.result()
    
    # Map over iterable
    results = executor.map(func, iterable)

# ProcessPoolExecutor (same API)
with ProcessPoolExecutor(max_workers=4) as executor:
    future = executor.submit(cpu_bound_func, data)

# Future methods
future.done()      # True if completed
future.cancelled() # True if cancelled
future.result()    # Get result (blocks)
future.exception() # Get exception if any
""")

# Example
def square(x):
    return x ** 2

print("ThreadPoolExecutor example:")
with ThreadPoolExecutor(max_workers=2) as executor:
    results = list(executor.map(square, range(5)))
    print(f"  Results: {results}")

# ══════════════════════════════════════════════════════════════════════════════
# 3.6 ASYNCIO BASICS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("3.6 ASYNCIO BASICS")
print("=" * 70)

print("""
📋 ASYNCIO (Cooperative Multitasking):

import asyncio

# Coroutine definition
async def fetch_data():
    await asyncio.sleep(1)  # Non-blocking sleep
    return "data"

# Running coroutines
asyncio.run(fetch_data())

# Multiple coroutines
async def main():
    # Sequential
    result1 = await fetch_data()
    result2 = await fetch_data()
    
    # Concurrent
    results = await asyncio.gather(
        fetch_data(),
        fetch_data()
    )

# Create tasks
async def main():
    task1 = asyncio.create_task(fetch_data())
    task2 = asyncio.create_task(fetch_data())
    
    result1 = await task1
    result2 = await task2
""")

# Simple async example
async def async_square(x):
    await asyncio.sleep(0.1)
    return x ** 2

async def async_main():
    results = await asyncio.gather(
        async_square(1),
        async_square(2),
        async_square(3)
    )
    return results

print("Asyncio example:")
results = asyncio.run(async_main())
print(f"  Results: {results}")

# ══════════════════════════════════════════════════════════════════════════════
# 3.7 GIL (Global Interpreter Lock)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("3.7 GIL (Global Interpreter Lock)")
print("=" * 70)

print("""
📋 GIL:

What is GIL?
  - Mutex that protects Python objects
  - Only one thread can execute Python bytecode at a time
  - CPython implementation detail

Implications:
  - Threading NOT effective for CPU-bound tasks
  - Threading IS effective for I/O-bound tasks
  - Use multiprocessing for CPU-bound parallelism

Workarounds:
  - multiprocessing (separate processes, no GIL sharing)
  - C extensions (can release GIL)
  - Alternative interpreters (Jython, IronPython)
  - asyncio for I/O-bound concurrency
""")

# ══════════════════════════════════════════════════════════════════════════════
# QUIZ
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("QUIZ")
print("=" * 70)

print("""
Q1. Threading is best for?
    → I/O-bound tasks

Q2. Multiprocessing is best for?
    → CPU-bound tasks

Q3. GIL affects?
    → Only threading, not multiprocessing

Q4. Lock prevents?
    → Race conditions

Q5. Queue.put() and Queue.get() are?
    → Thread-safe

Q6. asyncio.gather() does?
    → Runs coroutines concurrently

Q7. await keyword does?
    → Pauses coroutine until awaitable completes

Q8. ThreadPoolExecutor.map() returns?
    → Iterator of results
""")

print("\n" + "=" * 70)
print("CONCURRENCY MODULE COMPLETE!")
print("=" * 70)
