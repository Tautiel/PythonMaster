#!/usr/bin/env python3
"""
🔍 DEBUGGING METHODOLOGY MODULE
Systematic Debugging for Professional Developers

Duration: Integrated throughout course
Level: From Print() to Production Debugging
"""

import sys
import traceback
import logging
import cProfile
import pstats
import io
import time
import memory_profiler
import dis
from datetime import datetime
from typing import Any, List, Dict, Optional, Callable
from functools import wraps
from contextlib import contextmanager
import inspect

# ============================================================================
# PART 1: THE DEBUGGING FRAMEWORK
# ============================================================================

class SystematicDebugging:
    """Il framework RISOLVI per debugging sistematico"""
    
    def __init__(self):
        self.framework = """
        🎯 FRAMEWORK RISOLVI:
        
        R - Reproduce: Riprodurre il bug consistentemente
        I - Isolate: Isolare il componente problematico  
        S - Simplify: Semplificare al caso minimo
        O - Observe: Osservare stato e comportamento
        L - Logic: Logica - forma ipotesi
        V - Verify: Verificare la fix
        I - Integrate: Integrare e prevenire regressioni
        """
        
        self.debug_log = []
        
    def demonstrate_framework(self):
        """Esempio pratico del framework"""
        
        print(self.framework)
        print("\n" + "="*60)
        
        # Scenario: Trading bot che crasha randomicamente
        bug_scenario = """
        🐛 SCENARIO: Trading Bot Crash
        
        Sintomi: Bot crasha dopo 2-3 ore
        Frequenza: 30% delle volte
        Error: "KeyError: 'price'"
        """
        
        print(bug_scenario)
        print("\nAPPLICAZIONE FRAMEWORK:")
        
        steps = {
            "1. REPRODUCE": {
                "action": "Run bot con stesso dataset",
                "code": "python bot.py --replay data.csv",
                "result": "Crash dopo 2h 17min"
            },
            "2. ISOLATE": {
                "action": "Quale modulo crasha?",
                "code": "Aggiungi logging ad ogni modulo",
                "result": "PriceAnalyzer.get_price() fails"
            },
            "3. SIMPLIFY": {
                "action": "Test solo PriceAnalyzer",
                "code": "test_price_analyzer.py con minimal data",
                "result": "Fails quando symbol non in cache"
            },
            "4. OBSERVE": {
                "action": "Print state prima del crash",
                "code": "print(cache.keys()) prima di access",
                "result": "Symbol 'BTC-USD' diventa 'BTCUSD'"
            },
            "5. LOGIC": {
                "hypothesis": "API cambia formato symbol",
                "test": "Check API response format",
                "result": "Confirmed: formato inconsistente"
            },
            "6. VERIFY": {
                "fix": "Normalizza symbol format",
                "code": "symbol = symbol.replace('-', '')",
                "result": "No crash in 24h test"
            },
            "7. INTEGRATE": {
                "action": "Add regression test",
                "code": "test_symbol_formats.py",
                "prevention": "Symbol validator class"
            }
        }
        
        for step, details in steps.items():
            print(f"\n{step}:")
            for key, value in details.items():
                print(f"  {key:10} → {value}")

# ============================================================================
# PART 2: DEBUGGING TOOLS MASTERY
# ============================================================================

class DebuggingTools:
    """Padroneggiare tutti i debugging tools"""
    
    def __init__(self):
        self.tools = {
            "print": "Quick and dirty",
            "logging": "Production debugging", 
            "pdb/ipdb": "Interactive debugging",
            "debugpy": "VS Code debugging",
            "cProfile": "Performance profiling",
            "memory_profiler": "Memory debugging",
            "tracemalloc": "Memory allocation tracking",
            "py-spy": "Production profiling"
        }
    
    def advanced_print_debugging(self):
        """Print debugging fatto bene"""
        
        print("\n🖨️ ADVANCED PRINT DEBUGGING")
        print("=" * 60)
        
        # Debug print evoluto
        def debug_print(*args, **kwargs):
            """Print con context information"""
            frame = inspect.currentframe()
            caller = inspect.getframeinfo(frame.f_back)
            
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            location = f"{caller.filename}:{caller.lineno}"
            function = caller.function
            
            print(f"[{timestamp}] {location} in {function}():")
            print("  ", *args, **kwargs)
            
            # Mostra anche local variables
            local_vars = frame.f_back.f_locals
            if len(local_vars) <= 5:  # Non troppi
                print("  Locals:", {k: v for k, v in local_vars.items() 
                                   if not k.startswith('_')})
        
        # Esempio uso
        def calculate_profit(trades):
            total = 0
            for trade in trades:
                debug_print(f"Processing trade: {trade}")
                total += trade
            return total
        
        print("Advanced print con context:")
        print("""
        def debug_print(*args):
            # Mostra: timestamp, file:line, function
            # Plus: local variables!
        """)
        
        # Decorator per debug
        def debug_decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                print(f"\n→ Calling {func.__name__}")
                print(f"  Args: {args}")
                print(f"  Kwargs: {kwargs}")
                
                start = time.time()
                try:
                    result = func(*args, **kwargs)
                    elapsed = time.time() - start
                    print(f"← {func.__name__} returned: {result}")
                    print(f"  Time: {elapsed:.4f}s")
                    return result
                except Exception as e:
                    print(f"✗ {func.__name__} raised: {e}")
                    raise
            return wrapper
        
        return debug_decorator
    
    def logging_professional(self):
        """Logging per production"""
        
        print("\n📝 PROFESSIONAL LOGGING")
        print("=" * 60)
        
        # Setup logging professionale
        def setup_logger(name: str, level=logging.DEBUG):
            """Setup logger con file e console output"""
            
            logger = logging.getLogger(name)
            logger.setLevel(level)
            
            # File handler con rotation
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                f'{name}.log',
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            
            # Console handler
            console_handler = logging.StreamHandler()
            
            # Formatter dettagliato per file
            file_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - '
                '%(filename)s:%(lineno)d - %(funcName)s() - %(message)s'
            )
            
            # Formatter semplice per console
            console_format = logging.Formatter(
                '%(levelname)s - %(message)s'
            )
            
            file_handler.setFormatter(file_format)
            console_handler.setFormatter(console_format)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            
            return logger
        
        # Esempio pratico
        print("Logger Setup Example:")
        print("""
        logger = setup_logger('trading_bot')
        
        # Livelli di logging
        logger.debug('Detailed info for debugging')
        logger.info('General info')
        logger.warning('Warning: something unexpected')
        logger.error('Error occurred', exc_info=True)
        logger.critical('System is going down!')
        
        # Context manager per timing
        with log_timer(logger, 'fetch_prices'):
            prices = fetch_prices()
        """)
        
        # Log context manager
        @contextmanager
        def log_context(logger, operation):
            """Context manager per logging operations"""
            logger.info(f"Starting: {operation}")
            start = time.time()
            try:
                yield
                elapsed = time.time() - start
                logger.info(f"Completed: {operation} ({elapsed:.2f}s)")
            except Exception as e:
                elapsed = time.time() - start
                logger.error(f"Failed: {operation} ({elapsed:.2f}s): {e}")
                raise
    
    def pdb_mastery(self):
        """Master pdb/ipdb debugger"""
        
        print("\n🐞 PDB/IPDB MASTERY")
        print("=" * 60)
        
        pdb_commands = {
            "Navigation": {
                "l(ist)": "Show current code",
                "w(here)": "Show stack trace",
                "u(p)": "Go up in stack",
                "d(own)": "Go down in stack"
            },
            "Execution": {
                "n(ext)": "Next line (step over)",
                "s(tep)": "Step into function",
                "c(ontinue)": "Continue execution",
                "r(eturn)": "Continue until return",
                "j(ump) line": "Jump to line"
            },
            "Inspection": {
                "p expression": "Print expression",
                "pp expression": "Pretty print",
                "a(rgs)": "Print function arguments",
                "h(elp)": "Show help"
            },
            "Breakpoints": {
                "b(reak) line": "Set breakpoint",
                "b function": "Break at function",
                "cl(ear) n": "Clear breakpoint n",
                "disable n": "Disable breakpoint",
                "condition n expr": "Conditional break"
            },
            "Advanced": {
                "!statement": "Execute Python statement",
                "alias": "Create command alias",
                "debug code": "Debug nested code",
                "interact": "Start interactive shell"
            }
        }
        
        print("PDB Commands Reference:")
        for category, commands in pdb_commands.items():
            print(f"\n{category}:")
            for cmd, desc in commands.items():
                print(f"  {cmd:20} → {desc}")
        
        # Tricks professionali
        print("\n💡 PRO TRICKS:")
        tricks = [
            "1. Post-mortem debugging: python -m pdb script.py",
            "2. Conditional breakpoints: b 42, balance < 0",
            "3. Watch expressions: display variable",
            "4. Save breakpoints: alias save_bp ...",
            "5. Remote debugging: import rpdb; rpdb.set_trace()"
        ]
        for trick in tricks:
            print(f"  {trick}")

# ============================================================================
# PART 3: PERFORMANCE DEBUGGING
# ============================================================================

class PerformanceDebugging:
    """Trovare e fixare bottlenecks"""
    
    def profile_code(self):
        """Profiling con cProfile"""
        
        print("\n⚡ PERFORMANCE PROFILING")
        print("=" * 60)
        
        # Decorator per profiling
        def profile(sort_by='cumulative'):
            def decorator(func):
                @wraps(func)
                def wrapper(*args, **kwargs):
                    profiler = cProfile.Profile()
                    profiler.enable()
                    
                    result = func(*args, **kwargs)
                    
                    profiler.disable()
                    
                    # Analizza risultati
                    s = io.StringIO()
                    ps = pstats.Stats(profiler, stream=s)
                    ps.sort_stats(sort_by)
                    ps.print_stats(10)  # Top 10 functions
                    
                    print(f"\n📊 Profile of {func.__name__}:")
                    print(s.getvalue())
                    
                    return result
                return wrapper
            return decorator
        
        # Esempio: Ottimizzare trading algorithm
        @profile()
        def slow_trading_algorithm(prices):
            """Versione lenta O(n²)"""
            results = []
            for i in range(len(prices)):
                for j in range(i + 1, len(prices)):
                    if prices[j] > prices[i]:
                        profit = prices[j] - prices[i]
                        results.append((i, j, profit))
            return max(results, key=lambda x: x[2])
        
        @profile()
        def fast_trading_algorithm(prices):
            """Versione ottimizzata O(n)"""
            min_price = float('inf')
            max_profit = 0
            buy_day = 0
            sell_day = 0
            
            for i, price in enumerate(prices):
                if price < min_price:
                    min_price = price
                    buy_day = i
                elif price - min_price > max_profit:
                    max_profit = price - min_price
                    sell_day = i
            
            return (buy_day, sell_day, max_profit)
        
        print("Profile Comparison:")
        print("Slow: O(n²) nested loops")
        print("Fast: O(n) single pass")
        print("\nTools:")
        print("  - cProfile: Built-in profiler")
        print("  - line_profiler: Line-by-line")
        print("  - memory_profiler: Memory usage")
        print("  - py-spy: Production profiling")
    
    def memory_debugging(self):
        """Debug memory leaks e usage"""
        
        print("\n💾 MEMORY DEBUGGING")
        print("=" * 60)
        
        # Memory profiler decorator
        def memory_profile(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                import tracemalloc
                
                tracemalloc.start()
                snapshot1 = tracemalloc.take_snapshot()
                
                result = func(*args, **kwargs)
                
                snapshot2 = tracemalloc.take_snapshot()
                
                top_stats = snapshot2.compare_to(snapshot1, 'lineno')
                
                print(f"\n📊 Memory usage for {func.__name__}:")
                print("[ Top 5 differences ]")
                for stat in top_stats[:5]:
                    print(stat)
                
                current, peak = tracemalloc.get_traced_memory()
                print(f"\nCurrent memory: {current / 1024 / 1024:.2f} MB")
                print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")
                
                tracemalloc.stop()
                return result
            return wrapper
        
        # Esempio: Memory leak
        class MemoryLeakExample:
            """Esempio di memory leak comune"""
            
            cache = []  # Class variable - DANGER!
            
            def __init__(self):
                self.data = [0] * 1000000
                # Bug: aggiunge a class variable
                MemoryLeakExample.cache.append(self.data)
        
        class MemoryFixExample:
            """Versione corretta"""
            
            def __init__(self):
                self.data = [0] * 1000000
                self.cache = []  # Instance variable
                self.cache.append(self.data)
        
        print("Common Memory Issues:")
        issues = {
            "Global Cache": "Cache che cresce infinitamente",
            "Circular Refs": "A → B → A references",
            "Large Objects": "DataFrame non rilasciati",
            "Event Handlers": "Listeners non rimossi",
            "Thread Locals": "Threading data accumulation"
        }
        
        for issue, description in issues.items():
            print(f"  {issue:15} → {description}")

# ============================================================================
# PART 4: DEBUGGING PATTERNS
# ============================================================================

class DebuggingPatterns:
    """Pattern comuni e soluzioni"""
    
    def race_conditions(self):
        """Debug race conditions"""
        
        print("\n🏁 DEBUGGING RACE CONDITIONS")
        print("=" * 60)
        
        print("""
        Sintomi di Race Conditions:
        1. Bug intermittenti
        2. Funziona in debug, crasha in production
        3. Dipende dal timing
        4. Thread-related crashes
        
        Debug Strategy:
        """)
        
        import threading
        import time
        
        class RaceConditionDemo:
            def __init__(self):
                self.balance = 100
                self.lock = threading.Lock()
            
            def unsafe_withdraw(self, amount):
                """Versione con race condition"""
                if self.balance >= amount:
                    time.sleep(0.001)  # Simula delay
                    self.balance -= amount
                    return True
                return False
            
            def safe_withdraw(self, amount):
                """Versione thread-safe"""
                with self.lock:
                    if self.balance >= amount:
                        time.sleep(0.001)
                        self.balance -= amount
                        return True
                    return False
        
        print("Debug Tools:")
        print("  1. threading.Lock() per sincronizzazione")
        print("  2. logging con thread ID")
        print("  3. Thread sanitizer tools")
        print("  4. Stress testing con molti threads")
    
    def async_debugging(self):
        """Debug codice asincrono"""
        
        print("\n⚡ ASYNC DEBUGGING")
        print("=" * 60)
        
        print("""
        Async Debugging Challenges:
        1. Stack traces confusi
        2. Timing issues
        3. Callback hell
        4. Exception handling
        
        Best Practices:
        """)
        
        import asyncio
        
        # Async debug utilities
        def async_debug(func):
            """Decorator per debug async functions"""
            @wraps(func)
            async def wrapper(*args, **kwargs):
                print(f"→ Async {func.__name__} starting")
                try:
                    result = await func(*args, **kwargs)
                    print(f"← Async {func.__name__} completed")
                    return result
                except Exception as e:
                    print(f"✗ Async {func.__name__} failed: {e}")
                    raise
            return wrapper
        
        # Esempio pratico
        @async_debug
        async def fetch_prices_async(symbols):
            """Fetch prices asynchronously"""
            tasks = []
            for symbol in symbols:
                task = fetch_single_price(symbol)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            for symbol, result in zip(symbols, results):
                if isinstance(result, Exception):
                    print(f"Failed to fetch {symbol}: {result}")
            
            return results
        
        print("Async Debug Tips:")
        tips = [
            "Use asyncio.create_task() con name parameter",
            "Enable asyncio debug mode: asyncio.run(main(), debug=True)",
            "aiohttp con logging dettagliato",
            "pytest-asyncio per testing",
            "AsyncIO con VS Code debugger"
        ]
        for tip in tips:
            print(f"  • {tip}")

# ============================================================================
# PART 5: PRODUCTION DEBUGGING
# ============================================================================

class ProductionDebugging:
    """Debug in production environment"""
    
    def remote_debugging(self):
        """Debug remoto su server"""
        
        print("\n🌍 REMOTE DEBUGGING")
        print("=" * 60)
        
        print("""
        Remote Debug Options:
        
        1. SSH + tmux/screen
           ssh server
           tmux attach
           python -m pdb script.py
        
        2. Remote PDB
           import rpdb
           rpdb.set_trace()  # Telnet to port 4444
        
        3. VS Code Remote
           - Install Remote-SSH extension
           - Connect to server
           - Debug as local
        
        4. Logging aggregation
           - ELK Stack (Elasticsearch, Logstash, Kibana)
           - Datadog / New Relic
           - CloudWatch (AWS)
        """)
    
    def monitoring_and_alerting(self):
        """Monitoring per prevenire bugs"""
        
        print("\n📊 MONITORING & ALERTING")
        print("=" * 60)
        
        # Metriche da monitorare
        metrics = {
            "Performance": [
                "Response time (p50, p95, p99)",
                "Throughput (requests/sec)",
                "Error rate (%)",
                "CPU/Memory usage"
            ],
            "Business": [
                "Trades executed",
                "Order failures",
                "Balance discrepancies",
                "API rate limits"
            ],
            "System": [
                "Disk space",
                "Network latency",
                "Database connections",
                "Queue depth"
            ]
        }
        
        print("Key Metrics to Monitor:")
        for category, items in metrics.items():
            print(f"\n{category}:")
            for item in items:
                print(f"  • {item}")
        
        # Alert setup
        print("\n🚨 Alert Configuration:")
        print("""
        def setup_alerts():
            alerts = {
                "error_rate": {
                    "threshold": 1,  # 1%
                    "window": "5m",
                    "action": "page_oncall"
                },
                "response_time_p99": {
                    "threshold": 1000,  # 1s
                    "window": "1m",
                    "action": "email_team"
                }
            }
        """)

# ============================================================================
# PART 6: DEBUG PROJECTS
# ============================================================================

class DebugProjects:
    """Progetti pratici per debugging"""
    
    def project_1_debug_assistant(self):
        """Costruisci un Debug Assistant Tool"""
        
        print("\n🔨 PROJECT: Debug Assistant")
        print("=" * 60)
        
        class DebugAssistant:
            """AI-powered debugging helper"""
            
            def __init__(self):
                self.error_patterns = {}
                self.solutions = {}
                
            def analyze_error(self, error_trace):
                """Analizza error e suggerisci soluzioni"""
                # Parse traceback
                # Identify error type
                # Search knowledge base
                # Suggest fixes
                pass
            
            def collect_context(self):
                """Raccoglie context del bug"""
                context = {
                    "python_version": sys.version,
                    "installed_packages": self.get_packages(),
                    "environment_vars": dict(os.environ),
                    "recent_changes": self.get_git_diff(),
                    "system_info": self.get_system_info()
                }
                return context
            
            def suggest_fix(self, error_type, context):
                """Suggerisci fix basato su pattern"""
                suggestions = {
                    "KeyError": [
                        "Check if key exists: if key in dict",
                        "Use dict.get(key, default)",
                        "Verify data source consistency"
                    ],
                    "ImportError": [
                        "Check package installed: pip list",
                        "Verify PYTHONPATH",
                        "Check virtual environment"
                    ]
                }
                return suggestions.get(error_type, [])
        
        print("Features to implement:")
        print("✅ Error pattern recognition")
        print("✅ Context collection")
        print("✅ Fix suggestions")
        print("✅ Stack trace analysis")
        print("✅ Integration with IDE")

# ============================================================================
# EXERCISES
# ============================================================================

def debugging_exercises():
    """50 debugging exercises"""
    
    print("\n📚 DEBUGGING EXERCISES")
    print("=" * 60)
    
    exercises = {
        "Basic (1-10)": [
            "Debug syntax error in 100-line script",
            "Fix NameError in function",
            "Resolve ImportError",
            "Fix IndentationError",
            "Debug TypeError in calculation",
            "Fix KeyError in dictionary access",
            "Resolve IndexError in list",
            "Debug AttributeError",
            "Fix ValueError in conversion",
            "Debug ZeroDivisionError"
        ],
        
        "Intermediate (11-30)": [
            "Debug infinite loop",
            "Fix memory leak",
            "Resolve race condition",
            "Debug async function",
            "Fix circular import",
            "Debug recursion error",
            "Fix encoding issues",
            "Debug file permissions",
            "Resolve deadlock",
            "Fix cache invalidation",
            # ... più esercizi
        ],
        
        "Advanced (31-50)": [
            "Debug production crash from logs",
            "Profile and optimize slow function",
            "Debug memory leak in long-running process",
            "Fix race condition in threading",
            "Debug websocket disconnection",
            "Resolve database deadlock",
            "Debug Docker container issue",
            "Fix CI/CD pipeline failure",
            "Debug microservice communication",
            "Resolve distributed transaction issue",
            # ... più esercizi
        ]
    }
    
    for level, items in exercises.items():
        print(f"\n{level}:")
        for i, exercise in enumerate(items, 1):
            print(f"  {i}. {exercise}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run debugging methodology module"""
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║               🔍 DEBUGGING METHODOLOGY MODULE               ║
    ║            From Print() to Production Debugging             ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    modules = {
        "1": ("Systematic Framework", SystematicDebugging),
        "2": ("Debugging Tools", DebuggingTools),
        "3": ("Performance Debugging", PerformanceDebugging),
        "4": ("Debug Patterns", DebuggingPatterns),
        "5": ("Production Debug", ProductionDebugging),
        "6": ("Projects", DebugProjects),
        "7": ("Exercises", debugging_exercises)
    }
    
    while True:
        print("\n📚 SELECT MODULE:")
        for key, (name, _) in modules.items():
            print(f"  {key}. {name}")
        print("  Q. Quit")
        
        choice = input("\nChoice: ").strip().upper()
        
        if choice == 'Q':
            break
        elif choice in modules:
            if choice == '7':
                debugging_exercises()
            else:
                name, module_class = modules[choice]
                print(f"\n{'='*60}")
                print(f"Starting: {name}")
                print('='*60)
                
                module = module_class()
                
                # Run appropriate methods
                if hasattr(module, 'demonstrate_framework'):
                    module.demonstrate_framework()
                
                # Show all methods
                for method_name in dir(module):
                    if not method_name.startswith('_') and method_name != 'demonstrate_framework':
                        method = getattr(module, method_name)
                        if callable(method):
                            method()

if __name__ == "__main__":
    main()
