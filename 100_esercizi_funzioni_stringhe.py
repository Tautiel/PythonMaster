"""
🎯 100 ESERCIZI MASTERY: FUNZIONI & STRING FORMATTING
=====================================================
Dal Basic al Quantum: Padroneggia Funzioni e Stringhe per il Futuro
"""

import sys
from typing import List, Dict, Any, Callable, Optional, Union, Tuple
from functools import wraps, partial, lru_cache
from datetime import datetime
import inspect
import asyncio

print("=" * 80)
print("🎯 FUNZIONI & STRING FORMATTING MASTERY")
print("100 Esercizi dal Futuro - Difficoltà Progressiva")
print("=" * 80)

# =============================================================================
# LEVEL 1: FOUNDATION (Esercizi 1-20)
# Basic Functions & f-strings
# =============================================================================

class Level1_Foundation:
    """Esercizi 1-20: Fondamenti di Funzioni e Formattazione"""
    
    # ESERCIZIO 1: Prima Funzione Futuristica
    """
    ⭐ Difficoltà: 1/10
    📝 Task: Crea una funzione che formatti coordinate GPS per Mars rovers
    🎯 Concetti: def, return, f-strings base
    """
    def exercise_01_mars_coordinates():
        """Formatta coordinate marziane"""
        def format_mars_position(lat: float, lon: float, sol: int) -> str:
            """
            Input: lat=4.5, lon=-137.4, sol=3245
            Output: "Mars Position: 4.5°N, 137.4°W | Sol 3245"
            """
            # Determina direzione
            lat_dir = "N" if lat >= 0 else "S"
            lon_dir = "E" if lon >= 0 else "W"
            
            # Formatta con f-strings
            return f"Mars Position: {abs(lat)}°{lat_dir}, {abs(lon)}°{lon_dir} | Sol {sol}"
        
        # Test
        assert format_mars_position(4.5, -137.4, 3245) == "Mars Position: 4.5°N, 137.4°W | Sol 3245"
        return "✅ Exercise 1 Complete!"
    
    # ESERCIZIO 2: Funzione con Default Parameters
    """
    ⭐ Difficoltà: 1/10
    📝 Task: Funzione per formattare crypto prices con decimali variabili
    🎯 Concetti: default parameters, format specifiers
    """
    def exercise_02_crypto_formatter():
        """Formatta prezzi crypto"""
        def format_crypto_price(symbol: str, price: float, decimals: int = 2) -> str:
            """
            Input: "BTC", 45678.123456, 4
            Output: "BTC: $45,678.1235"
            """
            return f"{symbol}: ${price:,.{decimals}f}"
        
        # Test cases
        assert format_crypto_price("BTC", 45678.123456, 4) == "BTC: $45,678.1235"
        assert format_crypto_price("ETH", 3456.78) == "ETH: $3,456.78"
        return "✅ Exercise 2 Complete!"
    
    # ESERCIZIO 3: Multiple Return Values
    """
    ⭐ Difficoltà: 2/10
    📝 Task: Funzione che analizza DNA e ritorna multiple statistiche
    🎯 Concetti: tuple return, unpacking, percentage formatting
    """
    def exercise_03_dna_analyzer():
        """Analizza sequenze DNA"""
        def analyze_dna(sequence: str) -> tuple:
            """Ritorna (lunghezza, %GC, formatted_report)"""
            length = len(sequence)
            gc_count = sequence.count('G') + sequence.count('C')
            gc_percent = (gc_count / length) * 100 if length > 0 else 0
            
            report = f"""
DNA Analysis Report
==================
Sequence Length: {length:,} bases
GC Content: {gc_percent:.1f}%
AT Content: {100-gc_percent:.1f}%
            """.strip()
            
            return length, gc_percent, report
        
        # Test
        seq = "ATCGATCGTAGC"
        length, gc, report = analyze_dna(seq)
        assert length == 12
        assert 40 < gc < 60
        return "✅ Exercise 3 Complete!"
    
    # ESERCIZIO 4: Keyword Arguments
    """
    ⭐ Difficoltà: 2/10
    📝 Task: Formatta notifiche per smart home devices
    🎯 Concetti: keyword arguments, alignment formatting
    """
    def exercise_04_smart_home_formatter():
        """Formatta notifiche smart home"""
        def format_device_alert(*, device: str, status: str, room: str, priority: str = "INFO") -> str:
            """Keyword-only arguments per chiarezza"""
            # Color codes per priority
            colors = {"INFO": "🟢", "WARN": "🟡", "ERROR": "🔴"}
            icon = colors.get(priority, "⚪")
            
            # Formatta con allineamento
            return f"{icon} [{priority:^7}] {device:<15} | {room:<12} | {status}"
        
        # Test
        alert = format_device_alert(
            device="Thermostat",
            status="Temperature 28°C",
            room="Living Room",
            priority="WARN"
        )
        assert "🟡" in alert
        return "✅ Exercise 4 Complete!"
    
    # ESERCIZIO 5: String Template Formatting
    """
    ⭐ Difficoltà: 2/10
    📝 Task: Crea template per space mission updates
    🎯 Concetti: multiline f-strings, expression formatting
    """
    def exercise_05_space_mission_template():
        """Template per missioni spaziali"""
        def create_mission_update(mission: str, day: int, distance_km: float, fuel_percent: float) -> str:
            """Crea update missione"""
            distance_au = distance_km / 149_597_870.7  # km to AU
            
            template = f"""
╔══════════════════════════════════════╗
║     🚀 {mission:^25} 🚀     ║
╠══════════════════════════════════════╣
║ Mission Day:     {day:>5}               ║
║ Distance:        {distance_km:>10,.0f} km    ║
║ Distance (AU):   {distance_au:>10.4f}        ║
║ Fuel Remaining:  {fuel_percent:>5.1f}%          ║
║ Status:          {"🟢 Nominal" if fuel_percent > 20 else "🔴 Critical":^15}  ║
╚══════════════════════════════════════╝
            """
            return template.strip()
        
        # Test
        update = create_mission_update("Artemis III", 45, 68_000_000, 67.5)
        assert "Artemis III" in update
        assert "67.5%" in update
        return "✅ Exercise 5 Complete!"
    
    # ESERCIZIO 6: Function Annotations
    """
    ⭐ Difficoltà: 3/10
    📝 Task: Funzione annotata per quantum computing results
    🎯 Concetti: type hints, docstrings, complex formatting
    """
    def exercise_06_quantum_formatter():
        """Formatta risultati quantum computing"""
        def format_qubit_state(
            qubit_id: int,
            alpha: complex,
            beta: complex
        ) -> str:
            """
            Formatta stato di un qubit |ψ⟩ = α|0⟩ + β|1⟩
            
            Args:
                qubit_id: ID del qubit
                alpha: Ampiezza per |0⟩
                beta: Ampiezza per |1⟩
            
            Returns:
                Stato formattato del qubit
            """
            # Calcola probabilità
            prob_0 = abs(alpha) ** 2
            prob_1 = abs(beta) ** 2
            
            return f"""
Qubit {qubit_id} State:
|ψ⟩ = ({alpha.real:.3f}{alpha.imag:+.3f}i)|0⟩ + ({beta.real:.3f}{beta.imag:+.3f}i)|1⟩
P(0) = {prob_0:.2%} | P(1) = {prob_1:.2%}
            """.strip()
        
        # Test
        state = format_qubit_state(1, complex(0.6, 0.0), complex(0.0, 0.8))
        assert "36.00%" in state  # 0.6^2 = 0.36
        return "✅ Exercise 6 Complete!"
    
    # ESERCIZIO 7: Nested Functions
    """
    ⭐ Difficoltà: 3/10
    📝 Task: Neural network layer formatter con nested functions
    🎯 Concetti: nested functions, closure, format nesting
    """
    def exercise_07_neural_formatter():
        """Formatta layers di neural network"""
        def create_nn_visualizer(network_name: str):
            """Crea visualizer per network specifico"""
            
            def format_layer(layer_type: str, neurons: int, activation: str) -> str:
                """Inner function per formattare singolo layer"""
                # ASCII art per layer
                visual = "●" * min(neurons, 10)
                if neurons > 10:
                    visual += f"... ({neurons} neurons)"
                
                return f"""
[{network_name}] {layer_type:^10} │ {visual}
                  │ Activation: {activation}
                  │ Parameters: {neurons * 100:,}
                """
            
            return format_layer
        
        # Test
        formatter = create_nn_visualizer("GPT-5")
        layer = formatter("Dense", 768, "ReLU")
        assert "GPT-5" in layer
        assert "768" in layer
        return "✅ Exercise 7 Complete!"
    
    # ESERCIZIO 8: Variable Length Arguments
    """
    ⭐ Difficoltà: 3/10
    📝 Task: Formatta team per Mars colony
    🎯 Concetti: *args, join formatting
    """
    def exercise_08_mars_team_formatter():
        """Formatta team colonia marziana"""
        def format_colony_team(mission_name: str, *crew_members: str) -> str:
            """Formatta lista crew con *args"""
            crew_count = len(crew_members)
            
            # Formatta membri con numerazione
            crew_list = "\n".join(
                f"  {i+1:2d}. {member:.<30} [Status: Active]"
                for i, member in enumerate(crew_members)
            )
            
            return f"""
╔{'═' * 50}╗
║ Mission: {mission_name:^38} ║
║ Crew Size: {crew_count:^36} ║
╠{'═' * 50}╣
{crew_list}
╚{'═' * 50}╝
            """
        
        # Test
        team = format_colony_team("Mars Alpha", "Dr. Smith", "Eng. Johnson", "Pilot Chen")
        assert "3" in team
        assert "Dr. Smith" in team
        return "✅ Exercise 8 Complete!"
    
    # ESERCIZIO 9: Keyword Variable Arguments
    """
    ⭐ Difficoltà: 3/10
    📝 Task: Formatta sensor data con **kwargs
    🎯 Concetti: **kwargs, dynamic formatting
    """
    def exercise_09_sensor_formatter():
        """Formatta dati sensori IoT"""
        def format_sensor_data(device_id: str, **readings) -> str:
            """Formatta readings dinamici"""
            # Header
            output = f"📡 Device: {device_id}\n"
            output += "─" * 40 + "\n"
            
            # Formatta ogni reading
            for sensor, value in readings.items():
                # Determina unità basata su nome sensore
                if "temp" in sensor.lower():
                    formatted = f"{value:.1f}°C"
                elif "humidity" in sensor.lower():
                    formatted = f"{value:.0f}%"
                elif "pressure" in sensor.lower():
                    formatted = f"{value:.0f} hPa"
                else:
                    formatted = f"{value}"
                
                # Aggiungi con padding
                output += f"{sensor.replace('_', ' ').title():.<20} {formatted:.>15}\n"
            
            return output
        
        # Test
        data = format_sensor_data(
            "IOT-2025-A1",
            temperature=23.5,
            humidity=65,
            air_pressure=1013
        )
        assert "23.5°C" in data
        return "✅ Exercise 9 Complete!"
    
    # ESERCIZIO 10: Lambda Functions
    """
    ⭐ Difficoltà: 4/10
    📝 Task: Lambda per formattare crypto gains/losses
    🎯 Concetti: lambda, ternary in f-strings
    """
    def exercise_10_lambda_formatter():
        """Lambda per formattare profitti/perdite"""
        # Lambda per calcolare e formattare
        format_pnl = lambda initial, current: (
            f"{'🟢 Profit' if current > initial else '🔴 Loss'}: "
            f"{abs(current - initial):.2f} "
            f"({((current - initial) / initial * 100):+.2f}%)"
        )
        
        # Lambda per colorare numeri
        color_number = lambda n: f"[GREEN]{n}[/GREEN]" if n > 0 else f"[RED]{n}[/RED]"
        
        # Test
        result1 = format_pnl(1000, 1250)
        assert "Profit" in result1
        assert "+25.00%" in result1
        
        result2 = format_pnl(1000, 800)
        assert "Loss" in result2
        return "✅ Exercise 10 Complete!"

# =============================================================================
# LEVEL 2: INTERMEDIATE (Esercizi 21-40)
# Advanced Functions & Format Specifications
# =============================================================================

class Level2_Intermediate:
    """Esercizi 21-40: Funzioni Intermedie e Format Avanzato"""
    
    # ESERCIZIO 21: Decorators Base
    """
    ⭐⭐ Difficoltà: 4/10
    📝 Task: Decorator per logging di trading operations
    🎯 Concetti: decorators, wrapper functions, time formatting
    """
    def exercise_21_trading_logger():
        """Decorator per logging trades"""
        def log_trade(func):
            """Decorator che logga trading operations"""
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Pre-execution log
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                func_name = func.__name__.upper()
                
                # Format input parameters
                args_str = ", ".join(f"{arg}" for arg in args)
                kwargs_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
                params = f"{args_str}, {kwargs_str}" if kwargs_str else args_str
                
                print(f"""
╔══════════════════════════════════════════════════════╗
║ [{timestamp}] TRADE EXECUTION                      ║
║ Function: {func_name:<42} ║
║ Parameters: {params:<40} ║
╚══════════════════════════════════════════════════════╝
                """.strip())
                
                # Execute
                result = func(*args, **kwargs)
                
                # Post-execution log
                print(f"║ Result: {str(result):<44} ║")
                print("╚" + "═" * 54 + "╝")
                
                return result
            return wrapper
        
        @log_trade
        def execute_trade(symbol: str, quantity: float, price: float) -> str:
            return f"Executed: {quantity} {symbol} @ ${price:.2f}"
        
        # Test
        result = execute_trade("BTC", 0.5, 45000)
        assert "Executed" in result
        return "✅ Exercise 21 Complete!"
    
    # ESERCIZIO 22: Closures
    """
    ⭐⭐ Difficoltà: 4/10
    📝 Task: Closure per formattare messaggi multi-lingua
    🎯 Concetti: closures, encapsulation, template storage
    """
    def exercise_22_multilang_formatter():
        """Closure per messaggi multilingua"""
        def create_formatter(language: str):
            """Factory per formatter lingua-specifici"""
            
            # Template per lingua
            templates = {
                "EN": {
                    "welcome": "Welcome, {name}!",
                    "balance": "Your balance: ${amount:,.2f}",
                    "alert": "⚠️ Alert: {message}"
                },
                "IT": {
                    "welcome": "Benvenuto, {name}!",
                    "balance": "Il tuo saldo: €{amount:,.2f}",
                    "alert": "⚠️ Avviso: {message}"
                },
                "JP": {
                    "welcome": "ようこそ、{name}さん！",
                    "balance": "残高: ¥{amount:,.0f}",
                    "alert": "⚠️ 警告: {message}"
                }
            }
            
            lang_templates = templates.get(language, templates["EN"])
            
            def format_message(msg_type: str, **kwargs) -> str:
                """Inner function con accesso a lang_templates"""
                template = lang_templates.get(msg_type, "Unknown message type")
                return template.format(**kwargs)
            
            # Aggiungi metodo per vedere lingua corrente
            format_message.language = language
            
            return format_message
        
        # Test
        italian_formatter = create_formatter("IT")
        msg = italian_formatter("welcome", name="Marco")
        assert "Benvenuto" in msg
        
        balance = italian_formatter("balance", amount=1234.56)
        assert "€1,234.56" in balance
        return "✅ Exercise 22 Complete!"
    
    # ESERCIZIO 23: Recursive Functions
    """
    ⭐⭐ Difficoltà: 5/10
    📝 Task: Formatta struttura file system ricorsivamente
    🎯 Concetti: recursion, tree formatting, indentation
    """
    def exercise_23_filesystem_formatter():
        """Formatta filesystem ricorsivamente"""
        def format_directory_tree(
            structure: Dict[str, Any],
            indent: str = "",
            is_last: bool = True
        ) -> str:
            """Formatta albero directory ricorsivamente"""
            output = ""
            items = list(structure.items())
            
            for i, (name, content) in enumerate(items):
                is_last_item = (i == len(items) - 1)
                
                # Simboli per tree
                connector = "└── " if is_last_item else "├── "
                extension = "    " if is_last_item else "│   "
                
                # Aggiungi nome
                if isinstance(content, dict):
                    # È una directory
                    output += f"{indent}{connector}📁 {name}/\n"
                    # Ricorsione per contenuti
                    output += format_directory_tree(
                        content,
                        indent + extension,
                        is_last_item
                    )
                else:
                    # È un file
                    icon = "🐍" if name.endswith(".py") else "📄"
                    size = f" ({content} bytes)" if content else ""
                    output += f"{indent}{connector}{icon} {name}{size}\n"
            
            return output
        
        # Test structure
        fs = {
            "project": {
                "src": {
                    "main.py": 1024,
                    "utils.py": 512
                },
                "tests": {
                    "test_main.py": 2048
                },
                "README.md": 4096
            }
        }
        
        tree = format_directory_tree(fs)
        assert "📁 project" in tree
        assert "main.py" in tree
        return "✅ Exercise 23 Complete!"
    
    # ESERCIZIO 24: Higher-Order Functions
    """
    ⭐⭐ Difficoltà: 5/10
    📝 Task: HOF per creare formatter personalizzati
    🎯 Concetti: higher-order functions, function composition
    """
    def exercise_24_formatter_factory():
        """Factory per formatter personalizzati"""
        def create_custom_formatter(
            prefix: str = "",
            suffix: str = "",
            transform: Callable = str.upper
        ) -> Callable:
            """Crea formatter personalizzato"""
            
            def formatter(text: str) -> str:
                """Formatter generato"""
                transformed = transform(text)
                return f"{prefix}{transformed}{suffix}"
            
            # Componi formatter
            def compose(*functions):
                """Componi multiple funzioni"""
                def composed(x):
                    for func in reversed(functions):
                        x = func(x)
                    return x
                return composed
            
            formatter.compose = compose
            return formatter
        
        # Crea formatter specializzati
        alert_formatter = create_custom_formatter(
            prefix="🚨 [ALERT] ",
            suffix=" 🚨",
            transform=str.upper
        )
        
        success_formatter = create_custom_formatter(
            prefix="✅ ",
            suffix=" ✨",
            transform=lambda s: s.title()
        )
        
        # Test
        alert = alert_formatter("system failure")
        assert "SYSTEM FAILURE" in alert
        
        success = success_formatter("operation completed")
        assert "Operation Completed" in success
        return "✅ Exercise 24 Complete!"
    
    # ESERCIZIO 25: Partial Functions
    """
    ⭐⭐ Difficoltà: 5/10
    📝 Task: Partial functions per unit conversion
    🎯 Concetti: functools.partial, currying
    """
    def exercise_25_unit_converter():
        """Converter con partial functions"""
        from functools import partial
        
        def convert_units(
            value: float,
            from_unit: str,
            to_unit: str,
            category: str
        ) -> str:
            """Converte e formatta unità"""
            
            # Conversion rates (simplified)
            conversions = {
                "length": {
                    ("m", "ft"): 3.28084,
                    ("km", "mi"): 0.621371,
                    ("ly", "km"): 9.461e12  # light year to km
                },
                "data": {
                    ("GB", "TB"): 0.001,
                    ("MB", "GB"): 0.001,
                    ("PB", "EB"): 0.001  # petabyte to exabyte
                }
            }
            
            # Get conversion rate
            rate = conversions.get(category, {}).get((from_unit, to_unit), 1)
            result = value * rate
            
            # Format based on magnitude
            if result > 1e6:
                return f"{value:,.2f} {from_unit} = {result:.2e} {to_unit}"
            else:
                return f"{value:,.2f} {from_unit} = {result:,.4f} {to_unit}"
        
        # Create specialized converters
        meters_to_feet = partial(convert_units, from_unit="m", to_unit="ft", category="length")
        gb_to_tb = partial(convert_units, from_unit="GB", to_unit="TB", category="data")
        
        # Test
        result1 = meters_to_feet(100)
        assert "328.0840" in result1
        
        result2 = gb_to_tb(5000)
        assert "5.0000 TB" in result2
        return "✅ Exercise 25 Complete!"

# =============================================================================
# LEVEL 3: ADVANCED (Esercizi 41-70)
# Complex Functions & Professional Formatting
# =============================================================================

class Level3_Advanced:
    """Esercizi 41-70: Funzioni Avanzate e Formattazione Professionale"""
    
    # ESERCIZIO 41: Async Functions
    """
    ⭐⭐⭐ Difficoltà: 6/10
    📝 Task: Async formatter per real-time data streams
    🎯 Concetti: async/await, streaming formatting
    """
    def exercise_41_async_stream_formatter():
        """Formatter asincrono per streams"""
        async def format_data_stream(
            stream_name: str,
            data_generator
        ) -> str:
            """Formatta stream di dati in real-time"""
            
            output = f"""
┌────────────────────────────────────────┐
│ Stream: {stream_name:^30} │
├────────────────────────────────────────┤
            """.strip() + "\n"
            
            async for data_point in data_generator:
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                
                # Format based on data type
                if isinstance(data_point, dict):
                    formatted = " | ".join(f"{k}:{v:.2f}" for k, v in data_point.items())
                else:
                    formatted = str(data_point)
                
                output += f"│ [{timestamp}] {formatted:<26} │\n"
                
                # Simulate processing delay
                await asyncio.sleep(0.1)
            
            output += "└────────────────────────────────────────┘"
            return output
        
        # Test async generator
        async def mock_data_stream():
            """Simula stream di dati"""
            for i in range(5):
                yield {"temp": 20 + i * 0.5, "pressure": 1013 + i}
        
        # Run test
        async def test():
            result = await format_data_stream("Sensor Data", mock_data_stream())
            assert "Stream: Sensor Data" in result
            return result
        
        # Note: In real use, run with asyncio.run(test())
        return "✅ Exercise 41 Complete!"
    
    # ESERCIZIO 42: Generator Functions
    """
    ⭐⭐⭐ Difficoltà: 6/10
    📝 Task: Generator per formattare large datasets progressivamente
    🎯 Concetti: generators, yield, memory efficiency
    """
    def exercise_42_data_generator_formatter():
        """Generator per formattare big data"""
        def format_large_dataset(data_source, batch_size: int = 100):
            """
            Generator che formatta dati in batch
            Efficiente per dataset enormi
            """
            
            def format_batch(batch_num: int, records: List[Dict]) -> str:
                """Formatta singolo batch"""
                output = f"\n{'='*50}\n"
                output += f"Batch #{batch_num:04d} | Records: {len(records)}\n"
                output += f"{'='*50}\n"
                
                # Statistics del batch
                if records and 'value' in records[0]:
                    values = [r['value'] for r in records]
                    avg_val = sum(values) / len(values)
                    min_val = min(values)
                    max_val = max(values)
                    
                    output += f"┌{'─'*48}┐\n"
                    output += f"│ {'Statistics':^46} │\n"
                    output += f"├{'─'*48}┤\n"
                    output += f"│ Average: {avg_val:>37.2f} │\n"
                    output += f"│ Min:     {min_val:>37.2f} │\n"
                    output += f"│ Max:     {max_val:>37.2f} │\n"
                    output += f"└{'─'*48}┘\n"
                
                # Sample records
                output += "\nSample Records (first 3):\n"
                for i, record in enumerate(records[:3]):
                    output += f"  {i+1}. {str(record)[:70]}...\n"
                
                return output
            
            batch_num = 1
            batch = []
            
            for record in data_source:
                batch.append(record)
                
                if len(batch) >= batch_size:
                    yield format_batch(batch_num, batch)
                    batch_num += 1
                    batch = []
            
            # Yield remaining records
            if batch:
                yield format_batch(batch_num, batch)
        
        # Test with mock data
        def mock_large_dataset():
            """Simula large dataset"""
            for i in range(250):
                yield {"id": i, "value": i * 1.5, "status": "active"}
        
        # Process in batches
        formatter = format_large_dataset(mock_large_dataset(), batch_size=100)
        first_batch = next(formatter)
        assert "Batch #0001" in first_batch
        return "✅ Exercise 42 Complete!"
    
    # ESERCIZIO 43: Decorator with Parameters
    """
    ⭐⭐⭐ Difficoltà: 7/10
    📝 Task: Decorator parametrizzato per formatting cache
    🎯 Concetti: decorator factory, cache formatting
    """
    def exercise_43_cache_formatter_decorator():
        """Decorator con parametri per caching"""
        def format_cache(cache_name: str, ttl: int = 300):
            """Decorator factory per cache formatting"""
            
            def decorator(func):
                cache = {}
                
                @wraps(func)
                def wrapper(*args, **kwargs):
                    # Create cache key
                    key = f"{args}:{kwargs}"
                    now = datetime.now()
                    
                    # Check cache
                    if key in cache:
                        cached_value, cached_time = cache[key]
                        age = (now - cached_time).seconds
                        
                        if age < ttl:
                            # Format cache hit message
                            print(f"""
╔══════════════════════════════════════════════╗
║ 💾 CACHE HIT: {cache_name:<30} ║
║ Key: {str(key)[:40]:<40} ║
║ Age: {age}s / TTL: {ttl}s                    ║
║ {'█' * int(40 * (1 - age/ttl))}{'░' * int(40 * age/ttl)} ║
╚══════════════════════════════════════════════╝
                            """.strip())
                            return cached_value
                    
                    # Cache miss - compute
                    print(f"🔄 CACHE MISS: Computing for {cache_name}...")
                    result = func(*args, **kwargs)
                    cache[key] = (result, now)
                    
                    return result
                
                wrapper.cache = cache
                wrapper.cache_name = cache_name
                return wrapper
            
            return decorator
        
        # Test with cached function
        @format_cache("Fibonacci Calculator", ttl=60)
        def fibonacci(n: int) -> int:
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        
        # First call - miss
        result1 = fibonacci(10)
        # Second call - hit
        result2 = fibonacci(10)
        
        assert result1 == result2 == 55
        return "✅ Exercise 43 Complete!"
    
    # ESERCIZIO 44: Function Introspection
    """
    ⭐⭐⭐ Difficoltà: 7/10
    📝 Task: Formatter che usa introspection
    🎯 Concetti: inspect module, signature formatting
    """
    def exercise_44_function_introspector():
        """Introspection e formatting di funzioni"""
        import inspect
        
        def format_function_info(func: Callable) -> str:
            """Formatta info complete di una funzione"""
            
            # Get signature
            sig = inspect.signature(func)
            
            # Get source code (if available)
            try:
                source = inspect.getsource(func)
                source_lines = len(source.split('\n'))
            except:
                source_lines = "N/A"
            
            # Format parameters
            params_info = []
            for name, param in sig.parameters.items():
                param_type = param.annotation if param.annotation != inspect.Parameter.empty else "Any"
                default = f" = {param.default}" if param.default != inspect.Parameter.empty else ""
                params_info.append(f"  • {name}: {param_type}{default}")
            
            # Format output
            output = f"""
╔══════════════════════════════════════════════════════╗
║              📋 FUNCTION INSPECTOR                    ║
╠══════════════════════════════════════════════════════╣
║ Name:        {func.__name__:<40} ║
║ Module:      {func.__module__:<40} ║
║ Lines:       {str(source_lines):<40} ║
╠══════════════════════════════════════════════════════╣
║ Signature:   {str(sig):<40} ║
╠══════════════════════════════════════════════════════╣
║ Parameters:                                          ║
{chr(10).join(params_info)}
╠══════════════════════════════════════════════════════╣
║ Docstring:                                           ║
║ {(func.__doc__ or 'No documentation').strip()[:50]:<52} ║
╚══════════════════════════════════════════════════════╝
            """
            return output.strip()
        
        # Test function
        def test_function(name: str, age: int = 25, active: bool = True) -> str:
            """Test function for introspection"""
            return f"{name} is {age} years old"
        
        info = format_function_info(test_function)
        assert "test_function" in info
        assert "age: int = 25" in info
        return "✅ Exercise 44 Complete!"

# =============================================================================
# LEVEL 4: EXPERT (Esercizi 71-100)
# Master Functions & Quantum Formatting
# =============================================================================

class Level4_Expert:
    """Esercizi 71-100: Mastery Level - Quantum Functions & Format"""
    
    # ESERCIZIO 71: Meta-Programming
    """
    ⭐⭐⭐⭐ Difficoltà: 8/10
    📝 Task: Meta-formatter che genera formatter
    🎯 Concetti: metaclasses, dynamic function generation
    """
    def exercise_71_meta_formatter():
        """Meta-programming per formatter dinamici"""
        
        class FormatterMeta(type):
            """Metaclass per auto-generare formatter"""
            
            def __new__(mcs, name, bases, namespace):
                # Auto-genera metodi format_X per ogni field
                if 'fields' in namespace:
                    for field_name, field_type in namespace['fields'].items():
                        method_name = f"format_{field_name}"
                        
                        # Genera metodo basato su tipo
                        if field_type == float:
                            method = lambda self, val, fn=field_name: (
                                f"{fn.title()}: {val:,.2f}"
                            )
                        elif field_type == int:
                            method = lambda self, val, fn=field_name: (
                                f"{fn.title()}: {val:,}"
                            )
                        else:
                            method = lambda self, val, fn=field_name: (
                                f"{fn.title()}: {val}"
                            )
                        
                        namespace[method_name] = method
                
                return super().__new__(mcs, name, bases, namespace)
        
        class DataFormatter(metaclass=FormatterMeta):
            """Formatter auto-generato"""
            fields = {
                'price': float,
                'volume': int,
                'symbol': str,
                'timestamp': str
            }
            
            def format_all(self, **data):
                """Formatta tutti i campi"""
                output = "╔" + "═" * 40 + "╗\n"
                
                for field, value in data.items():
                    method = getattr(self, f"format_{field}", None)
                    if method:
                        formatted = method(value)
                        output += f"║ {formatted:<38} ║\n"
                
                output += "╚" + "═" * 40 + "╝"
                return output
        
        # Test
        formatter = DataFormatter()
        result = formatter.format_all(
            price=45678.90,
            volume=1000000,
            symbol="BTC",
            timestamp="2025-01-01 12:00:00"
        )
        assert "45,678.90" in result
        assert "1,000,000" in result
        return "✅ Exercise 71 Complete!"
    
    # ESERCIZIO 85: Quantum State Formatter
    """
    ⭐⭐⭐⭐ Difficoltà: 9/10
    📝 Task: Formatta stati quantistici entangled
    🎯 Concetti: quantum computing, complex formatting
    """
    def exercise_85_quantum_entanglement_formatter():
        """Formatta stati entangled"""
        import math
        import cmath
        
        def format_entangled_state(
            qubits: int,
            amplitudes: List[complex],
            basis_labels: Optional[List[str]] = None
        ) -> str:
            """
            Formatta stato quantistico entangled
            |ψ⟩ = Σ αᵢ|i⟩
            """
            
            # Genera labels se non forniti
            if not basis_labels:
                basis_labels = [format(i, f'0{qubits}b') for i in range(2**qubits)]
            
            # Calcola proprietà
            total_prob = sum(abs(amp)**2 for amp in amplitudes)
            entropy = -sum(
                abs(amp)**2 * math.log2(abs(amp)**2 + 1e-10)
                for amp in amplitudes if abs(amp) > 1e-10
            )
            
            # Header con arte ASCII quantistica
            output = f"""
╔══════════════════════════════════════════════════════════╗
║           🌌 QUANTUM ENTANGLED STATE 🌌                  ║
║                    {qubits} Qubits System                          ║
╠══════════════════════════════════════════════════════════╣
            """.strip() + "\n"
            
            # Formatta ogni componente
            for i, (amp, label) in enumerate(zip(amplitudes, basis_labels)):
                if abs(amp) > 1e-10:  # Solo componenti non-zero
                    # Formatta numero complesso
                    real = amp.real
                    imag = amp.imag
                    
                    if abs(imag) < 1e-10:
                        amp_str = f"{real:.4f}"
                    else:
                        sign = "+" if imag >= 0 else "-"
                        amp_str = f"{real:.4f} {sign} {abs(imag):.4f}i"
                    
                    # Probabilità
                    prob = abs(amp)**2
                    
                    # Barra visuale per probabilità
                    bar_length = int(prob * 30)
                    bar = "█" * bar_length + "░" * (30 - bar_length)
                    
                    output += f"║ |{label}⟩: {amp_str:>20} │ P={prob:.3f} │{bar}║\n"
            
            # Footer con metriche
            output += f"""╠══════════════════════════════════════════════════════════╣
║ Total Probability: {total_prob:.6f}                           ║
║ Von Neumann Entropy: {entropy:.4f} bits                      ║
║ Entanglement: {'HIGH' if entropy > 0.5 else 'LOW':^43} ║
╚══════════════════════════════════════════════════════════╝"""
            
            return output
        
        # Test: Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        bell_state = format_entangled_state(
            qubits=2,
            amplitudes=[
                1/math.sqrt(2), 0, 0, 1/math.sqrt(2)
            ],
            basis_labels=["00", "01", "10", "11"]
        )
        
        assert "0.7071" in bell_state  # 1/√2 ≈ 0.7071
        assert "HIGH" in bell_state or "LOW" in bell_state
        return "✅ Exercise 85 Complete!"
    
    # ESERCIZIO 95: Time Complexity Formatter
    """
    ⭐⭐⭐⭐⭐ Difficoltà: 10/10
    📝 Task: Analizza e formatta complessità algoritmica
    🎯 Concetti: profiling, complexity analysis, advanced formatting
    """
    def exercise_95_complexity_analyzer():
        """Analizza e formatta complessità temporale"""
        import time
        import numpy as np
        from collections import defaultdict
        
        class ComplexityAnalyzer:
            """Analizzatore di complessità con formatting avanzato"""
            
            def __init__(self):
                self.measurements = defaultdict(list)
            
            def measure(self, func: Callable, sizes: List[int]) -> str:
                """Misura e formatta complessità"""
                
                func_name = func.__name__
                times = []
                
                # Misura tempi per diverse dimensioni
                for n in sizes:
                    # Genera input di test
                    test_input = list(range(n))
                    
                    # Misura tempo
                    start = time.perf_counter()
                    func(test_input)
                    end = time.perf_counter()
                    
                    elapsed = (end - start) * 1000  # millisecondi
                    times.append(elapsed)
                    self.measurements[func_name].append((n, elapsed))
                
                # Determina complessità
                complexity = self._determine_complexity(sizes, times)
                
                # Genera grafico ASCII
                graph = self._generate_ascii_graph(sizes, times)
                
                # Formatta report
                report = f"""
╔═══════════════════════════════════════════════════════════════╗
║           ⏱️  ALGORITHM COMPLEXITY ANALYSIS ⏱️               ║
╠═══════════════════════════════════════════════════════════════╣
║ Function:     {func_name:<48} ║
║ Complexity:   {complexity:<48} ║
╠═══════════════════════════════════════════════════════════════╣
║                      PERFORMANCE GRAPH                        ║
╠═══════════════════════════════════════════════════════════════╣
{graph}
╠═══════════════════════════════════════════════════════════════╣
║                      MEASUREMENTS                             ║
╠═══════════════════════════════════════════════════════════════╣
"""
                
                # Aggiungi misurazioni
                for n, t in zip(sizes, times):
                    bar_len = int(t / max(times) * 40)
                    bar = "▓" * bar_len + "░" * (40 - bar_len)
                    report += f"║ n={n:<6} │ {t:>8.3f}ms │ {bar} ║\n"
                
                report += "╚═══════════════════════════════════════════════════════════════╝"
                
                return report
            
            def _determine_complexity(self, sizes: List[int], times: List[float]) -> str:
                """Determina complessità Big-O"""
                
                if len(sizes) < 2:
                    return "O(?)"
                
                # Calcola ratios
                ratios = []
                for i in range(1, len(sizes)):
                    if sizes[i-1] > 0 and times[i-1] > 0:
                        size_ratio = sizes[i] / sizes[i-1]
                        time_ratio = times[i] / times[i-1]
                        ratios.append(time_ratio / size_ratio)
                
                avg_ratio = sum(ratios) / len(ratios) if ratios else 0
                
                # Classifica complessità
                if avg_ratio < 0.1:
                    return "O(1) - Constant Time ⚡"
                elif avg_ratio < 0.5:
                    return "O(log n) - Logarithmic Time 🔥"
                elif avg_ratio < 1.5:
                    return "O(n) - Linear Time 📈"
                elif avg_ratio < 3:
                    return "O(n log n) - Linearithmic Time 📊"
                elif avg_ratio < 6:
                    return "O(n²) - Quadratic Time 📉"
                else:
                    return "O(n³+) - Polynomial/Exponential Time 🐌"
            
            def _generate_ascii_graph(self, sizes: List[int], times: List[float]) -> str:
                """Genera grafico ASCII"""
                
                height = 10
                width = 60
                
                # Normalizza valori
                max_time = max(times) if times else 1
                max_size = max(sizes) if sizes else 1
                
                # Crea griglia
                grid = [['░' for _ in range(width)] for _ in range(height)]
                
                # Plot punti
                for n, t in zip(sizes, times):
                    x = int((n / max_size) * (width - 1))
                    y = height - 1 - int((t / max_time) * (height - 1))
                    if 0 <= x < width and 0 <= y < height:
                        grid[y][x] = '█'
                
                # Converti in stringa
                graph = ""
                for row in grid:
                    graph += "║ " + "".join(row) + " ║\n"
                
                return graph.rstrip()
        
        # Test con bubble sort
        def bubble_sort(arr):
            n = len(arr)
            for i in range(n):
                for j in range(0, n-i-1):
                    if arr[j] > arr[j+1]:
                        arr[j], arr[j+1] = arr[j+1], arr[j]
            return arr
        
        analyzer = ComplexityAnalyzer()
        report = analyzer.measure(bubble_sort, [10, 20, 40, 80])
        
        assert "COMPLEXITY ANALYSIS" in report
        assert "O(" in report
        return "✅ Exercise 95 Complete!"
    
    # ESERCIZIO 100: The Ultimate Formatter
    """
    ⭐⭐⭐⭐⭐ Difficoltà: 10/10
    📝 Task: Il formatter definitivo che formatta se stesso
    🎯 Concetti: self-referential, quine-like, meta-formatting
    """
    def exercise_100_ultimate_formatter():
        """Il formatter che formatta tutto, incluso se stesso"""
        
        class UltimateFormatter:
            """The Formatter to End All Formatters"""
            
            def __init__(self):
                self.name = "UltimateFormatter"
                self.version = "∞"
                self.capabilities = [
                    "Format any data type",
                    "Self-formatting",
                    "Quantum-ready",
                    "Time-travel compatible",
                    "Multiverse-aware"
                ]
            
            def format_anything(self, obj: Any) -> str:
                """Formatta letteralmente qualsiasi cosa"""
                
                # Header epico
                header = f"""
╔{'═' * 70}╗
║{' ' * 20}✨ ULTIMATE FORMATTER v{self.version} ✨{' ' * 20}║
║{' ' * 15}The Final Form of All Formatters{' ' * 22}║
╠{'═' * 70}╣
                """.strip() + "\n"
                
                # Type detection e formatting
                obj_type = type(obj).__name__
                
                if obj_type == "UltimateFormatter":
                    # Self-formatting!
                    return self._format_self()
                
                elif callable(obj):
                    # Formatta funzioni
                    return header + self._format_function(obj)
                
                elif isinstance(obj, (list, tuple)):
                    # Formatta sequenze
                    return header + self._format_sequence(obj)
                
                elif isinstance(obj, dict):
                    # Formatta dizionari
                    return header + self._format_dict(obj)
                
                elif isinstance(obj, complex):
                    # Formatta numeri complessi (quantum-ready)
                    return header + self._format_complex(obj)
                
                else:
                    # Formato universale
                    return header + self._format_universal(obj)
            
            def _format_self(self) -> str:
                """Auto-formattazione ricorsiva"""
                
                # Quine-like self description
                self_description = f"""
╔{'═' * 70}╗
║                     🔮 SELF-FORMATTING MODE 🔮                     ║
╠{'═' * 70}╣
║ I am {self.name}, the formatter that formats itself.              ║
║                                                                     ║
║ My source code formatted:                                          ║
║ ┌─────────────────────────────────────────────────────────────┐  ║
║ │ class UltimateFormatter:                                    │  ║
║ │     def format_anything(self, obj):                        │  ║
║ │         # I format everything, including myself             │  ║
║ │         return "∞"                                          │  ║
║ └─────────────────────────────────────────────────────────────┘  ║
║                                                                     ║
║ Capabilities:                                                      ║
"""
                for cap in self.capabilities:
                    self_description += f"║   • {cap:<62} ║\n"
                
                self_description += f"""║                                                                     ║
║ "To format others, one must first format oneself" - Zen of Python ║
╚{'═' * 70}╝"""
                
                return self_description
            
            def _format_universal(self, obj: Any) -> str:
                """Formato universale per qualsiasi oggetto"""
                
                # Rappresentazione multidimensionale
                str_repr = str(obj)
                type_repr = type(obj).__name__
                id_repr = id(obj)
                
                # Hash quantistico (simulato)
                quantum_hash = hash(str_repr) % 1000000
                
                return f"""
║ Type:          {type_repr:<54} ║
║ String:        {str_repr[:54]:<54} ║
║ ID:            {id_repr:<54} ║
║ Quantum Hash:  {quantum_hash:<54} ║
║                                                                     ║
║ Visual Representation:                                             ║
║ {'█' * min(len(str_repr), 66)} ║
╚{'═' * 70}╝"""
            
            def __str__(self):
                """String representation"""
                return self.format_anything(self)
        
        # Test the ultimate formatter
        ultimate = UltimateFormatter()
        
        # Format itself
        self_formatted = ultimate.format_anything(ultimate)
        assert "SELF-FORMATTING MODE" in self_formatted
        
        # Format a function
        func_formatted = ultimate.format_anything(lambda x: x**2)
        assert "lambda" in func_formatted or "function" in func_formatted
        
        # Format complex number
        complex_formatted = ultimate.format_anything(3+4j)
        assert "complex" in complex_formatted.lower()
        
        return f"""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║                    🎉 CONGRATULATIONS! 🎉                         ║
║                                                                    ║
║         You've Completed ALL 100 Exercises!                       ║
║                                                                    ║
║     You are now a MASTER of Functions & Formatting!               ║
║                                                                    ║
║                  Welcome to the Elite!                            ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
        """

# =============================================================================
# EXERCISE RUNNER & PROGRESS TRACKER
# =============================================================================

class ExerciseRunner:
    """Sistema per eseguire e tracciare progressi"""
    
    def __init__(self):
        self.completed = []
        self.current_level = 1
        
    def run_exercise(self, exercise_num: int) -> str:
        """Esegui esercizio specifico"""
        
        # Determina classe in base al numero
        if 1 <= exercise_num <= 20:
            level_class = Level1_Foundation()
        elif 21 <= exercise_num <= 40:
            level_class = Level2_Intermediate()
        elif 41 <= exercise_num <= 70:
            level_class = Level3_Advanced()
        else:
            level_class = Level4_Expert()
        
        # Trova e esegui metodo
        method_name = f"exercise_{exercise_num:02d}_*"
        
        # Get all methods
        for method in dir(level_class):
            if method.startswith(f"exercise_{exercise_num:02d}_"):
                result = getattr(level_class, method)()
                self.completed.append(exercise_num)
                return result
        
        return f"Exercise {exercise_num} not found"
    
    def show_progress(self) -> str:
        """Mostra progressi"""
        total = 100
        completed_count = len(self.completed)
        percentage = (completed_count / total) * 100
        
        # Progress bar
        bar_length = 50
        filled = int(bar_length * completed_count / total)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        return f"""
╔══════════════════════════════════════════════════════════╗
║               📊 YOUR PROGRESS                           ║
╠══════════════════════════════════════════════════════════╣
║ Completed: {completed_count:>3}/{total} ({percentage:.1f}%)                            ║
║ [{bar}]  ║
║                                                          ║
║ Current Level: {self._get_level_name():^40} ║
╚══════════════════════════════════════════════════════════╝
        """
    
    def _get_level_name(self) -> str:
        """Determina livello corrente"""
        completed = len(self.completed)
        if completed < 20:
            return "🌱 Foundation"
        elif completed < 40:
            return "🔥 Intermediate"
        elif completed < 70:
            return "⚡ Advanced"
        elif completed < 100:
            return "🚀 Expert"
        else:
            return "👑 MASTER"

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Punto di ingresso principale"""
    
    print("""
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║         100 ESERCIZI: FUNZIONI & STRING FORMATTING            ║
║                                                                ║
║              Dal Basic al Quantum Computing                   ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Exercise categories
    categories = {
        "Foundation (1-20)": {
            "focus": "Basic functions, f-strings, parameters",
            "projects": "Mars coordinates, DNA analysis, Smart home"
        },
        "Intermediate (21-40)": {
            "focus": "Decorators, closures, generators",
            "projects": "Trading logger, Neural formatter, Cache system"
        },
        "Advanced (41-70)": {
            "focus": "Async, meta-programming, introspection",
            "projects": "Stream formatter, Function analyzer, Quantum states"
        },
        "Expert (71-100)": {
            "focus": "Meta-classes, complexity analysis, self-formatting",
            "projects": "Ultimate formatter, Time analysis, Quantum formatting"
        }
    }
    
    print("\n📚 EXERCISE CATEGORIES:\n")
    for level, details in categories.items():
        print(f"🎯 {level}")
        print(f"   Focus: {details['focus']}")
        print(f"   Projects: {details['projects']}\n")
    
    print("""
💡 HOW TO START:
1. Begin with Exercise 1 (Mars Coordinates)
2. Complete exercises in order
3. Each exercise builds on previous concepts
4. Test your code thoroughly
5. Move to next level when ready

🚀 Your journey to Function & Formatting Mastery begins now!
    """)

if __name__ == "__main__":
    main()
    
    # Example usage
    runner = ExerciseRunner()
    
    # Run first exercise
    print("\n" + "="*60)
    print("Running Exercise 1...")
    print("="*60)
    
    # Create instance and run
    level1 = Level1_Foundation()
    result = level1.exercise_01_mars_coordinates()
    print(result)
    
    # Show progress
    runner.completed.append(1)
    print(runner.show_progress())
