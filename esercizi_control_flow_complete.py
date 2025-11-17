"""
🎯 ESERCIZI CONTROL FLOW - 15 ESERCIZI COMPLETI
If/Else, Loops, Comprehensions, Flow Control
MANCAVANO NEL MATERIALE - ESSENZIALI PER LE BASI!
"""

print("=" * 60)
print("CONTROL FLOW: 15 ESERCIZI PROGRESSIVI")
print("=" * 60)

# ============================================
# SEZIONE 1: IF/ELIF/ELSE (Esercizi 1-5)
# ============================================

print("\n📚 CONDITIONAL STATEMENTS")
print("-" * 40)

# ESERCIZIO 1: If/Else Basics
print("\n📝 ESERCIZIO 1: Decisioni Base")
def basic_conditions():
    """
    Pratica con if/elif/else base
    """
    # Trading decision based on price
    btc_price = 45000
    buy_threshold = 44000
    sell_threshold = 46000
    
    # TODO: Decisione trading
    if btc_price < buy_threshold:
        action = "BUY"
        print(f"💰 Prezzo ${btc_price} < ${buy_threshold}: {action}")
    elif btc_price > sell_threshold:
        action = "SELL"
        print(f"📈 Prezzo ${btc_price} > ${sell_threshold}: {action}")
    else:
        action = "HOLD"
        print(f"⏸️ Prezzo ${btc_price} in range: {action}")
    
    # Controllo portfolio
    portfolio_value = 10000
    
    # TODO: Status basato su valore
    if portfolio_value < 1000:
        status = "CRITICAL"
        alert = True
    elif portfolio_value < 5000:
        status = "LOW"
        alert = True
    elif portfolio_value < 20000:
        status = "NORMAL"
        alert = False
    else:
        status = "EXCELLENT"
        alert = False
    
    print(f"\nPortfolio ${portfolio_value}: Status {status}")
    if alert:
        print("⚠️ ALERT: Portfolio needs attention!")
    
    # Password validation
    password = "MySecurePass123!"
    
    # TODO: Validazione completa
    is_long = len(password) >= 8
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(c in "!@#$%^&*" for c in password)
    
    if is_long and has_upper and has_lower and has_digit and has_special:
        print("\n✅ Password FORTE")
    elif is_long and (has_upper or has_lower) and has_digit:
        print("\n⚠️ Password MEDIA")
    else:
        print("\n❌ Password DEBOLE")
    
    return action

basic_conditions()


# ESERCIZIO 2: Nested Conditions
print("\n📝 ESERCIZIO 2: Condizioni Annidate")
def nested_conditions():
    """
    Condizioni complesse annidate
    """
    # Trading system complesso
    account_balance = 10000
    position_size = 0.1
    risk_per_trade = 0.02  # 2%
    market_open = True
    volatility = "HIGH"  # LOW, MEDIUM, HIGH
    
    print("SISTEMA TRADING AVANZATO")
    print(f"Balance: ${account_balance}")
    print(f"Volatility: {volatility}")
    
    # TODO: Logica trading complessa
    if market_open:
        if account_balance > 5000:
            if volatility == "LOW":
                max_position = position_size * 2
                print(f"✅ Posizione aumentata: {max_position}")
            elif volatility == "MEDIUM":
                max_position = position_size
                print(f"✅ Posizione normale: {max_position}")
            else:  # HIGH
                if account_balance > 15000:
                    max_position = position_size * 0.5
                    print(f"⚠️ Posizione ridotta: {max_position}")
                else:
                    max_position = 0
                    print(f"❌ NO TRADE - Volatilità troppo alta")
        else:
            print("❌ Balance insufficiente per tradare")
    else:
        print("❌ Mercato chiuso")
    
    # Sistema di scoring
    score = 75
    attendance = 85
    projects_completed = 8
    
    # TODO: Valutazione complessa
    if score >= 90:
        if attendance >= 90:
            grade = "A+"
        else:
            grade = "A"
    elif score >= 80:
        if attendance >= 85 and projects_completed >= 7:
            grade = "B+"
        elif attendance >= 80:
            grade = "B"
        else:
            grade = "B-"
    elif score >= 70:
        if projects_completed >= 5:
            grade = "C+"
        else:
            grade = "C"
    else:
        grade = "F"
    
    print(f"\n📊 Voto finale: {grade}")
    print(f"   Score: {score}, Attendance: {attendance}%, Projects: {projects_completed}")
    
    return grade

nested_conditions()


# ESERCIZIO 3: Operatori Logici
print("\n📝 ESERCIZIO 3: Operatori Logici (and, or, not)")
def logical_operators():
    """
    Combina condizioni con operatori logici
    """
    # User authentication
    username = "marco"
    password = "secure123"
    is_admin = False
    is_verified = True
    two_factor = True
    
    # TODO: Login con multiple condizioni
    basic_auth = username == "marco" and password == "secure123"
    
    if basic_auth and is_verified:
        if is_admin or two_factor:
            print("✅ Login autorizzato")
            access_level = "ADMIN" if is_admin else "USER"
            print(f"   Access level: {access_level}")
        else:
            print("⚠️ Richiesto 2FA")
    else:
        print("❌ Credenziali invalide")
    
    # Trading conditions
    price = 45000
    volume = 1500000
    rsi = 35  # Relative Strength Index
    macd_signal = True
    
    # TODO: Segnale di acquisto complesso
    oversold = rsi < 30
    high_volume = volume > 1000000
    price_range = 40000 < price < 50000
    
    strong_buy = oversold and high_volume and macd_signal
    moderate_buy = (oversold or macd_signal) and price_range
    no_trade = not high_volume or not price_range
    
    if strong_buy:
        signal = "STRONG BUY 🚀"
    elif moderate_buy and not no_trade:
        signal = "MODERATE BUY 📈"
    else:
        signal = "WAIT ⏸️"
    
    print(f"\n📊 Trading Signal: {signal}")
    print(f"   RSI: {rsi}, Volume: {volume:,}, MACD: {macd_signal}")
    
    # Eligibility check
    age = 25
    income = 50000
    credit_score = 720
    employment_years = 3
    
    # TODO: Loan eligibility
    age_ok = 18 <= age <= 65
    income_ok = income >= 30000
    credit_ok = credit_score >= 650
    employment_ok = employment_years >= 2
    
    eligible = age_ok and income_ok and (credit_ok or employment_ok)
    premium = credit_score >= 750 and income >= 70000
    
    if not age_ok:
        print("\n❌ Età non idonea")
    elif eligible and premium:
        print("\n✅ Eligible for PREMIUM loan")
    elif eligible:
        print("\n✅ Eligible for STANDARD loan")
    else:
        print("\n❌ Not eligible")
        if not income_ok:
            print("   - Income troppo basso")
        if not credit_ok:
            print("   - Credit score insufficiente")
    
    return signal

logical_operators()


# ESERCIZIO 4: Ternary Operator
print("\n📝 ESERCIZIO 4: Operatore Ternario")
def ternary_operator():
    """
    Conditional expressions (ternary operator)
    """
    # Sintassi: value_if_true if condition else value_if_false
    
    # TODO: Status account
    balance = 10000
    status = "ACTIVE" if balance > 0 else "INACTIVE"
    print(f"Account status: {status}")
    
    # TODO: Trading fee
    is_vip = True
    trade_value = 10000
    fee = trade_value * 0.001 if is_vip else trade_value * 0.002
    print(f"Trading fee: ${fee:.2f} (VIP: {is_vip})")
    
    # TODO: Messaggio personalizzato
    hour = 14
    greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"
    print(f"Greeting: {greeting}")
    
    # TODO: Validazione con ternario
    password = "short"
    valid = "✅ Valid" if len(password) >= 8 else "❌ Too short"
    print(f"Password: {valid}")
    
    # TODO: Calcolo con ternario
    quantity = 5
    price = 100
    discount_rate = 0.1 if quantity >= 10 else 0.05 if quantity >= 5 else 0
    final_price = price * quantity * (1 - discount_rate)
    print(f"Final price: ${final_price:.2f} (discount: {discount_rate*100:.0f}%)")
    
    # TODO: Nested ternary (usa con cautela!)
    score = 85
    grade = "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "F"
    print(f"Grade: {grade}")
    
    # List comprehension con ternario
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    labels = ["even" if n % 2 == 0 else "odd" for n in numbers]
    print(f"\nNumbers: {list(zip(numbers, labels))[:5]}...")
    
    return status

ternary_operator()


# ESERCIZIO 5: Match Statement (Python 3.10+)
print("\n📝 ESERCIZIO 5: Pattern Matching")
def pattern_matching():
    """
    Match statement (Python 3.10+)
    Nota: Se usi Python < 3.10, usa if/elif invece
    """
    import sys
    
    # Check Python version
    if sys.version_info >= (3, 10):
        print("✅ Python 3.10+ detected - using match statement")
        
        # TODO: Menu selection con match
        def process_command(command):
            match command.lower():
                case "buy":
                    return "📈 Executing BUY order"
                case "sell":
                    return "📉 Executing SELL order"
                case "hold":
                    return "⏸️ Holding position"
                case "exit" | "quit":
                    return "👋 Exiting program"
                case _:
                    return "❓ Unknown command"
        
        print(process_command("buy"))
        print(process_command("exit"))
        print(process_command("invalid"))
        
    else:
        print("⚠️ Python < 3.10 - using if/elif alternative")
        
        # Alternative con if/elif
        def process_command(command):
            cmd = command.lower()
            if cmd == "buy":
                return "📈 Executing BUY order"
            elif cmd == "sell":
                return "📉 Executing SELL order"
            elif cmd == "hold":
                return "⏸️ Holding position"
            elif cmd in ["exit", "quit"]:
                return "👋 Exiting program"
            else:
                return "❓ Unknown command"
        
        print(process_command("buy"))
        print(process_command("exit"))
        print(process_command("invalid"))
    
    # Grade calculator
    def calculate_grade(score):
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    scores = [95, 85, 75, 65, 55]
    for score in scores:
        grade = calculate_grade(score)
        print(f"Score {score} → Grade {grade}")
    
    return True

pattern_matching()


# ============================================
# SEZIONE 2: LOOPS (Esercizi 6-10)
# ============================================

print("\n\n📚 LOOPS")
print("-" * 40)

# ESERCIZIO 6: For Loop Basics
print("\n📝 ESERCIZIO 6: For Loop Base")
def for_loop_basics():
    """
    Cicli for fondamentali
    """
    # TODO: Itera su lista
    portfolio = ['BTC', 'ETH', 'ADA', 'DOT', 'LINK']
    print("Portfolio assets:")
    for asset in portfolio:
        print(f"  - {asset}")
    
    # TODO: Range con numeri
    print("\nCountdown:")
    for i in range(10, 0, -1):
        print(f"  {i}...")
    print("  🚀 Liftoff!")
    
    # TODO: Enumerate per indici
    print("\nRanked assets:")
    for rank, asset in enumerate(portfolio, 1):
        print(f"  #{rank}: {asset}")
    
    # TODO: Itera su dizionario
    prices = {
        'BTC': 45000,
        'ETH': 3000,
        'ADA': 1.5
    }
    
    print("\nPrices (3 modi):")
    
    # Solo chiavi
    print("Keys only:")
    for symbol in prices:
        print(f"  {symbol}")
    
    # Chiavi e valori
    print("Keys and values:")
    for symbol, price in prices.items():
        print(f"  {symbol}: ${price:,}")
    
    # Solo valori
    total = 0
    for price in prices.values():
        total += price
    print(f"Total value: ${total:,}")
    
    # TODO: Nested loops
    print("\nMultiplication table (3x3):")
    for i in range(1, 4):
        for j in range(1, 4):
            result = i * j
            print(f"{i}×{j}={result:2}", end="  ")
        print()  # Nuova riga
    
    # TODO: Loop con zip
    symbols = ['BTC', 'ETH', 'ADA']
    quantities = [0.5, 2.0, 1000]
    prices_list = [45000, 3000, 1.5]
    
    print("\nPortfolio value:")
    total_value = 0
    for sym, qty, price in zip(symbols, quantities, prices_list):
        value = qty * price
        total_value += value
        print(f"  {sym}: {qty} × ${price} = ${value:,.2f}")
    print(f"Total: ${total_value:,.2f}")
    
    return total_value

for_loop_basics()


# ESERCIZIO 7: While Loop
print("\n📝 ESERCIZIO 7: While Loop")
def while_loop_practice():
    """
    Cicli while per iterazioni condizionali
    """
    # TODO: While base
    counter = 0
    print("Counting to 5:")
    while counter < 5:
        counter += 1
        print(f"  Count: {counter}")
    
    # TODO: Trading simulation
    balance = 1000
    target = 1500
    trades = 0
    win_rate = 0.6  # 60% win rate
    
    print(f"\nTrading simulation (target: ${target}):")
    print(f"Starting balance: ${balance}")
    
    import random
    random.seed(42)  # Per risultati riproducibili
    
    while balance < target and balance > 500:
        trades += 1
        # Simula trade
        if random.random() < win_rate:
            # Win: +5%
            profit = balance * 0.05
            balance += profit
            print(f"  Trade #{trades}: WIN +${profit:.2f} → ${balance:.2f}")
        else:
            # Loss: -3%
            loss = balance * 0.03
            balance -= loss
            print(f"  Trade #{trades}: LOSS -${loss:.2f} → ${balance:.2f}")
        
        if trades >= 20:  # Limite trades
            print("  Max trades reached!")
            break
    
    if balance >= target:
        print(f"✅ Target reached in {trades} trades!")
    else:
        print(f"❌ Failed to reach target. Final: ${balance:.2f}")
    
    # TODO: Input validation con while
    def get_valid_input():
        """Simula input utente"""
        attempts = 0
        max_attempts = 3
        valid = False
        
        # Simula input
        test_inputs = ["abc", "-5", "42"]
        
        while not valid and attempts < max_attempts:
            # Simula input utente
            user_input = test_inputs[attempts]
            attempts += 1
            
            print(f"\nAttempt {attempts}: Input = '{user_input}'")
            
            try:
                number = int(user_input)
                if number > 0:
                    valid = True
                    print(f"✅ Valid input: {number}")
                else:
                    print("❌ Must be positive!")
            except ValueError:
                print("❌ Not a number!")
        
        if not valid:
            print("❌ Max attempts reached!")
            return None
        return number
    
    result = get_valid_input()
    
    # TODO: Infinite loop con break
    print("\nSearching for signal...")
    prices = [44000, 44500, 45000, 43000, 42000, 41000, 40000]
    index = 0
    
    while True:
        current_price = prices[index]
        print(f"  Checking ${current_price}...")
        
        if current_price <= 42000:
            print(f"  🎯 Buy signal at ${current_price}!")
            break
        
        index += 1
        if index >= len(prices):
            print("  No signal found")
            break
    
    return balance

while_loop_practice()


# ESERCIZIO 8: Loop Control (break, continue, else)
print("\n📝 ESERCIZIO 8: Controllo Loop")
def loop_control():
    """
    Break, continue, else nei loops
    """
    # TODO: Break - esci dal loop
    print("Searching for error:")
    logs = ["INFO: Started", "INFO: Processing", "ERROR: Failed", "INFO: Done"]
    
    for i, log in enumerate(logs):
        print(f"  Line {i+1}: {log}")
        if "ERROR" in log:
            print("  ❌ Error found! Stopping...")
            break
    
    # TODO: Continue - salta iterazione
    print("\nProcessing trades (skip small ones):")
    trades = [100, 5000, 50, 8000, 25, 10000]
    min_trade = 1000
    
    total = 0
    for trade in trades:
        if trade < min_trade:
            print(f"  ${trade} - Skipped (too small)")
            continue
        
        total += trade
        print(f"  ${trade} - Processed")
    
    print(f"Total processed: ${total}")
    
    # TODO: Else con for - eseguito se no break
    print("\nSearching for prime:")
    numbers = [4, 6, 8, 9, 10]
    
    for n in numbers:
        for divisor in range(2, n):
            if n % divisor == 0:
                print(f"  {n} is not prime (divisible by {divisor})")
                break
        else:  # No break occurred
            print(f"  {n} is prime!")
    
    # TODO: Else con while
    print("\nWaiting for condition:")
    attempts = 0
    max_attempts = 5
    
    while attempts < max_attempts:
        attempts += 1
        # Simula check
        success = attempts == 3  # Successo al 3° tentativo
        
        if success:
            print(f"  Attempt {attempts}: Success! ✅")
            break
        else:
            print(f"  Attempt {attempts}: Waiting...")
    else:  # No break (tutti falliti)
        print("  ❌ Max attempts reached without success")
    
    # TODO: Nested loops con break
    print("\nFinding target in matrix:")
    matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    target = 5
    found = False
    
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            print(f"  Checking [{i}][{j}] = {value}")
            if value == target:
                print(f"  ✅ Found {target} at position [{i}][{j}]")
                found = True
                break
        if found:
            break
    
    return found

loop_control()


# ESERCIZIO 9: Loop Patterns
print("\n📝 ESERCIZIO 9: Pattern con Loop")
def loop_patterns():
    """
    Pattern comuni con loops
    """
    # TODO: Accumulator pattern
    print("Accumulator pattern:")
    prices = [100, 150, 200, 175, 225]
    
    total = 0
    for price in prices:
        total += price
    average = total / len(prices)
    print(f"  Average price: ${average:.2f}")
    
    # TODO: Counter pattern
    print("\nCounter pattern:")
    trades = ['BUY', 'BUY', 'SELL', 'BUY', 'SELL', 'SELL', 'BUY']
    
    buy_count = 0
    sell_count = 0
    for trade in trades:
        if trade == 'BUY':
            buy_count += 1
        else:
            sell_count += 1
    
    print(f"  BUY: {buy_count}, SELL: {sell_count}")
    print(f"  Win rate: {buy_count/len(trades)*100:.1f}%")
    
    # TODO: Min/Max pattern
    print("\nMin/Max pattern:")
    temperatures = [22, 25, 19, 27, 21, 30, 18]
    
    min_temp = temperatures[0]
    max_temp = temperatures[0]
    
    for temp in temperatures:
        if temp < min_temp:
            min_temp = temp
        if temp > max_temp:
            max_temp = temp
    
    print(f"  Min: {min_temp}°C, Max: {max_temp}°C")
    
    # TODO: Filter pattern
    print("\nFilter pattern:")
    all_prices = [45000, 3000, 150, 25, 1.5, 0.5, 10000]
    high_value = []
    
    threshold = 1000
    for price in all_prices:
        if price >= threshold:
            high_value.append(price)
    
    print(f"  High value assets: {high_value}")
    
    # TODO: Transform pattern
    print("\nTransform pattern:")
    eur_prices = [100, 200, 300, 400]
    eur_to_usd = 1.18
    
    usd_prices = []
    for eur in eur_prices:
        usd = eur * eur_to_usd
        usd_prices.append(usd)
    
    print(f"  EUR: {eur_prices}")
    print(f"  USD: {usd_prices}")
    
    # TODO: Search pattern
    print("\nSearch pattern:")
    users = [
        {'id': 1, 'name': 'Alice', 'balance': 1000},
        {'id': 2, 'name': 'Bob', 'balance': 2000},
        {'id': 3, 'name': 'Charlie', 'balance': 1500}
    ]
    
    search_name = 'Bob'
    found_user = None
    
    for user in users:
        if user['name'] == search_name:
            found_user = user
            break
    
    if found_user:
        print(f"  Found: {found_user}")
    else:
        print(f"  User {search_name} not found")
    
    return True

loop_patterns()


# ESERCIZIO 10: Performance Considerations
print("\n📝 ESERCIZIO 10: Performance nei Loop")
def loop_performance():
    """
    Ottimizzazione e performance dei loop
    """
    import time
    
    # TODO: List vs Generator
    print("List vs Generator:")
    
    # List comprehension (tutto in memoria)
    start = time.time()
    squares_list = [x**2 for x in range(1000000)]
    list_time = time.time() - start
    
    # Generator (lazy evaluation)
    start = time.time()
    squares_gen = (x**2 for x in range(1000000))
    gen_time = time.time() - start
    
    print(f"  List creation: {list_time:.4f}s")
    print(f"  Generator creation: {gen_time:.6f}s")
    print(f"  Generator is {list_time/gen_time:.0f}x faster to create!")
    
    # TODO: Loop optimization
    print("\nLoop optimization:")
    data = list(range(10000))
    
    # Slow: function call in loop
    start = time.time()
    result = []
    for i in range(len(data)):
        result.append(data[i] * 2)
    slow_time = time.time() - start
    
    # Fast: direct iteration
    start = time.time()
    result = []
    for item in data:
        result.append(item * 2)
    fast_time = time.time() - start
    
    # Fastest: list comprehension
    start = time.time()
    result = [item * 2 for item in data]
    fastest_time = time.time() - start
    
    print(f"  Slow (index): {slow_time:.4f}s")
    print(f"  Fast (direct): {fast_time:.4f}s")
    print(f"  Fastest (comprehension): {fastest_time:.4f}s")
    
    # TODO: Early exit optimization
    print("\nEarly exit optimization:")
    large_list = list(range(1000000))
    target = 500000
    
    # Without early exit
    start = time.time()
    found = False
    for item in large_list:
        if item == target:
            found = True
    no_break_time = time.time() - start
    
    # With early exit
    start = time.time()
    found = False
    for item in large_list:
        if item == target:
            found = True
            break
    break_time = time.time() - start
    
    print(f"  Without break: {no_break_time:.4f}s")
    print(f"  With break: {break_time:.4f}s")
    print(f"  Break is {no_break_time/break_time:.1f}x faster!")
    
    print("\n💡 Best Practices:")
    print("  - Use generators for large datasets")
    print("  - Prefer comprehensions over manual loops")
    print("  - Exit early with break when possible")
    print("  - Avoid function calls in loop conditions")
    
    return True

loop_performance()


# ============================================
# SEZIONE 3: COMPREHENSIONS (Esercizi 11-15)
# ============================================

print("\n\n📚 COMPREHENSIONS")
print("-" * 40)

# ESERCIZIO 11: List Comprehensions Advanced
print("\n📝 ESERCIZIO 11: List Comprehensions Avanzate")
def advanced_list_comprehensions():
    """
    List comprehensions complesse
    """
    # TODO: Con condizione
    numbers = range(1, 21)
    evens = [n for n in numbers if n % 2 == 0]
    print(f"Even numbers: {evens}")
    
    # TODO: Con trasformazione e condizione
    prices = [100, 250, 75, 300, 50, 400]
    discounted_premium = [p * 0.8 for p in prices if p >= 200]
    print(f"Premium with discount: {discounted_premium}")
    
    # TODO: Con if-else (ternary)
    values = [1, -2, 3, -4, 5, -6]
    absolute = [x if x >= 0 else -x for x in values]
    print(f"Absolute values: {absolute}")
    
    # TODO: Nested comprehension
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    flattened = [item for row in matrix for item in row]
    print(f"Flattened: {flattened}")
    
    # TODO: Multiple conditions
    numbers = range(1, 101)
    special = [n for n in numbers if n % 3 == 0 if n % 5 == 0]
    print(f"Divisible by 3 AND 5: {special[:5]}...")
    
    # TODO: Con enumerate
    words = ['python', 'trading', 'bot', 'ai']
    indexed = [(i, word.upper()) for i, word in enumerate(words)]
    print(f"Indexed words: {indexed}")
    
    # TODO: Da multiple liste
    symbols = ['BTC', 'ETH', 'ADA']
    prices = [45000, 3000, 1.5]
    changes = [2.5, -1.2, 0.8]
    
    summary = [f"{s}: ${p:,.0f} ({c:+.1f}%)" 
               for s, p, c in zip(symbols, prices, changes)]
    print(f"\nMarket summary:")
    for item in summary:
        print(f"  {item}")
    
    return summary

advanced_list_comprehensions()


# ESERCIZIO 12: Dictionary Comprehensions Advanced
print("\n📝 ESERCIZIO 12: Dict Comprehensions Avanzate")
def advanced_dict_comprehensions():
    """
    Dictionary comprehensions complesse
    """
    # TODO: Transform dictionary
    prices_usd = {'BTC': 45000, 'ETH': 3000, 'ADA': 1.5}
    prices_eur = {k: v * 0.85 for k, v in prices_usd.items()}
    print(f"EUR prices: {prices_eur}")
    
    # TODO: Filter dictionary
    all_assets = {'BTC': 45000, 'ETH': 3000, 'DOGE': 0.1, 'ADA': 1.5}
    valuable = {k: v for k, v in all_assets.items() if v >= 100}
    print(f"Valuable assets: {valuable}")
    
    # TODO: Swap keys and values
    symbol_to_name = {'BTC': 'Bitcoin', 'ETH': 'Ethereum'}
    name_to_symbol = {v: k for k, v in symbol_to_name.items()}
    print(f"Name mapping: {name_to_symbol}")
    
    # TODO: From two lists
    symbols = ['BTC', 'ETH', 'ADA']
    holdings = [0.5, 2.0, 1000]
    portfolio = {s: h for s, h in zip(symbols, holdings) if h > 0}
    print(f"Portfolio: {portfolio}")
    
    # TODO: Nested dictionary
    data = {
        'BTC': {'price': 45000, 'volume': 1000000},
        'ETH': {'price': 3000, 'volume': 500000}
    }
    
    # Extract just prices
    just_prices = {k: v['price'] for k, v in data.items()}
    print(f"Just prices: {just_prices}")
    
    # TODO: Conditional values
    numbers = range(1, 11)
    squared_or_cubed = {n: n**2 if n % 2 == 0 else n**3 for n in numbers}
    print(f"Squared (even) or Cubed (odd): {dict(list(squared_or_cubed.items())[:5])}...")
    
    # TODO: Grouping with comprehension
    trades = [
        {'symbol': 'BTC', 'amount': 100},
        {'symbol': 'ETH', 'amount': 200},
        {'symbol': 'BTC', 'amount': 150},
        {'symbol': 'ETH', 'amount': 300}
    ]
    
    # Group by symbol (simple sum)
    grouped = {}
    for trade in trades:
        symbol = trade['symbol']
        grouped[symbol] = grouped.get(symbol, 0) + trade['amount']
    
    print(f"Grouped trades: {grouped}")
    
    return grouped

advanced_dict_comprehensions()


# ESERCIZIO 13: Set Comprehensions
print("\n📝 ESERCIZIO 13: Set Comprehensions")
def set_comprehensions():
    """
    Set comprehensions uniche
    """
    # TODO: Basic set comprehension
    numbers = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    unique_squares = {n**2 for n in numbers}
    print(f"Unique squares: {unique_squares}")
    
    # TODO: With condition
    words = ['python', 'java', 'javascript', 'rust', 'go', 'c++']
    short_words = {w.upper() for w in words if len(w) <= 4}
    print(f"Short words: {short_words}")
    
    # TODO: From string (unique chars)
    text = "cryptocurrency trading bot"
    vowels = {c for c in text.lower() if c in 'aeiou'}
    print(f"Vowels in text: {vowels}")
    
    # TODO: Mathematical set
    multiples_3 = {x for x in range(1, 31) if x % 3 == 0}
    multiples_5 = {x for x in range(1, 31) if x % 5 == 0}
    both = multiples_3 & multiples_5
    either = multiples_3 | multiples_5
    
    print(f"Multiples of 3: {multiples_3}")
    print(f"Multiples of 5: {multiples_5}")
    print(f"Both: {both}")
    print(f"Either: {either}")
    
    # TODO: Remove duplicates with logic
    prices = [100.0, 100.00, 200, 200.0, 150, 150.00]
    unique_prices = {float(p) for p in prices}
    print(f"Unique prices: {unique_prices}")
    
    return unique_prices

set_comprehensions()


# ESERCIZIO 14: Generator Expressions
print("\n📝 ESERCIZIO 14: Generator Expressions")
def generator_expressions():
    """
    Generator expressions per efficienza
    """
    import sys
    
    # TODO: List vs Generator memory
    list_comp = [x**2 for x in range(1000)]
    gen_exp = (x**2 for x in range(1000))
    
    print(f"List size: {sys.getsizeof(list_comp)} bytes")
    print(f"Generator size: {sys.getsizeof(gen_exp)} bytes")
    
    # TODO: Lazy evaluation
    def expensive_operation(n):
        """Simula operazione costosa"""
        return n ** 2
    
    # Generator - non calcola finché non richiesto
    gen = (expensive_operation(x) for x in range(5))
    print(f"\nGenerator created (not evaluated yet)")
    
    print("Getting first 3 values:")
    for i, val in enumerate(gen):
        print(f"  Value {i}: {val}")
        if i >= 2:
            break
    
    # TODO: Pipeline di generators
    numbers = range(1, 11)
    
    # Chain operations
    squared = (n**2 for n in numbers)
    filtered = (n for n in squared if n > 20)
    formatted = (f"Value: {n}" for n in filtered)
    
    print("\nPipeline results:")
    for result in formatted:
        print(f"  {result}")
    
    # TODO: sum/max/min con generator
    large_range = range(1, 1000001)
    
    # Efficiente con generator
    total = sum(x for x in large_range if x % 2 == 0)
    print(f"\nSum of even numbers 1-1M: {total:,}")
    
    # TODO: any/all con generator
    data = [2, 4, 6, 8, 10]
    all_even = all(x % 2 == 0 for x in data)
    has_negative = any(x < 0 for x in data)
    
    print(f"All even? {all_even}")
    print(f"Has negative? {has_negative}")
    
    return gen_exp

generator_expressions()


# ESERCIZIO 15: Nested Comprehensions
print("\n📝 ESERCIZIO 15: Comprehensions Annidate")
def nested_comprehensions():
    """
    Comprehensions complesse annidate
    """
    # TODO: Matrix creation
    matrix = [[i + j*3 for i in range(3)] for j in range(3)]
    print("Matrix 3x3:")
    for row in matrix:
        print(f"  {row}")
    
    # TODO: Matrix transpose
    transposed = [[matrix[j][i] for j in range(3)] for i in range(3)]
    print("\nTransposed:")
    for row in transposed:
        print(f"  {row}")
    
    # TODO: Flatten nested list
    nested = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    flat = [item for sublist in nested for item in sublist]
    print(f"\nFlattened: {flat}")
    
    # TODO: Cartesian product
    colors = ['red', 'blue']
    sizes = ['S', 'M', 'L']
    products = [(color, size) for color in colors for size in sizes]
    print(f"\nProducts: {products}")
    
    # TODO: Complex filtering
    data = [
        {'name': 'BTC', 'prices': [45000, 46000, 44000]},
        {'name': 'ETH', 'prices': [3000, 3100, 2900]},
        {'name': 'ADA', 'prices': [1.5, 1.6, 1.4]}
    ]
    
    # Get all prices above 1000
    high_prices = [
        price 
        for asset in data 
        for price in asset['prices'] 
        if price > 1000
    ]
    print(f"\nHigh prices: {high_prices}")
    
    # TODO: Dictionary of lists
    categories = {
        'high': [asset['name'] for asset in data if any(p > 10000 for p in asset['prices'])],
        'medium': [asset['name'] for asset in data if any(100 < p <= 10000 for p in asset['prices'])],
        'low': [asset['name'] for asset in data if all(p <= 100 for p in asset['prices'])]
    }
    print(f"\nCategorized: {categories}")
    
    # TODO: Comprehension readability
    # Bad - too complex
    # result = [[f(x) for x in row if condition(x)] for row in matrix if len(row) > 0]
    
    # Good - break it down
    filtered_matrix = [row for row in matrix if len(row) > 0]
    result = []
    for row in filtered_matrix:
        processed_row = [x * 2 for x in row if x > 0]
        result.append(processed_row)
    
    print("\n💡 Readability tip:")
    print("  Break complex comprehensions into steps!")
    
    return result

nested_comprehensions()


# ============================================
# PROGETTO FINALE: CONTROL FLOW MASTER
# ============================================

print("\n\n" + "=" * 60)
print("PROGETTO FINALE: Trading Strategy Simulator")
print("=" * 60)

class TradingStrategySimulator:
    """
    Simula una strategia di trading usando tutto il control flow
    """
    def __init__(self, initial_capital=10000):
        self.capital = initial_capital
        self.positions = []
        self.trades_history = []
        self.winning_trades = 0
        self.losing_trades = 0
    
    def analyze_market(self, price, volume, rsi, macd):
        """
        Analizza mercato con condizioni complesse
        """
        # Condizioni multiple
        oversold = rsi < 30
        overbought = rsi > 70
        high_volume = volume > 1000000
        macd_bullish = macd > 0
        macd_bearish = macd < 0
        
        # Logica decisionale
        if oversold and macd_bullish and high_volume:
            return "STRONG_BUY"
        elif oversold or (macd_bullish and rsi < 50):
            return "BUY"
        elif overbought and macd_bearish:
            return "STRONG_SELL"
        elif overbought or (macd_bearish and rsi > 50):
            return "SELL"
        else:
            return "HOLD"
    
    def execute_trades(self, market_data):
        """
        Esegue trades basati su analisi
        """
        for data in market_data:
            # Destructuring
            symbol, price, volume, rsi, macd = data
            
            # Analisi
            signal = self.analyze_market(price, volume, rsi, macd)
            
            # Esecuzione con ternary
            action = "EXECUTED" if signal != "HOLD" else "SKIPPED"
            
            # Trading logic
            if signal in ["STRONG_BUY", "BUY"]:
                # Position sizing con comprehension
                size = 0.2 if signal == "STRONG_BUY" else 0.1
                cost = self.capital * size
                
                if cost <= self.capital:
                    self.positions.append({
                        'symbol': symbol,
                        'entry': price,
                        'size': cost / price,
                        'signal': signal
                    })
                    self.capital -= cost
                    print(f"  📈 {signal}: {symbol} @ ${price}")
            
            elif signal in ["STRONG_SELL", "SELL"] and self.positions:
                # Close position
                for pos in self.positions[:]:  # Copy per modificare
                    if pos['symbol'] == symbol:
                        exit_price = price
                        pnl = (exit_price - pos['entry']) * pos['size']
                        self.capital += exit_price * pos['size']
                        
                        # Track stats
                        if pnl > 0:
                            self.winning_trades += 1
                        else:
                            self.losing_trades += 1
                        
                        self.trades_history.append({
                            'symbol': symbol,
                            'pnl': pnl,
                            'signal': signal
                        })
                        
                        self.positions.remove(pos)
                        print(f"  📉 {signal}: {symbol} @ ${price} (P&L: ${pnl:.2f})")
                        break
    
    def generate_report(self):
        """
        Genera report con comprehensions
        """
        # Stats con generator
        total_pnl = sum(t['pnl'] for t in self.trades_history)
        
        # Filter trades
        winners = [t for t in self.trades_history if t['pnl'] > 0]
        losers = [t for t in self.trades_history if t['pnl'] <= 0]
        
        # Win rate
        total_trades = len(self.trades_history)
        win_rate = (len(winners) / total_trades * 100) if total_trades > 0 else 0
        
        print("\n📊 TRADING REPORT")
        print("=" * 40)
        print(f"Initial Capital: ${10000:.2f}")
        print(f"Final Capital: ${self.capital:.2f}")
        print(f"Total P&L: ${total_pnl:.2f}")
        print(f"Total Trades: {total_trades}")
        print(f"Winners: {len(winners)}")
        print(f"Losers: {len(losers)}")
        print(f"Win Rate: {win_rate:.1f}%")
        
        # Best/Worst con condizioni
        if self.trades_history:
            best = max(self.trades_history, key=lambda x: x['pnl'])
            worst = min(self.trades_history, key=lambda x: x['pnl'])
            print(f"Best Trade: {best['symbol']} +${best['pnl']:.2f}")
            print(f"Worst Trade: {worst['symbol']} ${worst['pnl']:.2f}")


# Test del simulatore
if __name__ == "__main__":
    simulator = TradingStrategySimulator()
    
    # Market data: (symbol, price, volume, RSI, MACD)
    market_data = [
        ('BTC', 45000, 1500000, 25, 100),   # Oversold
        ('ETH', 3000, 800000, 45, 50),      # Neutral
        ('BTC', 46000, 1200000, 75, -50),   # Overbought
        ('ADA', 1.5, 500000, 80, -100),     # Very overbought
        ('ETH', 3100, 900000, 60, -20),     # Slightly overbought
    ]
    
    print("Executing trades...")
    simulator.execute_trades(market_data)
    simulator.generate_report()
    
    print("\n" + "=" * 60)
    print("🎉 COMPLIMENTI! Hai completato tutti gli esercizi Control Flow!")
    print("=" * 60)


"""
📚 RIEPILOGO CONTROL FLOW:

✅ CONDITIONALS:
- if/elif/else per decisioni
- Operatori logici: and, or, not
- Ternary operator per espressioni brevi
- Match statement (Python 3.10+)

✅ LOOPS:
- for: quando sai quante iterazioni
- while: quando dipende da condizione
- break: esci dal loop
- continue: salta iterazione
- else: eseguito se no break

✅ COMPREHENSIONS:
- List: [expr for item in iterable if condition]
- Dict: {key: val for item in iterable}
- Set: {expr for item in iterable}
- Generator: (expr for item in iterable)

🎯 BEST PRACTICES:
- Preferisci comprehensions per trasformazioni semplici
- Usa generators per grandi dataset
- Exit early con break quando possibile
- Keep it readable - non esagerare con comprehensions annidate
- Ternary operator solo per casi semplici

💡 PERFORMANCE TIPS:
- Comprehensions > manual loops
- Generators > lists per grandi dati
- Break early quando trovi risultato
- any/all per condizioni su iterabili
"""
