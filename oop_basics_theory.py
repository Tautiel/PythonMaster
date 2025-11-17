"""
🎓 OBJECT-ORIENTED PROGRAMMING BASICS
Modulo completo per imparare OOP dalle fondamenta
QUESTO MANCAVA NEL MATERIALE - ESSENZIALE PRIMA DI ADVANCED OOP!
"""

# ============================================
# PARTE 1: INTRODUZIONE ALL'OOP
# ============================================

"""
PERCHÉ OOP?

Prima di OOP, avevamo solo programmazione procedurale:
- Funzioni che operano su dati
- Dati e comportamenti separati
- Difficile mantenere codice complesso

OOP unisce dati (attributi) e comportamenti (metodi) in oggetti.

CONCETTI FONDAMENTALI:
1. Classe: Il "blueprint" o template
2. Oggetto: Un'istanza della classe
3. Attributi: Le variabili dell'oggetto
4. Metodi: Le funzioni dell'oggetto
5. Incapsulamento: Nascondere dettagli interni
6. Ereditarietà: Classi che estendono altre classi
7. Polimorfismo: Stesso metodo, comportamenti diversi
"""

# ============================================
# PARTE 2: LA TUA PRIMA CLASSE
# ============================================

print("=" * 60)
print("LEZIONE 1: CREARE UNA CLASSE")
print("=" * 60)

# Classe più semplice possibile
class Dog:
    pass

# Creare un'istanza
my_dog = Dog()
print(f"Tipo di my_dog: {type(my_dog)}")
print(f"È un Dog? {isinstance(my_dog, Dog)}")

# Classe con attributi
class Cat:
    # Attributi di classe (condivisi da tutte le istanze)
    species = "Felis catus"
    
    def __init__(self, name, age):
        # Attributi di istanza (unici per ogni oggetto)
        self.name = name
        self.age = age

# Creare istanze
cat1 = Cat("Micio", 3)
cat2 = Cat("Felix", 5)

print(f"\ncat1: {cat1.name}, {cat1.age} anni")
print(f"cat2: {cat2.name}, {cat2.age} anni")
print(f"Species (condivisa): {cat1.species}")

# ============================================
# PARTE 3: IL METODO __init__
# ============================================

print("\n" + "=" * 60)
print("LEZIONE 2: CONSTRUCTOR __init__")
print("=" * 60)

class TradingAccount:
    """
    Esempio pratico: Account per trading
    """
    def __init__(self, owner, initial_balance=0):
        """
        Il constructor viene chiamato automaticamente
        quando crei una nuova istanza
        
        Args:
            owner: Nome del proprietario
            initial_balance: Bilancio iniziale (default 0)
        """
        self.owner = owner
        self.balance = initial_balance
        self.transactions = []  # Lista vuota per storico
        self.created_at = "2024-01-01"  # Data creazione
        
        # Log creazione account
        print(f"✅ Account creato per {owner} con ${initial_balance}")

# Creare accounts
account1 = TradingAccount("Marco", 10000)
account2 = TradingAccount("Luigi")  # Usa default balance

print(f"\nAccount 1: {account1.owner} - ${account1.balance}")
print(f"Account 2: {account2.owner} - ${account2.balance}")

# ============================================
# PARTE 4: SELF - IL RIFERIMENTO ALL'ISTANZA
# ============================================

print("\n" + "=" * 60)
print("LEZIONE 3: CAPIRE SELF")
print("=" * 60)

"""
SELF è il riferimento all'istanza corrente.
- È SEMPRE il primo parametro dei metodi
- Python lo passa automaticamente
- Permette di accedere agli attributi dell'oggetto
"""

class Counter:
    def __init__(self):
        self.count = 0  # 'self' riferisce a questa istanza
    
    def increment(self):
        self.count += 1  # Modifica l'attributo di QUESTA istanza
        return self.count
    
    def reset(self):
        self.count = 0

# Due counter indipendenti
counter1 = Counter()
counter2 = Counter()

counter1.increment()
counter1.increment()
counter2.increment()

print(f"Counter1: {counter1.count}")  # 2
print(f"Counter2: {counter2.count}")  # 1
print("→ Ogni istanza ha il proprio stato!")

# ============================================
# PARTE 5: METODI - COMPORTAMENTI DEGLI OGGETTI
# ============================================

print("\n" + "=" * 60)
print("LEZIONE 4: METODI")
print("=" * 60)

class BankAccount:
    """
    Account bancario con metodi completi
    """
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance
        self.transactions = []
    
    def deposit(self, amount):
        """Deposita denaro"""
        if amount <= 0:
            print("❌ Importo deve essere positivo")
            return False
        
        self.balance += amount
        self.transactions.append(f"Deposito: +${amount}")
        print(f"✅ Depositati ${amount}. Nuovo bilancio: ${self.balance}")
        return True
    
    def withdraw(self, amount):
        """Preleva denaro"""
        if amount <= 0:
            print("❌ Importo deve essere positivo")
            return False
        
        if amount > self.balance:
            print(f"❌ Fondi insufficienti! Hai solo ${self.balance}")
            return False
        
        self.balance -= amount
        self.transactions.append(f"Prelievo: -${amount}")
        print(f"✅ Prelevati ${amount}. Nuovo bilancio: ${self.balance}")
        return True
    
    def get_balance(self):
        """Ritorna il bilancio attuale"""
        return self.balance
    
    def show_transactions(self):
        """Mostra lo storico transazioni"""
        print(f"\n📜 Transazioni di {self.owner}:")
        for t in self.transactions:
            print(f"  - {t}")
        print(f"  Bilancio finale: ${self.balance}")

# Test del BankAccount
marco_account = BankAccount("Marco", 1000)
marco_account.deposit(500)
marco_account.withdraw(200)
marco_account.withdraw(2000)  # Fallisce
marco_account.show_transactions()

# ============================================
# PARTE 6: ATTRIBUTI VS METODI
# ============================================

print("\n" + "=" * 60)
print("LEZIONE 5: ATTRIBUTI VS METODI")
print("=" * 60)

class Car:
    def __init__(self, brand, model, year):
        # ATTRIBUTI: Dati/Stato dell'oggetto
        self.brand = brand
        self.model = model
        self.year = year
        self.speed = 0
        self.engine_on = False
    
    # METODI: Azioni/Comportamenti dell'oggetto
    def start(self):
        """Avvia il motore"""
        if not self.engine_on:
            self.engine_on = True
            print(f"🚗 {self.brand} {self.model} avviata!")
        else:
            print("🚗 Auto già avviata")
    
    def accelerate(self, increment=10):
        """Accelera l'auto"""
        if not self.engine_on:
            print("❌ Devi prima avviare l'auto!")
            return
        
        self.speed += increment
        print(f"🚗 Velocità: {self.speed} km/h")
    
    def brake(self):
        """Frena completamente"""
        self.speed = 0
        print("🚗 Auto ferma")
    
    def info(self):
        """Mostra informazioni auto"""
        status = "Accesa" if self.engine_on else "Spenta"
        return f"{self.brand} {self.model} ({self.year}) - {status} - {self.speed} km/h"

# Test Car
my_car = Car("Tesla", "Model 3", 2023)
print(my_car.info())
my_car.accelerate()  # Fallisce - motore spento
my_car.start()
my_car.accelerate()
my_car.accelerate(30)
print(my_car.info())

# ============================================
# PARTE 7: ENCAPSULATION - PUBLIC VS PRIVATE
# ============================================

print("\n" + "=" * 60)
print("LEZIONE 6: ENCAPSULATION")
print("=" * 60)

"""
In Python:
- public: accessibile ovunque (normale)
- _protected: convenzione, uso interno (single underscore)
- __private: name mangling (double underscore)
"""

class SecureAccount:
    def __init__(self, owner, pin):
        # Pubblico
        self.owner = owner
        
        # Protected (convenzione)
        self._balance = 0
        
        # Private (name mangling)
        self.__pin = pin
    
    def _internal_method(self):
        """Metodo protected - uso interno"""
        print("Questo è un metodo interno")
    
    def __private_method(self):
        """Metodo private - molto privato"""
        print("Questo è molto privato")
    
    def verify_pin(self, pin):
        """Metodo pubblico per verificare PIN"""
        return pin == self.__pin
    
    def deposit(self, amount, pin):
        """Deposita solo con PIN corretto"""
        if self.verify_pin(pin):
            self._balance += amount
            print(f"✅ Depositati ${amount}")
        else:
            print("❌ PIN errato!")

# Test encapsulation
secure = SecureAccount("Marco", "1234")
print(f"Owner (pubblico): {secure.owner}")
print(f"Balance (protected): {secure._balance}")  # Funziona ma non dovremmo
# print(secure.__pin)  # AttributeError! 
# print(secure._SecureAccount__pin)  # Name mangling - funziona ma NON FARE!

secure.deposit(1000, "0000")  # PIN errato
secure.deposit(1000, "1234")  # PIN corretto

# ============================================
# PARTE 8: CLASS ATTRIBUTES VS INSTANCE ATTRIBUTES
# ============================================

print("\n" + "=" * 60)
print("LEZIONE 7: ATTRIBUTI DI CLASSE VS ISTANZA")
print("=" * 60)

class CryptoAsset:
    # Attributo di CLASSE (condiviso)
    exchange = "Binance"
    total_assets = 0
    
    def __init__(self, symbol, quantity):
        # Attributi di ISTANZA (unici)
        self.symbol = symbol
        self.quantity = quantity
        
        # Modifica attributo di classe
        CryptoAsset.total_assets += 1
    
    @classmethod
    def change_exchange(cls, new_exchange):
        """Metodo di classe - modifica attributo di classe"""
        cls.exchange = new_exchange
        print(f"📍 Exchange cambiato a {new_exchange}")

# Test
btc = CryptoAsset("BTC", 0.5)
eth = CryptoAsset("ETH", 10)

print(f"BTC exchange: {btc.exchange}")
print(f"ETH exchange: {eth.exchange}")
print(f"Assets totali: {CryptoAsset.total_assets}")

# Cambio exchange per TUTTI
CryptoAsset.change_exchange("Kraken")
print(f"BTC exchange dopo: {btc.exchange}")
print(f"ETH exchange dopo: {eth.exchange}")

# ============================================
# PARTE 9: INHERITANCE - EREDITARIETÀ BASE
# ============================================

print("\n" + "=" * 60)
print("LEZIONE 8: INHERITANCE BASE")
print("=" * 60)

# Classe base (parent/superclass)
class Animal:
    def __init__(self, name, species):
        self.name = name
        self.species = species
        self.alive = True
    
    def eat(self):
        print(f"{self.name} sta mangiando")
    
    def sleep(self):
        print(f"{self.name} sta dormendo")
    
    def make_sound(self):
        print(f"{self.name} fa un verso")

# Classe derivata (child/subclass)
class Dog(Animal):
    def __init__(self, name, breed):
        # Chiama il constructor del parent
        super().__init__(name, "Canis familiaris")
        self.breed = breed
    
    def make_sound(self):
        """Override del metodo parent"""
        print(f"{self.name} abbaia: Woof woof!")
    
    def fetch(self):
        """Metodo specifico del Dog"""
        print(f"{self.name} riporta la palla")

# Altra classe derivata
class Cat(Animal):
    def __init__(self, name, color):
        super().__init__(name, "Felis catus")
        self.color = color
    
    def make_sound(self):
        """Override del metodo parent"""
        print(f"{self.name} miagola: Miao miao!")
    
    def scratch(self):
        """Metodo specifico del Cat"""
        print(f"{self.name} graffia il divano")

# Test inheritance
dog = Dog("Buddy", "Golden Retriever")
cat = Cat("Whiskers", "nero")

# Metodi ereditati
dog.eat()
cat.sleep()

# Metodi override
dog.make_sound()
cat.make_sound()

# Metodi specifici
dog.fetch()
cat.scratch()

# Controllo tipo
print(f"\ndog è Animal? {isinstance(dog, Animal)}")  # True
print(f"dog è Dog? {isinstance(dog, Dog)}")  # True
print(f"dog è Cat? {isinstance(dog, Cat)}")  # False

# ============================================
# PARTE 10: ESEMPIO PRATICO - TRADING SYSTEM
# ============================================

print("\n" + "=" * 60)
print("LEZIONE 9: ESEMPIO COMPLETO - TRADING SYSTEM")
print("=" * 60)

class Position:
    """Rappresenta una posizione di trading"""
    
    def __init__(self, symbol, entry_price, quantity, position_type="LONG"):
        self.symbol = symbol
        self.entry_price = entry_price
        self.quantity = quantity
        self.position_type = position_type
        self.exit_price = None
        self.is_open = True
        self.pnl = 0
    
    def close(self, exit_price):
        """Chiude la posizione"""
        if not self.is_open:
            print("❌ Posizione già chiusa")
            return
        
        self.exit_price = exit_price
        self.is_open = False
        
        if self.position_type == "LONG":
            self.pnl = (exit_price - self.entry_price) * self.quantity
        else:  # SHORT
            self.pnl = (self.entry_price - exit_price) * self.quantity
        
        print(f"📊 Posizione chiusa: P&L = ${self.pnl:.2f}")
        return self.pnl
    
    def current_value(self, current_price):
        """Calcola valore attuale"""
        if self.position_type == "LONG":
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity
    
    def info(self):
        """Info sulla posizione"""
        status = "APERTA" if self.is_open else "CHIUSA"
        return (f"{self.symbol} {self.position_type} - "
                f"Entry: ${self.entry_price} x {self.quantity} - "
                f"Status: {status} - P&L: ${self.pnl:.2f}")


class TradingBot:
    """Bot di trading semplice"""
    
    def __init__(self, name, initial_capital):
        self.name = name
        self.capital = initial_capital
        self.positions = []
        self.closed_positions = []
        self.total_pnl = 0
    
    def open_position(self, symbol, price, quantity, position_type="LONG"):
        """Apre una nuova posizione"""
        cost = price * quantity
        if cost > self.capital:
            print(f"❌ Capitale insufficiente! Hai ${self.capital:.2f}")
            return None
        
        position = Position(symbol, price, quantity, position_type)
        self.positions.append(position)
        self.capital -= cost
        
        print(f"✅ Aperta posizione {position_type} su {symbol}")
        print(f"   Entry: ${price} x {quantity} = ${cost:.2f}")
        print(f"   Capitale rimanente: ${self.capital:.2f}")
        
        return position
    
    def close_position(self, position, exit_price):
        """Chiude una posizione"""
        if position not in self.positions:
            print("❌ Posizione non trovata")
            return
        
        pnl = position.close(exit_price)
        self.capital += (position.exit_price * position.quantity) + pnl
        self.total_pnl += pnl
        
        self.positions.remove(position)
        self.closed_positions.append(position)
        
        print(f"   Capitale dopo chiusura: ${self.capital:.2f}")
    
    def show_portfolio(self):
        """Mostra portfolio attuale"""
        print(f"\n📊 PORTFOLIO {self.name}")
        print("=" * 50)
        print(f"Capitale: ${self.capital:.2f}")
        print(f"P&L Totale: ${self.total_pnl:.2f}")
        
        print("\nPosizioni Aperte:")
        if not self.positions:
            print("  Nessuna")
        for pos in self.positions:
            print(f"  - {pos.info()}")
        
        print("\nPosizioni Chiuse:")
        if not self.closed_positions:
            print("  Nessuna")
        for pos in self.closed_positions:
            print(f"  - {pos.info()}")


# Test del trading system
bot = TradingBot("ScalpBot", 10000)

# Apri alcune posizioni
pos1 = bot.open_position("BTC", 45000, 0.1)
pos2 = bot.open_position("ETH", 3000, 1, "SHORT")

# Mostra portfolio
bot.show_portfolio()

# Chiudi posizioni
print("\n--- Chiusura Posizioni ---")
bot.close_position(pos1, 46000)  # Profit su LONG
bot.close_position(pos2, 2900)   # Profit su SHORT

# Portfolio finale
bot.show_portfolio()

# ============================================
# PARTE 11: SPECIAL METHODS (MAGIC METHODS)
# ============================================

print("\n" + "=" * 60)
print("LEZIONE 10: MAGIC METHODS")
print("=" * 60)

"""
Magic methods permettono di definire comportamenti speciali:
__init__ - Constructor
__str__ - String representation (per umani)
__repr__ - String representation (per debug)
__len__ - Lunghezza
__eq__ - Uguaglianza (==)
__lt__ - Less than (<)
__add__ - Addizione (+)
"""

class Trade:
    def __init__(self, symbol, price, quantity, trade_type):
        self.symbol = symbol
        self.price = price
        self.quantity = quantity
        self.trade_type = trade_type
        self.value = price * quantity
    
    def __str__(self):
        """Rappresentazione user-friendly"""
        return f"{self.trade_type} {self.quantity} {self.symbol} @ ${self.price}"
    
    def __repr__(self):
        """Rappresentazione per debug"""
        return f"Trade('{self.symbol}', {self.price}, {self.quantity}, '{self.trade_type}')"
    
    def __eq__(self, other):
        """Confronto uguaglianza"""
        if not isinstance(other, Trade):
            return False
        return (self.symbol == other.symbol and 
                self.price == other.price and
                self.quantity == other.quantity)
    
    def __lt__(self, other):
        """Confronto per ordinamento (per valore)"""
        return self.value < other.value
    
    def __add__(self, other):
        """Somma due trade dello stesso simbolo"""
        if self.symbol != other.symbol:
            raise ValueError("Non posso sommare trade di simboli diversi")
        
        avg_price = (self.value + other.value) / (self.quantity + other.quantity)
        total_qty = self.quantity + other.quantity
        return Trade(self.symbol, avg_price, total_qty, self.trade_type)

# Test magic methods
trade1 = Trade("BTC", 45000, 0.5, "BUY")
trade2 = Trade("BTC", 46000, 0.3, "BUY")
trade3 = Trade("ETH", 3000, 2, "BUY")

print(f"Trade 1: {trade1}")  # Usa __str__
print(f"Trade 1 repr: {repr(trade1)}")  # Usa __repr__

print(f"\nTrade1 == Trade2? {trade1 == trade2}")
print(f"Trade1 < Trade3? {trade1 < trade3}")

# Somma trades
combined = trade1 + trade2
print(f"\nTrade combinato: {combined}")

# Ordinamento
trades = [trade3, trade1, trade2]
trades.sort()  # Usa __lt__
print("\nTrades ordinati per valore:")
for t in trades:
    print(f"  - {t} (valore: ${t.value:.2f})")

# ============================================
# PARTE 12: PROPERTY DECORATOR
# ============================================

print("\n" + "=" * 60)
print("LEZIONE 11: PROPERTY DECORATOR")
print("=" * 60)

"""
@property permette di:
- Trasformare metodi in attributi
- Controllare accesso agli attributi
- Validare valori
"""

class Portfolio:
    def __init__(self):
        self._assets = {}  # {symbol: quantity}
        self._prices = {}  # {symbol: price}
    
    def add_asset(self, symbol, quantity, price):
        """Aggiunge asset al portfolio"""
        if symbol in self._assets:
            self._assets[symbol] += quantity
        else:
            self._assets[symbol] = quantity
        self._prices[symbol] = price
    
    @property
    def value(self):
        """Calcola valore totale (come attributo)"""
        total = 0
        for symbol, qty in self._assets.items():
            price = self._prices.get(symbol, 0)
            total += qty * price
        return total
    
    @property
    def assets(self):
        """Ritorna copia degli assets (read-only)"""
        return self._assets.copy()
    
    @property
    def symbols(self):
        """Lista dei simboli nel portfolio"""
        return list(self._assets.keys())
    
    def update_price(self, symbol, new_price):
        """Aggiorna prezzo di un asset"""
        if symbol in self._prices:
            self._prices[symbol] = new_price
            print(f"✅ Prezzo {symbol} aggiornato a ${new_price}")

# Test properties
portfolio = Portfolio()
portfolio.add_asset("BTC", 0.5, 45000)
portfolio.add_asset("ETH", 5, 3000)

print(f"Valore portfolio: ${portfolio.value:,.2f}")  # Come attributo!
print(f"Simboli: {portfolio.symbols}")
print(f"Assets: {portfolio.assets}")

# Aggiorna prezzi
portfolio.update_price("BTC", 46000)
print(f"Nuovo valore: ${portfolio.value:,.2f}")  # Ricalcolato automaticamente!

# ============================================
# PARTE 13: STATIC METHODS E CLASS METHODS
# ============================================

print("\n" + "=" * 60)
print("LEZIONE 12: STATIC E CLASS METHODS")
print("=" * 60)

class TradingUtils:
    """Utilities per trading"""
    
    commission_rate = 0.001  # 0.1% commissione
    
    @staticmethod
    def calculate_position_size(capital, risk_percent, stop_loss_percent):
        """
        Static method - non usa self né cls
        Calcola dimensione posizione basata su risk management
        """
        risk_amount = capital * (risk_percent / 100)
        position_size = risk_amount / (stop_loss_percent / 100)
        return position_size
    
    @classmethod
    def calculate_commission(cls, trade_value):
        """
        Class method - usa cls per accedere ad attributi di classe
        """
        return trade_value * cls.commission_rate
    
    @classmethod
    def update_commission_rate(cls, new_rate):
        """Aggiorna commissione per tutti"""
        cls.commission_rate = new_rate
        print(f"✅ Commissione aggiornata a {new_rate * 100:.2f}%")

# Test static method (non serve istanza)
size = TradingUtils.calculate_position_size(10000, 2, 5)
print(f"Position size: ${size:.2f}")

# Test class method
commission = TradingUtils.calculate_commission(1000)
print(f"Commissione su $1000: ${commission:.2f}")

# Modifica rate per tutti
TradingUtils.update_commission_rate(0.002)
new_commission = TradingUtils.calculate_commission(1000)
print(f"Nuova commissione su $1000: ${new_commission:.2f}")

# ============================================
# RIEPILOGO E BEST PRACTICES
# ============================================

print("\n" + "=" * 60)
print("RIEPILOGO OOP BASICS")
print("=" * 60)

"""
✅ HAI IMPARATO:

1. CLASSI E OGGETTI
   - Creare classi con 'class'
   - Istanziare oggetti
   - Differenza classe vs istanza

2. __init__ E self
   - Constructor per inizializzare
   - self riferisce all'istanza corrente
   - Parametri con defaults

3. ATTRIBUTI E METODI
   - Attributi = dati/stato
   - Metodi = comportamenti/azioni
   - Attributi di classe vs istanza

4. ENCAPSULATION
   - Public (normale)
   - Protected (_single)
   - Private (__double)

5. INHERITANCE
   - Extends functionality
   - super() per chiamare parent
   - Override di metodi

6. MAGIC METHODS
   - __str__, __repr__
   - __eq__, __lt__
   - Personalizzare comportamenti

7. PROPERTY DECORATOR
   - Metodi che sembrano attributi
   - Computed properties
   - Controllo accesso

8. STATIC/CLASS METHODS
   - @staticmethod - no self/cls
   - @classmethod - usa cls
   - Utilities e factory methods

📚 BEST PRACTICES:

1. Nomi classi in CamelCase: TradingBot, BankAccount
2. Nomi metodi/attributi in snake_case: get_balance, total_value
3. Usa docstrings per documentare
4. Keep it simple - non over-ingegnerizzare
5. Favor composition over inheritance
6. Una classe = una responsabilità
7. Metodi brevi e focalizzati
8. Validazione nell'__init__
9. Properties per computed values
10. Magic methods con cautela

🚀 PROSSIMI PASSI:

Ora sei pronto per:
- Design patterns
- Abstract classes
- Multiple inheritance
- Metaclasses (advanced)
- Decorators avanzati

Ma PRIMA consolida queste basi con gli esercizi!
"""

print("\n" + "=" * 60)
print("🎉 COMPLIMENTI! Hai completato OOP BASICS!")
print("Ora fai gli esercizi per consolidare!")
print("=" * 60)
