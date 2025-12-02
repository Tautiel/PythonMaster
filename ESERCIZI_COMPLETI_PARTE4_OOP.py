"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    🐍 PYTHON MASTER - SCHEDA ESERCIZI COMPLETA               ║
║                                                                              ║
║                    PARTE 4: OOP (Classi, Ereditarietà, Patterns)             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ==============================================================================
# SEZIONE 12: CLASSI BASE
# ==============================================================================

print("=" * 70)
print("SEZIONE 12: CLASSI BASE")
print("=" * 70)

# ------------------------------------------------------------------------------
# ESERCIZIO 12.1: Class Definition
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Crea una classe Stock con:
1. Attributi: symbol, price, quantity
2. Metodo per calcolare il valore totale
3. Metodo __repr__ per rappresentazione

💡 TEORIA:
Le classi sono blueprint per creare oggetti.
__init__ è il costruttore, self riferisce all'istanza.
__repr__ fornisce rappresentazione "ufficiale" dell'oggetto.

🎯 SKILLS: class, __init__, self, __repr__
⏱️ TEMPO: 10 minuti
🔢 LIVELLO: Principiante
"""

# ✅ SOLUZIONE:
def esercizio_12_1():
    """Class Definition - Definizione classi"""
    
    class Stock:
        """Rappresenta una posizione azionaria."""
        
        def __init__(self, symbol, price, quantity=0):
            """
            Inizializza una posizione.
            
            Args:
                symbol: Ticker dell'azione (es. "AAPL")
                price: Prezzo corrente
                quantity: Quantità posseduta (default 0)
            """
            self.symbol = symbol
            self.price = price
            self.quantity = quantity
        
        def total_value(self):
            """Calcola il valore totale della posizione."""
            return self.price * self.quantity
        
        def buy(self, qty, price=None):
            """Acquista azioni."""
            if price:
                self.price = price
            self.quantity += qty
        
        def sell(self, qty):
            """Vende azioni."""
            if qty > self.quantity:
                raise ValueError(f"Non puoi vendere {qty}, ne hai solo {self.quantity}")
            self.quantity -= qty
        
        def __repr__(self):
            """Rappresentazione ufficiale."""
            return f"Stock('{self.symbol}', price={self.price}, qty={self.quantity})"
        
        def __str__(self):
            """Rappresentazione user-friendly."""
            return f"{self.symbol}: {self.quantity} shares @ ${self.price:.2f} = ${self.total_value():.2f}"
    
    # Test della classe
    print("--- CREAZIONE STOCK ---")
    
    aapl = Stock("AAPL", 150.0, 100)
    print(f"  repr: {repr(aapl)}")
    print(f"  str:  {aapl}")
    print(f"  Valore: ${aapl.total_value():.2f}")
    
    print("\n--- OPERAZIONI ---")
    
    aapl.buy(50, 155.0)
    print(f"  Dopo buy(50, 155): {aapl}")
    
    aapl.sell(30)
    print(f"  Dopo sell(30): {aapl}")
    
    # Test errore
    try:
        aapl.sell(1000)
    except ValueError as e:
        print(f"  Errore atteso: {e}")
    
    return Stock

# 🧪 TEST:
if __name__ == "__main__":
    Stock = esercizio_12_1()
    s = Stock("TEST", 100, 10)
    assert s.total_value() == 1000
    print("✅ Esercizio 12.1 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 12.2: Class vs Instance Attributes
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Dimostra la differenza tra:
1. Attributi di classe (condivisi)
2. Attributi di istanza (specifici)
3. Metodi di classe e statici

💡 TEORIA:
- Attributi di classe: definiti nella classe, condivisi da tutte le istanze
- Attributi di istanza: definiti in __init__, specifici per istanza
- @classmethod: riceve cls, opera sulla classe
- @staticmethod: non riceve self/cls, utility function

🎯 SKILLS: class attributes, classmethod, staticmethod
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Intermedio
"""

# ✅ SOLUZIONE:
def esercizio_12_2():
    """Class vs Instance Attributes"""
    
    class Trade:
        """Rappresenta un trade con tracking globale."""
        
        # Attributi di CLASSE (condivisi)
        total_trades = 0
        commission_rate = 0.001  # 0.1%
        
        def __init__(self, symbol, side, quantity, price):
            # Attributi di ISTANZA (specifici)
            self.symbol = symbol
            self.side = side
            self.quantity = quantity
            self.price = price
            self.trade_id = Trade.total_trades
            
            # Incrementa contatore di classe
            Trade.total_trades += 1
        
        def gross_value(self):
            """Valore lordo del trade."""
            return self.quantity * self.price
        
        def commission(self):
            """Commissione basata su rate di classe."""
            return self.gross_value() * Trade.commission_rate
        
        def net_value(self):
            """Valore netto (gross - commission)."""
            return self.gross_value() - self.commission()
        
        @classmethod
        def set_commission_rate(cls, rate):
            """Cambia commission rate per TUTTI i trade."""
            cls.commission_rate = rate
            print(f"  Commission rate cambiato a {rate*100:.2f}%")
        
        @classmethod
        def get_stats(cls):
            """Statistiche globali."""
            return {
                'total_trades': cls.total_trades,
                'commission_rate': cls.commission_rate
            }
        
        @staticmethod
        def validate_side(side):
            """Valida il side di un trade (utility)."""
            valid_sides = ['BUY', 'SELL']
            if side.upper() not in valid_sides:
                raise ValueError(f"Side deve essere uno di {valid_sides}")
            return side.upper()
        
        def __repr__(self):
            return f"Trade#{self.trade_id}({self.side} {self.quantity} {self.symbol} @ {self.price})"
    
    # Test
    print("--- CLASS ATTRIBUTES ---")
    
    t1 = Trade("AAPL", "BUY", 100, 150.0)
    t2 = Trade("GOOGL", "SELL", 50, 140.0)
    t3 = Trade("MSFT", "BUY", 75, 380.0)
    
    print(f"  {t1}")
    print(f"  {t2}")
    print(f"  {t3}")
    print(f"  Totale trade creati: {Trade.total_trades}")
    
    print("\n--- INSTANCE vs CLASS ---")
    
    print(f"  t1.commission_rate: {t1.commission_rate}")
    print(f"  Trade.commission_rate: {Trade.commission_rate}")
    
    # Cambia per tutti
    Trade.set_commission_rate(0.002)
    print(f"  t1.commission_rate dopo cambio: {t1.commission_rate}")
    print(f"  t2.commission_rate dopo cambio: {t2.commission_rate}")
    
    print("\n--- CLASSMETHOD ---")
    
    stats = Trade.get_stats()
    print(f"  Stats: {stats}")
    
    print("\n--- STATICMETHOD ---")
    
    print(f"  validate_side('buy'): {Trade.validate_side('buy')}")
    try:
        Trade.validate_side('INVALID')
    except ValueError as e:
        print(f"  Errore: {e}")
    
    print("\n--- CALCOLI ---")
    
    print(f"  {t1}: Gross=${t1.gross_value():.2f}, Comm=${t1.commission():.2f}, Net=${t1.net_value():.2f}")
    
    return Trade

# 🧪 TEST:
if __name__ == "__main__":
    Trade = esercizio_12_2()
    print("✅ Esercizio 12.2 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 12.3: Properties
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Implementa una classe con:
1. Property getter
2. Property setter con validazione
3. Property computed (calcolata)
4. Property deleter

💡 TEORIA:
@property trasforma un metodo in attributo "virtuale".
Permette validazione, calcoli lazy, e encapsulation.

🎯 SKILLS: @property, getter, setter, deleter
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Intermedio
"""

# ✅ SOLUZIONE:
def esercizio_12_3():
    """Properties - Proprietà gestite"""
    
    class Position:
        """Posizione trading con properties."""
        
        def __init__(self, symbol, entry_price, quantity, stop_loss_pct=2.0):
            self.symbol = symbol
            self._entry_price = entry_price  # Underscore = "privato"
            self._quantity = quantity
            self._stop_loss_pct = stop_loss_pct
            self._current_price = entry_price
        
        # 1. PROPERTY GETTER SEMPLICE
        @property
        def entry_price(self):
            """Prezzo di entrata (read-only)."""
            return self._entry_price
        
        # 2. PROPERTY CON SETTER E VALIDAZIONE
        @property
        def quantity(self):
            """Quantità della posizione."""
            return self._quantity
        
        @quantity.setter
        def quantity(self, value):
            if value < 0:
                raise ValueError("Quantity non può essere negativa")
            self._quantity = value
        
        @property
        def current_price(self):
            """Prezzo corrente."""
            return self._current_price
        
        @current_price.setter
        def current_price(self, value):
            if value <= 0:
                raise ValueError("Prezzo deve essere positivo")
            self._current_price = value
        
        # 3. PROPERTY COMPUTED (calcolata)
        @property
        def market_value(self):
            """Valore di mercato corrente."""
            return self._current_price * self._quantity
        
        @property
        def cost_basis(self):
            """Costo totale di acquisto."""
            return self._entry_price * self._quantity
        
        @property
        def pnl(self):
            """Profit/Loss in dollari."""
            return self.market_value - self.cost_basis
        
        @property
        def pnl_percent(self):
            """Profit/Loss in percentuale."""
            if self.cost_basis == 0:
                return 0.0
            return (self.pnl / self.cost_basis) * 100
        
        @property
        def stop_loss_price(self):
            """Prezzo di stop loss."""
            return self._entry_price * (1 - self._stop_loss_pct / 100)
        
        @property
        def is_stopped_out(self):
            """True se il prezzo ha raggiunto lo stop loss."""
            return self._current_price <= self.stop_loss_price
        
        # 4. PROPERTY CON DELETER
        @property
        def stop_loss_pct(self):
            """Percentuale stop loss."""
            return self._stop_loss_pct
        
        @stop_loss_pct.setter
        def stop_loss_pct(self, value):
            if not 0 < value < 100:
                raise ValueError("Stop loss deve essere tra 0 e 100")
            self._stop_loss_pct = value
        
        @stop_loss_pct.deleter
        def stop_loss_pct(self):
            """Rimuove lo stop loss."""
            print(f"  ⚠️ Stop loss rimosso per {self.symbol}")
            self._stop_loss_pct = None
        
        def __repr__(self):
            return f"Position({self.symbol}, entry={self._entry_price}, qty={self._quantity})"
    
    # Test
    print("--- PROPERTIES BASE ---")
    
    pos = Position("AAPL", 150.0, 100)
    print(f"  Posizione: {pos}")
    print(f"  Entry price: ${pos.entry_price}")
    print(f"  Quantity: {pos.quantity}")
    
    print("\n--- SETTER CON VALIDAZIONE ---")
    
    pos.quantity = 150
    print(f"  Nuova quantity: {pos.quantity}")
    
    try:
        pos.quantity = -50
    except ValueError as e:
        print(f"  Errore validazione: {e}")
    
    print("\n--- COMPUTED PROPERTIES ---")
    
    pos.current_price = 165.0
    print(f"  Current price: ${pos.current_price}")
    print(f"  Cost basis: ${pos.cost_basis:.2f}")
    print(f"  Market value: ${pos.market_value:.2f}")
    print(f"  P&L: ${pos.pnl:.2f} ({pos.pnl_percent:+.2f}%)")
    
    print("\n--- STOP LOSS ---")
    
    print(f"  Stop loss %: {pos.stop_loss_pct}%")
    print(f"  Stop loss price: ${pos.stop_loss_price:.2f}")
    print(f"  Is stopped out: {pos.is_stopped_out}")
    
    # Simula drop
    pos.current_price = 145.0
    print(f"\n  Dopo drop a ${pos.current_price}:")
    print(f"  Is stopped out: {pos.is_stopped_out}")
    print(f"  P&L: ${pos.pnl:.2f} ({pos.pnl_percent:+.2f}%)")
    
    print("\n--- DELETER ---")
    del pos.stop_loss_pct
    
    return Position

# 🧪 TEST:
if __name__ == "__main__":
    Position = esercizio_12_3()
    p = Position("TEST", 100, 10)
    assert p.cost_basis == 1000
    p.current_price = 110
    assert p.pnl == 100
    print("✅ Esercizio 12.3 completato!\n")


# ==============================================================================
# SEZIONE 13: MAGIC METHODS
# ==============================================================================

print("\n" + "=" * 70)
print("SEZIONE 13: MAGIC METHODS")
print("=" * 70)

# ------------------------------------------------------------------------------
# ESERCIZIO 13.1: Comparison Methods
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Implementa metodi di confronto per una classe Money:
1. __eq__, __ne__ (uguaglianza)
2. __lt__, __le__, __gt__, __ge__ (ordinamento)
3. Usa @total_ordering per semplificare

💡 TEORIA:
I magic methods (dunder) definiscono comportamenti speciali.
@total_ordering genera automaticamente i metodi mancanti.

🎯 SKILLS: __eq__, __lt__, @total_ordering
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Intermedio
"""

# ✅ SOLUZIONE:
def esercizio_13_1():
    """Comparison Methods - Metodi di confronto"""
    
    from functools import total_ordering
    from decimal import Decimal
    
    @total_ordering  # Genera __le__, __gt__, __ge__ automaticamente
    class Money:
        """Rappresenta un importo monetario."""
        
        def __init__(self, amount, currency="USD"):
            self.amount = Decimal(str(amount))
            self.currency = currency
        
        def __eq__(self, other):
            """Uguaglianza."""
            if not isinstance(other, Money):
                return NotImplemented
            if self.currency != other.currency:
                raise ValueError(f"Cannot compare {self.currency} with {other.currency}")
            return self.amount == other.amount
        
        def __lt__(self, other):
            """Minore di."""
            if not isinstance(other, Money):
                return NotImplemented
            if self.currency != other.currency:
                raise ValueError(f"Cannot compare {self.currency} with {other.currency}")
            return self.amount < other.amount
        
        def __hash__(self):
            """Per usare Money in set/dict."""
            return hash((self.amount, self.currency))
        
        def __repr__(self):
            return f"Money({self.amount}, '{self.currency}')"
        
        def __str__(self):
            symbols = {'USD': '$', 'EUR': '€', 'GBP': '£'}
            symbol = symbols.get(self.currency, self.currency)
            return f"{symbol}{self.amount:.2f}"
    
    # Test
    print("--- CONFRONTI ---")
    
    m1 = Money(100, "USD")
    m2 = Money(100, "USD")
    m3 = Money(150, "USD")
    
    print(f"  {m1} == {m2}: {m1 == m2}")
    print(f"  {m1} != {m3}: {m1 != m3}")
    print(f"  {m1} < {m3}: {m1 < m3}")
    print(f"  {m1} <= {m2}: {m1 <= m2}")  # Generato da @total_ordering
    print(f"  {m3} > {m1}: {m3 > m1}")    # Generato da @total_ordering
    
    print("\n--- SORTING ---")
    
    amounts = [Money(50), Money(200), Money(75), Money(150), Money(25)]
    sorted_amounts = sorted(amounts)
    print(f"  Originale: {[str(m) for m in amounts]}")
    print(f"  Ordinato: {[str(m) for m in sorted_amounts]}")
    
    print("\n--- SET (richiede __hash__) ---")
    
    money_set = {Money(100), Money(100), Money(200)}
    print(f"  Set: {[str(m) for m in money_set]}")
    
    print("\n--- ERRORE CURRENCY ---")
    
    usd = Money(100, "USD")
    eur = Money(100, "EUR")
    try:
        result = usd < eur
    except ValueError as e:
        print(f"  Errore atteso: {e}")
    
    return Money

# 🧪 TEST:
if __name__ == "__main__":
    Money = esercizio_13_1()
    assert Money(100) == Money(100)
    assert Money(50) < Money(100)
    print("✅ Esercizio 13.1 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 13.2: Arithmetic Methods
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Implementa operazioni aritmetiche per Money:
1. __add__, __sub__ (addizione, sottrazione)
2. __mul__, __truediv__ (moltiplicazione, divisione)
3. __neg__, __abs__ (negazione, valore assoluto)
4. __radd__, __rmul__ (operazioni riflesse)

💡 TEORIA:
__radd__ viene chiamato quando l'operando sinistro non supporta l'operazione.
Es: 5 + Money(10) → Money.__radd__(5)

🎯 SKILLS: Arithmetic dunder methods, reflected operations
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Intermedio-Avanzato
"""

# ✅ SOLUZIONE:
def esercizio_13_2():
    """Arithmetic Methods - Operazioni aritmetiche"""
    
    from decimal import Decimal
    
    class Money:
        """Money con operazioni aritmetiche."""
        
        def __init__(self, amount, currency="USD"):
            self.amount = Decimal(str(amount))
            self.currency = currency
        
        def _check_currency(self, other):
            if self.currency != other.currency:
                raise ValueError(f"Currency mismatch: {self.currency} vs {other.currency}")
        
        # ADDIZIONE
        def __add__(self, other):
            if isinstance(other, Money):
                self._check_currency(other)
                return Money(self.amount + other.amount, self.currency)
            elif isinstance(other, (int, float, Decimal)):
                return Money(self.amount + Decimal(str(other)), self.currency)
            return NotImplemented
        
        def __radd__(self, other):
            """Permette: 0 + Money (utile per sum())"""
            return self.__add__(other)
        
        # SOTTRAZIONE
        def __sub__(self, other):
            if isinstance(other, Money):
                self._check_currency(other)
                return Money(self.amount - other.amount, self.currency)
            elif isinstance(other, (int, float, Decimal)):
                return Money(self.amount - Decimal(str(other)), self.currency)
            return NotImplemented
        
        # MOLTIPLICAZIONE (Money * numero)
        def __mul__(self, other):
            if isinstance(other, (int, float, Decimal)):
                return Money(self.amount * Decimal(str(other)), self.currency)
            return NotImplemented
        
        def __rmul__(self, other):
            """Permette: 2 * Money"""
            return self.__mul__(other)
        
        # DIVISIONE
        def __truediv__(self, other):
            if isinstance(other, (int, float, Decimal)):
                if other == 0:
                    raise ZeroDivisionError("Cannot divide money by zero")
                return Money(self.amount / Decimal(str(other)), self.currency)
            elif isinstance(other, Money):
                # Money / Money = ratio (numero)
                self._check_currency(other)
                return float(self.amount / other.amount)
            return NotImplemented
        
        # NEGAZIONE E VALORE ASSOLUTO
        def __neg__(self):
            return Money(-self.amount, self.currency)
        
        def __abs__(self):
            return Money(abs(self.amount), self.currency)
        
        def __pos__(self):
            return Money(+self.amount, self.currency)
        
        # IN-PLACE (opzionale, modifica l'oggetto)
        def __iadd__(self, other):
            result = self.__add__(other)
            self.amount = result.amount
            return self
        
        def __repr__(self):
            return f"Money({self.amount}, '{self.currency}')"
        
        def __str__(self):
            return f"${self.amount:.2f}"
    
    # Test
    print("--- ADDIZIONE ---")
    
    m1 = Money(100)
    m2 = Money(50)
    
    print(f"  {m1} + {m2} = {m1 + m2}")
    print(f"  {m1} + 25 = {m1 + 25}")
    print(f"  25 + {m1} = {25 + m1}")  # Usa __radd__
    
    print("\n--- SUM (usa __radd__) ---")
    
    amounts = [Money(10), Money(20), Money(30)]
    total = sum(amounts, Money(0))  # Start con Money(0)
    print(f"  sum({[str(m) for m in amounts]}) = {total}")
    
    print("\n--- SOTTRAZIONE ---")
    
    print(f"  {m1} - {m2} = {m1 - m2}")
    
    print("\n--- MOLTIPLICAZIONE ---")
    
    print(f"  {m1} * 2 = {m1 * 2}")
    print(f"  3 * {m2} = {3 * m2}")  # Usa __rmul__
    
    print("\n--- DIVISIONE ---")
    
    print(f"  {m1} / 4 = {m1 / 4}")
    print(f"  {m1} / {m2} = {m1 / m2}")  # Ratio
    
    print("\n--- NEGAZIONE E ABS ---")
    
    profit = Money(100)
    loss = Money(-50)
    
    print(f"  -{profit} = {-profit}")
    print(f"  abs({loss}) = {abs(loss)}")
    
    print("\n--- IN-PLACE ---")
    
    balance = Money(1000)
    print(f"  Balance iniziale: {balance}")
    balance += Money(500)
    print(f"  Dopo += 500: {balance}")
    
    return Money

# 🧪 TEST:
if __name__ == "__main__":
    Money = esercizio_13_2()
    assert str(Money(100) + Money(50)) == "$150.00"
    assert str(Money(100) * 2) == "$200.00"
    print("✅ Esercizio 13.2 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 13.3: Container Methods
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Implementa una classe Portfolio che si comporta come container:
1. __len__ (lunghezza)
2. __getitem__, __setitem__, __delitem__ (accesso)
3. __iter__ (iterazione)
4. __contains__ (in operator)

💡 TEORIA:
Questi metodi rendono la classe utilizzabile come lista/dict.

🎯 SKILLS: Container protocol, iteration protocol
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Intermedio-Avanzato
"""

# ✅ SOLUZIONE:
def esercizio_13_3():
    """Container Methods - Metodi container"""
    
    class Portfolio:
        """Portfolio che si comporta come container."""
        
        def __init__(self, name="My Portfolio"):
            self.name = name
            self._positions = {}  # symbol -> {'qty': int, 'price': float}
        
        # LUNGHEZZA
        def __len__(self):
            return len(self._positions)
        
        # ACCESSO CON []
        def __getitem__(self, symbol):
            if symbol not in self._positions:
                raise KeyError(f"No position in {symbol}")
            return self._positions[symbol]
        
        def __setitem__(self, symbol, position):
            """Imposta o aggiorna posizione."""
            if isinstance(position, dict):
                self._positions[symbol] = position
            elif isinstance(position, tuple) and len(position) == 2:
                qty, price = position
                self._positions[symbol] = {'qty': qty, 'price': price}
            else:
                raise ValueError("Position must be dict or (qty, price) tuple")
        
        def __delitem__(self, symbol):
            if symbol not in self._positions:
                raise KeyError(f"No position in {symbol}")
            del self._positions[symbol]
        
        # MEMBERSHIP TEST
        def __contains__(self, symbol):
            return symbol in self._positions
        
        # ITERAZIONE
        def __iter__(self):
            return iter(self._positions)
        
        def items(self):
            """Itera su (symbol, position)."""
            return self._positions.items()
        
        def values(self):
            """Itera su posizioni."""
            return self._positions.values()
        
        # CALCOLI
        @property
        def total_value(self):
            return sum(p['qty'] * p['price'] for p in self._positions.values())
        
        def __repr__(self):
            return f"Portfolio('{self.name}', {len(self)} positions)"
        
        def __str__(self):
            lines = [f"📊 {self.name} ({len(self)} positions)"]
            lines.append("-" * 40)
            for symbol, pos in self._positions.items():
                value = pos['qty'] * pos['price']
                lines.append(f"  {symbol}: {pos['qty']} @ ${pos['price']:.2f} = ${value:.2f}")
            lines.append("-" * 40)
            lines.append(f"  Total: ${self.total_value:.2f}")
            return "\n".join(lines)
    
    # Test
    print("--- CREAZIONE E ACCESSO ---")
    
    portfolio = Portfolio("Tech Portfolio")
    
    # Usa __setitem__
    portfolio["AAPL"] = (100, 150.0)
    portfolio["GOOGL"] = {'qty': 50, 'price': 140.0}
    portfolio["MSFT"] = (75, 380.0)
    
    print(f"  {repr(portfolio)}")
    print(f"  portfolio['AAPL']: {portfolio['AAPL']}")
    
    print("\n--- LEN E CONTAINS ---")
    
    print(f"  len(portfolio): {len(portfolio)}")
    print(f"  'AAPL' in portfolio: {'AAPL' in portfolio}")
    print(f"  'TSLA' in portfolio: {'TSLA' in portfolio}")
    
    print("\n--- ITERAZIONE ---")
    
    print("  Symbols:")
    for symbol in portfolio:
        print(f"    {symbol}")
    
    print("\n  Items:")
    for symbol, pos in portfolio.items():
        print(f"    {symbol}: {pos}")
    
    print("\n--- STR ---")
    print(portfolio)
    
    print("\n--- DELETE ---")
    
    del portfolio["GOOGL"]
    print(f"  Dopo delete GOOGL: {len(portfolio)} posizioni")
    
    return Portfolio

# 🧪 TEST:
if __name__ == "__main__":
    Portfolio = esercizio_13_3()
    p = Portfolio()
    p["AAPL"] = (10, 100)
    assert len(p) == 1
    assert "AAPL" in p
    print("✅ Esercizio 13.3 completato!\n")


# ==============================================================================
# SEZIONE 14: EREDITARIETÀ
# ==============================================================================

print("\n" + "=" * 70)
print("SEZIONE 14: EREDITARIETÀ")
print("=" * 70)

# ------------------------------------------------------------------------------
# ESERCIZIO 14.1: Basic Inheritance
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Crea gerarchia di ordini trading:
1. Classe base Order
2. MarketOrder, LimitOrder, StopOrder che ereditano
3. Usa super() correttamente

💡 TEORIA:
L'ereditarietà permette di riusare e specializzare codice.
super() chiama il metodo della classe parent.

🎯 SKILLS: inheritance, super(), method overriding
⏱️ TEMPO: 20 minuti
🔢 LIVELLO: Intermedio
"""

# ✅ SOLUZIONE:
def esercizio_14_1():
    """Basic Inheritance - Ereditarietà base"""
    
    from datetime import datetime
    from abc import ABC, abstractmethod
    
    class Order(ABC):
        """Classe base astratta per ordini."""
        
        _order_counter = 0
        
        def __init__(self, symbol, side, quantity):
            Order._order_counter += 1
            self.order_id = Order._order_counter
            self.symbol = symbol
            self.side = side.upper()
            self.quantity = quantity
            self.status = "PENDING"
            self.created_at = datetime.now()
            self.filled_price = None
            
            self._validate()
        
        def _validate(self):
            """Validazione base."""
            if self.side not in ['BUY', 'SELL']:
                raise ValueError(f"Invalid side: {self.side}")
            if self.quantity <= 0:
                raise ValueError("Quantity must be positive")
        
        @abstractmethod
        def get_execution_price(self, market_price):
            """Determina il prezzo di esecuzione."""
            pass
        
        @abstractmethod
        def can_execute(self, market_price):
            """Verifica se l'ordine può essere eseguito."""
            pass
        
        def execute(self, market_price):
            """Esegue l'ordine se possibile."""
            if not self.can_execute(market_price):
                return False
            
            self.filled_price = self.get_execution_price(market_price)
            self.status = "FILLED"
            return True
        
        def cancel(self):
            """Cancella l'ordine."""
            if self.status == "FILLED":
                raise ValueError("Cannot cancel filled order")
            self.status = "CANCELLED"
        
        def __repr__(self):
            return f"{self.__class__.__name__}(#{self.order_id}, {self.side} {self.quantity} {self.symbol})"
    
    
    class MarketOrder(Order):
        """Ordine market - esegue al prezzo corrente."""
        
        def __init__(self, symbol, side, quantity):
            super().__init__(symbol, side, quantity)
            self.order_type = "MARKET"
        
        def get_execution_price(self, market_price):
            return market_price
        
        def can_execute(self, market_price):
            return True  # Sempre eseguibile
    
    
    class LimitOrder(Order):
        """Ordine limit - esegue solo al prezzo specificato o migliore."""
        
        def __init__(self, symbol, side, quantity, limit_price):
            self.limit_price = limit_price
            super().__init__(symbol, side, quantity)
            self.order_type = "LIMIT"
        
        def get_execution_price(self, market_price):
            # Esegue al limite o meglio
            if self.side == "BUY":
                return min(self.limit_price, market_price)
            else:
                return max(self.limit_price, market_price)
        
        def can_execute(self, market_price):
            if self.side == "BUY":
                return market_price <= self.limit_price
            else:
                return market_price >= self.limit_price
        
        def __repr__(self):
            return f"{super().__repr__()} @ {self.limit_price}"
    
    
    class StopOrder(Order):
        """Ordine stop - si attiva quando il prezzo raggiunge lo stop."""
        
        def __init__(self, symbol, side, quantity, stop_price):
            self.stop_price = stop_price
            self.triggered = False
            super().__init__(symbol, side, quantity)
            self.order_type = "STOP"
        
        def get_execution_price(self, market_price):
            return market_price  # Esegue a market dopo trigger
        
        def can_execute(self, market_price):
            if not self.triggered:
                # Verifica trigger
                if self.side == "BUY" and market_price >= self.stop_price:
                    self.triggered = True
                elif self.side == "SELL" and market_price <= self.stop_price:
                    self.triggered = True
            return self.triggered
        
        def __repr__(self):
            status = "TRIGGERED" if self.triggered else "WAITING"
            return f"{super().__repr__()} stop={self.stop_price} [{status}]"
    
    # Test
    print("--- MARKET ORDER ---")
    
    market = MarketOrder("AAPL", "BUY", 100)
    print(f"  {market}")
    market.execute(150.0)
    print(f"  Dopo execute: status={market.status}, filled_price={market.filled_price}")
    
    print("\n--- LIMIT ORDER ---")
    
    limit_buy = LimitOrder("GOOGL", "BUY", 50, 135.0)
    print(f"  {limit_buy}")
    
    # Prezzo troppo alto
    result = limit_buy.execute(140.0)
    print(f"  Execute @ 140: {result} (prezzo > limit)")
    
    # Prezzo OK
    result = limit_buy.execute(130.0)
    print(f"  Execute @ 130: {result}, filled={limit_buy.filled_price}")
    
    print("\n--- STOP ORDER ---")
    
    stop_loss = StopOrder("MSFT", "SELL", 75, 370.0)
    print(f"  {stop_loss}")
    
    # Prezzo sopra stop
    result = stop_loss.execute(380.0)
    print(f"  Execute @ 380: {result}")
    print(f"  {stop_loss}")
    
    # Prezzo sotto stop
    result = stop_loss.execute(365.0)
    print(f"  Execute @ 365: {result}")
    print(f"  {stop_loss}")
    
    print("\n--- POLIMORFISMO ---")
    
    orders = [
        MarketOrder("AAPL", "BUY", 100),
        LimitOrder("GOOGL", "SELL", 50, 145.0),
        StopOrder("TSLA", "SELL", 25, 240.0),
    ]
    
    market_price = 145.0
    print(f"  Market price: ${market_price}")
    
    for order in orders:
        can_exec = order.can_execute(market_price)
        print(f"  {order}: can_execute={can_exec}")
    
    return Order, MarketOrder, LimitOrder, StopOrder

# 🧪 TEST:
if __name__ == "__main__":
    classes = esercizio_14_1()
    MarketOrder = classes[1]
    mo = MarketOrder("TEST", "BUY", 10)
    assert mo.can_execute(100)
    print("✅ Esercizio 14.1 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 14.2: Multiple Inheritance e MRO
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Implementa:
1. Mixin classes per funzionalità aggiuntive
2. Comprendi il Method Resolution Order (MRO)
3. Usa super() con multiple inheritance

💡 TEORIA:
Python supporta ereditarietà multipla.
MRO (C3 linearization) determina l'ordine di ricerca metodi.
I Mixin aggiungono funzionalità senza essere classi "principali".

🎯 SKILLS: Multiple inheritance, MRO, mixins
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Avanzato
"""

# ✅ SOLUZIONE:
def esercizio_14_2():
    """Multiple Inheritance e MRO"""
    
    import json
    from datetime import datetime
    
    # MIXIN CLASSES
    class TimestampMixin:
        """Aggiunge timestamp di creazione e modifica."""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.created_at = datetime.now()
            self.updated_at = datetime.now()
        
        def touch(self):
            """Aggiorna timestamp."""
            self.updated_at = datetime.now()
    
    
    class SerializableMixin:
        """Aggiunge serializzazione JSON."""
        
        def to_dict(self):
            """Converte a dizionario."""
            result = {}
            for key, value in self.__dict__.items():
                if isinstance(value, datetime):
                    result[key] = value.isoformat()
                else:
                    result[key] = value
            return result
        
        def to_json(self):
            """Converte a JSON string."""
            return json.dumps(self.to_dict(), indent=2)
        
        @classmethod
        def from_dict(cls, data):
            """Crea istanza da dizionario."""
            # Implementazione base, può essere override
            raise NotImplementedError("Subclass must implement from_dict")
    
    
    class ValidatableMixin:
        """Aggiunge validazione."""
        
        def validate(self):
            """Esegue tutte le validazioni."""
            errors = []
            
            # Chiama tutti i metodi che iniziano con _validate_
            for attr_name in dir(self):
                if attr_name.startswith('_validate_') and callable(getattr(self, attr_name)):
                    error = getattr(self, attr_name)()
                    if error:
                        errors.append(error)
            
            if errors:
                raise ValueError(f"Validation failed: {'; '.join(errors)}")
            return True
    
    
    # CLASSE CHE USA I MIXIN
    class Trade(TimestampMixin, SerializableMixin, ValidatableMixin):
        """Trade con tutte le funzionalità dei mixin."""
        
        def __init__(self, symbol, side, quantity, price):
            self.symbol = symbol
            self.side = side
            self.quantity = quantity
            self.price = price
            super().__init__()  # Chiama i mixin!
        
        def _validate_symbol(self):
            if not self.symbol or len(self.symbol) > 10:
                return "Invalid symbol"
        
        def _validate_side(self):
            if self.side not in ['BUY', 'SELL']:
                return "Side must be BUY or SELL"
        
        def _validate_quantity(self):
            if self.quantity <= 0:
                return "Quantity must be positive"
        
        def _validate_price(self):
            if self.price <= 0:
                return "Price must be positive"
        
        @property
        def value(self):
            return self.quantity * self.price
        
        def __repr__(self):
            return f"Trade({self.side} {self.quantity} {self.symbol} @ {self.price})"
    
    # Test
    print("--- MRO (Method Resolution Order) ---")
    
    print(f"  Trade MRO:")
    for cls in Trade.__mro__:
        print(f"    {cls.__name__}")
    
    print("\n--- TRADE CON MIXIN ---")
    
    trade = Trade("AAPL", "BUY", 100, 150.0)
    print(f"  {trade}")
    
    # TimestampMixin
    print(f"\n  created_at: {trade.created_at}")
    
    # ValidatableMixin
    print(f"\n  Validazione:")
    try:
        trade.validate()
        print("  ✅ Trade valido")
    except ValueError as e:
        print(f"  ❌ {e}")
    
    # SerializableMixin
    print(f"\n  JSON:")
    print(trade.to_json())
    
    print("\n--- VALIDAZIONE FALLITA ---")
    
    try:
        bad_trade = Trade("AAPL", "INVALID", -10, 0)
        bad_trade.validate()
    except ValueError as e:
        print(f"  ❌ {e}")
    
    # Diamond problem demo
    print("\n--- DIAMOND PROBLEM ---")
    
    class A:
        def method(self):
            print("    A.method")
    
    class B(A):
        def method(self):
            print("    B.method")
            super().method()
    
    class C(A):
        def method(self):
            print("    C.method")
            super().method()
    
    class D(B, C):
        def method(self):
            print("    D.method")
            super().method()
    
    print(f"  D MRO: {[c.__name__ for c in D.__mro__]}")
    print("  D().method():")
    D().method()
    
    return Trade

# 🧪 TEST:
if __name__ == "__main__":
    Trade = esercizio_14_2()
    t = Trade("AAPL", "BUY", 10, 100)
    assert hasattr(t, 'created_at')
    assert hasattr(t, 'to_json')
    print("✅ Esercizio 14.2 completato!\n")


# ==============================================================================
# SEZIONE 15: DATACLASSES E DESIGN PATTERNS
# ==============================================================================

print("\n" + "=" * 70)
print("SEZIONE 15: DATACLASSES E DESIGN PATTERNS")
print("=" * 70)

# ------------------------------------------------------------------------------
# ESERCIZIO 15.1: Dataclasses
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Usa dataclasses per creare:
1. Classe OHLC per candlestick
2. Classe con default_factory
3. Classe frozen (immutabile)

💡 TEORIA:
dataclass genera automaticamente __init__, __repr__, __eq__, ecc.
Riduce boilerplate per classi che sono principalmente "data containers".

🎯 SKILLS: @dataclass, field(), frozen, post_init
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Intermedio
"""

# ✅ SOLUZIONE:
def esercizio_15_1():
    """Dataclasses - Classi dati"""
    
    from dataclasses import dataclass, field, asdict, astuple
    from typing import List, Optional
    from datetime import datetime
    
    # 1. DATACLASS BASE
    @dataclass
    class OHLC:
        """Candlestick OHLC."""
        timestamp: datetime
        open: float
        high: float
        low: float
        close: float
        volume: int = 0
        
        @property
        def body(self):
            """Corpo della candela."""
            return abs(self.close - self.open)
        
        @property
        def is_bullish(self):
            """True se candela verde."""
            return self.close > self.open
        
        @property
        def range(self):
            """Range high-low."""
            return self.high - self.low
    
    # 2. CON DEFAULT FACTORY
    @dataclass
    class Portfolio:
        """Portfolio con default factory per liste."""
        name: str
        positions: List[str] = field(default_factory=list)
        created_at: datetime = field(default_factory=datetime.now)
        metadata: dict = field(default_factory=dict, repr=False)  # Non in repr
        
        def add_position(self, symbol):
            self.positions.append(symbol)
    
    # 3. FROZEN (immutabile)
    @dataclass(frozen=True)
    class TradeSignal:
        """Segnale trading immutabile."""
        symbol: str
        side: str
        price: float
        timestamp: datetime = field(default_factory=datetime.now)
        
        def __hash__(self):
            return hash((self.symbol, self.side, self.price))
    
    # 4. CON POST_INIT
    @dataclass
    class Position:
        """Posizione con validazione post-init."""
        symbol: str
        quantity: int
        entry_price: float
        current_price: float = field(default=0.0)
        
        # Campi calcolati
        pnl: float = field(init=False)
        pnl_percent: float = field(init=False)
        
        def __post_init__(self):
            """Chiamato dopo __init__."""
            if self.current_price == 0.0:
                self.current_price = self.entry_price
            self._update_pnl()
        
        def _update_pnl(self):
            cost = self.quantity * self.entry_price
            value = self.quantity * self.current_price
            self.pnl = value - cost
            self.pnl_percent = (self.pnl / cost * 100) if cost else 0.0
        
        def update_price(self, new_price):
            self.current_price = new_price
            self._update_pnl()
    
    # Test
    print("--- OHLC ---")
    
    candle = OHLC(
        timestamp=datetime.now(),
        open=100.0,
        high=105.0,
        low=98.0,
        close=103.0,
        volume=1000000
    )
    print(f"  {candle}")
    print(f"  Bullish: {candle.is_bullish}, Body: {candle.body}, Range: {candle.range}")
    
    print("\n--- PORTFOLIO CON DEFAULT FACTORY ---")
    
    p1 = Portfolio("Tech")
    p2 = Portfolio("Crypto")
    
    p1.add_position("AAPL")
    p1.add_position("GOOGL")
    p2.add_position("BTC")
    
    print(f"  p1: {p1}")
    print(f"  p2: {p2}")
    print(f"  (Liste separate!)")
    
    print("\n--- FROZEN (IMMUTABILE) ---")
    
    signal = TradeSignal("AAPL", "BUY", 150.0)
    print(f"  {signal}")
    
    try:
        signal.price = 160.0  # Errore!
    except Exception as e:
        print(f"  Errore modifica: {type(e).__name__}")
    
    # Può essere usato in set
    signals = {signal, TradeSignal("GOOGL", "SELL", 140.0)}
    print(f"  Set di signals: {len(signals)}")
    
    print("\n--- POSITION CON POST_INIT ---")
    
    pos = Position("AAPL", 100, 150.0)
    print(f"  {pos}")
    
    pos.update_price(165.0)
    print(f"  Dopo update a 165:")
    print(f"  P&L: ${pos.pnl:.2f} ({pos.pnl_percent:+.2f}%)")
    
    print("\n--- CONVERSIONE ---")
    
    print(f"  asdict: {asdict(candle)}")
    print(f"  astuple: {astuple(candle)}")
    
    return OHLC, Position

# 🧪 TEST:
if __name__ == "__main__":
    OHLC, Position = esercizio_15_1()
    from datetime import datetime
    c = OHLC(datetime.now(), 100, 110, 90, 105)
    assert c.is_bullish == True
    print("✅ Esercizio 15.1 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 15.2: Design Patterns - Strategy
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Implementa il pattern Strategy per strategie di trading:
1. Interfaccia Strategy
2. Implementazioni concrete (SMA, RSI, MACD)
3. Context che usa le strategie

💡 TEORIA:
Strategy pattern permette di scegliere algoritmi a runtime.
Separa l'algoritmo dal codice che lo usa.

🎯 SKILLS: Strategy pattern, ABC, dependency injection
⏱️ TEMPO: 20 minuti
🔢 LIVELLO: Avanzato
"""

# ✅ SOLUZIONE:
def esercizio_15_2():
    """Design Patterns - Strategy Pattern"""
    
    from abc import ABC, abstractmethod
    from dataclasses import dataclass
    from typing import List
    from enum import Enum
    
    class Signal(Enum):
        BUY = "BUY"
        SELL = "SELL"
        HOLD = "HOLD"
    
    # INTERFACCIA STRATEGY
    class TradingStrategy(ABC):
        """Interfaccia per strategie di trading."""
        
        @property
        @abstractmethod
        def name(self) -> str:
            pass
        
        @abstractmethod
        def generate_signal(self, prices: List[float]) -> Signal:
            """Genera segnale basato sui prezzi."""
            pass
    
    # IMPLEMENTAZIONI CONCRETE
    class SMAStrategy(TradingStrategy):
        """Strategia basata su SMA crossover."""
        
        def __init__(self, fast_period=10, slow_period=20):
            self.fast_period = fast_period
            self.slow_period = slow_period
        
        @property
        def name(self):
            return f"SMA({self.fast_period}/{self.slow_period})"
        
        def _sma(self, prices, period):
            if len(prices) < period:
                return None
            return sum(prices[-period:]) / period
        
        def generate_signal(self, prices):
            fast_sma = self._sma(prices, self.fast_period)
            slow_sma = self._sma(prices, self.slow_period)
            
            if fast_sma is None or slow_sma is None:
                return Signal.HOLD
            
            if fast_sma > slow_sma * 1.01:  # 1% sopra
                return Signal.BUY
            elif fast_sma < slow_sma * 0.99:  # 1% sotto
                return Signal.SELL
            else:
                return Signal.HOLD
    
    
    class RSIStrategy(TradingStrategy):
        """Strategia basata su RSI."""
        
        def __init__(self, period=14, oversold=30, overbought=70):
            self.period = period
            self.oversold = oversold
            self.overbought = overbought
        
        @property
        def name(self):
            return f"RSI({self.period})"
        
        def _calculate_rsi(self, prices):
            if len(prices) < self.period + 1:
                return None
            
            changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            recent_changes = changes[-self.period:]
            
            gains = [c for c in recent_changes if c > 0]
            losses = [-c for c in recent_changes if c < 0]
            
            avg_gain = sum(gains) / self.period if gains else 0
            avg_loss = sum(losses) / self.period if losses else 0.0001
            
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))
        
        def generate_signal(self, prices):
            rsi = self._calculate_rsi(prices)
            
            if rsi is None:
                return Signal.HOLD
            
            if rsi < self.oversold:
                return Signal.BUY
            elif rsi > self.overbought:
                return Signal.SELL
            else:
                return Signal.HOLD
    
    
    class MomentumStrategy(TradingStrategy):
        """Strategia momentum semplice."""
        
        def __init__(self, lookback=5, threshold=0.02):
            self.lookback = lookback
            self.threshold = threshold
        
        @property
        def name(self):
            return f"Momentum({self.lookback})"
        
        def generate_signal(self, prices):
            if len(prices) < self.lookback:
                return Signal.HOLD
            
            momentum = (prices[-1] - prices[-self.lookback]) / prices[-self.lookback]
            
            if momentum > self.threshold:
                return Signal.BUY
            elif momentum < -self.threshold:
                return Signal.SELL
            else:
                return Signal.HOLD
    
    
    # CONTEXT
    class TradingBot:
        """Bot che usa una strategia configurabile."""
        
        def __init__(self, strategy: TradingStrategy):
            self._strategy = strategy
            self._prices = []
            self._signals = []
        
        @property
        def strategy(self):
            return self._strategy
        
        @strategy.setter
        def strategy(self, strategy: TradingStrategy):
            """Permette di cambiare strategia a runtime."""
            print(f"  Strategia cambiata: {self._strategy.name} → {strategy.name}")
            self._strategy = strategy
        
        def add_price(self, price):
            self._prices.append(price)
        
        def get_signal(self):
            signal = self._strategy.generate_signal(self._prices)
            self._signals.append(signal)
            return signal
        
        def run(self, prices):
            """Esegue la strategia su una serie di prezzi."""
            signals = []
            for price in prices:
                self.add_price(price)
                signal = self.get_signal()
                signals.append((price, signal))
            return signals
    
    # Test
    print("--- STRATEGIE ---")
    
    # Prezzi simulati (trend up)
    prices = [100, 101, 99, 102, 103, 101, 104, 106, 105, 108, 
              110, 109, 112, 115, 113, 116, 118, 120, 119, 122,
              125, 123, 126, 128, 130]
    
    strategies = [
        SMAStrategy(5, 10),
        RSIStrategy(14),
        MomentumStrategy(5),
    ]
    
    for strategy in strategies:
        bot = TradingBot(strategy)
        results = bot.run(prices.copy())
        
        buys = sum(1 for _, s in results if s == Signal.BUY)
        sells = sum(1 for _, s in results if s == Signal.SELL)
        
        print(f"\n  {strategy.name}:")
        print(f"    BUY signals: {buys}")
        print(f"    SELL signals: {sells}")
        print(f"    Ultimo segnale: {results[-1][1].value}")
    
    print("\n--- CAMBIO STRATEGIA A RUNTIME ---")
    
    bot = TradingBot(SMAStrategy())
    bot.run(prices[:10])
    
    # Cambia strategia
    bot.strategy = RSIStrategy()
    bot.run(prices[10:])
    
    return TradingStrategy, TradingBot

# 🧪 TEST:
if __name__ == "__main__":
    TradingStrategy, TradingBot = esercizio_15_2()
    print("✅ Esercizio 15.2 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 15.3: Design Patterns - Observer
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Implementa il pattern Observer per:
1. Publisher di prezzi (PriceFeed)
2. Subscriber che reagiscono (AlertSystem, Logger, Trader)

💡 TEORIA:
Observer pattern permette notifiche one-to-many.
Utile per event-driven systems.

🎯 SKILLS: Observer pattern, event handling, loose coupling
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Avanzato
"""

# ✅ SOLUZIONE:
def esercizio_15_3():
    """Design Patterns - Observer Pattern"""
    
    from abc import ABC, abstractmethod
    from dataclasses import dataclass
    from datetime import datetime
    from typing import List, Dict
    
    @dataclass
    class PriceUpdate:
        """Evento di aggiornamento prezzo."""
        symbol: str
        price: float
        timestamp: datetime
    
    # INTERFACCIA OBSERVER
    class PriceObserver(ABC):
        """Interfaccia per osservatori di prezzi."""
        
        @abstractmethod
        def on_price_update(self, update: PriceUpdate):
            pass
    
    # SUBJECT (PUBLISHER)
    class PriceFeed:
        """Feed di prezzi che notifica gli observer."""
        
        def __init__(self):
            self._observers: List[PriceObserver] = []
            self._prices: Dict[str, float] = {}
        
        def subscribe(self, observer: PriceObserver):
            self._observers.append(observer)
            print(f"    + {observer.__class__.__name__} subscribed")
        
        def unsubscribe(self, observer: PriceObserver):
            self._observers.remove(observer)
            print(f"    - {observer.__class__.__name__} unsubscribed")
        
        def update_price(self, symbol: str, price: float):
            """Aggiorna prezzo e notifica observers."""
            self._prices[symbol] = price
            
            update = PriceUpdate(
                symbol=symbol,
                price=price,
                timestamp=datetime.now()
            )
            
            self._notify(update)
        
        def _notify(self, update: PriceUpdate):
            for observer in self._observers:
                observer.on_price_update(update)
    
    # OBSERVERS CONCRETI
    class PriceLogger(PriceObserver):
        """Logga tutti gli aggiornamenti."""
        
        def __init__(self):
            self.log = []
        
        def on_price_update(self, update: PriceUpdate):
            entry = f"[{update.timestamp.strftime('%H:%M:%S')}] {update.symbol}: ${update.price:.2f}"
            self.log.append(entry)
            print(f"      📝 LOG: {entry}")
    
    
    class AlertSystem(PriceObserver):
        """Genera alert basati su soglie."""
        
        def __init__(self):
            self.alerts = {}  # symbol -> {'above': price, 'below': price}
            self.triggered = []
        
        def set_alert(self, symbol, above=None, below=None):
            self.alerts[symbol] = {'above': above, 'below': below}
        
        def on_price_update(self, update: PriceUpdate):
            if update.symbol not in self.alerts:
                return
            
            alert = self.alerts[update.symbol]
            
            if alert['above'] and update.price > alert['above']:
                msg = f"🔔 ALERT: {update.symbol} above ${alert['above']}"
                print(f"      {msg}")
                self.triggered.append(msg)
            
            if alert['below'] and update.price < alert['below']:
                msg = f"🔔 ALERT: {update.symbol} below ${alert['below']}"
                print(f"      {msg}")
                self.triggered.append(msg)
    
    
    class SimpleTrader(PriceObserver):
        """Trader che reagisce ai prezzi."""
        
        def __init__(self, symbol, buy_below, sell_above):
            self.symbol = symbol
            self.buy_below = buy_below
            self.sell_above = sell_above
            self.position = 0
            self.trades = []
        
        def on_price_update(self, update: PriceUpdate):
            if update.symbol != self.symbol:
                return
            
            if update.price < self.buy_below and self.position <= 0:
                self.position = 100
                trade = f"BUY 100 {self.symbol} @ ${update.price:.2f}"
                self.trades.append(trade)
                print(f"      🟢 {trade}")
            
            elif update.price > self.sell_above and self.position > 0:
                trade = f"SELL 100 {self.symbol} @ ${update.price:.2f}"
                self.trades.append(trade)
                self.position = 0
                print(f"      🔴 {trade}")
    
    # Test
    print("--- SETUP ---")
    
    feed = PriceFeed()
    
    logger = PriceLogger()
    alerts = AlertSystem()
    trader = SimpleTrader("AAPL", buy_below=148, sell_above=155)
    
    feed.subscribe(logger)
    feed.subscribe(alerts)
    feed.subscribe(trader)
    
    alerts.set_alert("AAPL", above=154, below=147)
    
    print("\n--- PRICE UPDATES ---")
    
    prices = [150.0, 149.0, 147.5, 146.0, 148.0, 152.0, 154.5, 156.0, 155.0]
    
    for price in prices:
        print(f"\n  Update: AAPL @ ${price}")
        feed.update_price("AAPL", price)
    
    print("\n--- SUMMARY ---")
    
    print(f"  Log entries: {len(logger.log)}")
    print(f"  Alerts triggered: {len(alerts.triggered)}")
    print(f"  Trades executed: {len(trader.trades)}")
    
    return PriceFeed, PriceObserver

# 🧪 TEST:
if __name__ == "__main__":
    PriceFeed, PriceObserver = esercizio_15_3()
    print("✅ Esercizio 15.3 completato!\n")


# ==============================================================================
# RIEPILOGO SEZIONE 12-15: OOP
# ==============================================================================

print("\n" + "=" * 70)
print("RIEPILOGO: ESERCIZI OOP COMPLETATI")
print("=" * 70)

print("""
ESERCIZI COMPLETATI IN QUESTA PARTE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SEZIONE 12 - Classi Base:
  ✅ 12.1 Class Definition
  ✅ 12.2 Class vs Instance Attributes
  ✅ 12.3 Properties

SEZIONE 13 - Magic Methods:
  ✅ 13.1 Comparison Methods
  ✅ 13.2 Arithmetic Methods
  ✅ 13.3 Container Methods

SEZIONE 14 - Ereditarietà:
  ✅ 14.1 Basic Inheritance
  ✅ 14.2 Multiple Inheritance e MRO

SEZIONE 15 - Dataclasses e Design Patterns:
  ✅ 15.1 Dataclasses
  ✅ 15.2 Design Patterns - Strategy
  ✅ 15.3 Design Patterns - Observer

TOTALE QUESTA PARTE: 11 esercizi
TOTALE CUMULATIVO: 45 esercizi
""")

if __name__ == "__main__":
    print("\n🎉 TUTTI GLI ESERCIZI DELLA PARTE 4 COMPLETATI!")
