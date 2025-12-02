"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    🐍 PYTHON MASTER - SCHEDA ESERCIZI COMPLETA               ║
║                                                                              ║
║                    PARTE 7: TESTING (pytest, TDD, Mocking)                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ==============================================================================
# SEZIONE 20: PYTEST BASICS
# ==============================================================================

print("=" * 70)
print("SEZIONE 20: PYTEST BASICS")
print("=" * 70)

# ------------------------------------------------------------------------------
# ESERCIZIO 20.1: Test Functions
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Scrivi test per funzioni:
1. Test base con assert
2. Naming conventions
3. Organizzazione test

💡 TEORIA:
pytest usa assert nativo di Python.
I test devono iniziare con test_ o finire con _test.
Un file test = un modulo testato.

🎯 SKILLS: pytest basics, assert, test organization
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Intermedio

NOTA: Questi test possono essere eseguiti con: pytest nome_file.py
"""

# ✅ CODICE DA TESTARE:
def calculate_pnl(entry_price, exit_price, quantity, side='LONG'):
    """Calcola Profit & Loss di un trade."""
    if quantity <= 0:
        raise ValueError("Quantity must be positive")
    if entry_price <= 0 or exit_price <= 0:
        raise ValueError("Prices must be positive")
    
    if side.upper() == 'LONG':
        return (exit_price - entry_price) * quantity
    elif side.upper() == 'SHORT':
        return (entry_price - exit_price) * quantity
    else:
        raise ValueError(f"Invalid side: {side}")


def calculate_position_size(capital, risk_percent, entry_price, stop_loss_price):
    """Calcola position size basato sul rischio."""
    if capital <= 0:
        raise ValueError("Capital must be positive")
    if not 0 < risk_percent <= 100:
        raise ValueError("Risk percent must be between 0 and 100")
    
    risk_per_share = abs(entry_price - stop_loss_price)
    if risk_per_share == 0:
        return 0
    
    risk_amount = capital * (risk_percent / 100)
    return int(risk_amount / risk_per_share)


# ✅ TEST:
def test_calculate_pnl_long_profit():
    """Test PnL per trade long in profitto."""
    result = calculate_pnl(100, 110, 10, 'LONG')
    assert result == 100  # (110-100) * 10


def test_calculate_pnl_long_loss():
    """Test PnL per trade long in perdita."""
    result = calculate_pnl(100, 90, 10, 'LONG')
    assert result == -100  # (90-100) * 10


def test_calculate_pnl_short_profit():
    """Test PnL per trade short in profitto."""
    result = calculate_pnl(100, 90, 10, 'SHORT')
    assert result == 100  # (100-90) * 10


def test_calculate_pnl_short_loss():
    """Test PnL per trade short in perdita."""
    result = calculate_pnl(100, 110, 10, 'SHORT')
    assert result == -100


def test_calculate_pnl_case_insensitive():
    """Test che side sia case insensitive."""
    assert calculate_pnl(100, 110, 10, 'long') == 100
    assert calculate_pnl(100, 110, 10, 'LONG') == 100
    assert calculate_pnl(100, 110, 10, 'Long') == 100


def test_calculate_pnl_invalid_quantity():
    """Test errore per quantity invalida."""
    import pytest
    # Questa sintassi richiede pytest
    # with pytest.raises(ValueError):
    #     calculate_pnl(100, 110, 0, 'LONG')
    
    try:
        calculate_pnl(100, 110, 0, 'LONG')
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected


def test_position_size_basic():
    """Test calcolo position size base."""
    # Capital 10000, risk 1%, entry 100, SL 95 (risk $5 per share)
    # Risk amount = 10000 * 0.01 = 100
    # Position size = 100 / 5 = 20 shares
    result = calculate_position_size(10000, 1.0, 100, 95)
    assert result == 20


def test_position_size_no_risk():
    """Test quando entry == stop loss."""
    result = calculate_position_size(10000, 1.0, 100, 100)
    assert result == 0


def esercizio_20_1():
    """Esegui test manualmente per demo."""
    print("--- PYTEST TEST FUNCTIONS ---")
    
    tests = [
        test_calculate_pnl_long_profit,
        test_calculate_pnl_long_loss,
        test_calculate_pnl_short_profit,
        test_calculate_pnl_short_loss,
        test_calculate_pnl_case_insensitive,
        test_calculate_pnl_invalid_quantity,
        test_position_size_basic,
        test_position_size_no_risk,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            print(f"  ✅ {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ {test.__name__}: {type(e).__name__}: {e}")
            failed += 1
    
    print(f"\n  Risultato: {passed} passed, {failed} failed")
    return passed, failed


# 🧪 ESECUZIONE:
if __name__ == "__main__":
    esercizio_20_1()
    print("✅ Esercizio 20.1 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 20.2: Fixtures
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Implementa fixtures per:
1. Setup dati di test
2. Fixture con scope diversi
3. Fixture parametrizzate

💡 TEORIA:
Le fixtures forniscono dati/oggetti riusabili ai test.
Scopes: function (default), class, module, session.
yield per setup/teardown.

🎯 SKILLS: fixtures, scope, yield fixtures
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Intermedio
"""

# ✅ SIMULAZIONE FIXTURES (senza pytest decorator):
class Portfolio:
    """Classe da testare."""
    
    def __init__(self, initial_balance):
        self.balance = initial_balance
        self.positions = {}
    
    def buy(self, symbol, quantity, price):
        cost = quantity * price
        if cost > self.balance:
            raise ValueError("Insufficient funds")
        self.balance -= cost
        self.positions[symbol] = self.positions.get(symbol, 0) + quantity
    
    def sell(self, symbol, quantity, price):
        if symbol not in self.positions or self.positions[symbol] < quantity:
            raise ValueError("Insufficient shares")
        self.positions[symbol] -= quantity
        self.balance += quantity * price
    
    def total_value(self, prices):
        value = self.balance
        for symbol, qty in self.positions.items():
            if symbol in prices:
                value += qty * prices[symbol]
        return value


# Fixture simulata
def portfolio_fixture():
    """Crea portfolio di test."""
    return Portfolio(10000)


def sample_prices():
    """Prezzi di esempio."""
    return {
        'AAPL': 150.0,
        'GOOGL': 140.0,
        'MSFT': 380.0,
    }


# Test che usano le "fixtures"
def test_portfolio_buy():
    portfolio = portfolio_fixture()
    prices = sample_prices()
    
    portfolio.buy('AAPL', 10, prices['AAPL'])
    
    assert portfolio.positions['AAPL'] == 10
    assert portfolio.balance == 10000 - (10 * 150)


def test_portfolio_sell():
    portfolio = portfolio_fixture()
    prices = sample_prices()
    
    portfolio.buy('AAPL', 10, prices['AAPL'])
    portfolio.sell('AAPL', 5, prices['AAPL'] + 10)
    
    assert portfolio.positions['AAPL'] == 5
    assert portfolio.balance == 10000 - 1500 + 800  # -1500 buy, +800 sell


def test_portfolio_insufficient_funds():
    portfolio = portfolio_fixture()
    
    try:
        portfolio.buy('AAPL', 1000, 150)  # Cost = 150000 > 10000
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "Insufficient funds" in str(e)


def test_portfolio_total_value():
    portfolio = portfolio_fixture()
    prices = sample_prices()
    
    portfolio.buy('AAPL', 10, prices['AAPL'])  # -1500
    portfolio.buy('GOOGL', 5, prices['GOOGL'])  # -700
    
    # Balance: 10000 - 1500 - 700 = 7800
    # Positions: 10*150 + 5*140 = 1500 + 700 = 2200
    # Total: 7800 + 2200 = 10000
    
    total = portfolio.total_value(prices)
    assert total == 10000


def esercizio_20_2():
    """Esegui test con fixtures."""
    print("--- FIXTURES ---")
    
    tests = [
        test_portfolio_buy,
        test_portfolio_sell,
        test_portfolio_insufficient_funds,
        test_portfolio_total_value,
    ]
    
    passed = 0
    for test in tests:
        try:
            test()
            print(f"  ✅ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {test.__name__}: {e}")
    
    print(f"\n  Risultato: {passed}/{len(tests)} passed")
    
    # Esempio di come sarebbero le fixtures con pytest
    print("\n  Esempio fixtures pytest:")
    print("""
    @pytest.fixture
    def portfolio():
        return Portfolio(10000)
    
    @pytest.fixture
    def sample_prices():
        return {'AAPL': 150.0, 'GOOGL': 140.0}
    
    def test_buy(portfolio, sample_prices):
        portfolio.buy('AAPL', 10, sample_prices['AAPL'])
        assert portfolio.positions['AAPL'] == 10
    """)
    
    return passed


if __name__ == "__main__":
    esercizio_20_2()
    print("✅ Esercizio 20.2 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 20.3: Parametrized Tests
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Implementa test parametrizzati:
1. Test con input multipli
2. Test con expected output
3. Test con ID descrittivi

💡 TEORIA:
@pytest.mark.parametrize esegue lo stesso test con input diversi.
Riduce duplicazione e aumenta copertura.

🎯 SKILLS: parametrize, test cases
⏱️ TEMPO: 10 minuti
🔢 LIVELLO: Intermedio
"""

# ✅ SIMULAZIONE PARAMETRIZED:
def test_pnl_parametrized():
    """Test PnL con parametri multipli."""
    
    # (entry, exit, qty, side, expected)
    test_cases = [
        (100, 110, 10, 'LONG', 100),    # Long profit
        (100, 90, 10, 'LONG', -100),    # Long loss
        (100, 90, 10, 'SHORT', 100),    # Short profit
        (100, 110, 10, 'SHORT', -100),  # Short loss
        (50, 75, 20, 'LONG', 500),      # Bigger trade
        (200, 180, 5, 'SHORT', 100),    # Short profit
    ]
    
    print("  Test cases:")
    for entry, exit_p, qty, side, expected in test_cases:
        result = calculate_pnl(entry, exit_p, qty, side)
        status = "✅" if result == expected else "❌"
        print(f"    {status} PnL({entry}, {exit_p}, {qty}, {side}) = {result} (expected {expected})")


def test_position_size_parametrized():
    """Test position size con parametri multipli."""
    
    # (capital, risk_pct, entry, sl, expected)
    test_cases = [
        (10000, 1.0, 100, 95, 20),    # Standard
        (10000, 2.0, 100, 95, 40),    # Double risk
        (20000, 1.0, 100, 95, 40),    # Double capital
        (10000, 1.0, 100, 90, 10),    # Wider stop
        (10000, 1.0, 50, 48, 50),     # Different price
        (10000, 1.0, 100, 100, 0),    # No risk (entry == SL)
    ]
    
    print("  Position size test cases:")
    for capital, risk, entry, sl, expected in test_cases:
        result = calculate_position_size(capital, risk, entry, sl)
        status = "✅" if result == expected else "❌"
        print(f"    {status} size({capital}, {risk}%, {entry}, {sl}) = {result} (expected {expected})")


def esercizio_20_3():
    """Esegui test parametrizzati."""
    print("--- PARAMETRIZED TESTS ---")
    
    test_pnl_parametrized()
    print()
    test_position_size_parametrized()
    
    # Esempio pytest.mark.parametrize
    print("\n  Esempio con pytest:")
    print("""
    @pytest.mark.parametrize("entry,exit_p,qty,side,expected", [
        (100, 110, 10, 'LONG', 100),
        (100, 90, 10, 'LONG', -100),
        (100, 90, 10, 'SHORT', 100),
    ])
    def test_pnl(entry, exit_p, qty, side, expected):
        assert calculate_pnl(entry, exit_p, qty, side) == expected
    """)


if __name__ == "__main__":
    esercizio_20_3()
    print("✅ Esercizio 20.3 completato!\n")


# ==============================================================================
# SEZIONE 21: MOCKING
# ==============================================================================

print("\n" + "=" * 70)
print("SEZIONE 21: MOCKING")
print("=" * 70)

# ------------------------------------------------------------------------------
# ESERCIZIO 21.1: Mock Basics
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Usa mock per:
1. Sostituire funzioni/metodi
2. Verificare chiamate
3. Controllare return values

💡 TEORIA:
Mock sostituisce oggetti reali con oggetti controllabili.
Utile per testare codice che dipende da risorse esterne.

🎯 SKILLS: Mock, MagicMock, patch
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Avanzato
"""

# ✅ SOLUZIONE:
def esercizio_21_1():
    """Mock Basics"""
    
    from unittest.mock import Mock, MagicMock, patch
    
    # 1. MOCK BASE
    print("--- MOCK BASE ---")
    
    # Crea un mock
    mock_api = Mock()
    
    # Configura return value
    mock_api.get_price.return_value = 150.0
    
    # Usa il mock
    price = mock_api.get_price('AAPL')
    print(f"  mock_api.get_price('AAPL') = {price}")
    
    # Verifica che sia stato chiamato
    mock_api.get_price.assert_called_once_with('AAPL')
    print(f"  assert_called_once_with('AAPL') ✅")
    
    # 2. MOCK CON SIDE EFFECT
    print("\n--- SIDE EFFECT ---")
    
    # Return values diversi per chiamate successive
    mock_api.get_price.side_effect = [100, 105, 102]
    
    prices = [mock_api.get_price('X') for _ in range(3)]
    print(f"  Chiamate successive: {prices}")
    
    # Side effect come eccezione
    mock_api.get_price.side_effect = ConnectionError("API down")
    
    try:
        mock_api.get_price('AAPL')
    except ConnectionError as e:
        print(f"  Exception raised: {e}")
    
    # Side effect come funzione
    def price_lookup(symbol):
        prices = {'AAPL': 150, 'GOOGL': 140}
        return prices.get(symbol, 0)
    
    mock_api.get_price.side_effect = price_lookup
    print(f"  AAPL: {mock_api.get_price('AAPL')}")
    print(f"  GOOGL: {mock_api.get_price('GOOGL')}")
    print(f"  UNKNOWN: {mock_api.get_price('UNKNOWN')}")
    
    # 3. CALL TRACKING
    print("\n--- CALL TRACKING ---")
    
    mock_api = Mock()
    
    mock_api.place_order('AAPL', 100, 'BUY')
    mock_api.place_order('GOOGL', 50, 'SELL')
    mock_api.place_order('AAPL', 200, 'BUY')
    
    print(f"  call_count: {mock_api.place_order.call_count}")
    print(f"  call_args: {mock_api.place_order.call_args}")  # Ultima chiamata
    print(f"  call_args_list: {mock_api.place_order.call_args_list}")
    
    # 4. MAGIC MOCK
    print("\n--- MAGIC MOCK ---")
    
    # MagicMock supporta magic methods
    magic = MagicMock()
    
    magic.__len__.return_value = 5
    magic.__getitem__.return_value = 'item'
    
    print(f"  len(magic) = {len(magic)}")
    print(f"  magic[0] = {magic[0]}")
    
    # 5. PATCH (context manager)
    print("\n--- PATCH ---")
    
    # Simula una funzione che chiama API
    class TradingClient:
        def __init__(self, api):
            self.api = api
        
        def get_portfolio_value(self, symbols, quantities):
            total = 0
            for symbol, qty in zip(symbols, quantities):
                price = self.api.get_price(symbol)
                total += price * qty
            return total
    
    # Test con mock
    mock_api = Mock()
    mock_api.get_price.side_effect = lambda s: {'AAPL': 150, 'GOOGL': 140}[s]
    
    client = TradingClient(mock_api)
    value = client.get_portfolio_value(['AAPL', 'GOOGL'], [10, 5])
    
    print(f"  Portfolio value: ${value}")
    print(f"  API calls: {mock_api.get_price.call_count}")
    
    return Mock

if __name__ == "__main__":
    esercizio_21_1()
    print("✅ Esercizio 21.1 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 21.2: Testing Trading Strategies
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Scrivi test per una strategia di trading:
1. Test segnali corretti
2. Mock price data
3. Test edge cases

💡 TEORIA:
Testare strategie richiede dati controllati.
Mock permette di simulare scenari specifici.

🎯 SKILLS: Strategy testing, data mocking
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Avanzato
"""

# ✅ STRATEGIA DA TESTARE:
class SMAStrategy:
    """Simple Moving Average Crossover Strategy."""
    
    def __init__(self, fast_period=10, slow_period=20):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.prices = []
    
    def add_price(self, price):
        self.prices.append(price)
    
    def sma(self, period):
        if len(self.prices) < period:
            return None
        return sum(self.prices[-period:]) / period
    
    def get_signal(self):
        fast = self.sma(self.fast_period)
        slow = self.sma(self.slow_period)
        
        if fast is None or slow is None:
            return 'WAIT'
        
        if fast > slow:
            return 'BUY'
        elif fast < slow:
            return 'SELL'
        else:
            return 'HOLD'


# ✅ TEST:
def test_sma_calculation():
    """Test calcolo SMA corretto."""
    strategy = SMAStrategy(fast_period=3, slow_period=5)
    
    prices = [10, 20, 30, 40, 50]
    for p in prices:
        strategy.add_price(p)
    
    # SMA(3) = (30+40+50)/3 = 40
    assert strategy.sma(3) == 40
    
    # SMA(5) = (10+20+30+40+50)/5 = 30
    assert strategy.sma(5) == 30
    
    print("  ✅ test_sma_calculation")


def test_signal_wait_insufficient_data():
    """Test che ritorna WAIT con dati insufficienti."""
    strategy = SMAStrategy(fast_period=10, slow_period=20)
    
    for p in [100, 101, 102]:
        strategy.add_price(p)
    
    assert strategy.get_signal() == 'WAIT'
    print("  ✅ test_signal_wait_insufficient_data")


def test_signal_buy():
    """Test segnale BUY quando fast > slow."""
    strategy = SMAStrategy(fast_period=2, slow_period=4)
    
    # Trend up: fast SMA > slow SMA
    prices = [100, 102, 104, 106, 108, 110]
    for p in prices:
        strategy.add_price(p)
    
    # fast = (108+110)/2 = 109
    # slow = (104+106+108+110)/4 = 107
    signal = strategy.get_signal()
    assert signal == 'BUY', f"Expected BUY, got {signal}"
    print("  ✅ test_signal_buy")


def test_signal_sell():
    """Test segnale SELL quando fast < slow."""
    strategy = SMAStrategy(fast_period=2, slow_period=4)
    
    # Trend down: fast SMA < slow SMA
    prices = [110, 108, 106, 104, 102, 100]
    for p in prices:
        strategy.add_price(p)
    
    # fast = (102+100)/2 = 101
    # slow = (106+104+102+100)/4 = 103
    signal = strategy.get_signal()
    assert signal == 'SELL', f"Expected SELL, got {signal}"
    print("  ✅ test_signal_sell")


def test_signal_sequence():
    """Test sequenza di segnali."""
    strategy = SMAStrategy(fast_period=2, slow_period=3)
    
    signals = []
    prices = [100, 100, 100, 105, 110, 115, 110, 105, 100, 95]
    
    for p in prices:
        strategy.add_price(p)
        signals.append(strategy.get_signal())
    
    print(f"  Prices: {prices}")
    print(f"  Signals: {signals}")
    
    # Primi segnali dovrebbero essere WAIT
    assert signals[0] == 'WAIT'
    assert signals[1] == 'WAIT'
    
    print("  ✅ test_signal_sequence")


def esercizio_21_2():
    """Esegui test strategia."""
    print("--- TESTING TRADING STRATEGY ---")
    
    tests = [
        test_sma_calculation,
        test_signal_wait_insufficient_data,
        test_signal_buy,
        test_signal_sell,
        test_signal_sequence,
    ]
    
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {test.__name__}: {e}")
    
    print(f"\n  Risultato: {passed}/{len(tests)} passed")


if __name__ == "__main__":
    esercizio_21_2()
    print("✅ Esercizio 21.2 completato!\n")


# ==============================================================================
# SEZIONE 22: TDD (Test Driven Development)
# ==============================================================================

print("\n" + "=" * 70)
print("SEZIONE 22: TDD")
print("=" * 70)

# ------------------------------------------------------------------------------
# ESERCIZIO 22.1: TDD Workflow
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Pratica il ciclo TDD:
1. RED: Scrivi test che fallisce
2. GREEN: Implementa il minimo per passare
3. REFACTOR: Migliora il codice

💡 TEORIA:
TDD = Test Driven Development
Scrivi il test PRIMA del codice.
Ciclo: Red → Green → Refactor → Repeat

🎯 SKILLS: TDD workflow, incremental development
⏱️ TEMPO: 20 minuti
🔢 LIVELLO: Avanzato
"""

# ✅ ESEMPIO TDD: Sviluppiamo un Risk Calculator

# STEP 1: RED - Scrivi test che fallisce
print("--- TDD WORKFLOW ---")

# Test che vogliamo far passare
def test_risk_calculator_max_position():
    """Calcola massima posizione basata su rischio."""
    calc = RiskCalculator(capital=10000, max_risk_percent=1.0)
    
    # Con SL al 5%, posso rischiare $100 (1% di 10000)
    # Position size = 100 / (100 * 0.05) = 20 shares
    result = calc.max_position(price=100, stop_loss_percent=5)
    assert result == 20


def test_risk_calculator_max_loss():
    """Calcola perdita massima per una posizione."""
    calc = RiskCalculator(capital=10000, max_risk_percent=1.0)
    
    # Max loss = 1% di 10000 = 100
    assert calc.max_loss() == 100


def test_risk_calculator_position_risk():
    """Calcola rischio di una specifica posizione."""
    calc = RiskCalculator(capital=10000, max_risk_percent=1.0)
    
    # 50 shares @ 100 con SL 5% = 50 * 100 * 0.05 = 250
    risk = calc.position_risk(quantity=50, price=100, stop_loss_percent=5)
    assert risk == 250


def test_risk_calculator_is_within_risk():
    """Verifica se posizione è nei limiti di rischio."""
    calc = RiskCalculator(capital=10000, max_risk_percent=1.0)
    
    # 20 shares ok (risk = 100)
    assert calc.is_within_risk(20, 100, 5) == True
    
    # 50 shares troppo (risk = 250)
    assert calc.is_within_risk(50, 100, 5) == False


# STEP 2: GREEN - Implementa il minimo
class RiskCalculator:
    """Calcolatore di rischio per position sizing."""
    
    def __init__(self, capital, max_risk_percent):
        self.capital = capital
        self.max_risk_percent = max_risk_percent
    
    def max_loss(self):
        """Massima perdita accettabile."""
        return self.capital * (self.max_risk_percent / 100)
    
    def position_risk(self, quantity, price, stop_loss_percent):
        """Rischio di una specifica posizione."""
        return quantity * price * (stop_loss_percent / 100)
    
    def max_position(self, price, stop_loss_percent):
        """Massima posizione mantenendo il rischio nei limiti."""
        risk_per_share = price * (stop_loss_percent / 100)
        if risk_per_share == 0:
            return 0
        return int(self.max_loss() / risk_per_share)
    
    def is_within_risk(self, quantity, price, stop_loss_percent):
        """Verifica se posizione è nei limiti."""
        risk = self.position_risk(quantity, price, stop_loss_percent)
        return risk <= self.max_loss()


# STEP 3: Esegui test
def esercizio_22_1():
    """Esegui TDD workflow."""
    print("\n  Eseguendo test TDD:")
    
    tests = [
        test_risk_calculator_max_position,
        test_risk_calculator_max_loss,
        test_risk_calculator_position_risk,
        test_risk_calculator_is_within_risk,
    ]
    
    passed = 0
    for test in tests:
        try:
            test()
            print(f"  ✅ {test.__name__}")
            passed += 1
        except (AssertionError, NameError) as e:
            print(f"  ❌ {test.__name__}: {e}")
    
    print(f"\n  TDD Result: {passed}/{len(tests)} GREEN")
    
    # Spiegazione workflow
    print("\n  TDD Workflow:")
    print("  1. 🔴 RED: Scrivi test che fallisce")
    print("  2. 🟢 GREEN: Scrivi codice minimo per passare")
    print("  3. 🔵 REFACTOR: Migliora senza rompere test")
    print("  4. 🔄 REPEAT")
    
    return RiskCalculator


if __name__ == "__main__":
    esercizio_22_1()
    print("✅ Esercizio 22.1 completato!\n")


# ==============================================================================
# RIEPILOGO SEZIONE 20-22: TESTING
# ==============================================================================

print("\n" + "=" * 70)
print("RIEPILOGO: ESERCIZI TESTING COMPLETATI")
print("=" * 70)

print("""
ESERCIZI COMPLETATI IN QUESTA PARTE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SEZIONE 20 - Pytest Basics:
  ✅ 20.1 Test Functions
  ✅ 20.2 Fixtures
  ✅ 20.3 Parametrized Tests

SEZIONE 21 - Mocking:
  ✅ 21.1 Mock Basics
  ✅ 21.2 Testing Trading Strategies

SEZIONE 22 - TDD:
  ✅ 22.1 TDD Workflow

TOTALE QUESTA PARTE: 6 esercizi
TOTALE CUMULATIVO: 66 esercizi
""")

if __name__ == "__main__":
    print("\n🎉 TUTTI GLI ESERCIZI DELLA PARTE 7 COMPLETATI!")
