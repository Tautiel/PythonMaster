"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║            PROFESSIONAL PYTHON PROGRAMMER - MODULE 1                         ║
║                       Testing & Test-Driven Development                      ║
║                                                                              ║
║                     Allineato al Syllabus PCPP2                              ║
║                         OpenEDG Python Institute                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

PCPP2 Exam Section: Testing (~25%)

STRUTTURA MODULO:
├── Section 1.1: unittest Framework
├── Section 1.2: pytest Framework
├── Section 1.3: Mocking & Patching
├── Section 1.4: Test-Driven Development (TDD)
├── Section 1.5: Code Coverage
└── Module 1 Test (20 domande)

TEMPO STIMATO: 5-6 ore

═══════════════════════════════════════════════════════════════════════════════
"""

# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 1.1: UNITTEST FRAMEWORK
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    1.1 TEORIA: UNITTEST                                      │
└──────────────────────────────────────────────────────────────────────────────┘

unittest: Framework di testing incluso in Python (standard library).
Ispirato a JUnit (Java).

TERMINOLOGIA:
─────────────
- Test Case: Singolo test
- Test Suite: Collezione di test cases
- Test Runner: Componente che esegue i test
- Test Fixture: Setup/teardown per i test
"""

import unittest

# ═══════════════════════════════════════════════════════════════════════════
# CLASSE DA TESTARE
# ═══════════════════════════════════════════════════════════════════════════

class Calculator:
    """Classe esempio da testare."""
    
    def add(self, a: float, b: float) -> float:
        return a + b
    
    def subtract(self, a: float, b: float) -> float:
        return a - b
    
    def multiply(self, a: float, b: float) -> float:
        return a * b
    
    def divide(self, a: float, b: float) -> float:
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS CON UNITTEST
# ═══════════════════════════════════════════════════════════════════════════

class TestCalculator(unittest.TestCase):
    """Test per la classe Calculator."""
    
    # ───────────────────────────────────────────────────────────────────────
    # FIXTURES (Setup e Teardown)
    # ───────────────────────────────────────────────────────────────────────
    
    @classmethod
    def setUpClass(cls):
        """Eseguito UNA VOLTA prima di tutti i test della classe."""
        print("\n=== setUpClass: Setup iniziale ===")
        cls.shared_resource = "resource"
    
    @classmethod
    def tearDownClass(cls):
        """Eseguito UNA VOLTA dopo tutti i test della classe."""
        print("\n=== tearDownClass: Cleanup finale ===")
    
    def setUp(self):
        """Eseguito PRIMA DI OGNI test."""
        self.calc = Calculator()
    
    def tearDown(self):
        """Eseguito DOPO OGNI test."""
        pass  # Cleanup se necessario
    
    # ───────────────────────────────────────────────────────────────────────
    # TEST METHODS (devono iniziare con 'test_')
    # ───────────────────────────────────────────────────────────────────────
    
    def test_add_positive_numbers(self):
        """Test addizione con numeri positivi."""
        result = self.calc.add(2, 3)
        self.assertEqual(result, 5)
    
    def test_add_negative_numbers(self):
        """Test addizione con numeri negativi."""
        result = self.calc.add(-2, -3)
        self.assertEqual(result, -5)
    
    def test_subtract(self):
        """Test sottrazione."""
        self.assertEqual(self.calc.subtract(5, 3), 2)
    
    def test_multiply(self):
        """Test moltiplicazione."""
        self.assertEqual(self.calc.multiply(3, 4), 12)
    
    def test_divide(self):
        """Test divisione."""
        self.assertEqual(self.calc.divide(10, 2), 5)
    
    def test_divide_by_zero_raises_error(self):
        """Test che la divisione per zero solleva eccezione."""
        with self.assertRaises(ValueError) as context:
            self.calc.divide(10, 0)
        
        self.assertEqual(str(context.exception), "Cannot divide by zero")
    
    # ───────────────────────────────────────────────────────────────────────
    # TUTTI GLI ASSERT METHODS
    # ───────────────────────────────────────────────────────────────────────
    
    def test_all_assertions(self):
        """Dimostra tutti i metodi di asserzione."""
        
        # Uguaglianza
        self.assertEqual(1 + 1, 2)
        self.assertNotEqual(1 + 1, 3)
        
        # Booleani
        self.assertTrue(1 < 2)
        self.assertFalse(1 > 2)
        
        # None
        self.assertIsNone(None)
        self.assertIsNotNone("value")
        
        # Identità (is)
        a = [1, 2, 3]
        b = a
        c = [1, 2, 3]
        self.assertIs(a, b)
        self.assertIsNot(a, c)
        
        # Membership (in)
        self.assertIn(2, [1, 2, 3])
        self.assertNotIn(4, [1, 2, 3])
        
        # Tipo
        self.assertIsInstance("hello", str)
        self.assertNotIsInstance("hello", int)
        
        # Comparazioni
        self.assertGreater(5, 3)
        self.assertGreaterEqual(5, 5)
        self.assertLess(3, 5)
        self.assertLessEqual(5, 5)
        
        # Float (con tolleranza)
        self.assertAlmostEqual(0.1 + 0.2, 0.3, places=7)
        
        # Regex
        self.assertRegex("hello world", r"hello.*")
        
        # Collections
        self.assertCountEqual([1, 2, 3], [3, 2, 1])  # Stessi elementi
        self.assertSequenceEqual([1, 2], [1, 2])    # Stessa sequenza
        self.assertDictEqual({'a': 1}, {'a': 1})
        self.assertSetEqual({1, 2}, {2, 1})
    
    # ───────────────────────────────────────────────────────────────────────
    # SKIP E EXPECTED FAILURES
    # ───────────────────────────────────────────────────────────────────────
    
    @unittest.skip("Test non implementato ancora")
    def test_future_feature(self):
        pass
    
    @unittest.skipIf(True, "Condizione vera")
    def test_conditional_skip(self):
        pass
    
    @unittest.skipUnless(False, "Condizione falsa")
    def test_another_skip(self):
        pass
    
    @unittest.expectedFailure
    def test_known_bug(self):
        """Questo test fallirà, ma è atteso."""
        self.assertEqual(1, 2)


# Eseguire i test
# python -m unittest test_module.py
# python -m unittest test_module.TestCalculator
# python -m unittest test_module.TestCalculator.test_add_positive_numbers
# python -m unittest discover  # Trova tutti i test


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 1.2: PYTEST FRAMEWORK
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    1.2 TEORIA: PYTEST                                        │
└──────────────────────────────────────────────────────────────────────────────┘

pytest: Framework di testing moderno, più semplice di unittest.
pip install pytest

VANTAGGI:
- Usa assert standard (non metodi speciali)
- Auto-discovery dei test
- Fixtures potenti
- Parametrizzazione facile
- Plugin ecosystem
"""

# File: test_calculator_pytest.py

def test_add():
    """Test semplice con pytest."""
    calc = Calculator()
    assert calc.add(2, 3) == 5


def test_divide_by_zero():
    """Test eccezioni con pytest."""
    import pytest
    calc = Calculator()
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        calc.divide(10, 0)


# ═══════════════════════════════════════════════════════════════════════════
# PYTEST FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

PYTEST_FIXTURES_EXAMPLE = '''
import pytest

@pytest.fixture
def calculator():
    """Fixture che fornisce un Calculator."""
    return Calculator()

@pytest.fixture
def sample_data():
    """Fixture con dati di test."""
    return [1, 2, 3, 4, 5]

def test_add_with_fixture(calculator):
    """Il fixture viene passato automaticamente."""
    assert calculator.add(2, 3) == 5

# Fixture con setup e teardown
@pytest.fixture
def database():
    """Fixture con cleanup."""
    db = Database()
    db.connect()
    yield db  # Il test usa db qui
    db.disconnect()  # Cleanup dopo il test

# Fixture con scope
@pytest.fixture(scope="module")
def expensive_resource():
    """Creato una volta per modulo."""
    return create_expensive_resource()

# Scopes disponibili:
# - "function" (default): per ogni test
# - "class": per classe
# - "module": per file
# - "session": per tutta la sessione
'''


# ═══════════════════════════════════════════════════════════════════════════
# PYTEST PARAMETRIZE
# ═══════════════════════════════════════════════════════════════════════════

PYTEST_PARAMETRIZE_EXAMPLE = '''
import pytest

@pytest.mark.parametrize("a, b, expected", [
    (2, 3, 5),
    (-1, 1, 0),
    (0, 0, 0),
    (100, 200, 300),
])
def test_add_parametrized(a, b, expected):
    """Un test, molti casi."""
    calc = Calculator()
    assert calc.add(a, b) == expected

@pytest.mark.parametrize("input_val", [1, 2, 3])
@pytest.mark.parametrize("multiplier", [10, 20])
def test_combinations(input_val, multiplier):
    """Testa tutte le combinazioni (6 test)."""
    result = input_val * multiplier
    assert result > 0
'''


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 1.3: MOCKING & PATCHING
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    1.3 TEORIA: MOCKING                                       │
└──────────────────────────────────────────────────────────────────────────────┘

MOCK: Oggetto che simula il comportamento di un oggetto reale.
Utile per:
- Isolare il codice da testare
- Evitare dipendenze esterne (DB, API, file)
- Controllare il comportamento delle dipendenze
"""

from unittest.mock import Mock, MagicMock, patch, call

def mocking_examples():
    """Esempi di mocking."""
    
    # ═══════════════════════════════════════════════════════════════════════
    # MOCK BASE
    # ═══════════════════════════════════════════════════════════════════════
    
    # Creare un mock
    mock = Mock()
    
    # Chiamare metodi (non esistenti!)
    mock.some_method()
    mock.another_method(1, 2, key='value')
    
    # Verificare chiamate
    mock.some_method.assert_called()
    mock.another_method.assert_called_once()
    mock.another_method.assert_called_with(1, 2, key='value')
    
    # Configurare return value
    mock.get_value.return_value = 42
    assert mock.get_value() == 42
    
    # Side effects
    mock.failing_method.side_effect = ValueError("Error!")
    # mock.failing_method()  # Solleva ValueError
    
    # Side effect come funzione
    mock.compute.side_effect = lambda x: x * 2
    assert mock.compute(5) == 10
    
    # Side effect come lista (valori sequenziali)
    mock.next_val.side_effect = [1, 2, 3]
    assert mock.next_val() == 1
    assert mock.next_val() == 2
    assert mock.next_val() == 3
    
    # ═══════════════════════════════════════════════════════════════════════
    # MAGICMOCK
    # ═══════════════════════════════════════════════════════════════════════
    
    # MagicMock supporta magic methods
    magic = MagicMock()
    magic.__len__.return_value = 5
    assert len(magic) == 5
    
    magic.__getitem__.return_value = 'item'
    assert magic[0] == 'item'
    
    # ═══════════════════════════════════════════════════════════════════════
    # PATCH
    # ═══════════════════════════════════════════════════════════════════════
    
    # Esempio: funzione che usa requests
    # def fetch_data(url):
    #     response = requests.get(url)
    #     return response.json()
    
    # Test con patch
    # with patch('module.requests.get') as mock_get:
    #     mock_get.return_value.json.return_value = {'data': 'test'}
    #     result = fetch_data('http://example.com')
    #     assert result == {'data': 'test'}


# ═══════════════════════════════════════════════════════════════════════════
# ESEMPIO PRATICO: TEST CON MOCK
# ═══════════════════════════════════════════════════════════════════════════

class UserService:
    """Servizio che dipende da un database."""
    
    def __init__(self, db):
        self.db = db
    
    def get_user(self, user_id: int):
        user = self.db.find_by_id(user_id)
        if user is None:
            raise ValueError(f"User {user_id} not found")
        return user
    
    def create_user(self, name: str, email: str):
        if self.db.find_by_email(email):
            raise ValueError("Email already exists")
        return self.db.insert({'name': name, 'email': email})


class TestUserService(unittest.TestCase):
    """Test UserService con mock del database."""
    
    def setUp(self):
        self.mock_db = Mock()
        self.service = UserService(self.mock_db)
    
    def test_get_user_success(self):
        """Test get_user quando l'utente esiste."""
        self.mock_db.find_by_id.return_value = {'id': 1, 'name': 'Marco'}
        
        user = self.service.get_user(1)
        
        self.assertEqual(user['name'], 'Marco')
        self.mock_db.find_by_id.assert_called_once_with(1)
    
    def test_get_user_not_found(self):
        """Test get_user quando l'utente non esiste."""
        self.mock_db.find_by_id.return_value = None
        
        with self.assertRaises(ValueError):
            self.service.get_user(999)
    
    def test_create_user_success(self):
        """Test creazione utente."""
        self.mock_db.find_by_email.return_value = None
        self.mock_db.insert.return_value = {'id': 1, 'name': 'Marco'}
        
        user = self.service.create_user('Marco', 'marco@example.com')
        
        self.mock_db.find_by_email.assert_called_once_with('marco@example.com')
        self.mock_db.insert.assert_called_once()
    
    def test_create_user_email_exists(self):
        """Test creazione fallisce se email esiste."""
        self.mock_db.find_by_email.return_value = {'id': 1}
        
        with self.assertRaises(ValueError):
            self.service.create_user('Marco', 'existing@example.com')
        
        # insert NON deve essere chiamato
        self.mock_db.insert.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 1.4: TEST-DRIVEN DEVELOPMENT
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    1.4 TEORIA: TDD                                           │
└──────────────────────────────────────────────────────────────────────────────┘

TDD = Test-Driven Development

CICLO RED-GREEN-REFACTOR:
─────────────────────────
1. RED: Scrivi un test che FALLISCE
2. GREEN: Scrivi il codice MINIMO per far passare il test
3. REFACTOR: Migliora il codice mantenendo i test verdi

BENEFICI:
- Codice testato al 100%
- Design migliore (forzato a pensare all'interfaccia)
- Documentazione vivente
- Confidenza nel refactoring
"""

# ═══════════════════════════════════════════════════════════════════════════
# ESEMPIO TDD: Sviluppare una Stack
# ═══════════════════════════════════════════════════════════════════════════

# STEP 1: Scrivi il test PRIMA del codice

class TestStack(unittest.TestCase):
    """TDD per una classe Stack."""
    
    def test_new_stack_is_empty(self):
        """Una nuova stack è vuota."""
        stack = Stack()
        self.assertTrue(stack.is_empty())
    
    def test_push_adds_element(self):
        """Push aggiunge un elemento."""
        stack = Stack()
        stack.push(1)
        self.assertFalse(stack.is_empty())
    
    def test_pop_removes_last_element(self):
        """Pop rimuove e restituisce l'ultimo elemento."""
        stack = Stack()
        stack.push(1)
        stack.push(2)
        self.assertEqual(stack.pop(), 2)
        self.assertEqual(stack.pop(), 1)
    
    def test_pop_empty_stack_raises_error(self):
        """Pop su stack vuota solleva errore."""
        stack = Stack()
        with self.assertRaises(IndexError):
            stack.pop()
    
    def test_peek_returns_without_removing(self):
        """Peek restituisce senza rimuovere."""
        stack = Stack()
        stack.push(1)
        self.assertEqual(stack.peek(), 1)
        self.assertFalse(stack.is_empty())
    
    def test_size_returns_count(self):
        """Size restituisce il numero di elementi."""
        stack = Stack()
        self.assertEqual(stack.size(), 0)
        stack.push(1)
        stack.push(2)
        self.assertEqual(stack.size(), 2)


# STEP 2: Implementa il codice MINIMO per far passare i test

class Stack:
    """Stack implementata seguendo TDD."""
    
    def __init__(self):
        self._items = []
    
    def is_empty(self) -> bool:
        return len(self._items) == 0
    
    def push(self, item) -> None:
        self._items.append(item)
    
    def pop(self):
        if self.is_empty():
            raise IndexError("Pop from empty stack")
        return self._items.pop()
    
    def peek(self):
        if self.is_empty():
            raise IndexError("Peek from empty stack")
        return self._items[-1]
    
    def size(self) -> int:
        return len(self._items)


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 1.5: CODE COVERAGE
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    1.5 TEORIA: CODE COVERAGE                                 │
└──────────────────────────────────────────────────────────────────────────────┘

Coverage = Percentuale di codice eseguita dai test.

TIPI DI COVERAGE:
─────────────────
- Line coverage: % di righe eseguite
- Branch coverage: % di branch (if/else) testati
- Function coverage: % di funzioni chiamate
- Condition coverage: % di condizioni booleane testate

TOOL: coverage.py
pip install coverage

COMANDI:
coverage run -m pytest
coverage report
coverage html  # Report HTML dettagliato

TARGET REALISTICO: 80-90%
(100% spesso non vale lo sforzo)
"""

COVERAGE_CONFIG = '''
# .coveragerc o pyproject.toml

[tool.coverage.run]
source = ["src"]
branch = true
omit = ["tests/*", "*/__init__.py"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
]
fail_under = 80
'''


# ══════════════════════════════════════════════════════════════════════════════
#                    MODULE 1 TEST
# ══════════════════════════════════════════════════════════════════════════════

MODULE_1_TEST = """
══════════════════════════════════════════════════════════════════════════════
                    PP MODULE 1 - TEST FINALE
                    20 domande - Target: 70%
══════════════════════════════════════════════════════════════════════════════

Q1. In unittest, i metodi di test devono iniziare con:
    A) test_    B) _test    C) check_    D) Qualsiasi

Q2. setUp() viene eseguito:
    A) Una volta    B) Prima di ogni test    C) Dopo ogni test    D) Mai

Q3. setUpClass() viene eseguito:
    A) Una volta per classe    B) Per ogni test    C) Dopo tutti    D) Mai

Q4. assertEqual(a, b) verifica:
    A) a is b    B) a == b    C) a in b    D) type(a) == type(b)

Q5. assertRaises si usa per:
    A) Sollevare eccezioni    B) Verificare eccezioni    C) Ignorare errori    D) Log

Q6. In pytest, le fixtures sono definite con:
    A) @fixture    B) @pytest.fixture    C) def fixture    D) @mock

Q7. @pytest.mark.parametrize serve per:
    A) Parametri fixture    B) Test multipli    C) Skip    D) Mock

Q8. Mock.return_value imposta:
    A) Eccezione    B) Valore di ritorno    C) Side effect    D) Nome

Q9. Mock.side_effect può essere:
    A) Solo eccezione    B) Solo funzione    C) Eccezione, funzione, lista    D) Solo lista

Q10. patch() si usa per:
     A) Fixare bug    B) Sostituire temporaneamente    C) Skip test    D) Coverage

Q11. MagicMock supporta:
     A) Solo metodi    B) Magic methods    C) Solo attributi    D) Niente

Q12. Il ciclo TDD è:
     A) Green-Red-Refactor    B) Red-Green-Refactor    C) Refactor-Test-Code    D) Code-Test-Debug

Q13. In TDD, si scrive PRIMA:
     A) Il codice    B) Il test    C) La documentazione    D) Il design

Q14. coverage.py misura:
     A) Performance    B) Memoria    C) Linee testate    D) Complessità

Q15. Branch coverage testa:
     A) Funzioni    B) if/else    C) Import    D) Classi

Q16. assert_called_once() verifica:
     A) Mai chiamato    B) Chiamato 1 volta    C) Chiamato N volte    D) Return value

Q17. assert_not_called() verifica:
     A) Mai chiamato    B) Chiamato    C) Errore    D) Return None

Q18. @unittest.skip serve per:
     A) Eseguire    B) Saltare    C) Fallire    D) Parametrizzare

Q19. @unittest.expectedFailure indica:
     A) Test rotto    B) Bug noto    C) Test skip    D) Success

Q20. pytest auto-discover trova file che iniziano con:
     A) test_    B) _test    C) check_    D) spec_


══════════════════════════════════════════════════════════════════════════════
"""

MODULE_1_TEST_ANSWERS = """
══════════════════════════════════════════════════════════════════════════════
                         RISPOSTE TEST MODULE 1
══════════════════════════════════════════════════════════════════════════════

Q1:  A) test_
Q2:  B) Prima di ogni test
Q3:  A) Una volta per classe
Q4:  B) a == b (uguaglianza, non identità)
Q5:  B) Verificare eccezioni
Q6:  B) @pytest.fixture
Q7:  B) Test multipli (stessa logica, dati diversi)
Q8:  B) Valore di ritorno
Q9:  C) Eccezione, funzione, lista
Q10: B) Sostituire temporaneamente (monkey patching)
Q11: B) Magic methods (__len__, __getitem__, etc.)
Q12: B) Red-Green-Refactor
Q13: B) Il test
Q14: C) Linee testate (code coverage)
Q15: B) if/else (ogni branch)
Q16: B) Chiamato esattamente 1 volta
Q17: A) Mai chiamato
Q18: B) Saltare il test
Q19: B) Bug noto (test fallisce ma è atteso)
Q20: A) test_ (o finiscono con _test)

══════════════════════════════════════════════════════════════════════════════
"""


if __name__ == "__main__":
    print("=" * 70)
    print("PP MODULE 1: Testing & TDD")
    print("=" * 70)
    print("""
    Comandi:
    print(MODULE_1_TEST)         # Test finale
    print(MODULE_1_TEST_ANSWERS) # Risposte
    
    Esegui test:
    python -m unittest pp_m1_testing
    """)
    
    # Eseguire i test del modulo
    # unittest.main(verbosity=2)
