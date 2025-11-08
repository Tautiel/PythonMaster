"""
🚀 SESSIONE 2 - PARTE 4: TESTING & BEST PRACTICES
==================================================
Testing Professionale con pytest, mocking, TDD
Durata: 60 minuti di testing e best practices
"""

import unittest
import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import Any, List, Dict, Optional
import tempfile
import os
import json
from dataclasses import dataclass
from datetime import datetime
import logging
from contextlib import contextmanager
import time

print("="*80)
print("🧪 SESSIONE 2 PARTE 4: TESTING & BEST PRACTICES")
print("="*80)

# ==============================================================================
# SEZIONE 1: UNIT TESTING CON UNITTEST
# ==============================================================================

print("\n" + "="*60)
print("🔬 SEZIONE 1: UNIT TESTING CON UNITTEST")
print("="*60)

# Code to test
class Calculator:
    """Calculator class per testing"""
    
    def add(self, a: float, b: float) -> float:
        return a + b
    
    def divide(self, a: float, b: float) -> float:
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    
    def calculate_average(self, numbers: List[float]) -> float:
        if not numbers:
            raise ValueError("Cannot calculate average of empty list")
        return sum(numbers) / len(numbers)

class TestCalculator(unittest.TestCase):
    """Test suite per Calculator"""
    
    def setUp(self):
        """Setup eseguito prima di ogni test"""
        self.calc = Calculator()
    
    def tearDown(self):
        """Cleanup eseguito dopo ogni test"""
        del self.calc
    
    def test_add_positive_numbers(self):
        """Test addition con numeri positivi"""
        result = self.calc.add(2, 3)
        self.assertEqual(result, 5)
    
    def test_add_negative_numbers(self):
        """Test addition con numeri negativi"""
        result = self.calc.add(-2, -3)
        self.assertEqual(result, -5)
    
    def test_divide_normal(self):
        """Test divisione normale"""
        result = self.calc.divide(10, 2)
        self.assertEqual(result, 5)
    
    def test_divide_by_zero(self):
        """Test divisione per zero solleva eccezione"""
        with self.assertRaises(ValueError) as context:
            self.calc.divide(10, 0)
        
        self.assertEqual(str(context.exception), "Cannot divide by zero")
    
    def test_calculate_average(self):
        """Test calcolo media"""
        numbers = [1, 2, 3, 4, 5]
        result = self.calc.calculate_average(numbers)
        self.assertEqual(result, 3)
    
    def test_calculate_average_empty_list(self):
        """Test media con lista vuota"""
        with self.assertRaises(ValueError):
            self.calc.calculate_average([])
    
    @unittest.skip("Skipping this test")
    def test_skipped(self):
        """Questo test viene saltato"""
        pass
    
    @unittest.skipIf(os.name == 'nt', "Skipping on Windows")
    def test_unix_only(self):
        """Test solo per Unix"""
        self.assertTrue(True)

# ==============================================================================
# SEZIONE 2: PYTEST ADVANCED
# ==============================================================================

print("\n" + "="*60)
print("🚀 SEZIONE 2: PYTEST ADVANCED")
print("="*60)

# Fixtures
@pytest.fixture
def sample_data():
    """Fixture che fornisce dati di test"""
    return {
        "users": ["Alice", "Bob", "Charlie"],
        "scores": [95, 87, 92],
        "timestamp": datetime.now()
    }

@pytest.fixture
def temp_file():
    """Fixture per file temporaneo"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("Test content")
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)

# Parametrized tests
@pytest.mark.parametrize("input,expected", [
    (2, 4),
    (3, 9),
    (4, 16),
    (-2, 4)
])
def test_square(input, expected):
    """Test parametrizzato per quadrato"""
    assert input ** 2 == expected

# Custom markers
@pytest.mark.slow
def test_slow_operation():
    """Test marcato come slow"""
    time.sleep(1)
    assert True

@pytest.mark.integration
def test_integration():
    """Test di integrazione"""
    # Simula test di integrazione
    assert True

# Monkeypatch
def test_monkeypatch(monkeypatch):
    """Test con monkeypatch"""
    def mock_getcwd():
        return "/fake/path"
    
    monkeypatch.setattr(os, "getcwd", mock_getcwd)
    assert os.getcwd() == "/fake/path"

# ==============================================================================
# SEZIONE 3: MOCKING
# ==============================================================================

print("\n" + "="*60)
print("🎭 SEZIONE 3: MOCKING")
print("="*60)

# Code to test with external dependencies
class EmailService:
    """Servizio email da mockare"""
    
    def send_email(self, to: str, subject: str, body: str) -> bool:
        # In produzione invierebbe davvero l'email
        print(f"Sending email to {to}")
        # Simula invio
        return True

class UserService:
    """Servizio utenti che usa EmailService"""
    
    def __init__(self, email_service: EmailService):
        self.email_service = email_service
        self.users = {}
    
    def register_user(self, username: str, email: str) -> bool:
        if username in self.users:
            return False
        
        self.users[username] = {"email": email}
        
        # Invia email di benvenuto
        self.email_service.send_email(
            email,
            "Welcome!",
            f"Welcome {username}!"
        )
        
        return True
    
    def get_user(self, username: str) -> Optional[Dict]:
        return self.users.get(username)

class TestUserService(unittest.TestCase):
    """Test UserService con mocking"""
    
    def setUp(self):
        """Setup con mock"""
        self.email_service_mock = Mock(spec=EmailService)
        self.email_service_mock.send_email.return_value = True
        self.user_service = UserService(self.email_service_mock)
    
    def test_register_user_success(self):
        """Test registrazione utente con successo"""
        result = self.user_service.register_user("alice", "alice@example.com")
        
        self.assertTrue(result)
        self.assertIn("alice", self.user_service.users)
        
        # Verifica che email sia stata inviata
        self.email_service_mock.send_email.assert_called_once_with(
            "alice@example.com",
            "Welcome!",
            "Welcome alice!"
        )
    
    def test_register_duplicate_user(self):
        """Test registrazione utente duplicato"""
        self.user_service.register_user("bob", "bob@example.com")
        result = self.user_service.register_user("bob", "bob2@example.com")
        
        self.assertFalse(result)
        # Email non dovrebbe essere inviata per duplicati
        self.email_service_mock.send_email.assert_called_once()
    
    @patch('time.time')
    def test_with_patch(self, mock_time):
        """Test con patch decorator"""
        mock_time.return_value = 1234567890
        
        # Ora time.time() ritorna sempre 1234567890
        self.assertEqual(time.time(), 1234567890)
    
    def test_mock_side_effect(self):
        """Test con side_effect"""
        # Mock che solleva eccezione
        self.email_service_mock.send_email.side_effect = Exception("Email failed")
        
        with self.assertRaises(Exception):
            self.user_service.register_user("charlie", "charlie@example.com")

# ==============================================================================
# SEZIONE 4: ASYNC TESTING
# ==============================================================================

print("\n" + "="*60)
print("⚡ SEZIONE 4: ASYNC TESTING")
print("="*60)

# Async code to test
class AsyncDataFetcher:
    """Fetcher di dati asincrono"""
    
    async def fetch_data(self, id: int) -> Dict:
        """Fetch data asincrono"""
        await asyncio.sleep(0.1)  # Simula I/O
        return {"id": id, "data": f"Data for {id}"}
    
    async def fetch_multiple(self, ids: List[int]) -> List[Dict]:
        """Fetch multipli dati"""
        tasks = [self.fetch_data(id) for id in ids]
        return await asyncio.gather(*tasks)

class TestAsyncDataFetcher(unittest.TestCase):
    """Test per codice asincrono"""
    
    def setUp(self):
        self.fetcher = AsyncDataFetcher()
    
    def test_fetch_data(self):
        """Test fetch singolo"""
        async def run_test():
            result = await self.fetcher.fetch_data(1)
            self.assertEqual(result["id"], 1)
            self.assertIn("data", result)
        
        asyncio.run(run_test())
    
    def test_fetch_multiple(self):
        """Test fetch multiplo"""
        async def run_test():
            results = await self.fetcher.fetch_multiple([1, 2, 3])
            self.assertEqual(len(results), 3)
            for i, result in enumerate(results, 1):
                self.assertEqual(result["id"], i)
        
        asyncio.run(run_test())
    
    def test_with_async_mock(self):
        """Test con AsyncMock"""
        async def run_test():
            mock_fetch = AsyncMock(return_value={"id": 99, "data": "Mocked"})
            self.fetcher.fetch_data = mock_fetch
            
            result = await self.fetcher.fetch_data(99)
            self.assertEqual(result["id"], 99)
            mock_fetch.assert_called_once_with(99)
        
        asyncio.run(run_test())

# Pytest async testing
@pytest.mark.asyncio
async def test_async_with_pytest():
    """Test asincrono con pytest"""
    fetcher = AsyncDataFetcher()
    result = await fetcher.fetch_data(42)
    assert result["id"] == 42

# ==============================================================================
# SEZIONE 5: TEST PATTERNS E BEST PRACTICES
# ==============================================================================

print("\n" + "="*60)
print("🎯 SEZIONE 5: TEST PATTERNS & BEST PRACTICES")
print("="*60)

# 5.1 AAA PATTERN
class TestAAAPattern(unittest.TestCase):
    """Test seguendo pattern Arrange-Act-Assert"""
    
    def test_aaa_example(self):
        """Esempio di AAA pattern"""
        # ARRANGE - Setup dei dati
        calculator = Calculator()
        a, b = 10, 5
        
        # ACT - Esegui l'azione
        result = calculator.add(a, b)
        
        # ASSERT - Verifica risultato
        self.assertEqual(result, 15)

# 5.2 TEST DOUBLES
class TestDoubles:
    """Esempi di test doubles"""
    
    def test_stub(self):
        """Stub - ritorna risposte predefinite"""
        stub = Mock()
        stub.get_data.return_value = {"status": "ok"}
        
        assert stub.get_data() == {"status": "ok"}
    
    def test_spy(self):
        """Spy - registra chiamate"""
        spy = Mock()
        spy.process(1, 2, 3)
        
        spy.process.assert_called_with(1, 2, 3)
        assert spy.process.call_count == 1
    
    def test_fake(self):
        """Fake - implementazione semplificata"""
        class FakeDatabase:
            def __init__(self):
                self.data = {}
            
            def save(self, key, value):
                self.data[key] = value
            
            def get(self, key):
                return self.data.get(key)
        
        fake_db = FakeDatabase()
        fake_db.save("user1", {"name": "Alice"})
        assert fake_db.get("user1") == {"name": "Alice"}

# 5.3 TEST DATA BUILDERS
class UserBuilder:
    """Builder per creare utenti di test"""
    
    def __init__(self):
        self.username = "testuser"
        self.email = "test@example.com"
        self.age = 25
        self.is_active = True
    
    def with_username(self, username: str):
        self.username = username
        return self
    
    def with_email(self, email: str):
        self.email = email
        return self
    
    def with_age(self, age: int):
        self.age = age
        return self
    
    def inactive(self):
        self.is_active = False
        return self
    
    def build(self) -> Dict:
        return {
            "username": self.username,
            "email": self.email,
            "age": self.age,
            "is_active": self.is_active
        }

def test_with_builder():
    """Test usando builder pattern"""
    # Crea utente di test personalizzato
    user = UserBuilder()\
        .with_username("alice")\
        .with_age(30)\
        .inactive()\
        .build()
    
    assert user["username"] == "alice"
    assert user["age"] == 30
    assert user["is_active"] == False

# 5.4 PROPERTY-BASED TESTING
from hypothesis import given, strategies as st

@given(st.integers(), st.integers())
def test_addition_commutative(a, b):
    """Property: addition è commutativa"""
    calc = Calculator()
    assert calc.add(a, b) == calc.add(b, a)

@given(st.lists(st.floats(min_value=0.1, max_value=1000)))
def test_average_bounds(numbers):
    """Property: media è tra min e max"""
    if numbers:
        calc = Calculator()
        avg = calc.calculate_average(numbers)
        assert min(numbers) <= avg <= max(numbers)

# 5.5 INTEGRATION TEST EXAMPLE
class IntegrationTest(unittest.TestCase):
    """Test di integrazione esempio"""
    
    @classmethod
    def setUpClass(cls):
        """Setup per tutta la classe (costoso)"""
        # Simula setup database
        cls.db_connection = "db_connection"
        print("Setting up database connection")
    
    @classmethod
    def tearDownClass(cls):
        """Cleanup per tutta la classe"""
        # Simula chiusura database
        cls.db_connection = None
        print("Closing database connection")
    
    def test_full_workflow(self):
        """Test workflow completo"""
        # Simula test end-to-end
        # 1. Create user
        user = {"id": 1, "name": "Test User"}
        
        # 2. Create post
        post = {"user_id": 1, "title": "Test Post"}
        
        # 3. Verify relationship
        self.assertEqual(post["user_id"], user["id"])

# ==============================================================================
# SEZIONE 6: PERFORMANCE TESTING
# ==============================================================================

print("\n" + "="*60)
print("⚡ SEZIONE 6: PERFORMANCE TESTING")
print("="*60)

class PerformanceTest:
    """Performance testing utilities"""
    
    @staticmethod
    @contextmanager
    def timer(name: str):
        """Context manager per misurare tempo"""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            print(f"  {name} took {elapsed:.4f} seconds")
    
    @staticmethod
    def benchmark(func, *args, iterations: int = 1000, **kwargs):
        """Benchmark di una funzione"""
        times = []
        
        for _ in range(iterations):
            start = time.perf_counter()
            func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        return {
            "avg": avg_time,
            "min": min_time,
            "max": max_time,
            "iterations": iterations
        }

def test_performance():
    """Test di performance"""
    
    def slow_function():
        time.sleep(0.001)
    
    def fast_function():
        pass
    
    # Benchmark functions
    perf = PerformanceTest()
    
    with perf.timer("Slow function"):
        slow_function()
    
    with perf.timer("Fast function"):
        fast_function()
    
    # Benchmark con statistiche
    results = perf.benchmark(fast_function, iterations=1000)
    print(f"\n  Benchmark results:")
    print(f"    Average: {results['avg']*1000:.4f}ms")
    print(f"    Min: {results['min']*1000:.4f}ms")
    print(f"    Max: {results['max']*1000:.4f}ms")

# ==============================================================================
# SEZIONE 7: TEST COVERAGE E QUALITY
# ==============================================================================

print("\n" + "="*60)
print("📊 SEZIONE 7: TEST COVERAGE & QUALITY")
print("="*60)

print("""
📝 TEST COVERAGE BEST PRACTICES:

1. COVERAGE TOOLS:
   • pytest-cov per pytest
   • coverage.py per unittest
   
   Esempio:
   pytest --cov=myproject --cov-report=html

2. COVERAGE TARGETS:
   • Minimo 70% per codice normale
   • 90%+ per codice critico
   • 100% per librerie pubbliche

3. COSA NON TESTARE:
   • Configurazioni semplici
   • Getter/setter triviali
   • Codice generato
   • Main entry points

4. MUTATION TESTING:
   • Verifica qualità dei test
   • Tool: mutmut, cosmic-ray
   
5. TEST PYRAMID:
   
        /\\
       /  \\     E2E Tests (Few)
      /    \\
     /______\\   Integration Tests (Some)
    /        \\
   /__________\\ Unit Tests (Many)

6. CI/CD INTEGRATION:
   • Run tests su ogni commit
   • Block merge se test falliscono
   • Report coverage automatici
""")

# ==============================================================================
# MAIN - Demo runner
# ==============================================================================

def run_demos():
    """Esegue tutti i demo di testing"""
    
    print("\n" + "="*60)
    print("🧪 RUNNING TEST DEMONSTRATIONS")
    print("="*60)
    
    # 1. Run unittest example
    print("\n1️⃣ UNITTEST EXAMPLE")
    print("-"*40)
    
    # Create test suite
    suite = unittest.TestSuite()
    suite.addTest(TestCalculator('test_add_positive_numbers'))
    suite.addTest(TestCalculator('test_divide_by_zero'))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    # 2. Mock example
    print("\n2️⃣ MOCKING EXAMPLE")
    print("-"*40)
    
    mock_service = Mock()
    mock_service.get_data.return_value = {"status": "mocked"}
    
    print(f"Mock call result: {mock_service.get_data()}")
    print(f"Mock called: {mock_service.get_data.called}")
    print(f"Call count: {mock_service.get_data.call_count}")
    
    # 3. Async test example
    print("\n3️⃣ ASYNC TEST EXAMPLE")
    print("-"*40)
    
    async def async_test():
        fetcher = AsyncDataFetcher()
        result = await fetcher.fetch_data(999)
        print(f"Async fetch result: {result}")
    
    asyncio.run(async_test())
    
    # 4. Performance test
    print("\n4️⃣ PERFORMANCE TEST")
    print("-"*40)
    
    test_performance()
    
    # 5. Test builder example
    print("\n5️⃣ TEST BUILDER PATTERN")
    print("-"*40)
    
    user = UserBuilder()\
        .with_username("demo_user")\
        .with_email("demo@test.com")\
        .with_age(35)\
        .build()
    
    print(f"Built user: {json.dumps(user, indent=2)}")
    
    print("\n" + "="*60)
    print("✅ ALL TEST DEMONSTRATIONS COMPLETED!")
    print("="*60)

if __name__ == "__main__":
    run_demos()
