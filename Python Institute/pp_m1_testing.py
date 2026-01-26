#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PYTHON PROFESSIONAL 2 - MODULE 1                          ║
║                    TESTING (unittest, pytest)                                 ║
║                    PCPP2 Prep - Testing is expected to be ~25% of exam       ║
╚══════════════════════════════════════════════════════════════════════════════╝

NOTE: PCPP2 exam is still "In Development" per Python Institute.
This module covers expected testing topics.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

# ══════════════════════════════════════════════════════════════════════════════
# 1.1 UNITTEST BASICS
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("1.1 UNITTEST BASICS")
print("=" * 70)

print("""
📋 UNITTEST STRUCTURE:

import unittest

class TestMyFunction(unittest.TestCase):
    
    def setUp(self):
        '''Runs before EACH test'''
        self.data = [1, 2, 3]
    
    def tearDown(self):
        '''Runs after EACH test'''
        pass
    
    def test_something(self):
        '''Test method must start with 'test_' '''
        self.assertEqual(1 + 1, 2)

if __name__ == '__main__':
    unittest.main()
""")

# Demo test class
class Calculator:
    def add(self, a, b):
        return a + b
    
    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

class TestCalculator(unittest.TestCase):
    def setUp(self):
        self.calc = Calculator()
    
    def test_add(self):
        self.assertEqual(self.calc.add(2, 3), 5)
    
    def test_add_negative(self):
        self.assertEqual(self.calc.add(-1, 1), 0)
    
    def test_divide(self):
        self.assertEqual(self.calc.divide(10, 2), 5)
    
    def test_divide_by_zero(self):
        with self.assertRaises(ValueError):
            self.calc.divide(10, 0)

# ══════════════════════════════════════════════════════════════════════════════
# 1.2 ASSERTION METHODS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("1.2 ASSERTION METHODS")
print("=" * 70)

print("""
📋 COMMON ASSERTIONS:

EQUALITY:
  assertEqual(a, b)       - a == b
  assertNotEqual(a, b)    - a != b
  assertAlmostEqual(a,b)  - round(a-b, 7) == 0

BOOLEAN:
  assertTrue(x)           - bool(x) is True
  assertFalse(x)          - bool(x) is False

IDENTITY:
  assertIs(a, b)          - a is b
  assertIsNot(a, b)       - a is not b
  assertIsNone(x)         - x is None
  assertIsNotNone(x)      - x is not None

MEMBERSHIP:
  assertIn(a, b)          - a in b
  assertNotIn(a, b)       - a not in b

TYPE:
  assertIsInstance(a, b)  - isinstance(a, b)

EXCEPTIONS:
  assertRaises(exc)       - Raises exception
  assertRaisesRegex(exc, r) - Raises with message matching regex

COMPARISON:
  assertGreater(a, b)     - a > b
  assertGreaterEqual(a,b) - a >= b
  assertLess(a, b)        - a < b
  assertLessEqual(a, b)   - a <= b
""")

# ══════════════════════════════════════════════════════════════════════════════
# 1.3 TEST FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("1.3 TEST FIXTURES")
print("=" * 70)

print("""
📋 FIXTURE METHODS:

METHOD LEVEL:
  setUp()      - Before each test method
  tearDown()   - After each test method

CLASS LEVEL:
  setUpClass()    - Before all tests in class (classmethod)
  tearDownClass() - After all tests in class (classmethod)

MODULE LEVEL:
  setUpModule()    - Before all tests in module (function)
  tearDownModule() - After all tests in module (function)

EXAMPLE:
class TestDatabase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        '''Connect to test database once'''
        cls.db = connect_db()
    
    @classmethod
    def tearDownClass(cls):
        '''Disconnect once'''
        cls.db.close()
    
    def setUp(self):
        '''Start transaction before each test'''
        self.db.begin()
    
    def tearDown(self):
        '''Rollback after each test'''
        self.db.rollback()
""")

# ══════════════════════════════════════════════════════════════════════════════
# 1.4 MOCKING
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("1.4 MOCKING")
print("=" * 70)

print("""
📋 MOCK OBJECTS:

from unittest.mock import Mock, patch, MagicMock

# Basic Mock
mock = Mock()
mock.method.return_value = 42
result = mock.method()  # Returns 42

# Assert calls
mock.method.assert_called()
mock.method.assert_called_once()
mock.method.assert_called_with(arg1, arg2)

# Side effects
mock.method.side_effect = ValueError("Error!")
mock.method.side_effect = [1, 2, 3]  # Returns sequentially
""")

# Mock example
def get_user_data(user_id, api_client):
    response = api_client.get(f"/users/{user_id}")
    return response.json()

# Test with mock
mock_api = Mock()
mock_api.get.return_value.json.return_value = {"name": "Marco", "id": 1}

result = get_user_data(1, mock_api)
print(f"Mock result: {result}")
mock_api.get.assert_called_once_with("/users/1")

# Patch decorator
print("""
📋 PATCH DECORATOR:

@patch('module.ClassName')
def test_something(self, MockClass):
    instance = MockClass.return_value
    instance.method.return_value = 'mocked'
    # Test code...

# Or as context manager
with patch('module.function') as mock_func:
    mock_func.return_value = 42
    result = module.function()
""")

# ══════════════════════════════════════════════════════════════════════════════
# 1.5 PYTEST BASICS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("1.5 PYTEST BASICS")
print("=" * 70)

print("""
📋 PYTEST vs UNITTEST:

UNITTEST:
- Class-based
- Verbose assertions (assertEqual, assertTrue)
- Built into Python

PYTEST:
- Function-based (simpler)
- Plain assert statements
- Better output, plugins
- pip install pytest

PYTEST EXAMPLE:
```python
# test_calculator.py

def test_add():
    assert 1 + 1 == 2

def test_divide():
    assert 10 / 2 == 5

def test_divide_by_zero():
    import pytest
    with pytest.raises(ZeroDivisionError):
        1 / 0
```

RUN: pytest test_calculator.py
""")

# ══════════════════════════════════════════════════════════════════════════════
# 1.6 PYTEST FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("1.6 PYTEST FIXTURES")
print("=" * 70)

print("""
📋 PYTEST FIXTURES:

import pytest

@pytest.fixture
def sample_data():
    '''Fixture function'''
    return [1, 2, 3, 4, 5]

def test_sum(sample_data):
    '''sample_data is injected automatically'''
    assert sum(sample_data) == 15

@pytest.fixture(scope='module')
def database():
    '''Scope: function, class, module, session'''
    db = connect()
    yield db  # yield = teardown after
    db.close()

# conftest.py - Shared fixtures
# Place fixtures in conftest.py for automatic discovery
""")

# ══════════════════════════════════════════════════════════════════════════════
# 1.7 PARAMETRIZED TESTS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("1.7 PARAMETRIZED TESTS")
print("=" * 70)

print("""
📋 PYTEST PARAMETRIZE:

import pytest

@pytest.mark.parametrize("input,expected", [
    (1, 1),
    (2, 4),
    (3, 9),
    (4, 16),
])
def test_square(input, expected):
    assert input ** 2 == expected

# Multiple parameters
@pytest.mark.parametrize("a,b,expected", [
    (1, 2, 3),
    (0, 0, 0),
    (-1, 1, 0),
])
def test_add(a, b, expected):
    assert a + b == expected
""")

# ══════════════════════════════════════════════════════════════════════════════
# 1.8 TEST COVERAGE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("1.8 TEST COVERAGE")
print("=" * 70)

print("""
📋 COVERAGE:

pip install coverage pytest-cov

# With unittest
coverage run -m unittest discover
coverage report
coverage html  # Generates HTML report

# With pytest
pytest --cov=mypackage tests/
pytest --cov=mypackage --cov-report=html tests/

# Coverage targets:
- 80%+ is good
- 90%+ is excellent
- 100% is ideal but not always practical
""")

# ══════════════════════════════════════════════════════════════════════════════
# QUIZ
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("QUIZ")
print("=" * 70)

print("""
Q1. Test method in unittest must start with?
    → test_

Q2. setUp() runs when?
    → Before each test method

Q3. @classmethod setUpClass runs when?
    → Once before all tests in class

Q4. Mock.return_value sets?
    → What mock returns when called

Q5. patch() decorator does what?
    → Replaces object temporarily during test

Q6. pytest fixture scope='module' means?
    → Fixture created once per module

Q7. @pytest.mark.parametrize does what?
    → Runs test with multiple inputs

Q8. assertRaises checks?
    → That exception is raised
""")

print("\n" + "=" * 70)
print("TESTING MODULE COMPLETE!")
print("=" * 70)
