#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PYTHON PROFESSIONAL 1 - MODULE 2                          ║
║                    CODING CONVENTIONS AND BEST PRACTICES                      ║
║                    PCPP1-32-101 Section 2: 12% (5 domande)                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

SYLLABUS:
├── PCPP1 2.1 - PEP 1, PEP 8 (Style Guide)
├── PCPP1 2.2 - PEP 20 (Zen of Python)
├── PCPP1 2.3 - PEP 257 (Docstrings)
└── PCPP1 2.4 - PEP 484 (Type Hints)
"""

# ══════════════════════════════════════════════════════════════════════════════
# 2.1 PEP OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("2.1 PEP OVERVIEW")
print("=" * 70)

print("""
📋 PEP = Python Enhancement Proposal

PEP 1   - PEP Purpose and Guidelines
PEP 8   - Style Guide for Python Code
PEP 20  - The Zen of Python
PEP 257 - Docstring Conventions
PEP 484 - Type Hints
""")

# ══════════════════════════════════════════════════════════════════════════════
# 2.2 PEP 20 - THE ZEN OF PYTHON (ESAME!)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("2.2 PEP 20 - THE ZEN OF PYTHON (MEMORIZZA!)")
print("=" * 70)

import this  # Stampa lo Zen

print("""
📋 PRINCIPI CHIAVE (ESAME!):

1. Beautiful is better than ugly.
2. Explicit is better than implicit.
3. Simple is better than complex.
4. Complex is better than complicated.
5. Flat is better than nested.
6. Sparse is better than dense.
7. Readability counts.
8. Special cases aren't special enough to break the rules.
9. Although practicality beats purity.
10. Errors should never pass silently.
11. Unless explicitly silenced.
12. In the face of ambiguity, refuse the temptation to guess.
13. There should be one-- and preferably only one --obvious way to do it.
14. Now is better than never.
15. Although never is often better than *right* now.
16. If the implementation is hard to explain, it's a bad idea.
17. If the implementation is easy to explain, it may be a good idea.
18. Namespaces are one honking great idea -- let's do more of those!
""")

# ══════════════════════════════════════════════════════════════════════════════
# 2.3 PEP 8 - STYLE GUIDE (ESAME!)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("2.3 PEP 8 - STYLE GUIDE (ESAME!)")
print("=" * 70)

print("""
═══════════════════════════════════════════════════════════════════════
INDENTATION
═══════════════════════════════════════════════════════════════════════
✅ Use 4 spaces per indentation level
✅ Continuation lines: align with opening delimiter OR use hanging indent

# Good
def long_function_name(
        var_one, var_two,
        var_three, var_four):
    print(var_one)

# Good (aligned)
foo = long_function_name(var_one, var_two,
                         var_three, var_four)

═══════════════════════════════════════════════════════════════════════
LINE LENGTH
═══════════════════════════════════════════════════════════════════════
✅ Maximum 79 characters per line
✅ Docstrings/comments: 72 characters
✅ Use parentheses for line continuation

# Good
income = (gross_wages
          + taxable_interest
          + (dividends - qualified_dividends))

═══════════════════════════════════════════════════════════════════════
BLANK LINES
═══════════════════════════════════════════════════════════════════════
✅ 2 blank lines before/after top-level definitions (classes, functions)
✅ 1 blank line between methods in a class
✅ Use blank lines sparingly inside functions

═══════════════════════════════════════════════════════════════════════
IMPORTS
═══════════════════════════════════════════════════════════════════════
✅ Imports on separate lines
✅ Order: standard library, third-party, local
✅ Absolute imports preferred

# Good
import os
import sys

from subprocess import Popen, PIPE

import mypackage

# Bad
import os, sys

═══════════════════════════════════════════════════════════════════════
WHITESPACE
═══════════════════════════════════════════════════════════════════════
✅ No space inside parentheses: func(arg), not func( arg )
✅ No space before comma: func(a, b), not func(a , b)
✅ Space around operators: x = 1, not x=1
✅ No space around = in keyword arguments: func(arg=value)

# Good
spam(ham[1], {eggs: 2})
x = 1
y = x + 1
def func(arg=default):

# Bad
spam( ham[ 1 ], { eggs: 2 } )
x=1
y = x+1
def func(arg = default):

═══════════════════════════════════════════════════════════════════════
NAMING CONVENTIONS
═══════════════════════════════════════════════════════════════════════
┌─────────────────────┬────────────────────────────────────────┐
│ Type                │ Convention                             │
├─────────────────────┼────────────────────────────────────────┤
│ Packages            │ lowercase (no underscores if possible) │
│ Modules             │ lowercase_with_underscores             │
│ Classes             │ CapitalizedWords (CamelCase)           │
│ Exceptions          │ CapitalizedWords (Error suffix)        │
│ Functions           │ lowercase_with_underscores             │
│ Variables           │ lowercase_with_underscores             │
│ Constants           │ UPPERCASE_WITH_UNDERSCORES             │
│ Instance vars       │ lowercase_with_underscores             │
│ Protected           │ _single_leading_underscore             │
│ Private             │ __double_leading_underscore            │
│ Magic methods       │ __double_underscore__                  │
└─────────────────────┴────────────────────────────────────────┘
""")

# ══════════════════════════════════════════════════════════════════════════════
# 2.4 PEP 257 - DOCSTRINGS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("2.4 PEP 257 - DOCSTRINGS")
print("=" * 70)

print('''
📋 DOCSTRING CONVENTIONS:

# One-line docstring
def simple_function():
    """Return the sum of two numbers."""
    pass

# Multi-line docstring
def complex_function(arg1, arg2):
    """
    Summary line.
    
    Extended description of function.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: If arg1 is invalid
    """
    pass

# Class docstring
class MyClass:
    """
    Summary of class.
    
    Longer description if needed.
    
    Attributes:
        attr1: Description
        attr2: Description
    """
    pass
''')

# Example with docstring
def calculate_area(width: float, height: float) -> float:
    """
    Calculate the area of a rectangle.
    
    Args:
        width: The width of the rectangle
        height: The height of the rectangle
    
    Returns:
        The area of the rectangle
    
    Raises:
        ValueError: If width or height is negative
    """
    if width < 0 or height < 0:
        raise ValueError("Dimensions must be positive")
    return width * height

print(f"calculate_area.__doc__:\n{calculate_area.__doc__}")

# ══════════════════════════════════════════════════════════════════════════════
# 2.5 PEP 484 - TYPE HINTS (ESAME!)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("2.5 PEP 484 - TYPE HINTS (ESAME!)")
print("=" * 70)

from typing import List, Dict, Tuple, Optional, Union, Callable, Any

print("""
📋 BASIC TYPE HINTS:
""")

# Variable annotations
name: str = "Marco"
age: int = 25
prices: List[float] = [19.99, 29.99]
scores: Dict[str, int] = {"math": 95, "english": 87}

print(f"name: str = {name}")
print(f"prices: List[float] = {prices}")
print(f"scores: Dict[str, int] = {scores}")

# Function annotations
def greet(name: str, times: int = 1) -> str:
    """Function with type hints."""
    return f"Hello, {name}! " * times

print(f"\ngreet('Marco', 2) = {greet('Marco', 2)}")

# Complex types
def process(
    items: List[int],
    callback: Callable[[int], str],
    config: Optional[Dict[str, Any]] = None
) -> Tuple[str, int]:
    """Function with complex type hints."""
    return ("result", len(items))

print("""
📋 TYPING MODULE:

List[int]           - Lista di interi
Dict[str, int]      - Dict con chiavi str, valori int
Tuple[int, str]     - Tupla con int e str
Optional[str]       - str oppure None (equivale a Union[str, None])
Union[int, str]     - int oppure str
Callable[[int], str]- Funzione che prende int, restituisce str
Any                 - Qualsiasi tipo
""")

# Optional = può essere None
def find_user(user_id: int) -> Optional[str]:
    """Returns username or None if not found."""
    users = {1: "Marco", 2: "Anna"}
    return users.get(user_id)

print(f"find_user(1) = {find_user(1)}")
print(f"find_user(99) = {find_user(99)}")

# Union = uno dei tipi elencati
def process_id(id: Union[int, str]) -> str:
    """Accepts int or str."""
    return f"Processing: {id}"

print(f"process_id(123) = {process_id(123)}")
print(f"process_id('abc') = {process_id('abc')}")

# Type aliases
Vector = List[float]
Matrix = List[Vector]

def scale(v: Vector, factor: float) -> Vector:
    return [x * factor for x in v]

# ══════════════════════════════════════════════════════════════════════════════
# 2.6 LINTERS AND TOOLS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("2.6 LINTERS AND TOOLS")
print("=" * 70)

print("""
📋 TOOLS FOR CODE QUALITY:

LINTERS (check style):
  - pylint     - Comprehensive checker
  - flake8     - Style guide enforcement
  - pycodestyle (formerly pep8) - PEP 8 checker

FORMATTERS (auto-fix style):
  - black      - Opinionated formatter
  - autopep8   - PEP 8 auto-formatter
  - yapf       - Yet Another Python Formatter

TYPE CHECKERS:
  - mypy       - Static type checker
  - pyright    - Microsoft type checker
  - pytype     - Google type checker

USAGE:
  $ pip install pylint black mypy
  $ pylint mycode.py
  $ black mycode.py
  $ mypy mycode.py
""")

# ══════════════════════════════════════════════════════════════════════════════
# QUIZ
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("QUIZ DI VERIFICA")
print("=" * 70)

print("""
Q1. Quanti spazi per indentazione raccomanda PEP 8?
    A) 2  B) 4  C) Tab  D) 8
    → RISPOSTA: B

Q2. Lunghezza massima linea secondo PEP 8?
    A) 72  B) 79  C) 80  D) 120
    → RISPOSTA: B

Q3. Come si nominano le classi secondo PEP 8?
    A) lowercase  B) UPPERCASE  C) CamelCase  D) snake_case
    → RISPOSTA: C

Q4. Come si nominano le costanti?
    A) lowercase  B) UPPERCASE  C) CamelCase  D) snake_case
    → RISPOSTA: B

Q5. Optional[str] equivale a?
    A) str  B) None  C) Union[str, None]  D) List[str]
    → RISPOSTA: C

Q6. Quale tool controlla i type hints staticamente?
    A) pylint  B) black  C) mypy  D) flake8
    → RISPOSTA: C

Q7. "import this" mostra?
    A) PEP 8  B) PEP 20  C) PEP 257  D) PEP 484
    → RISPOSTA: B (Zen of Python)

Q8. Quante linee vuote tra funzioni top-level?
    A) 0  B) 1  C) 2  D) 3
    → RISPOSTA: C
""")

print("\n" + "=" * 70)
print("BEST PRACTICES MODULE COMPLETATO!")
print("=" * 70)
