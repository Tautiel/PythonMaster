#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PYTHON ESSENTIALS 2 - MODULE 1                            ║
║                    MODULES AND PACKAGES                                       ║
║                    PCAP-31-03 Section 1: 12% (6 domande)                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

SYLLABUS:
├── PCAP 1.1 - Import variants (import, from, as, *)
├── PCAP 1.2 - math module (ceil, floor, trunc, factorial, hypot, sqrt)
├── PCAP 1.3 - random module (random, seed, choice, sample)
├── PCAP 1.4 - platform module (platform, machine, processor, system, version)
└── PCAP 1.5 - User-defined modules/packages (__pycache__, __name__, __init__.py)
"""

import math
import random
import platform
import sys

# ══════════════════════════════════════════════════════════════════════════════
# 1.1 IMPORT VARIANTS
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("1.1 IMPORT VARIANTS")
print("=" * 70)

# Variant 1: import module
import math
print(f"import math → math.pi = {math.pi}")

# Variant 2: from module import name
from math import sqrt, ceil
print(f"from math import sqrt → sqrt(16) = {sqrt(16)}")

# Variant 3: import as (alias)
import math as m
print(f"import math as m → m.e = {m.e}")

from math import factorial as fact
print(f"from math import factorial as fact → fact(5) = {fact(5)}")

# Variant 4: from module import * (SCONSIGLIATO)
# from math import *  # Importa TUTTO - namespace pollution!
print("from module import * → SCONSIGLIATO (namespace pollution)")

# ══════════════════════════════════════════════════════════════════════════════
# 1.2 dir() AND sys.path
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("1.2 dir() AND sys.path")
print("=" * 70)

# dir() - lista nomi nel modulo
print(f"dir(math)[:5] = {dir(math)[:5]}")
print(f"Totale nomi in math: {len(dir(math))}")

# sys.path - dove Python cerca i moduli
print(f"\nsys.path[0] = {sys.path[0]}")
print(f"Totale percorsi: {len(sys.path)}")

# ══════════════════════════════════════════════════════════════════════════════
# 1.3 MATH MODULE (ESAME!)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("1.3 MATH MODULE (MEMORIZZA QUESTI!)")
print("=" * 70)

# CEIL - Arrotonda verso l'ALTO
print("\n📐 math.ceil()")
print(f"   ceil(4.1) = {math.ceil(4.1)}")      # 5
print(f"   ceil(-4.1) = {math.ceil(-4.1)}")    # -4 (verso alto!)

# FLOOR - Arrotonda verso il BASSO
print("\n📐 math.floor()")
print(f"   floor(4.9) = {math.floor(4.9)}")    # 4
print(f"   floor(-4.9) = {math.floor(-4.9)}")  # -5 (verso basso!)

# TRUNC - Tronca verso ZERO
print("\n📐 math.trunc()")
print(f"   trunc(4.9) = {math.trunc(4.9)}")    # 4
print(f"   trunc(-4.9) = {math.trunc(-4.9)}")  # -4 (verso zero!)

# ⚠️ DIFFERENZA CRITICA!
print("\n⚠️ DIFFERENZA floor vs trunc (NEGATIVI):")
print(f"   floor(-4.5) = {math.floor(-4.5)}")  # -5
print(f"   trunc(-4.5) = {math.trunc(-4.5)}")  # -4

# FACTORIAL
print("\n📐 math.factorial()")
print(f"   factorial(0) = {math.factorial(0)}")  # 1
print(f"   factorial(5) = {math.factorial(5)}")  # 120

# HYPOT - Ipotenusa (sqrt(a² + b²))
print("\n📐 math.hypot()")
print(f"   hypot(3, 4) = {math.hypot(3, 4)}")   # 5.0

# SQRT
print("\n📐 math.sqrt()")
print(f"   sqrt(16) = {math.sqrt(16)}")         # 4.0

# ══════════════════════════════════════════════════════════════════════════════
# 1.4 RANDOM MODULE (ESAME!)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("1.4 RANDOM MODULE (MEMORIZZA QUESTI!)")
print("=" * 70)

# RANDOM - Float tra 0.0 e 1.0
print("\n🎲 random.random()")
print(f"   random() = {random.random():.4f}")

# SEED - Riproducibilità
print("\n🎲 random.seed()")
random.seed(42)
r1 = random.random()
random.seed(42)
r2 = random.random()
print(f"   Stesso seed, stesso risultato: {r1 == r2}")

# CHOICE - UN elemento casuale
print("\n🎲 random.choice()")
colors = ["red", "green", "blue"]
print(f"   choice({colors}) = {random.choice(colors)}")

# SAMPLE - N elementi SENZA ripetizione
print("\n🎲 random.sample()")
nums = [1, 2, 3, 4, 5]
print(f"   sample({nums}, 3) = {random.sample(nums, 3)}")

# Altre utili
print("\n🎲 Altre funzioni:")
print(f"   randint(1, 10) = {random.randint(1, 10)}")  # Include 10!
print(f"   randrange(1, 10) = {random.randrange(1, 10)}")  # Esclude 10

# ══════════════════════════════════════════════════════════════════════════════
# 1.5 PLATFORM MODULE (ESAME!)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("1.5 PLATFORM MODULE")
print("=" * 70)

print(f"\n💻 platform() = {platform.platform()}")
print(f"💻 machine() = {platform.machine()}")
print(f"💻 processor() = {platform.processor()}")
print(f"💻 system() = {platform.system()}")
print(f"💻 version() = {platform.version()[:50]}...")
print(f"💻 python_implementation() = {platform.python_implementation()}")
print(f"💻 python_version_tuple() = {platform.python_version_tuple()}")

# ══════════════════════════════════════════════════════════════════════════════
# 1.6 USER-DEFINED MODULES & PACKAGES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("1.6 USER-DEFINED MODULES & PACKAGES")
print("=" * 70)

print("""
📁 STRUTTURA PACKAGE:
    mypackage/
    ├── __init__.py      ← OBBLIGATORIO! Rende directory un package
    ├── module1.py
    └── subpkg/
        ├── __init__.py  ← Anche per subpackage
        └── module2.py

📋 __name__ VARIABLE:
    - Se file eseguito: __name__ == "__main__"
    - Se file importato: __name__ == "nome_modulo"
    
    Pattern comune:
    if __name__ == "__main__":
        main()  # Esegue solo se run direttamente

📋 __pycache__:
    - Directory con bytecode compilato (.pyc)
    - Velocizza import successivi
    - Puoi eliminarla, viene ricreata
""")

print(f"\nIn questo file: __name__ = '{__name__}'")

# ══════════════════════════════════════════════════════════════════════════════
# QUIZ SECTION 1
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("QUIZ DI VERIFICA")
print("=" * 70)

quiz = """
Q1. Quale import per usare sqrt() senza prefisso?
    A) import math  B) from math import sqrt  C) import sqrt  D) math.sqrt
    → RISPOSTA: B

Q2. math.floor(-4.5) = ?
    A) -4  B) -5  C) 4  D) 5
    → RISPOSTA: B (verso il basso)

Q3. math.trunc(-4.5) = ?
    A) -4  B) -5  C) 4  D) 5
    → RISPOSTA: A (verso zero)

Q4. Quale funzione sceglie N elementi SENZA ripetizione?
    A) choice()  B) sample()  C) randint()  D) random()
    → RISPOSTA: B

Q5. Cosa restituisce platform.python_version_tuple()?
    A) String  B) Int  C) Tuple  D) List
    → RISPOSTA: C

Q6. Quale file rende directory un package?
    A) package.py  B) __main__.py  C) __init__.py  D) setup.py
    → RISPOSTA: C

Q7. Quando __name__ == "__main__"?
    A) Sempre  B) Se importato  C) Se eseguito direttamente  D) Mai
    → RISPOSTA: C

Q8. from module import * importa nomi con _?
    A) Sì  B) No  C) Solo pubblici  D) Error
    → RISPOSTA: B
"""
print(quiz)

# ══════════════════════════════════════════════════════════════════════════════
# ESERCIZI
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ESERCIZI PRATICI")
print("=" * 70)

# Esercizio 1
print("\n📝 Esercizio 1: Calcola l'ipotenusa di un triangolo con cateti 5 e 12")
result = math.hypot(5, 12)
print(f"   Soluzione: math.hypot(5, 12) = {result}")

# Esercizio 2
print("\n📝 Esercizio 2: Genera 3 numeri casuali tra 1 e 100 senza ripetizione")
result = random.sample(range(1, 101), 3)
print(f"   Soluzione: random.sample(range(1, 101), 3) = {result}")

# Esercizio 3
print("\n📝 Esercizio 3: Arrotonda -7.3 verso l'alto, il basso, e verso zero")
print(f"   ceil(-7.3) = {math.ceil(-7.3)}")
print(f"   floor(-7.3) = {math.floor(-7.3)}")
print(f"   trunc(-7.3) = {math.trunc(-7.3)}")

print("\n" + "=" * 70)
print("MODULO 1 COMPLETATO! → Prossimo: pe2_m2_strings.py")
print("=" * 70)
