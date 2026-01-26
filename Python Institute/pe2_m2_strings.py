#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PYTHON ESSENTIALS 2 - MODULE 2                            ║
║                    STRINGS (ADVANCED)                                         ║
║                    PCAP-31-03 Section 3: 18% (8 domande)                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

SYLLABUS:
├── PCAP 3.1 - Encoding: ASCII, Unicode, UTF-8, code points, escape sequences
├── PCAP 3.2 - String operations: ord(), chr(), indexing, slicing, iteration
└── PCAP 3.3 - String methods: .isxxx(), .join(), .split(), .strip(), .find()
"""

# ══════════════════════════════════════════════════════════════════════════════
# 3.1 ENCODING STANDARDS
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("3.1 ENCODING: ASCII, UNICODE, UTF-8")
print("=" * 70)

print("""
📋 ASCII (American Standard Code for Information Interchange)
   - 128 caratteri (0-127)
   - 7 bit per carattere
   - Solo caratteri inglesi base
   - 'A' = 65, 'a' = 97, '0' = 48

📋 UNICODE
   - Standard universale per TUTTI i caratteri
   - 143,000+ caratteri da tutte le lingue
   - Ogni carattere ha un "code point" (es. U+0041 = 'A')
   - Python 3 usa Unicode internamente per le stringhe

📋 UTF-8 (Unicode Transformation Format - 8 bit)
   - Encoding per salvare Unicode in file/rete
   - Variabile: 1-4 byte per carattere
   - ASCII compatibile (primi 128 caratteri = 1 byte)
   - Standard de facto per il web
""")

# ══════════════════════════════════════════════════════════════════════════════
# 3.2 ord() AND chr() - FONDAMENTALI!
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("3.2 ord() AND chr() - CONVERSIONE CARATTERE ↔ CODICE")
print("=" * 70)

# ord(char) → code point (intero)
print("\n📐 ord(char) → restituisce il code point (intero)")
print(f"   ord('A') = {ord('A')}")      # 65
print(f"   ord('a') = {ord('a')}")      # 97
print(f"   ord('0') = {ord('0')}")      # 48
print(f"   ord(' ') = {ord(' ')}")      # 32
print(f"   ord('€') = {ord('€')}")      # 8364

# chr(code) → carattere
print("\n📐 chr(code) → restituisce il carattere")
print(f"   chr(65) = '{chr(65)}'")      # 'A'
print(f"   chr(97) = '{chr(97)}'")      # 'a'
print(f"   chr(48) = '{chr(48)}'")      # '0'
print(f"   chr(8364) = '{chr(8364)}'")  # '€'

# Relazione importante
print("\n⚠️ RELAZIONE FONDAMENTALE:")
print(f"   ord('A') + 32 = {ord('A') + 32} = ord('a')")
print(f"   Differenza maiuscola-minuscola = 32")

# ══════════════════════════════════════════════════════════════════════════════
# 3.3 ESCAPE SEQUENCES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("3.3 ESCAPE SEQUENCES")
print("=" * 70)

print("""
┌────────────┬─────────────────────────────────────┐
│ Sequenza   │ Significato                         │
├────────────┼─────────────────────────────────────┤
│ \\n         │ Newline (a capo)                    │
│ \\t         │ Tab                                 │
│ \\\\         │ Backslash letterale                 │
│ \\'         │ Apostrofo                           │
│ \\"         │ Virgolette                          │
│ \\r         │ Carriage return                     │
│ \\0         │ Null character                      │
│ \\xHH       │ Carattere hex (es. \\x41 = 'A')     │
│ \\uHHHH     │ Unicode 16-bit (es. \\u0041 = 'A')  │
└────────────┴─────────────────────────────────────┘
""")

print("Esempi:")
print("   'Hello\\nWorld' →")
print("Hello\nWorld")
print(f"   'Tab\\there' → 'Tab\there'")
print(f"   '\\x41\\x42\\x43' → '{'ABC'}'")

# Raw strings (no escape)
print("\n📋 RAW STRINGS (r'...'):")
print(f"   r'C:\\new\\folder' = {r'C:\new\folder'}")

# ══════════════════════════════════════════════════════════════════════════════
# 3.4 STRING INDEXING AND SLICING
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("3.4 STRING INDEXING AND SLICING")
print("=" * 70)

s = "PYTHON"
print(f"\nStringa: '{s}'")
print("Indici:   0 1 2 3 4 5")
print("Negativi:-6-5-4-3-2-1")

print(f"\n📐 Indexing:")
print(f"   s[0] = '{s[0]}'")
print(f"   s[-1] = '{s[-1]}'")
print(f"   s[-2] = '{s[-2]}'")

print(f"\n📐 Slicing [start:stop:step]:")
print(f"   s[0:3] = '{s[0:3]}'")       # PYT
print(f"   s[2:5] = '{s[2:5]}'")       # THO
print(f"   s[:3] = '{s[:3]}'")         # PYT
print(f"   s[3:] = '{s[3:]}'")         # HON
print(f"   s[::2] = '{s[::2]}'")       # PTO (ogni 2)
print(f"   s[::-1] = '{s[::-1]}'")     # NOHTYP (reverse!)
print(f"   s[1:5:2] = '{s[1:5:2]}'")   # YH

# Slice fuori range → niente errore!
print(f"\n⚠️ Slice fuori range:")
print(f"   s[100:] = '{s[100:]}'")     # '' (stringa vuota)

# ══════════════════════════════════════════════════════════════════════════════
# 3.5 STRING IMMUTABILITY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("3.5 STRING IMMUTABILITY")
print("=" * 70)

print("""
⚠️ LE STRINGHE SONO IMMUTABILI!

   s = "Hello"
   s[0] = 'J'  # TypeError: 'str' object does not support item assignment

   Per modificare, crea una NUOVA stringa:
   s = "J" + s[1:]  # "Jello"
""")

# ══════════════════════════════════════════════════════════════════════════════
# 3.6 STRING METHODS - isXXX()
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("3.6 STRING METHODS - isXXX() (ESAME!)")
print("=" * 70)

print("""
┌────────────────┬─────────────────────────────────────┐
│ Metodo         │ True se...                          │
├────────────────┼─────────────────────────────────────┤
│ .isalpha()     │ Solo lettere (a-z, A-Z)             │
│ .isdigit()     │ Solo cifre (0-9)                    │
│ .isalnum()     │ Lettere O cifre                     │
│ .isspace()     │ Solo spazi/tab/newline              │
│ .isupper()     │ Tutte maiuscole                     │
│ .islower()     │ Tutte minuscole                     │
│ .istitle()     │ Title Case (Es: "Hello World")      │
└────────────────┴─────────────────────────────────────┘
""")

# Esempi
print("Esempi:")
print(f"   'Hello'.isalpha() = {'Hello'.isalpha()}")      # True
print(f"   'Hello123'.isalpha() = {'Hello123'.isalpha()}")  # False
print(f"   '123'.isdigit() = {'123'.isdigit()}")          # True
print(f"   'Hello123'.isalnum() = {'Hello123'.isalnum()}")  # True
print(f"   '   '.isspace() = {'   '.isspace()}")          # True
print(f"   'HELLO'.isupper() = {'HELLO'.isupper()}")      # True
print(f"   'hello'.islower() = {'hello'.islower()}")      # True
print(f"   'Hello World'.istitle() = {'Hello World'.istitle()}")  # True

# ⚠️ Stringhe vuote!
print(f"\n⚠️ Stringa vuota: ''.isalpha() = {''.isalpha()}")  # False

# ══════════════════════════════════════════════════════════════════════════════
# 3.7 STRING METHODS - CASE CONVERSION
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("3.7 CASE CONVERSION")
print("=" * 70)

s = "Hello World"
print(f"Stringa: '{s}'")
print(f"   .upper() = '{s.upper()}'")      # HELLO WORLD
print(f"   .lower() = '{s.lower()}'")      # hello world
print(f"   .title() = '{s.title()}'")      # Hello World
print(f"   .capitalize() = '{s.capitalize()}'")  # Hello world
print(f"   .swapcase() = '{s.swapcase()}'")      # hELLO wORLD

# ══════════════════════════════════════════════════════════════════════════════
# 3.8 STRING METHODS - SPLIT AND JOIN (ESAME!)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("3.8 SPLIT AND JOIN (ESAME!)")
print("=" * 70)

# SPLIT - Divide stringa in lista
print("\n📐 .split(separator)")
s = "one,two,three"
print(f"   '{s}'.split(',') = {s.split(',')}")

s = "hello world python"
print(f"   '{s}'.split() = {s.split()}")  # Default: spazi

s = "a::b::c"
print(f"   '{s}'.split('::') = {s.split('::')}")

# Split con maxsplit
s = "a,b,c,d,e"
print(f"   '{s}'.split(',', 2) = {s.split(',', 2)}")  # Max 2 split

# JOIN - Unisce lista in stringa
print("\n📐 separator.join(list)")
words = ['one', 'two', 'three']
print(f"   ','.join({words}) = '{','.join(words)}'")
print(f"   ' - '.join({words}) = '{' - '.join(words)}'")
print(f"   ''.join({words}) = '{''.join(words)}'")

# ══════════════════════════════════════════════════════════════════════════════
# 3.9 STRING METHODS - STRIP (ESAME!)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("3.9 STRIP (ESAME!)")
print("=" * 70)

s = "   Hello World   "
print(f"Stringa: '{s}'")
print(f"   .strip() = '{s.strip()}'")      # Rimuove entrambi i lati
print(f"   .lstrip() = '{s.lstrip()}'")    # Solo sinistra
print(f"   .rstrip() = '{s.rstrip()}'")    # Solo destra

# Strip con caratteri specifici
s = "***Hello***"
print(f"\n'{s}'.strip('*') = '{s.strip('*')}'")

# ══════════════════════════════════════════════════════════════════════════════
# 3.10 STRING METHODS - FIND AND INDEX (ESAME!)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("3.10 FIND, RFIND, INDEX (ESAME!)")
print("=" * 70)

s = "Hello World Hello"
print(f"Stringa: '{s}'")

# FIND - Restituisce indice o -1 se non trovato
print("\n📐 .find(sub) - Restituisce -1 se non trovato")
print(f"   .find('World') = {s.find('World')}")    # 6
print(f"   .find('Hello') = {s.find('Hello')}")    # 0 (primo)
print(f"   .find('xyz') = {s.find('xyz')}")        # -1

# RFIND - Cerca da destra
print("\n📐 .rfind(sub) - Cerca da DESTRA")
print(f"   .rfind('Hello') = {s.rfind('Hello')}")  # 12 (ultimo)

# INDEX - Come find ma solleva ValueError se non trovato
print("\n📐 .index(sub) - Solleva ValueError se non trovato")
print(f"   .index('World') = {s.index('World')}")  # 6
# s.index('xyz')  # ValueError!

# ══════════════════════════════════════════════════════════════════════════════
# 3.11 STRING METHODS - REPLACE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("3.11 REPLACE")
print("=" * 70)

s = "Hello World World"
print(f"Stringa: '{s}'")
print(f"   .replace('World', 'Python') = '{s.replace('World', 'Python')}'")
print(f"   .replace('World', 'Python', 1) = '{s.replace('World', 'Python', 1)}'")

# ══════════════════════════════════════════════════════════════════════════════
# 3.12 STRING METHODS - STARTSWITH, ENDSWITH
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("3.12 STARTSWITH, ENDSWITH")
print("=" * 70)

s = "Hello World"
print(f"Stringa: '{s}'")
print(f"   .startswith('Hello') = {s.startswith('Hello')}")  # True
print(f"   .startswith('World') = {s.startswith('World')}")  # False
print(f"   .endswith('World') = {s.endswith('World')}")      # True
print(f"   .endswith(('World', 'Python')) = {s.endswith(('World', 'Python'))}")  # Tupla!

# ══════════════════════════════════════════════════════════════════════════════
# 3.13 STRING FORMATTING
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("3.13 STRING FORMATTING")
print("=" * 70)

name = "Marco"
age = 25

# f-strings (Python 3.6+)
print(f"\n📐 f-strings:")
print(f"   f'Name: {{name}}' = f'Name: {name}'")
print(f"   f'Age: {{age}}' = f'Age: {age}'")
print(f"   f'{{2+2}}' = f'{2+2}'")

# .format()
print("\n📐 .format():")
print(f"   'Name: {{}}'.format(name) = '{'Name: {}'.format(name)}'")

# % formatting (old style)
print("\n📐 % formatting:")
print(f"   'Name: %s' % name = '{'Name: %s' % name}'")

# ══════════════════════════════════════════════════════════════════════════════
# QUIZ
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("QUIZ DI VERIFICA")
print("=" * 70)

print("""
Q1. ord('A') = ?
    A) 97  B) 65  C) 48  D) 32
    → RISPOSTA: B

Q2. chr(97) = ?
    A) 'A'  B) 'a'  C) '9'  D) ' '
    → RISPOSTA: B

Q3. 'Hello'[1:4] = ?
    A) 'Hell'  B) 'ell'  C) 'ello'  D) 'Hel'
    → RISPOSTA: B

Q4. 'Python'[::-1] = ?
    A) 'Python'  B) 'nohtyP'  C) 'Pytho'  D) Error
    → RISPOSTA: B

Q5. '123'.isdigit() = ?
    A) True  B) False  C) '123'  D) Error
    → RISPOSTA: A

Q6. 'hello world'.split() = ?
    A) ['hello world']  B) ['hello', 'world']  C) 'hello world'  D) Error
    → RISPOSTA: B

Q7. '-'.join(['a', 'b', 'c']) = ?
    A) 'abc'  B) 'a-b-c'  C) ['a-b-c']  D) Error
    → RISPOSTA: B

Q8. 'Hello World'.find('xyz') = ?
    A) 0  B) -1  C) None  D) ValueError
    → RISPOSTA: B
""")

print("\n" + "=" * 70)
print("MODULO 2 COMPLETATO! → Prossimo: pe2_m3_oop.py")
print("=" * 70)
