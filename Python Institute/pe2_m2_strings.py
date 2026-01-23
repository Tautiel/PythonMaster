"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    PYTHON ESSENTIALS 2 - MODULE 2                            ║
║                    Strings, String Methods, Exceptions                       ║
║                                                                              ║
║                     Allineato al Syllabus PCAP-31-03                         ║
║                         OpenEDG Python Institute                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

PCAP-31-03 Exam Block 2: Strings (18%)

STRUTTURA MODULO:
├── Section 2.1: String Nature and Operations
├── Section 2.2: String Methods
├── Section 2.3: String Comparison and Sorting
├── Section 2.4: Advanced Exceptions
├── Labs (10 esercizi)
└── Module 2 Quiz (30 domande)

TEMPO STIMATO: 6-7 ore

═══════════════════════════════════════════════════════════════════════════════
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 2.1: STRING NATURE
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    2.1 TEORIA: NATURA DELLE STRINGHE                         │
└──────────────────────────────────────────────────────────────────────────────┘

Le stringhe in Python sono:
1. IMMUTABILI - non puoi modificare i caratteri
2. SEQUENZE - supportano indexing, slicing, iterazione
3. UNICODE - supportano caratteri internazionali
"""

s = "Hello"

# Indexing
print(s[0])    # 'H'
print(s[-1])   # 'o'

# Slicing
print(s[1:4])  # 'ell'
print(s[::-1]) # 'olleH' (reverse)

# IMMUTABILITÀ
# s[0] = 'h'  # TypeError: 'str' object does not support item assignment

# Per "modificare", crea una NUOVA stringa
s = 'h' + s[1:]
print(s)  # 'hello'

# Iterazione
for char in "abc":
    print(char)  # a, b, c

# in
print('ell' in 'Hello')  # True
print('x' in 'Hello')    # False

# Lunghezza
print(len("Hello"))  # 5


"""
OPERAZIONI:
───────────
"""
# Concatenazione
print("Hello" + " " + "World")  # 'Hello World'

# Ripetizione
print("ab" * 3)  # 'ababab'

# ATTENZIONE: non puoi concatenare str e int!
# print("Age: " + 25)  # TypeError!
print("Age: " + str(25))  # OK


"""
ORD() e CHR() - Conversione carattere ↔ codice Unicode:
───────────────────────────────────────────────────────
"""
print(ord('A'))    # 65
print(ord('a'))    # 97
print(ord('0'))    # 48

print(chr(65))     # 'A'
print(chr(97))     # 'a'
print(chr(8364))   # '€'


"""
MIN() e MAX() con stringhe:
───────────────────────────
Confrontano per valore Unicode (ord)
"""
print(min("Hello"))  # 'H' (72 < 101 < 108 < 111)
print(max("Hello"))  # 'o'
print(min("aAbB"))   # 'A' (65 < 66 < 97 < 98)


# ══════════════════════════════════════════════════════════════════════════════
#                     SECTION 2.1 - QUIZ
# ══════════════════════════════════════════════════════════════════════════════

QUIZ_2_1 = """
══════════════════════════════════════════════════════════════════════════════
QUIZ 2.1.1
══════════════════════════════════════════════════════════════════════════════
s = "Python"
s[0] = 'p'
print(s)

Cosa succede?
A) 'python'
B) 'Python'
C) TypeError
D) 'pPython'

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.1.2
══════════════════════════════════════════════════════════════════════════════
print(ord('A'), ord('a'))

Stampa:
A) 65 97
B) A a
C) 97 65
D) 1 1

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.1.3
══════════════════════════════════════════════════════════════════════════════
print(min("aAbB"))

Stampa:
A) 'a'
B) 'A'
C) 'b'
D) 'B'

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.1.4
══════════════════════════════════════════════════════════════════════════════
print("ab" * 3)

Stampa:
A) 'ab3'
B) 'ababab'
C) 'ab ab ab'
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.1.5
══════════════════════════════════════════════════════════════════════════════
print("Hello"[::-1])

Stampa:
A) 'Hello'
B) 'olleH'
C) 'H'
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.1.6
══════════════════════════════════════════════════════════════════════════════
print('ell' in 'Hello')

Stampa:
A) True
B) False
C) 1
D) Error

Tua risposta: ___
"""


RISPOSTE_2_1 = """
══════════════════════════════════════════════════════════════════════════════
RISPOSTE QUIZ 2.1
══════════════════════════════════════════════════════════════════════════════

2.1.1: C) TypeError
       Le stringhe sono IMMUTABILI, non puoi modificare i caratteri.

2.1.2: A) 65 97
       'A' = 65, 'a' = 97 in Unicode/ASCII.

2.1.3: B) 'A'
       min usa ord(): 'A'=65, 'B'=66, 'a'=97, 'b'=98. Min è 65='A'.

2.1.4: B) 'ababab'
       * ripete la stringa.

2.1.5: B) 'olleH'
       [::-1] inverte la stringa.

2.1.6: A) True
       'ell' è una sottostringa di 'Hello'.
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 2.2: STRING METHODS
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    2.2 TEORIA: METODI DELLE STRINGHE                         │
└──────────────────────────────────────────────────────────────────────────────┘

TUTTI i metodi stringa restituiscono NUOVE stringhe (immutabilità!)

CASE METHODS:
─────────────
"""
s = "Hello World"
print(s.upper())       # 'HELLO WORLD'
print(s.lower())       # 'hello world'
print(s.capitalize())  # 'Hello world' (solo prima lettera maiuscola)
print(s.title())       # 'Hello World' (prima lettera ogni parola)
print(s.swapcase())    # 'hELLO wORLD'


"""
SEARCH METHODS:
───────────────
"""
s = "Hello World"

# find(sub) - restituisce indice o -1 se non trovato
print(s.find('o'))     # 4 (prima occorrenza)
print(s.find('x'))     # -1 (non trovato)
print(s.find('o', 5))  # 7 (cerca da indice 5)

# rfind(sub) - cerca da destra
print(s.rfind('o'))    # 7

# index(sub) - come find ma solleva ValueError se non trovato!
print(s.index('o'))    # 4
# print(s.index('x'))  # ValueError!

# count(sub) - conta occorrenze
print(s.count('o'))    # 2
print(s.count('l'))    # 3


"""
CHECK METHODS (restituiscono bool):
───────────────────────────────────
"""
# startswith / endswith
print("Hello".startswith('He'))  # True
print("Hello".endswith('lo'))    # True

# Contenuto
print("Hello".isalpha())   # True (solo lettere)
print("Hello1".isalpha())  # False
print("123".isdigit())     # True (solo cifre)
print("12.3".isdigit())    # False (il punto!)
print("Hello1".isalnum())  # True (lettere o cifre)
print("   ".isspace())     # True (solo whitespace)

# Case
print("HELLO".isupper())   # True
print("hello".islower())   # True


"""
STRIP METHODS (rimuovono caratteri):
────────────────────────────────────
"""
s = "   Hello   "
print(s.strip())       # 'Hello' (rimuove spazi da entrambi i lati)
print(s.lstrip())      # 'Hello   ' (solo sinistra)
print(s.rstrip())      # '   Hello' (solo destra)

# Puoi specificare caratteri da rimuovere
print("...Hello...".strip('.'))  # 'Hello'


"""
REPLACE METHOD:
───────────────
"""
s = "Hello World"
print(s.replace('o', '0'))       # 'Hell0 W0rld' (tutte le occorrenze)
print(s.replace('o', '0', 1))    # 'Hell0 World' (solo prima occorrenza)


"""
SPLIT e JOIN:
─────────────
"""
# split() - stringa → lista
s = "one,two,three"
print(s.split(','))     # ['one', 'two', 'three']

s = "Hello World"
print(s.split())        # ['Hello', 'World'] (split su whitespace)
print(s.split(' ', 1))  # ['Hello', 'World'] (max 1 split)

# join() - lista → stringa
lst = ['a', 'b', 'c']
print(','.join(lst))    # 'a,b,c'
print(''.join(lst))     # 'abc'
print(' '.join(lst))    # 'a b c'

# ATTENZIONE: join richiede una lista di STRINGHE!
# ','.join([1, 2, 3])  # TypeError!
print(','.join(map(str, [1, 2, 3])))  # '1,2,3'


"""
ALTRI METODI UTILI:
───────────────────
"""
# center, ljust, rjust - padding
print("Hello".center(11))    # '   Hello   '
print("Hello".center(11, '-'))  # '---Hello---'
print("Hello".ljust(10, '.'))   # 'Hello.....'
print("Hello".rjust(10, '.'))   # '.....Hello'

# zfill - padding con zeri (utile per numeri)
print("42".zfill(5))    # '00042'


# ══════════════════════════════════════════════════════════════════════════════
#                     SECTION 2.2 - QUIZ
# ══════════════════════════════════════════════════════════════════════════════

QUIZ_2_2 = """
══════════════════════════════════════════════════════════════════════════════
QUIZ 2.2.1
══════════════════════════════════════════════════════════════════════════════
print("Hello".find('x'))

Stampa:
A) False
B) None
C) -1
D) ValueError

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.2.2
══════════════════════════════════════════════════════════════════════════════
print("Hello".index('x'))

Cosa succede?
A) False
B) None
C) -1
D) ValueError

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.2.3
══════════════════════════════════════════════════════════════════════════════
print("hello world".title())

Stampa:
A) 'HELLO WORLD'
B) 'Hello World'
C) 'Hello world'
D) 'hello World'

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.2.4
══════════════════════════════════════════════════════════════════════════════
print("Hello".count('l'))

Stampa:
A) 1
B) 2
C) 3
D) True

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.2.5
══════════════════════════════════════════════════════════════════════════════
print("12.5".isdigit())

Stampa:
A) True
B) False
C) 12.5
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.2.6
══════════════════════════════════════════════════════════════════════════════
print(','.join(['a', 'b', 'c']))

Stampa:
A) 'abc'
B) 'a,b,c'
C) ['a', 'b', 'c']
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.2.7
══════════════════════════════════════════════════════════════════════════════
print("one,two,three".split(',', 1))

Stampa:
A) ['one', 'two', 'three']
B) ['one', 'two,three']
C) ['one']
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.2.8
══════════════════════════════════════════════════════════════════════════════
print("   Hello   ".strip())

Stampa:
A) 'Hello   '
B) '   Hello'
C) 'Hello'
D) '  Hello  '

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.2.9
══════════════════════════════════════════════════════════════════════════════
s = "Hello"
s.upper()
print(s)

Stampa:
A) 'HELLO'
B) 'Hello'
C) None
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.2.10
══════════════════════════════════════════════════════════════════════════════
print("hello world".capitalize())

Stampa:
A) 'HELLO WORLD'
B) 'Hello World'
C) 'Hello world'
D) 'hELLO WORLD'

Tua risposta: ___
"""


RISPOSTE_2_2 = """
══════════════════════════════════════════════════════════════════════════════
RISPOSTE QUIZ 2.2
══════════════════════════════════════════════════════════════════════════════

2.2.1: C) -1
       find() restituisce -1 se non trova la sottostringa.

2.2.2: D) ValueError
       index() solleva ValueError se non trova (diverso da find!).

2.2.3: B) 'Hello World'
       title() mette maiuscola la prima lettera di OGNI parola.

2.2.4: B) 2
       "Hello" contiene 2 'l'.

2.2.5: B) False
       isdigit() restituisce False se c'è il punto decimale.

2.2.6: B) 'a,b,c'
       join() unisce con il separatore specificato.

2.2.7: B) ['one', 'two,three']
       Il secondo argomento limita il numero di split.

2.2.8: C) 'Hello'
       strip() rimuove spazi da ENTRAMBI i lati.

2.2.9: B) 'Hello'
       upper() restituisce una NUOVA stringa, non modifica l'originale!
       Stringhe sono IMMUTABILI.

2.2.10: C) 'Hello world'
        capitalize() mette maiuscola SOLO la prima lettera della stringa.
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 2.3: STRING COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    2.3 TEORIA: CONFRONTO STRINGHE                            │
└──────────────────────────────────────────────────────────────────────────────┘

Le stringhe si confrontano CARATTERE PER CARATTERE usando i valori Unicode.
"""

# Confronto carattere per carattere
print('apple' < 'banana')  # True ('a' < 'b')
print('apple' < 'Apple')   # False ('a'=97 > 'A'=65)

# Se una stringa è prefisso dell'altra, la più corta è "minore"
print('app' < 'apple')     # True

# Numeri come stringhe
print('10' < '9')          # True! ('1' < '9' nel primo carattere)
print('10' < '2')          # True! (confronto lessicografico, non numerico)


"""
ORDINAMENTO:
────────────
"""
words = ['banana', 'Apple', 'cherry']
print(sorted(words))  # ['Apple', 'banana', 'cherry'] (maiuscole prima!)

# Case-insensitive sort
print(sorted(words, key=str.lower))  # ['Apple', 'banana', 'cherry']


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 2.4: ADVANCED EXCEPTIONS
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    2.4 TEORIA: ECCEZIONI AVANZATE                            │
└──────────────────────────────────────────────────────────────────────────────┘

GERARCHIA COMPLETA (parziale):
──────────────────────────────
BaseException
├── SystemExit
├── KeyboardInterrupt
├── GeneratorExit
└── Exception
    ├── StopIteration
    ├── ArithmeticError
    │   ├── FloatingPointError
    │   ├── OverflowError
    │   └── ZeroDivisionError
    ├── LookupError
    │   ├── IndexError
    │   └── KeyError
    ├── ValueError
    ├── TypeError
    ├── AttributeError
    └── ...
"""

# Catturare eccezioni dalla gerarchia
try:
    x = {}['key']
except LookupError:  # Cattura KeyError E IndexError
    print("Lookup failed")


"""
ECCEZIONI PERSONALIZZATE:
─────────────────────────
"""
class MyError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(message)

# Uso:
# raise MyError("Something went wrong")


"""
args DELL'ECCEZIONE:
────────────────────
"""
try:
    raise ValueError("Invalid value", 42)
except ValueError as e:
    print(e.args)  # ('Invalid value', 42)


"""
ASSERT:
───────
Verifica condizioni durante lo sviluppo
"""
x = 10
assert x > 0, "x must be positive"  # OK
# assert x < 0, "x must be negative"  # AssertionError!

# assert è disabilitato con python -O (ottimizzato)


# ══════════════════════════════════════════════════════════════════════════════
#                     SECTION 2.3-2.4 - QUIZ
# ══════════════════════════════════════════════════════════════════════════════

QUIZ_2_3_2_4 = """
══════════════════════════════════════════════════════════════════════════════
QUIZ 2.3.1
══════════════════════════════════════════════════════════════════════════════
print('apple' < 'Apple')

Stampa:
A) True
B) False
C) Error
D) None

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.3.2
══════════════════════════════════════════════════════════════════════════════
print('10' < '9')

Stampa:
A) True
B) False
C) Error
D) Dipende

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.3.3
══════════════════════════════════════════════════════════════════════════════
Quale eccezione è parent di IndexError e KeyError?

A) Exception
B) ValueError
C) LookupError
D) BaseException

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.3.4
══════════════════════════════════════════════════════════════════════════════
x = 5
assert x < 0, "negative required"

Cosa succede?
A) Stampa "negative required"
B) AssertionError
C) ValueError
D) Niente

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.3.5
══════════════════════════════════════════════════════════════════════════════
print(sorted(['B', 'a', 'C']))

Stampa:
A) ['a', 'B', 'C']
B) ['B', 'C', 'a']
C) ['a', 'b', 'c']
D) ['A', 'B', 'C']

Tua risposta: ___
"""


RISPOSTE_2_3_2_4 = """
══════════════════════════════════════════════════════════════════════════════
RISPOSTE QUIZ 2.3-2.4
══════════════════════════════════════════════════════════════════════════════

2.3.1: B) False
       'a' (97) > 'A' (65), quindi 'apple' > 'Apple'.

2.3.2: A) True
       Confronto lessicografico: '1' < '9' (primo carattere).
       NON è confronto numerico!

2.3.3: C) LookupError
       LookupError è la classe base per IndexError e KeyError.

2.3.4: B) AssertionError
       x < 0 è False, quindi assert fallisce.

2.3.5: B) ['B', 'C', 'a']
       Maiuscole (65-90) vengono prima di minuscole (97-122).
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    MODULE 2 - TEST FINALE
# ══════════════════════════════════════════════════════════════════════════════

MODULE_2_TEST = """
══════════════════════════════════════════════════════════════════════════════
                     PE2 MODULE 2 - TEST FINALE
                    30 domande - Target: 70% (21/30)
══════════════════════════════════════════════════════════════════════════════

Q1. s="Hello"; s[0]='h' → ?
    A) 'hello'    B) TypeError    C) 'hHello'    D) None

Q2. print(ord('a') - ord('A')) = ?
    A) 0    B) 32    C) -32    D) 26

Q3. print(min("aAzZ")) = ?
    A) 'a'    B) 'A'    C) 'z'    D) 'Z'

Q4. print("Hello".find('x')) = ?
    A) None    B) False    C) -1    D) ValueError

Q5. print("Hello".index('x')) → ?
    A) None    B) False    C) -1    D) ValueError

Q6. print("hello world".title()) = ?
    A) 'Hello world'    B) 'Hello World'    C) 'HELLO WORLD'    D) 'hELLO wORLD'

Q7. print("hello world".capitalize()) = ?
    A) 'Hello world'    B) 'Hello World'    C) 'HELLO WORLD'    D) 'hELLO WORLD'

Q8. print("12.5".isdigit()) = ?
    A) True    B) False    C) 12.5    D) Error

Q9. print("ABC".isalpha()) = ?
    A) True    B) False    C) 'ABC'    D) Error

Q10. print("Hello".replace('l', 'L', 1)) = ?
     A) 'HeLLo'    B) 'HeLlo'    C) 'Hello'    D) 'HELLO'

Q11. print("a,b,c".split(',')) = ?
     A) 'a b c'    B) ['a','b','c']    C) ('a','b','c')    D) Error

Q12. print('-'.join(['x','y'])) = ?
     A) 'x-y'    B) '-xy'    C) ['x','-','y']    D) Error

Q13. print("  Hi  ".strip()) = ?
     A) 'Hi  '    B) '  Hi'    C) 'Hi'    D) 'Hi'

Q14. s="Hi"; s.upper(); print(s) = ?
     A) 'HI'    B) 'Hi'    C) None    D) Error

Q15. print("Hello".count('l')) = ?
     A) 1    B) 2    C) 3    D) True

Q16. print("abc".center(7,'-')) = ?
     A) '--abc--'    B) '---abc'    C) 'abc----'    D) '-abc--'

Q17. print("42".zfill(5)) = ?
     A) '42000'    B) '00042'    C) '42   '    D) '   42'

Q18. print('apple' < 'banana') = ?
     A) True    B) False    C) Error    D) None

Q19. print('Zoo' < 'apple') = ?
     A) True    B) False    C) Error    D) Dipende

Q20. print('10' < '2') = ?
     A) True    B) False    C) Error    D) Dipende

Q21. print("Hello".startswith('He')) = ?
     A) True    B) False    C) 'He'    D) 0

Q22. print("Hello".rfind('l')) = ?
     A) 2    B) 3    C) 4    D) -1

Q23. LookupError è parent di:
     A) ValueError    B) TypeError    C) KeyError    D) ZeroDivisionError

Q24. assert True, "msg" → ?
     A) Stampa "msg"    B) AssertionError    C) Niente    D) True

Q25. assert False, "msg" → ?
     A) Stampa "msg"    B) AssertionError    C) Niente    D) False

Q26. print("abc"*2+"d") = ?
     A) 'abcabcd'    B) 'abcd2'    C) Error    D) 'abc2d'

Q27. print("Hi\\nBye".split()) = ?
     A) ['Hi','Bye']    B) ['Hi\\nBye']    C) ['Hi\\n','Bye']    D) Error

Q28. print(chr(ord('a')+1)) = ?
     A) 'b'    B) 'a1'    C) 98    D) Error

Q29. sorted(['B','a','A','b']) = ?
     A) ['A','B','a','b']    B) ['a','A','b','B']    C) ['A','a','B','b']    D) Error

Q30. try: raise ValueError("x",1); except ValueError as e: print(type(e.args))
     A) str    B) tuple    C) list    D) Error


══════════════════════════════════════════════════════════════════════════════
                              FINE TEST
══════════════════════════════════════════════════════════════════════════════
"""


MODULE_2_TEST_ANSWERS = """
══════════════════════════════════════════════════════════════════════════════
                    PE2 MODULE 2 - RISPOSTE
══════════════════════════════════════════════════════════════════════════════

Q1:  B) TypeError (stringhe immutabili)
Q2:  B) 32 (97-65)
Q3:  B) 'A' (65 < 90 < 97 < 122)
Q4:  C) -1
Q5:  D) ValueError
Q6:  B) 'Hello World' (title)
Q7:  A) 'Hello world' (capitalize)
Q8:  B) False (punto!)
Q9:  A) True
Q10: B) 'HeLlo' (solo prima occorrenza)
Q11: B) ['a','b','c']
Q12: A) 'x-y'
Q13: C) 'Hi'
Q14: B) 'Hi' (immutabile!)
Q15: B) 2
Q16: A) '--abc--'
Q17: B) '00042'
Q18: A) True
Q19: A) True ('Z'=90 < 'a'=97)
Q20: A) True (lessicografico!)
Q21: A) True
Q22: B) 3 (ultimo 'l')
Q23: C) KeyError
Q24: C) Niente (True passa)
Q25: B) AssertionError
Q26: A) 'abcabcd'
Q27: A) ['Hi','Bye'] (\\n è whitespace)
Q28: A) 'b'
Q29: A) ['A','B','a','b']
Q30: B) tuple

PUNTEGGIO: ___/30
Target: 21/30 (70%)
"""


if __name__ == "__main__":
    print("=" * 78)
    print("PYTHON ESSENTIALS 2 - MODULE 2")
    print("Strings and String Methods")
    print("=" * 78)
    print("""
    CONTENUTO:
    - Section 2.1: String Nature (6 quiz)
    - Section 2.2: String Methods (10 quiz)
    - Section 2.3-2.4: Comparison & Exceptions (5 quiz)
    - Test finale: 30 domande
    
    TRAPPOLE:
    ⚠️  Stringhe IMMUTABILI - i metodi restituiscono NUOVE stringhe
    ⚠️  find() → -1, index() → ValueError
    ⚠️  '10' < '9' è TRUE (lessicografico)
    ⚠️  capitalize() vs title()
    """)
