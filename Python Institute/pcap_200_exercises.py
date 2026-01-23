"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    200 ESERCIZI PCAP (PCAP-31-03)                            ║
║                                                                              ║
║                 OOP, Modules, Strings, Files, Exceptions                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Esercizi pratici per prepararsi al PCAP.
Scrivi le soluzioni PRIMA di vedere quelle proposte.

═══════════════════════════════════════════════════════════════════════════════
"""

# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 1: OOP BASICS (50 exercises)
# ══════════════════════════════════════════════════════════════════════════════

OOP_EXERCISES = """
════════════════════════════════════════════════════════════════════════════════
                    SECTION 1: OOP BASICS (1-50)
════════════════════════════════════════════════════════════════════════════════

EXERCISE 1: Basic Class
────────────────────────────────────────────────────────────────────────────────
Crea una classe Person con attributi name e age.

class Person:
    # Il tuo codice
    pass

# Test:
# p = Person("Marco", 30)
# print(p.name, p.age)  → Marco 30


EXERCISE 2: Method
────────────────────────────────────────────────────────────────────────────────
Aggiungi metodo greet() che restituisce "Hello, I'm {name}".

class Person:
    def __init__(self, name, age):
        pass
    
    def greet(self):
        pass

# Test: Person("Marco", 30).greet() → "Hello, I'm Marco"


EXERCISE 3: Class Variable
────────────────────────────────────────────────────────────────────────────────
Crea Counter con variabile di classe che conta le istanze.

class Counter:
    count = 0
    # ...

# Test:
# a = Counter()
# b = Counter()
# print(Counter.count)  → 2


EXERCISE 4: __str__ Method
────────────────────────────────────────────────────────────────────────────────
Implementa __str__ per Book.

class Book:
    def __init__(self, title, author):
        pass
    
    def __str__(self):
        pass

# Test: print(Book("1984", "Orwell")) → "1984 by Orwell"


EXERCISE 5: __repr__ Method
────────────────────────────────────────────────────────────────────────────────
Implementa __repr__ che restituisce codice per ricreare l'oggetto.

class Point:
    def __init__(self, x, y):
        pass
    
    def __repr__(self):
        pass

# Test: repr(Point(3, 4)) → "Point(3, 4)"


EXERCISE 6: Comparison __eq__
────────────────────────────────────────────────────────────────────────────────
Implementa __eq__ per confrontare due Point.

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        pass

# Test: Point(1, 2) == Point(1, 2) → True


EXERCISE 7: Arithmetic __add__
────────────────────────────────────────────────────────────────────────────────
Implementa __add__ per sommare due Vector.

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        pass

# Test: Vector(1,2) + Vector(3,4) → Vector(4,6)


EXERCISE 8: __len__
────────────────────────────────────────────────────────────────────────────────
Implementa __len__ per Playlist.

class Playlist:
    def __init__(self, songs):
        self.songs = songs
    
    def __len__(self):
        pass

# Test: len(Playlist(["a", "b", "c"])) → 3


EXERCISE 9: __getitem__
────────────────────────────────────────────────────────────────────────────────
Implementa __getitem__ per accesso con [].

class MyList:
    def __init__(self, items):
        self.items = items
    
    def __getitem__(self, index):
        pass

# Test: MyList([1,2,3])[1] → 2


EXERCISE 10: __contains__
────────────────────────────────────────────────────────────────────────────────
Implementa __contains__ per operatore in.

class Bag:
    def __init__(self, items):
        self.items = items
    
    def __contains__(self, item):
        pass

# Test: "apple" in Bag(["apple", "banana"]) → True


EXERCISE 11: Simple Inheritance
────────────────────────────────────────────────────────────────────────────────
Crea Student che eredita da Person.

class Person:
    def __init__(self, name):
        self.name = name

class Student(Person):
    def __init__(self, name, student_id):
        # chiama parent __init__
        pass

# Test: s = Student("Marco", "S123")
#       print(s.name, s.student_id) → Marco S123


EXERCISE 12: Method Override
────────────────────────────────────────────────────────────────────────────────
Override del metodo speak().

class Animal:
    def speak(self):
        return "..."

class Dog(Animal):
    def speak(self):
        pass

class Cat(Animal):
    def speak(self):
        pass

# Test: Dog().speak() → "Woof"
#       Cat().speak() → "Meow"


EXERCISE 13: super() Call
────────────────────────────────────────────────────────────────────────────────
Usa super() per estendere metodo parent.

class Vehicle:
    def describe(self):
        return "I am a vehicle"

class Car(Vehicle):
    def describe(self):
        # Chiama parent e aggiungi " with 4 wheels"
        pass

# Test: Car().describe() → "I am a vehicle with 4 wheels"


EXERCISE 14: Multiple Inheritance
────────────────────────────────────────────────────────────────────────────────
Crea classe che eredita da due parent.

class Flyable:
    def fly(self):
        return "Flying"

class Swimmable:
    def swim(self):
        return "Swimming"

class Duck(Flyable, Swimmable):
    pass

# Test: d = Duck()
#       d.fly() → "Flying"
#       d.swim() → "Swimming"


EXERCISE 15: isinstance Check
────────────────────────────────────────────────────────────────────────────────
Scrivi funzione che accetta solo Animal.

class Animal:
    pass

class Dog(Animal):
    pass

def make_sound(animal):
    # Verifica isinstance, altrimenti solleva TypeError
    pass


EXERCISE 16: Private Attribute
────────────────────────────────────────────────────────────────────────────────
Crea classe con attributo "privato" __balance.

class BankAccount:
    def __init__(self, balance):
        self.__balance = balance
    
    def get_balance(self):
        pass
    
    def deposit(self, amount):
        pass

# Test: acc = BankAccount(100)
#       acc.deposit(50)
#       acc.get_balance() → 150


EXERCISE 17: Property Decorator
────────────────────────────────────────────────────────────────────────────────
Usa @property per getter/setter.

class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        pass
    
    @radius.setter
    def radius(self, value):
        # Solo valori positivi
        pass
    
    @property
    def area(self):
        pass

# Test: c = Circle(5)
#       c.area → ~78.54


EXERCISE 18: Class Method
────────────────────────────────────────────────────────────────────────────────
Crea class method come factory.

class Date:
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day
    
    @classmethod
    def from_string(cls, date_string):
        # Parse "YYYY-MM-DD"
        pass

# Test: Date.from_string("2024-01-15").year → 2024


EXERCISE 19: Static Method
────────────────────────────────────────────────────────────────────────────────
Crea static method helper.

class Math:
    @staticmethod
    def is_even(n):
        pass
    
    @staticmethod
    def is_prime(n):
        pass

# Test: Math.is_even(4) → True
#       Math.is_prime(7) → True


EXERCISE 20: Abstract Base Class
────────────────────────────────────────────────────────────────────────────────
Crea ABC con metodo astratto.

from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        pass
    
    def area(self):
        pass

# Test: Rectangle(3, 4).area() → 12
#       Shape() → Error


... (altri 30 esercizi OOP)
"""

# ══════════════════════════════════════════════════════════════════════════════
#                    OOP SOLUTIONS (1-20)
# ══════════════════════════════════════════════════════════════════════════════

# Exercise 1
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

# Exercise 2
class PersonWithGreet:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def greet(self):
        return f"Hello, I'm {self.name}"

# Exercise 3
class Counter:
    count = 0
    
    def __init__(self):
        Counter.count += 1

# Exercise 4
class Book:
    def __init__(self, title, author):
        self.title = title
        self.author = author
    
    def __str__(self):
        return f"{self.title} by {self.author}"

# Exercise 5
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f"Point({self.x}, {self.y})"
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

# Exercise 7
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

# Exercise 8
class Playlist:
    def __init__(self, songs):
        self.songs = songs
    
    def __len__(self):
        return len(self.songs)

# Exercise 9
class MyList:
    def __init__(self, items):
        self.items = items
    
    def __getitem__(self, index):
        return self.items[index]

# Exercise 10
class Bag:
    def __init__(self, items):
        self.items = items
    
    def __contains__(self, item):
        return item in self.items

# Exercise 11
class PersonBase:
    def __init__(self, name):
        self.name = name

class Student(PersonBase):
    def __init__(self, name, student_id):
        super().__init__(name)
        self.student_id = student_id

# Exercise 12
class Animal:
    def speak(self):
        return "..."

class Dog(Animal):
    def speak(self):
        return "Woof"

class Cat(Animal):
    def speak(self):
        return "Meow"

# Exercise 13
class Vehicle:
    def describe(self):
        return "I am a vehicle"

class Car(Vehicle):
    def describe(self):
        return super().describe() + " with 4 wheels"

# Exercise 16
class BankAccount:
    def __init__(self, balance):
        self.__balance = balance
    
    def get_balance(self):
        return self.__balance
    
    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount

# Exercise 17
import math

class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, value):
        if value > 0:
            self._radius = value
        else:
            raise ValueError("Radius must be positive")
    
    @property
    def area(self):
        return math.pi * self._radius ** 2

# Exercise 18
class Date:
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day
    
    @classmethod
    def from_string(cls, date_string):
        year, month, day = map(int, date_string.split("-"))
        return cls(year, month, day)

# Exercise 19
class MathUtils:
    @staticmethod
    def is_even(n):
        return n % 2 == 0
    
    @staticmethod
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 2: STRINGS (50 exercises)
# ══════════════════════════════════════════════════════════════════════════════

STRING_EXERCISES = """
════════════════════════════════════════════════════════════════════════════════
                    SECTION 2: STRINGS (51-100)
════════════════════════════════════════════════════════════════════════════════

EXERCISE 51: Capitalize Words
────────────────────────────────────────────────────────────────────────────────
Capitalizza la prima lettera di ogni parola.

def capitalize_words(s):
    pass

# Test: capitalize_words("hello world") → "Hello World"


EXERCISE 52: Count Substring
────────────────────────────────────────────────────────────────────────────────
Conta le occorrenze di una sottostringa.

def count_substring(s, sub):
    pass

# Test: count_substring("abcabc", "abc") → 2


EXERCISE 53: Is Palindrome (ignore spaces)
────────────────────────────────────────────────────────────────────────────────
Verifica palindromo ignorando spazi e maiuscole.

def is_palindrome(s):
    pass

# Test: is_palindrome("A man a plan a canal Panama") → True


EXERCISE 54: Remove Vowels
────────────────────────────────────────────────────────────────────────────────
Rimuovi tutte le vocali.

def remove_vowels(s):
    pass

# Test: remove_vowels("Hello World") → "Hll Wrld"


EXERCISE 55: Compress String
────────────────────────────────────────────────────────────────────────────────
Comprimi "aaabbc" → "a3b2c1".

def compress(s):
    pass

# Test: compress("aaabbc") → "a3b2c1"


EXERCISE 56: Decompress String
────────────────────────────────────────────────────────────────────────────────
Decomprimi "a3b2c1" → "aaabbc".

def decompress(s):
    pass

# Test: decompress("a3b2c1") → "aaabbc"


EXERCISE 57: Caesar Cipher
────────────────────────────────────────────────────────────────────────────────
Implementa cifrario di Cesare.

def caesar(s, shift):
    pass

# Test: caesar("abc", 1) → "bcd"


EXERCISE 58: Find All Occurrences
────────────────────────────────────────────────────────────────────────────────
Trova tutti gli indici di una sottostringa.

def find_all(s, sub):
    pass

# Test: find_all("abcabc", "abc") → [0, 3]


EXERCISE 59: Reverse Words
────────────────────────────────────────────────────────────────────────────────
Inverti l'ordine delle parole.

def reverse_words(s):
    pass

# Test: reverse_words("Hello World") → "World Hello"


EXERCISE 60: Is Anagram
────────────────────────────────────────────────────────────────────────────────
Verifica se due stringhe sono anagrammi.

def is_anagram(s1, s2):
    pass

# Test: is_anagram("listen", "silent") → True


... (altri 40 esercizi stringhe)
"""

# String Solutions
def capitalize_words(s):
    return s.title()

def count_substring(s, sub):
    return s.count(sub)

def is_palindrome_advanced(s):
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]

def remove_vowels(s):
    vowels = "aeiouAEIOU"
    return ''.join(c for c in s if c not in vowels)

def compress(s):
    if not s:
        return ""
    result = []
    count = 1
    for i in range(1, len(s)):
        if s[i] == s[i-1]:
            count += 1
        else:
            result.append(f"{s[i-1]}{count}")
            count = 1
    result.append(f"{s[-1]}{count}")
    return ''.join(result)

def decompress(s):
    result = []
    i = 0
    while i < len(s):
        char = s[i]
        i += 1
        num = ""
        while i < len(s) and s[i].isdigit():
            num += s[i]
            i += 1
        result.append(char * int(num))
    return ''.join(result)

def caesar(s, shift):
    result = []
    for c in s:
        if c.isalpha():
            base = ord('a') if c.islower() else ord('A')
            result.append(chr((ord(c) - base + shift) % 26 + base))
        else:
            result.append(c)
    return ''.join(result)

def find_all_occurrences(s, sub):
    indices = []
    start = 0
    while True:
        idx = s.find(sub, start)
        if idx == -1:
            break
        indices.append(idx)
        start = idx + 1
    return indices

def reverse_words(s):
    return ' '.join(s.split()[::-1])

def is_anagram(s1, s2):
    return sorted(s1.lower()) == sorted(s2.lower())


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 3: FILES & EXCEPTIONS (50 exercises)
# ══════════════════════════════════════════════════════════════════════════════

FILES_EXERCISES = """
════════════════════════════════════════════════════════════════════════════════
                    SECTION 3: FILES & EXCEPTIONS (101-150)
════════════════════════════════════════════════════════════════════════════════

EXERCISE 101: Read File
────────────────────────────────────────────────────────────────────────────────
Leggi un file e restituisci il contenuto.

def read_file(filename):
    pass

# Test: contenuto del file


EXERCISE 102: Write File
────────────────────────────────────────────────────────────────────────────────
Scrivi contenuto in un file.

def write_file(filename, content):
    pass


EXERCISE 103: Count Lines
────────────────────────────────────────────────────────────────────────────────
Conta le righe non vuote.

def count_lines(filename):
    pass


EXERCISE 104: Word Frequency
────────────────────────────────────────────────────────────────────────────────
Conta frequenza parole in un file.

def word_frequency(filename):
    pass

# Restituisce dict {word: count}


EXERCISE 105: Copy File
────────────────────────────────────────────────────────────────────────────────
Copia contenuto da un file all'altro.

def copy_file(src, dst):
    pass


EXERCISE 106: Safe Division
────────────────────────────────────────────────────────────────────────────────
Divisione con gestione ZeroDivisionError.

def safe_divide(a, b):
    pass

# Test: safe_divide(10, 0) → None (o messaggio errore)


EXERCISE 107: Parse Integer
────────────────────────────────────────────────────────────────────────────────
Converti stringa in int, restituisci default se fallisce.

def parse_int(s, default=0):
    pass

# Test: parse_int("abc") → 0
#       parse_int("123") → 123


EXERCISE 108: Custom Exception
────────────────────────────────────────────────────────────────────────────────
Crea eccezione personalizzata.

class ValidationError(Exception):
    pass

def validate_age(age):
    # Solleva ValidationError se age < 0
    pass


EXERCISE 109: Multiple Exceptions
────────────────────────────────────────────────────────────────────────────────
Gestisci multiple eccezioni.

def process_file(filename):
    # Gestisci FileNotFoundError e PermissionError
    pass


EXERCISE 110: Finally Cleanup
────────────────────────────────────────────────────────────────────────────────
Usa finally per cleanup.

def process_with_cleanup(filename):
    # Apri file, processa, chiudi sempre
    pass


... (altri 40 esercizi files/exceptions)
"""

# File Solutions
def read_file(filename):
    with open(filename, 'r') as f:
        return f.read()

def write_file(filename, content):
    with open(filename, 'w') as f:
        f.write(content)

def count_lines(filename):
    with open(filename, 'r') as f:
        return sum(1 for line in f if line.strip())

def word_frequency(filename):
    freq = {}
    with open(filename, 'r') as f:
        for line in f:
            for word in line.lower().split():
                word = ''.join(c for c in word if c.isalnum())
                if word:
                    freq[word] = freq.get(word, 0) + 1
    return freq

def copy_file(src, dst):
    with open(src, 'r') as f_in:
        with open(dst, 'w') as f_out:
            f_out.write(f_in.read())

def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return None

def parse_int(s, default=0):
    try:
        return int(s)
    except ValueError:
        return default

class ValidationError(Exception):
    pass

def validate_age(age):
    if age < 0:
        raise ValidationError("Age cannot be negative")
    return age


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 4: MODULES & ADVANCED (50 exercises)
# ══════════════════════════════════════════════════════════════════════════════

MODULES_EXERCISES = """
════════════════════════════════════════════════════════════════════════════════
                    SECTION 4: MODULES & ADVANCED (151-200)
════════════════════════════════════════════════════════════════════════════════

EXERCISE 151: Import Math
────────────────────────────────────────────────────────────────────────────────
Calcola radice quadrata e logaritmo.

import math

def calculate(n):
    # Restituisci (sqrt, log10)
    pass

# Test: calculate(100) → (10.0, 2.0)


EXERCISE 152: Random Sampling
────────────────────────────────────────────────────────────────────────────────
Estrai n elementi casuali.

import random

def sample_items(items, n):
    pass

# Test: sample_items([1,2,3,4,5], 3) → [random subset]


EXERCISE 153: Date Calculation
────────────────────────────────────────────────────────────────────────────────
Calcola giorni tra due date.

from datetime import date

def days_between(date1, date2):
    # date1, date2 sono stringhe "YYYY-MM-DD"
    pass

# Test: days_between("2024-01-01", "2024-01-10") → 9


EXERCISE 154: Generator Function
────────────────────────────────────────────────────────────────────────────────
Crea generatore di numeri di Fibonacci.

def fib_generator(n):
    # Yield primi n numeri di Fibonacci
    pass

# Test: list(fib_generator(5)) → [0, 1, 1, 2, 3]


EXERCISE 155: List Comprehension Advanced
────────────────────────────────────────────────────────────────────────────────
Genera matrice con list comprehension.

def matrix(n):
    # Matrice n×n con elemento (i,j) = i*n + j
    pass

# Test: matrix(3) → [[0,1,2], [3,4,5], [6,7,8]]


EXERCISE 156: Dict Comprehension
────────────────────────────────────────────────────────────────────────────────
Crea dict da liste.

def create_dict(keys, values):
    pass

# Test: create_dict(['a','b'], [1,2]) → {'a': 1, 'b': 2}


EXERCISE 157: Lambda with Filter
────────────────────────────────────────────────────────────────────────────────
Filtra con lambda.

def filter_positive(numbers):
    pass

# Test: filter_positive([1,-2,3,-4,5]) → [1,3,5]


EXERCISE 158: Lambda with Map
────────────────────────────────────────────────────────────────────────────────
Trasforma con lambda.

def double_all(numbers):
    pass

# Test: double_all([1,2,3]) → [2,4,6]


EXERCISE 159: Lambda with Sorted
────────────────────────────────────────────────────────────────────────────────
Ordina per criterio custom.

def sort_by_length(strings):
    pass

# Test: sort_by_length(["aaa", "b", "cc"]) → ["b", "cc", "aaa"]


EXERCISE 160: Closure
────────────────────────────────────────────────────────────────────────────────
Crea closure.

def make_multiplier(n):
    # Restituisci funzione che moltiplica per n
    pass

# Test: triple = make_multiplier(3)
#       triple(5) → 15


... (altri 40 esercizi advanced)
"""

# Module Solutions
def calculate_math(n):
    import math
    return (math.sqrt(n), math.log10(n))

def sample_items(items, n):
    import random
    return random.sample(items, min(n, len(items)))

def days_between(date1_str, date2_str):
    from datetime import datetime
    d1 = datetime.strptime(date1_str, "%Y-%m-%d")
    d2 = datetime.strptime(date2_str, "%Y-%m-%d")
    return abs((d2 - d1).days)

def fib_generator(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

def matrix(n):
    return [[i*n + j for j in range(n)] for i in range(n)]

def create_dict(keys, values):
    return {k: v for k, v in zip(keys, values)}

def filter_positive(numbers):
    return list(filter(lambda x: x > 0, numbers))

def double_all(numbers):
    return list(map(lambda x: x * 2, numbers))

def sort_by_length(strings):
    return sorted(strings, key=len)

def make_multiplier(n):
    def multiplier(x):
        return x * n
    return multiplier


# ══════════════════════════════════════════════════════════════════════════════
#                    MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("200 ESERCIZI PCAP")
    print("=" * 70)
    print("""
    Sezioni:
    print(OOP_EXERCISES)      # Esercizi 1-50 (OOP)
    print(STRING_EXERCISES)   # Esercizi 51-100 (Strings)
    print(FILES_EXERCISES)    # Esercizi 101-150 (Files/Exceptions)
    print(MODULES_EXERCISES)  # Esercizi 151-200 (Modules/Advanced)
    
    Le soluzioni sono funzioni definite nel modulo.
    Testa ogni soluzione prima di vedere quella proposta!
    """)
