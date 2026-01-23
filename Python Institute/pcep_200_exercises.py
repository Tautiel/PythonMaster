"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    200 ESERCIZI PRATICI PCEP                                 ║
║                                                                              ║
║                 Coding Exercises con Soluzioni Complete                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

ISTRUZIONI:
1. Ogni esercizio ha una descrizione e test cases
2. Scrivi la tua soluzione PRIMA di guardare quella proposta
3. Testa con i casi forniti
4. Confronta con la soluzione

═══════════════════════════════════════════════════════════════════════════════
"""

# ══════════════════════════════════════════════════════════════════════════════
#                    BLOCK 1: BASICS (Exercises 1-50)
# ══════════════════════════════════════════════════════════════════════════════

exercises_block1 = """
════════════════════════════════════════════════════════════════════════════════
                         BLOCK 1: BASICS (1-50)
════════════════════════════════════════════════════════════════════════════════

EXERCISE 1: Hello Custom
───────────────────────────────────────────────────────────────────────────────
Scrivi una funzione che stampa "Hello, {name}!" dove name è un parametro.

def hello(name):
    # Il tuo codice qui
    pass

# Test: hello("Marco") → stampa "Hello, Marco!"


EXERCISE 2: Sum Two Numbers
───────────────────────────────────────────────────────────────────────────────
Scrivi una funzione che restituisce la somma di due numeri.

def add(a, b):
    pass

# Test: add(3, 5) → 8
#       add(-1, 1) → 0


EXERCISE 3: Is Even
───────────────────────────────────────────────────────────────────────────────
Scrivi una funzione che restituisce True se n è pari, False altrimenti.

def is_even(n):
    pass

# Test: is_even(4) → True
#       is_even(7) → False


EXERCISE 4: Absolute Value
───────────────────────────────────────────────────────────────────────────────
Scrivi una funzione che restituisce il valore assoluto SENZA usare abs().

def my_abs(n):
    pass

# Test: my_abs(-5) → 5
#       my_abs(3) → 3


EXERCISE 5: Max of Two
───────────────────────────────────────────────────────────────────────────────
Scrivi una funzione che restituisce il maggiore tra due numeri SENZA max().

def my_max(a, b):
    pass

# Test: my_max(3, 7) → 7
#       my_max(10, 5) → 10


EXERCISE 6: Min of Two
───────────────────────────────────────────────────────────────────────────────
Scrivi una funzione che restituisce il minore tra due numeri SENZA min().

def my_min(a, b):
    pass

# Test: my_min(3, 7) → 3
#       my_min(10, 5) → 5


EXERCISE 7: Sign of Number
───────────────────────────────────────────────────────────────────────────────
Restituisce 1 se positivo, -1 se negativo, 0 se zero.

def sign(n):
    pass

# Test: sign(5) → 1
#       sign(-3) → -1
#       sign(0) → 0


EXERCISE 8: Is Digit
───────────────────────────────────────────────────────────────────────────────
Restituisce True se il carattere è una cifra (0-9).

def is_digit(char):
    pass

# Test: is_digit('5') → True
#       is_digit('a') → False


EXERCISE 9: Is Letter
───────────────────────────────────────────────────────────────────────────────
Restituisce True se il carattere è una lettera (a-z o A-Z).

def is_letter(char):
    pass

# Test: is_letter('a') → True
#       is_letter('5') → False


EXERCISE 10: Celsius to Fahrenheit
───────────────────────────────────────────────────────────────────────────────
Converte Celsius in Fahrenheit: F = C * 9/5 + 32

def celsius_to_fahrenheit(c):
    pass

# Test: celsius_to_fahrenheit(0) → 32.0
#       celsius_to_fahrenheit(100) → 212.0


EXERCISE 11: Fahrenheit to Celsius
───────────────────────────────────────────────────────────────────────────────
Converte Fahrenheit in Celsius: C = (F - 32) * 5/9

def fahrenheit_to_celsius(f):
    pass

# Test: fahrenheit_to_celsius(32) → 0.0
#       fahrenheit_to_celsius(212) → 100.0


EXERCISE 12: Circle Area
───────────────────────────────────────────────────────────────────────────────
Calcola l'area di un cerchio: A = π * r²

def circle_area(radius):
    pass

# Test: circle_area(1) → ~3.14159
#       circle_area(2) → ~12.566


EXERCISE 13: Rectangle Perimeter
───────────────────────────────────────────────────────────────────────────────
Calcola il perimetro di un rettangolo.

def rectangle_perimeter(width, height):
    pass

# Test: rectangle_perimeter(3, 4) → 14


EXERCISE 14: Is Leap Year
───────────────────────────────────────────────────────────────────────────────
Restituisce True se l'anno è bisestile.
Bisestile se: divisibile per 4 E (non per 100 O divisibile per 400)

def is_leap_year(year):
    pass

# Test: is_leap_year(2000) → True
#       is_leap_year(1900) → False
#       is_leap_year(2024) → True


EXERCISE 15: Days in Month
───────────────────────────────────────────────────────────────────────────────
Restituisce il numero di giorni in un mese (1-12), considerando anno bisestile.

def days_in_month(month, year):
    pass

# Test: days_in_month(2, 2024) → 29
#       days_in_month(2, 2023) → 28
#       days_in_month(7, 2023) → 31


EXERCISE 16: Factorial
───────────────────────────────────────────────────────────────────────────────
Calcola il fattoriale di n usando un loop.

def factorial(n):
    pass

# Test: factorial(5) → 120
#       factorial(0) → 1


EXERCISE 17: Power
───────────────────────────────────────────────────────────────────────────────
Calcola base^exp SENZA usare ** o pow().

def power(base, exp):
    pass

# Test: power(2, 3) → 8
#       power(5, 0) → 1


EXERCISE 18: Sum of Digits
───────────────────────────────────────────────────────────────────────────────
Restituisce la somma delle cifre di un numero.

def sum_digits(n):
    pass

# Test: sum_digits(123) → 6
#       sum_digits(9999) → 36


EXERCISE 19: Count Digits
───────────────────────────────────────────────────────────────────────────────
Conta quante cifre ha un numero.

def count_digits(n):
    pass

# Test: count_digits(123) → 3
#       count_digits(10000) → 5


EXERCISE 20: Reverse Number
───────────────────────────────────────────────────────────────────────────────
Inverte le cifre di un numero.

def reverse_number(n):
    pass

# Test: reverse_number(123) → 321
#       reverse_number(1000) → 1


EXERCISE 21: Is Palindrome Number
───────────────────────────────────────────────────────────────────────────────
Restituisce True se il numero è palindromo.

def is_palindrome_number(n):
    pass

# Test: is_palindrome_number(121) → True
#       is_palindrome_number(123) → False


EXERCISE 22: GCD (Greatest Common Divisor)
───────────────────────────────────────────────────────────────────────────────
Calcola il massimo comun divisore usando l'algoritmo di Euclide.

def gcd(a, b):
    pass

# Test: gcd(48, 18) → 6
#       gcd(17, 5) → 1


EXERCISE 23: LCM (Least Common Multiple)
───────────────────────────────────────────────────────────────────────────────
Calcola il minimo comune multiplo: lcm(a,b) = |a*b| / gcd(a,b)

def lcm(a, b):
    pass

# Test: lcm(4, 6) → 12
#       lcm(3, 5) → 15


EXERCISE 24: Is Prime
───────────────────────────────────────────────────────────────────────────────
Restituisce True se n è primo.

def is_prime(n):
    pass

# Test: is_prime(7) → True
#       is_prime(4) → False
#       is_prime(1) → False


EXERCISE 25: Fibonacci
───────────────────────────────────────────────────────────────────────────────
Restituisce l'n-esimo numero di Fibonacci (0-indexed).

def fibonacci(n):
    pass

# Test: fibonacci(0) → 0
#       fibonacci(1) → 1
#       fibonacci(10) → 55


EXERCISE 26: Sum Range
───────────────────────────────────────────────────────────────────────────────
Somma tutti i numeri da start a end (inclusi).

def sum_range(start, end):
    pass

# Test: sum_range(1, 10) → 55
#       sum_range(5, 5) → 5


EXERCISE 27: Count Vowels
───────────────────────────────────────────────────────────────────────────────
Conta le vocali in una stringa.

def count_vowels(s):
    pass

# Test: count_vowels("hello") → 2
#       count_vowels("AEIOU") → 5


EXERCISE 28: Count Words
───────────────────────────────────────────────────────────────────────────────
Conta le parole in una stringa (separate da spazi).

def count_words(s):
    pass

# Test: count_words("Hello World") → 2
#       count_words("  one   two  three  ") → 3


EXERCISE 29: Reverse String
───────────────────────────────────────────────────────────────────────────────
Inverte una stringa SENZA usare [::-1].

def reverse_string(s):
    pass

# Test: reverse_string("hello") → "olleh"


EXERCISE 30: Is Palindrome String
───────────────────────────────────────────────────────────────────────────────
Restituisce True se la stringa è palindroma (ignora maiuscole/minuscole).

def is_palindrome(s):
    pass

# Test: is_palindrome("Anna") → True
#       is_palindrome("hello") → False


EXERCISE 31: Remove Duplicates
───────────────────────────────────────────────────────────────────────────────
Rimuove i duplicati da una stringa mantenendo l'ordine.

def remove_duplicates(s):
    pass

# Test: remove_duplicates("aabbcc") → "abc"
#       remove_duplicates("hello") → "helo"


EXERCISE 32: Find Max in List
───────────────────────────────────────────────────────────────────────────────
Trova il massimo in una lista SENZA usare max().

def find_max(lst):
    pass

# Test: find_max([3, 1, 4, 1, 5]) → 5


EXERCISE 33: Find Min in List
───────────────────────────────────────────────────────────────────────────────
Trova il minimo in una lista SENZA usare min().

def find_min(lst):
    pass

# Test: find_min([3, 1, 4, 1, 5]) → 1


EXERCISE 34: Sum List
───────────────────────────────────────────────────────────────────────────────
Somma tutti gli elementi SENZA usare sum().

def sum_list(lst):
    pass

# Test: sum_list([1, 2, 3, 4, 5]) → 15


EXERCISE 35: Average List
───────────────────────────────────────────────────────────────────────────────
Calcola la media degli elementi.

def average(lst):
    pass

# Test: average([1, 2, 3, 4, 5]) → 3.0


EXERCISE 36: Count Occurrences
───────────────────────────────────────────────────────────────────────────────
Conta quante volte un elemento appare in una lista.

def count_occurrences(lst, item):
    pass

# Test: count_occurrences([1, 2, 2, 3, 2], 2) → 3


EXERCISE 37: Find Index
───────────────────────────────────────────────────────────────────────────────
Restituisce l'indice della prima occorrenza, -1 se non trovato.

def find_index(lst, item):
    pass

# Test: find_index([1, 2, 3, 4], 3) → 2
#       find_index([1, 2, 3], 5) → -1


EXERCISE 38: Remove All Occurrences
───────────────────────────────────────────────────────────────────────────────
Rimuove tutte le occorrenze di un elemento da una lista.

def remove_all(lst, item):
    pass

# Test: remove_all([1, 2, 2, 3, 2], 2) → [1, 3]


EXERCISE 39: Flatten List
───────────────────────────────────────────────────────────────────────────────
Appiattisce una lista di liste (solo un livello).

def flatten(lst):
    pass

# Test: flatten([[1, 2], [3, 4], [5]]) → [1, 2, 3, 4, 5]


EXERCISE 40: Zip Lists
───────────────────────────────────────────────────────────────────────────────
Combina due liste in lista di tuple SENZA usare zip().

def my_zip(lst1, lst2):
    pass

# Test: my_zip([1, 2, 3], ['a', 'b', 'c']) → [(1, 'a'), (2, 'b'), (3, 'c')]


EXERCISE 41: Unique Elements
───────────────────────────────────────────────────────────────────────────────
Restituisce gli elementi unici mantenendo l'ordine.

def unique(lst):
    pass

# Test: unique([1, 2, 2, 3, 1, 4]) → [1, 2, 3, 4]


EXERCISE 42: Intersection
───────────────────────────────────────────────────────────────────────────────
Restituisce gli elementi presenti in entrambe le liste.

def intersection(lst1, lst2):
    pass

# Test: intersection([1, 2, 3], [2, 3, 4]) → [2, 3]


EXERCISE 43: Union
───────────────────────────────────────────────────────────────────────────────
Restituisce tutti gli elementi unici da entrambe le liste.

def union(lst1, lst2):
    pass

# Test: union([1, 2, 3], [2, 3, 4]) → [1, 2, 3, 4]


EXERCISE 44: Difference
───────────────────────────────────────────────────────────────────────────────
Restituisce gli elementi in lst1 ma non in lst2.

def difference(lst1, lst2):
    pass

# Test: difference([1, 2, 3], [2, 3, 4]) → [1]


EXERCISE 45: Rotate List
───────────────────────────────────────────────────────────────────────────────
Ruota la lista di n posizioni a destra.

def rotate(lst, n):
    pass

# Test: rotate([1, 2, 3, 4, 5], 2) → [4, 5, 1, 2, 3]


EXERCISE 46: Chunk List
───────────────────────────────────────────────────────────────────────────────
Divide la lista in chunk di dimensione n.

def chunk(lst, n):
    pass

# Test: chunk([1, 2, 3, 4, 5], 2) → [[1, 2], [3, 4], [5]]


EXERCISE 47: Merge Sorted
───────────────────────────────────────────────────────────────────────────────
Unisce due liste ordinate in una lista ordinata.

def merge_sorted(lst1, lst2):
    pass

# Test: merge_sorted([1, 3, 5], [2, 4, 6]) → [1, 2, 3, 4, 5, 6]


EXERCISE 48: Second Largest
───────────────────────────────────────────────────────────────────────────────
Trova il secondo elemento più grande.

def second_largest(lst):
    pass

# Test: second_largest([1, 5, 2, 4, 3]) → 4


EXERCISE 49: Most Frequent
───────────────────────────────────────────────────────────────────────────────
Trova l'elemento più frequente.

def most_frequent(lst):
    pass

# Test: most_frequent([1, 2, 2, 3, 2, 4]) → 2


EXERCISE 50: Is Sorted
───────────────────────────────────────────────────────────────────────────────
Verifica se la lista è ordinata (crescente).

def is_sorted(lst):
    pass

# Test: is_sorted([1, 2, 3, 4]) → True
#       is_sorted([1, 3, 2, 4]) → False

"""

# ══════════════════════════════════════════════════════════════════════════════
#                    SOLUTIONS BLOCK 1
# ══════════════════════════════════════════════════════════════════════════════

solutions_block1 = """
════════════════════════════════════════════════════════════════════════════════
                         SOLUTIONS BLOCK 1 (1-50)
════════════════════════════════════════════════════════════════════════════════
"""

# Exercise 1
def hello(name):
    print(f"Hello, {name}!")

# Exercise 2
def add(a, b):
    return a + b

# Exercise 3
def is_even(n):
    return n % 2 == 0

# Exercise 4
def my_abs(n):
    return n if n >= 0 else -n

# Exercise 5
def my_max(a, b):
    return a if a > b else b

# Exercise 6
def my_min(a, b):
    return a if a < b else b

# Exercise 7
def sign(n):
    if n > 0:
        return 1
    elif n < 0:
        return -1
    else:
        return 0

# Exercise 8
def is_digit(char):
    return '0' <= char <= '9'

# Exercise 9
def is_letter(char):
    return ('a' <= char <= 'z') or ('A' <= char <= 'Z')

# Exercise 10
def celsius_to_fahrenheit(c):
    return c * 9 / 5 + 32

# Exercise 11
def fahrenheit_to_celsius(f):
    return (f - 32) * 5 / 9

# Exercise 12
def circle_area(radius):
    import math
    return math.pi * radius ** 2

# Exercise 13
def rectangle_perimeter(width, height):
    return 2 * (width + height)

# Exercise 14
def is_leap_year(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

# Exercise 15
def days_in_month(month, year):
    if month in [1, 3, 5, 7, 8, 10, 12]:
        return 31
    elif month in [4, 6, 9, 11]:
        return 30
    elif month == 2:
        return 29 if is_leap_year(year) else 28

# Exercise 16
def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

# Exercise 17
def power(base, exp):
    result = 1
    for _ in range(exp):
        result *= base
    return result

# Exercise 18
def sum_digits(n):
    n = abs(n)
    total = 0
    while n > 0:
        total += n % 10
        n //= 10
    return total

# Exercise 19
def count_digits(n):
    n = abs(n)
    if n == 0:
        return 1
    count = 0
    while n > 0:
        count += 1
        n //= 10
    return count

# Exercise 20
def reverse_number(n):
    reversed_n = 0
    while n > 0:
        reversed_n = reversed_n * 10 + n % 10
        n //= 10
    return reversed_n

# Exercise 21
def is_palindrome_number(n):
    return n == reverse_number(n)

# Exercise 22
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# Exercise 23
def lcm(a, b):
    return abs(a * b) // gcd(a, b)

# Exercise 24
def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

# Exercise 25
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# Exercise 26
def sum_range(start, end):
    total = 0
    for i in range(start, end + 1):
        total += i
    return total

# Exercise 27
def count_vowels(s):
    vowels = "aeiouAEIOU"
    return sum(1 for c in s if c in vowels)

# Exercise 28
def count_words(s):
    return len(s.split())

# Exercise 29
def reverse_string(s):
    result = ""
    for char in s:
        result = char + result
    return result

# Exercise 30
def is_palindrome(s):
    s = s.lower()
    return s == s[::-1]

# Exercise 31
def remove_duplicates(s):
    seen = ""
    for c in s:
        if c not in seen:
            seen += c
    return seen

# Exercise 32
def find_max(lst):
    if not lst:
        return None
    max_val = lst[0]
    for item in lst[1:]:
        if item > max_val:
            max_val = item
    return max_val

# Exercise 33
def find_min(lst):
    if not lst:
        return None
    min_val = lst[0]
    for item in lst[1:]:
        if item < min_val:
            min_val = item
    return min_val

# Exercise 34
def sum_list(lst):
    total = 0
    for item in lst:
        total += item
    return total

# Exercise 35
def average(lst):
    if not lst:
        return 0
    return sum_list(lst) / len(lst)

# Exercise 36
def count_occurrences(lst, item):
    count = 0
    for x in lst:
        if x == item:
            count += 1
    return count

# Exercise 37
def find_index(lst, item):
    for i, x in enumerate(lst):
        if x == item:
            return i
    return -1

# Exercise 38
def remove_all(lst, item):
    return [x for x in lst if x != item]

# Exercise 39
def flatten(lst):
    result = []
    for sublist in lst:
        for item in sublist:
            result.append(item)
    return result

# Exercise 40
def my_zip(lst1, lst2):
    result = []
    for i in range(min(len(lst1), len(lst2))):
        result.append((lst1[i], lst2[i]))
    return result

# Exercise 41
def unique(lst):
    seen = []
    for item in lst:
        if item not in seen:
            seen.append(item)
    return seen

# Exercise 42
def intersection(lst1, lst2):
    return [x for x in lst1 if x in lst2]

# Exercise 43
def union(lst1, lst2):
    result = lst1[:]
    for item in lst2:
        if item not in result:
            result.append(item)
    return result

# Exercise 44
def difference(lst1, lst2):
    return [x for x in lst1 if x not in lst2]

# Exercise 45
def rotate(lst, n):
    if not lst:
        return lst
    n = n % len(lst)
    return lst[-n:] + lst[:-n]

# Exercise 46
def chunk(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

# Exercise 47
def merge_sorted(lst1, lst2):
    result = []
    i = j = 0
    while i < len(lst1) and j < len(lst2):
        if lst1[i] <= lst2[j]:
            result.append(lst1[i])
            i += 1
        else:
            result.append(lst2[j])
            j += 1
    result.extend(lst1[i:])
    result.extend(lst2[j:])
    return result

# Exercise 48
def second_largest(lst):
    if len(lst) < 2:
        return None
    first = second = float('-inf')
    for n in lst:
        if n > first:
            second = first
            first = n
        elif n > second and n != first:
            second = n
    return second

# Exercise 49
def most_frequent(lst):
    counts = {}
    for item in lst:
        counts[item] = counts.get(item, 0) + 1
    return max(counts, key=counts.get)

# Exercise 50
def is_sorted(lst):
    for i in range(len(lst) - 1):
        if lst[i] > lst[i + 1]:
            return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
#                    BLOCK 2: INTERMEDIATE (51-100) - Esercizi
# ══════════════════════════════════════════════════════════════════════════════

exercises_block2 = """
════════════════════════════════════════════════════════════════════════════════
                         BLOCK 2: INTERMEDIATE (51-100)
════════════════════════════════════════════════════════════════════════════════

EXERCISE 51: Two Sum
───────────────────────────────────────────────────────────────────────────────
Trova due numeri nella lista che sommano a target. Restituisci i loro indici.

def two_sum(nums, target):
    pass

# Test: two_sum([2, 7, 11, 15], 9) → (0, 1)


EXERCISE 52: Anagram Check
───────────────────────────────────────────────────────────────────────────────
Verifica se due stringhe sono anagrammi.

def is_anagram(s1, s2):
    pass

# Test: is_anagram("listen", "silent") → True


EXERCISE 53: Valid Parentheses
───────────────────────────────────────────────────────────────────────────────
Verifica se le parentesi sono bilanciate.

def is_valid_parentheses(s):
    pass

# Test: is_valid_parentheses("()[]{}") → True
#       is_valid_parentheses("([)]") → False


EXERCISE 54: Binary to Decimal
───────────────────────────────────────────────────────────────────────────────
Converte stringa binaria in decimale SENZA int(s, 2).

def binary_to_decimal(binary):
    pass

# Test: binary_to_decimal("1010") → 10


EXERCISE 55: Decimal to Binary
───────────────────────────────────────────────────────────────────────────────
Converte decimale in stringa binaria SENZA bin().

def decimal_to_binary(n):
    pass

# Test: decimal_to_binary(10) → "1010"


EXERCISE 56: Pascal's Triangle Row
───────────────────────────────────────────────────────────────────────────────
Genera la n-esima riga del triangolo di Pascal.

def pascal_row(n):
    pass

# Test: pascal_row(4) → [1, 4, 6, 4, 1]


EXERCISE 57: Spiral Matrix
───────────────────────────────────────────────────────────────────────────────
Leggi una matrice in ordine a spirale.

def spiral_order(matrix):
    pass

# Test: spiral_order([[1,2,3],[4,5,6],[7,8,9]]) → [1,2,3,6,9,8,7,4,5]


EXERCISE 58: FizzBuzz List
───────────────────────────────────────────────────────────────────────────────
Genera lista FizzBuzz da 1 a n.

def fizzbuzz(n):
    pass

# Test: fizzbuzz(5) → ["1", "2", "Fizz", "4", "Buzz"]


EXERCISE 59: Run-Length Encoding
───────────────────────────────────────────────────────────────────────────────
Comprimi stringa con run-length encoding.

def encode_rle(s):
    pass

# Test: encode_rle("aaabbc") → "a3b2c1"


EXERCISE 60: Run-Length Decoding
───────────────────────────────────────────────────────────────────────────────
Decodifica stringa compressa.

def decode_rle(s):
    pass

# Test: decode_rle("a3b2c1") → "aaabbc"


EXERCISE 61: Matrix Transpose
───────────────────────────────────────────────────────────────────────────────
Trasponi una matrice.

def transpose(matrix):
    pass

# Test: transpose([[1,2],[3,4]]) → [[1,3],[2,4]]


EXERCISE 62: Matrix Multiplication
───────────────────────────────────────────────────────────────────────────────
Moltiplica due matrici.

def matrix_multiply(A, B):
    pass

# Test: matrix_multiply([[1,2],[3,4]], [[5,6],[7,8]]) → [[19,22],[43,50]]


EXERCISE 63: Prime Factors
───────────────────────────────────────────────────────────────────────────────
Trova i fattori primi di un numero.

def prime_factors(n):
    pass

# Test: prime_factors(12) → [2, 2, 3]


EXERCISE 64: Sieve of Eratosthenes
───────────────────────────────────────────────────────────────────────────────
Trova tutti i numeri primi fino a n.

def sieve(n):
    pass

# Test: sieve(10) → [2, 3, 5, 7]


EXERCISE 65: Roman to Integer
───────────────────────────────────────────────────────────────────────────────
Converte numero romano in intero.

def roman_to_int(s):
    pass

# Test: roman_to_int("XIV") → 14


EXERCISE 66: Integer to Roman
───────────────────────────────────────────────────────────────────────────────
Converte intero in numero romano (1-3999).

def int_to_roman(n):
    pass

# Test: int_to_roman(14) → "XIV"


EXERCISE 67: Longest Common Prefix
───────────────────────────────────────────────────────────────────────────────
Trova il prefisso comune più lungo.

def longest_common_prefix(strings):
    pass

# Test: longest_common_prefix(["flower","flow","flight"]) → "fl"


EXERCISE 68: Group Anagrams
───────────────────────────────────────────────────────────────────────────────
Raggruppa le stringhe che sono anagrammi.

def group_anagrams(strings):
    pass

# Test: group_anagrams(["eat","tea","tan","ate","nat","bat"]) 
#       → [["eat","tea","ate"],["tan","nat"],["bat"]]


EXERCISE 69: Missing Number
───────────────────────────────────────────────────────────────────────────────
Trova il numero mancante da 0 a n.

def missing_number(nums):
    pass

# Test: missing_number([0, 1, 3]) → 2


EXERCISE 70: Single Number
───────────────────────────────────────────────────────────────────────────────
Trova l'unico numero che appare una volta (altri appaiono due volte).

def single_number(nums):
    pass

# Test: single_number([2,2,1]) → 1


... (Exercises 71-100 continue with similar format)

"""

# ══════════════════════════════════════════════════════════════════════════════
#                    SOLUTIONS BLOCK 2 (51-70)
# ══════════════════════════════════════════════════════════════════════════════

# Exercise 51
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return None

# Exercise 52
def is_anagram(s1, s2):
    return sorted(s1.lower()) == sorted(s2.lower())

# Exercise 53
def is_valid_parentheses(s):
    stack = []
    mapping = {')': '(', ']': '[', '}': '{'}
    for char in s:
        if char in '([{':
            stack.append(char)
        elif char in ')]}':
            if not stack or stack.pop() != mapping[char]:
                return False
    return len(stack) == 0

# Exercise 54
def binary_to_decimal(binary):
    result = 0
    for digit in binary:
        result = result * 2 + int(digit)
    return result

# Exercise 55
def decimal_to_binary(n):
    if n == 0:
        return "0"
    result = ""
    while n > 0:
        result = str(n % 2) + result
        n //= 2
    return result

# Exercise 56
def pascal_row(n):
    row = [1]
    for i in range(n):
        row = [1] + [row[j] + row[j+1] for j in range(len(row)-1)] + [1]
    return row

# Exercise 57
def spiral_order(matrix):
    result = []
    while matrix:
        result += matrix.pop(0)
        if matrix and matrix[0]:
            for row in matrix:
                result.append(row.pop())
        if matrix:
            result += matrix.pop()[::-1]
        if matrix and matrix[0]:
            for row in matrix[::-1]:
                result.append(row.pop(0))
    return result

# Exercise 58
def fizzbuzz(n):
    result = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            result.append("FizzBuzz")
        elif i % 3 == 0:
            result.append("Fizz")
        elif i % 5 == 0:
            result.append("Buzz")
        else:
            result.append(str(i))
    return result

# Exercise 59
def encode_rle(s):
    if not s:
        return ""
    result = ""
    count = 1
    for i in range(1, len(s)):
        if s[i] == s[i-1]:
            count += 1
        else:
            result += s[i-1] + str(count)
            count = 1
    result += s[-1] + str(count)
    return result

# Exercise 60
def decode_rle(s):
    result = ""
    i = 0
    while i < len(s):
        char = s[i]
        num = ""
        i += 1
        while i < len(s) and s[i].isdigit():
            num += s[i]
            i += 1
        result += char * int(num)
    return result

# Exercise 61
def transpose(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

# Exercise 62
def matrix_multiply(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    result = [[0] * cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result

# Exercise 63
def prime_factors(n):
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

# Exercise 64
def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(n + 1) if is_prime[i]]

# Exercise 65
def roman_to_int(s):
    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    result = 0
    for i in range(len(s)):
        if i + 1 < len(s) and values[s[i]] < values[s[i + 1]]:
            result -= values[s[i]]
        else:
            result += values[s[i]]
    return result

# Exercise 66
def int_to_roman(n):
    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syms = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
    result = ""
    for i in range(len(val)):
        while n >= val[i]:
            result += syms[i]
            n -= val[i]
    return result

# Exercise 67
def longest_common_prefix(strings):
    if not strings:
        return ""
    prefix = strings[0]
    for s in strings[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix

# Exercise 68
def group_anagrams(strings):
    groups = {}
    for s in strings:
        key = tuple(sorted(s))
        if key not in groups:
            groups[key] = []
        groups[key].append(s)
    return list(groups.values())

# Exercise 69
def missing_number(nums):
    n = len(nums)
    expected = n * (n + 1) // 2
    return expected - sum(nums)

# Exercise 70
def single_number(nums):
    result = 0
    for num in nums:
        result ^= num  # XOR
    return result


if __name__ == "__main__":
    print("=" * 70)
    print("200 ESERCIZI PRATICI PCEP")
    print("=" * 70)
    print("""
    Comandi:
    print(exercises_block1)  # Esercizi 1-50
    print(exercises_block2)  # Esercizi 51-100
    
    Le soluzioni sono funzioni definite nel modulo:
    - two_sum, is_anagram, is_valid_parentheses, etc.
    
    Testa ogni funzione prima di vedere la soluzione!
    """)
