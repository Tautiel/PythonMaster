#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PCEP-30-02 EXAM SIMULATIONS                               ║
║                    3 Complete Mock Exams (90 Questions)                      ║
║                    Allineato 100% al Syllabus Ufficiale                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

FORMATO ESAME REALE:
- 30 domande | 40 minuti | 70% per passare (21/30)
- Block 1: Fundamentals (18%) ~5 domande
- Block 2: Control Flow (29%) ~9 domande  
- Block 3: Data Collections (25%) ~8 domande
- Block 4: Functions/Exceptions (28%) ~8 domande

ISTRUZIONI:
1. Timer 40 minuti
2. NO esecuzione codice
3. Scrivi risposte su foglio
4. Target: 80%+ prima dell'esame reale
"""

# ══════════════════════════════════════════════════════════════════════════════
#                           SIMULATION 1 - STANDARD
# ══════════════════════════════════════════════════════════════════════════════

SIM1_QUESTIONS = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SIMULATION 1 - 30 Questions - 40 Minutes                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

═══ BLOCK 1: FUNDAMENTALS (Q1-5) ═══

Q1. What is the output?
    print("A", "B", sep="-", end="!")
A) A B!    B) A-B!    C) A-B !    D) AB!

Q2. Which is a valid variable name?
A) 2var    B) my-var    C) _private    D) class

Q3. Result of: 17 // 5
A) 3.4    B) 3    C) 4    D) 2

Q4. Result of: 2 ** 3 ** 2
A) 64    B) 512    C) 36    D) 81

Q5. Result of: 0b1010 + 0o12 + 0xA
A) 30    B) 32    C) 1032    D) Error

═══ BLOCK 2: CONTROL FLOW (Q6-14) ═══

Q6. What is the output?
    x = 10
    if x > 5:
        if x > 15:
            print("A")
        else:
            print("B")
    else:
        print("C")
A) A    B) B    C) C    D) No output

Q7. What is the output?
    for i in range(2, 10, 3):
        print(i, end=" ")
A) 2 5 8    B) 2 5 8 11    C) 2 3 4 5 6 7 8 9    D) 3 6 9

Q8. What is the output?
    x = 0
    while x < 3:
        x += 1
    else:
        print("Done")
A) No output    B) Done    C) 3    D) Error

Q9. What is the output?
    for i in range(3):
        if i == 1:
            break
    else:
        print("Complete")
    print("End")
A) Complete End    B) End    C) Complete    D) No output

Q10. Result of: not True or False and True
A) True    B) False    C) Error    D) None

Q11. Result of: 5 & 3
A) 8    B) 1    C) 5    D) 3

Q12. Result of: 5 | 3
A) 8    B) 7    C) 5    D) 3

Q13. What is the output?
    for i in range(5):
        if i % 2 == 0:
            print(i, end=" ")
A) 0 2 4    B) 1 3 5    C) 0 1 2 3 4    D) 2 4

Q14. What does pass do?
A) Exits loop    B) Skips iteration    C) Does nothing    D) Raises exception

═══ BLOCK 3: DATA COLLECTIONS (Q15-22) ═══

Q15. What is the output?
    my_list = [1, 2, 3, 4, 5]
    print(my_list[1:4])
A) [1, 2, 3, 4]    B) [2, 3, 4]    C) [2, 3, 4, 5]    D) [1, 2, 3]

Q16. What is the output?
    my_tuple = (1, 2, 3)
    my_tuple[0] = 10
    print(my_tuple)
A) (10, 2, 3)    B) (1, 2, 3)    C) Error    D) [10, 2, 3]

Q17. What is the output?
    my_dict = {"a": 1, "b": 2}
    print(my_dict.get("c", 0))
A) None    B) 0    C) Error    D) "c"

Q18. What is the output?
    my_list = [1, 2, 3]
    my_list.append([4, 5])
    print(len(my_list))
A) 3    B) 4    C) 5    D) Error

Q19. What is the output?
    text = "Python"
    print(text[-2])
A) P    B) o    C) n    D) h

Q20. What is the output?
    result = [x**2 for x in range(4)]
    print(result)
A) [0, 1, 4, 9]    B) [1, 4, 9, 16]    C) [0, 1, 2, 3]    D) [1, 2, 3, 4]

Q21. What is the output?
    my_list = [1, 2, 3, 4, 5]
    print(my_list[::2])
A) [1, 3, 5]    B) [2, 4]    C) [1, 2]    D) [5, 4, 3, 2, 1]

Q22. What is the output?
    my_list = [1, 2, 3, 4, 5]
    print(my_list[::-1])
A) [1, 2, 3, 4, 5]    B) [5, 4, 3, 2, 1]    C) [5]    D) Error

═══ BLOCK 4: FUNCTIONS & EXCEPTIONS (Q23-30) ═══

Q23. What is the output?
    def greet(name="World"):
        return "Hello, " + name
    print(greet())
A) Hello,    B) Hello, World    C) Error    D) None

Q24. What is the output?
    def func(a, b=2, c=3):
        return a + b + c
    print(func(1, c=5))
A) 6    B) 8    C) 10    D) Error

Q25. What is the output?
    x = 10
    def change():
        x = 20
        return x
    change()
    print(x)
A) 10    B) 20    C) None    D) Error

Q26. What is the output?
    def factorial(n):
        if n <= 1:
            return 1
        return n * factorial(n - 1)
    print(factorial(4))
A) 4    B) 10    C) 24    D) 120

Q27. What is the output?
    try:
        x = 10 / 0
    except ZeroDivisionError:
        print("A")
    except:
        print("B")
    else:
        print("C")
A) A    B) B    C) C    D) A B

Q28. What is the output?
    def test():
        return
    result = test()
    print(result)
A) Nothing    B) None    C) Error    D) 0

Q29. Which exception for invalid list index?
A) ValueError    B) KeyError    C) IndexError    D) TypeError

Q30. What is the output?
    x = lambda a, b: a * b
    print(x(3, 4))
A) (3, 4)    B) 12    C) lambda    D) Error
"""

SIM1_ANSWERS = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SIMULATION 1 - ANSWERS                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Q1:  B) A-B!         | sep="-" replaces space, end="!" replaces newline
Q2:  C) _private     | Valid: starts with _ or letter, no hyphens, not keyword
Q3:  B) 3            | Floor division 17//5 = 3
Q4:  B) 512          | Right-associative: 3**2=9, then 2**9=512
Q5:  A) 30           | Binary 10 + Octal 10 + Hex 10 = 30

Q6:  B) B            | 10>5 True, 10>15 False → else prints B
Q7:  A) 2 5 8        | range(2,10,3) = 2, 5, 8
Q8:  B) Done         | while-else runs when loop completes normally
Q9:  B) End          | break skips else clause
Q10: B) False        | not True=False, False and True=False, False or False=False
Q11: B) 1            | 101 & 011 = 001 = 1
Q12: B) 7            | 101 | 011 = 111 = 7
Q13: A) 0 2 4        | Even numbers in range(5)
Q14: C) Does nothing | Placeholder statement

Q15: B) [2, 3, 4]    | Indices 1, 2, 3
Q16: C) Error        | Tuples are immutable
Q17: B) 0            | .get() returns default if key missing
Q18: B) 4            | append adds [4,5] as ONE element
Q19: B) o            | Index -2 = second from end
Q20: A) [0, 1, 4, 9] | 0², 1², 2², 3²
Q21: A) [1, 3, 5]    | Step 2 from start
Q22: B) [5,4,3,2,1]  | Step -1 reverses

Q23: B) Hello, World | Default parameter used
Q24: B) 8            | 1 + 2 + 5 = 8
Q25: A) 10           | Local x doesn't affect global
Q26: C) 24           | 4! = 4×3×2×1 = 24
Q27: A) A            | ZeroDivisionError caught
Q28: B) None         | Function without return value returns None
Q29: C) IndexError   | For list/tuple index errors
Q30: B) 12           | Lambda: 3 * 4 = 12

SCORE: ___/30  |  PASS: 21+  |  TARGET: 24+
"""

# ══════════════════════════════════════════════════════════════════════════════
#                           SIMULATION 2 - STANDARD
# ══════════════════════════════════════════════════════════════════════════════

SIM2_QUESTIONS = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SIMULATION 2 - 30 Questions - 40 Minutes                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

═══ BLOCK 1: FUNDAMENTALS (Q1-5) ═══

Q1. What is the output?
    print(type(3.14).__name__)
A) float    B) <class 'float'>    C) 3.14    D) number

Q2. What is the output?
    x = "10"
    y = 5
    print(int(x) + y)
A) "105"    B) 15    C) Error    D) 105

Q3. Result of: -3 % 2
A) -1    B) 1    C) -2    D) 0

Q4. Result of: ~5
A) -5    B) -6    C) 4    D) 5

Q5. What is the output?
    x = 1.1 + 2.2
    print(x == 3.3)
A) True    B) False    C) 3.3    D) Error

═══ BLOCK 2: CONTROL FLOW (Q6-14) ═══

Q6. What is the output?
    x = 0
    if x:
        print("True")
    else:
        print("False")
A) True    B) False    C) 0    D) Error

Q7. What is the output?
    for i in range(5, 0, -2):
        print(i, end=" ")
A) 5 3 1    B) 5 4 3 2 1    C) 1 3 5    D) 5 3 1 -1

Q8. What is the output?
    i = 5
    while i > 0:
        i -= 1
        if i == 2:
            break
        print(i, end=" ")
A) 4 3    B) 4 3 2 1 0    C) 5 4 3    D) 4 3 2

Q9. What is the output?
    x = 5
    y = 10
    z = 15
    print(x < y < z)
A) True    B) False    C) Error    D) None

Q10. What is the output?
    for letter in "Python":
        if letter == "h":
            continue
        print(letter, end="")
A) Python    B) Pyton    C) Pytho    D) ython

Q11. Result of: 8 >> 2
A) 2    B) 4    C) 32    D) 16

Q12. Result of: 2 << 3
A) 6    B) 8    C) 16    D) 5

Q13. What is the output?
    x, y = 10, 20
    x, y = y, x
    print(x, y)
A) 10 20    B) 20 10    C) 20 20    D) Error

Q14. Result of: True + True + False
A) True    B) 2    C) 1    D) Error

═══ BLOCK 3: DATA COLLECTIONS (Q15-22) ═══

Q15. What is the output?
    my_list = [1, 2, 3, 4, 5]
    print(my_list[-3:])
A) [3, 4, 5]    B) [1, 2]    C) [3, 4]    D) [1, 2, 3]

Q16. What is the output?
    my_list = [[1, 2], [3, 4]]
    print(my_list[1][0])
A) 1    B) 2    C) 3    D) 4

Q17. What is the output?
    my_dict = {"a": 1, "b": 2}
    my_dict["c"] = 3
    print(len(my_dict))
A) 2    B) 3    C) 4    D) Error

Q18. What is the output?
    my_list = [1, 2, 3]
    my_list.insert(1, 10)
    print(my_list)
A) [10, 1, 2, 3]    B) [1, 10, 2, 3]    C) [1, 2, 10, 3]    D) [1, 2, 3, 10]

Q19. What is the output?
    my_list = [1, 2, 3]
    my_list.extend([4, 5])
    print(len(my_list))
A) 3    B) 4    C) 5    D) Error

Q20. What is the output?
    my_list = [1, 2, 3]
    print(my_list * 2)
A) [2, 4, 6]    B) [1, 2, 3, 1, 2, 3]    C) [[1,2,3],[1,2,3]]    D) Error

Q21. What is the output?
    t1 = (1, 2)
    t2 = (3, 4)
    print(t1 + t2)
A) (4, 6)    B) (1, 2, 3, 4)    C) ((1,2), (3,4))    D) Error

Q22. What is the output?
    d = {1: "a", 2: "b"}
    print(list(d.keys()))
A) ["a", "b"]    B) [1, 2]    C) [(1,"a"), (2,"b")]    D) {1, 2}

═══ BLOCK 4: FUNCTIONS & EXCEPTIONS (Q23-30) ═══

Q23. What is the output?
    def func(*args):
        return sum(args)
    print(func(1, 2, 3, 4))
A) (1, 2, 3, 4)    B) 10    C) [1, 2, 3, 4]    D) Error

Q24. What is the output?
    def outer(x):
        def inner(y):
            return x + y
        return inner
    f = outer(10)
    print(f(5))
A) 10    B) 5    C) 15    D) Error

Q25. What is the output?
    x = 10
    def change():
        global x
        x = 20
    change()
    print(x)
A) 10    B) 20    C) Error    D) None

Q26. What is the output?
    try:
        x = int("abc")
    except ValueError:
        print("A")
    finally:
        print("B")
A) A    B) B    C) A then B    D) Error

Q27. Base class for ALL built-in exceptions?
A) Exception    B) BaseException    C) Error    D) StandardError

Q28. What is the output?
    def func(a, b, c):
        return a + b * c
    print(func(c=2, a=1, b=3))
A) 7    B) 9    C) 8    D) Error

Q29. What is the output?
    def gen():
        yield 1
        yield 2
    g = gen()
    print(next(g), next(g))
A) 1 2    B) 1 1    C) (1, 2)    D) Error

Q30. Exception for missing dictionary key?
A) IndexError    B) KeyError    C) ValueError    D) LookupError
"""

SIM2_ANSWERS = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SIMULATION 2 - ANSWERS                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Q1:  A) float        | __name__ returns class name as string
Q2:  B) 15           | int("10")=10, 10+5=15
Q3:  B) 1            | Python modulo returns sign of divisor
Q4:  B) -6           | Bitwise NOT: ~n = -(n+1)
Q5:  B) False        | Floating point precision issue

Q6:  B) False        | 0 is falsy
Q7:  A) 5 3 1        | range(5,0,-2) = 5, 3, 1
Q8:  A) 4 3          | Prints 4, 3, then breaks at i==2
Q9:  A) True         | Chained comparison: 5<10<15
Q10: B) Pyton        | continue skips 'h'
Q11: A) 2            | 8 (1000) >> 2 = 2 (0010)
Q12: C) 16           | 2 (0010) << 3 = 16 (10000)
Q13: B) 20 10        | Tuple swap
Q14: B) 2            | True=1, False=0: 1+1+0=2

Q15: A) [3, 4, 5]    | Last 3 elements
Q16: C) 3            | [1][0] = second list, first element
Q17: B) 3            | Added new key
Q18: B) [1, 10, 2, 3]| Insert at index 1
Q19: C) 5            | extend adds individual elements
Q20: B) [1,2,3,1,2,3]| Repetition
Q21: B) (1, 2, 3, 4) | Tuple concatenation
Q22: B) [1, 2]       | Keys as list

Q23: B) 10           | *args collects args, sum=10
Q24: C) 15           | Closure: 10 + 5
Q25: B) 20           | global allows modification
Q26: C) A then B     | except then finally
Q27: B) BaseException| Root of exception hierarchy
Q28: A) 7            | 1 + 3*2 = 7
Q29: A) 1 2          | Generator yields sequentially
Q30: B) KeyError     | For missing dict keys

SCORE: ___/30  |  PASS: 21+  |  TARGET: 24+
"""

# ══════════════════════════════════════════════════════════════════════════════
#                           SIMULATION 3 - HARDER
# ══════════════════════════════════════════════════════════════════════════════

SIM3_QUESTIONS = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SIMULATION 3 - HARDER - Final Practice                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

═══ BLOCK 1: FUNDAMENTALS (Q1-5) ═══

Q1. What is the output?
    print(1, 2, 3, sep="", end="!")
A) 1 2 3!    B) 123!    C) 1, 2, 3!    D) 123 !

Q2. What is the output?
    x = y = z = 0
    x += 1
    print(x, y, z)
A) 1 1 1    B) 1 0 0    C) 0 0 0    D) Error

Q3. Result of: 8 ^ 3
A) 24    B) 11    C) 512    D) 5

Q4. What is the output?
    print(bool([]), bool({}), bool(""))
A) True True True    B) False False False    C) True False False    D) Error

Q5. Result of: 15 % -4
A) 3    B) -1    C) -3    D) 1

═══ BLOCK 2: CONTROL FLOW (Q6-14) ═══

Q6. What is the output?
    x = None
    if x:
        print("A")
    elif x is None:
        print("B")
    else:
        print("C")
A) A    B) B    C) C    D) No output

Q7. What is the output?
    for i in range(3):
        for j in range(3):
            if i == j:
                break
            print(i, j, end=" ")
A) 1 0 2 0 2 1    B) 0 1 0 2 1 2    C) Nothing    D) Error

Q8. What is the output?
    result = []
    for i in range(4):
        if i == 2:
            continue
        result.append(i)
    else:
        result.append("end")
    print(result)
A) [0, 1, 3, "end"]    B) [0, 1, 2, 3, "end"]    C) [0, 1, 3]    D) Error

Q9. What is the output?
    x = 5
    y = x if x > 3 else 0
    print(y)
A) 5    B) 0    C) True    D) 3

Q10. Result of: bool([]) or bool({}) or bool(0) or bool("")
A) True    B) False    C) 0    D) []

Q11. What is the output?
    count = 0
    while count < 5:
        count += 1
        if count == 3:
            continue
    print(count)
A) 3    B) 4    C) 5    D) 2

Q12. What is the output?
    a = [1, 2, 3]
    b = a
    b[0] = 10
    print(a[0])
A) 1    B) 10    C) [10, 2, 3]    D) Error

Q13. Result of: 0 or "" or [] or "hello" or 0
A) 0    B) ""    C) []    D) "hello"

Q14. What is the output?
    for i in "abc":
        pass
    print(i)
A) a    B) c    C) abc    D) Error

═══ BLOCK 3: DATA COLLECTIONS (Q15-22) ═══

Q15. What is the output?
    text = "abcdef"
    print(text[1:5:2])
A) bd    B) bcd    C) ace    D) bcde

Q16. What is the output?
    d = {}
    d[(1, 2)] = "tuple"
    print(d[(1, 2)])
A) tuple    B) Error    C) (1, 2)    D) None

Q17. What is the output?
    my_list = [3, 1, 4, 1, 5]
    my_list.sort()
    print(my_list[2])
A) 4    B) 3    C) 1    D) 5

Q18. What is the output?
    my_dict = {"a": 1, "b": 2}
    for k, v in my_dict.items():
        print(v, end="")
A) ab    B) 12    C) a1b2    D) Error

Q19. What is the output?
    t = (1,)
    print(type(t).__name__)
A) int    B) tuple    C) list    D) Error

Q20. What is the output?
    my_list = [1, 2, 3, 4, 5]
    del my_list[1:3]
    print(my_list)
A) [1, 4, 5]    B) [1, 2, 5]    C) [3, 4, 5]    D) Error

Q21. What is the output?
    "x" in {"x": 1, "y": 2}
A) True    B) False    C) 1    D) Error

Q22. What is the output?
    s = "hello"
    print(s[10:])
A) Error    B) ""    C) "hello"    D) None

═══ BLOCK 4: FUNCTIONS & EXCEPTIONS (Q23-30) ═══

Q23. What is the output?
    def func(a, b=[]):
        b.append(a)
        return b
    print(func(1))
    print(func(2))
A) [1] [2]    B) [1] [1, 2]    C) [1, 2] [1, 2]    D) Error

Q24. What is the output?
    def func(**kwargs):
        return len(kwargs)
    print(func(a=1, b=2, c=3))
A) (a, b, c)    B) 3    C) {"a":1,"b":2,"c":3}    D) Error

Q25. What is the output?
    def f1():
        return f2()
    def f2():
        return "Hello"
    print(f1())
A) Hello    B) None    C) Error    D) f2()

Q26. What is the output?
    try:
        x = 1 / 0
    except ArithmeticError:
        print("A")
    except ZeroDivisionError:
        print("B")
A) A    B) B    C) A B    D) Error

Q27. What is the output?
    def func(x):
        try:
            return x / 0
        finally:
            return "finally"
    print(func(10))
A) Error    B) finally    C) inf    D) None

Q28. What is the output?
    def factorial(n):
        if n == 0:
            return 1
        return n * factorial(n - 1)
    print(factorial(0))
A) 0    B) 1    C) Error    D) None

Q29. What is the output?
    x = 5
    def outer():
        x = 10
        def inner():
            print(x)
        inner()
    outer()
A) 5    B) 10    C) Error    D) None

Q30. What is the output?
    try:
        raise ValueError("test")
    except Exception as e:
        print(type(e).__name__)
A) Exception    B) ValueError    C) test    D) Error
"""

SIM3_ANSWERS = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SIMULATION 3 - ANSWERS                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Q1:  B) 123!         | sep="" removes spaces
Q2:  B) 1 0 0        | Integers are immutable, x+=1 creates new object
Q3:  B) 11           | XOR: 1000 ^ 0011 = 1011 = 11
Q4:  B) False×3      | Empty containers are falsy
Q5:  B) -1           | Result has sign of divisor

Q6:  B) B            | None is falsy, but "is None" is True
Q7:  A) 1 0 2 0 2 1  | Break only exits inner loop
Q8:  A) [0,1,3,"end"]| continue skips 2, else runs
Q9:  A) 5            | Ternary: 5>3 so returns 5
Q10: B) False        | All are falsy, or returns last falsy
Q11: C) 5            | continue doesn't exit, loop completes
Q12: B) 10           | b=a creates reference, not copy
Q13: D) "hello"      | or returns first truthy value
Q14: B) c            | Loop variable persists after loop

Q15: A) bd           | [1:5:2] = indices 1, 3
Q16: A) tuple        | Tuples can be dict keys
Q17: B) 3            | Sorted: [1,1,3,4,5], index 2 = 3
Q18: B) 12           | .items() gives (k,v) pairs, print v
Q19: B) tuple        | (1,) with comma is tuple
Q20: A) [1, 4, 5]    | del removes slice [1:3]
Q21: A) True         | 'in' checks keys
Q22: B) ""           | Out of range slice returns empty

Q23: B) [1] [1, 2]   | Mutable default argument trap!
Q24: B) 3            | **kwargs is dict, len=3
Q25: A) Hello        | f1 calls f2 which returns "Hello"
Q26: A) A            | ZeroDivisionError is subclass, first match wins
Q27: B) finally      | finally return overrides exception
Q28: B) 1            | Base case: 0! = 1
Q29: B) 10           | inner uses enclosing scope's x
Q30: B) ValueError   | type(e).__name__ gives class name

SCORE: ___/30  |  PASS: 21+  |  TARGET: 24+

════════════════════════════════════════════════════════════════════════════════
PASS ALL 3 SIMULATIONS WITH 80%+ BEFORE BOOKING YOUR EXAM!
════════════════════════════════════════════════════════════════════════════════
"""

# ══════════════════════════════════════════════════════════════════════════════
#                               MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PCEP-30-02 EXAM SIMULATIONS                               ║
║                                                                              ║
║  1. Simulation 1 (Standard)                                                  ║
║  2. Simulation 2 (Standard)                                                  ║
║  3. Simulation 3 (Harder - Final Practice)                                   ║
║  4. Show All Answers Only                                                    ║
║                                                                              ║
║  Instructions: Set 40-min timer, NO code execution, write on paper          ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    sims = {
        "1": (SIM1_QUESTIONS, SIM1_ANSWERS),
        "2": (SIM2_QUESTIONS, SIM2_ANSWERS),
        "3": (SIM3_QUESTIONS, SIM3_ANSWERS),
    }
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "4":
        print(SIM1_ANSWERS)
        print(SIM2_ANSWERS)
        print(SIM3_ANSWERS)
    elif choice in sims:
        questions, answers = sims[choice]
        print(questions)
        input("\n\nPress ENTER when done to see answers...")
        print(answers)
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
