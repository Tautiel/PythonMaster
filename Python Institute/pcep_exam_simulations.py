"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    PCEP-30-02 EXAM SIMULATION                                ║
║                    200 Domande Formato Pearson VUE                           ║
║                                                                              ║
║                         5 Simulazioni Complete                               ║
║                       40 domande × 45 minuti each                            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

ISTRUZIONI:
- Rispondi SENZA eseguire codice
- Tempo: 45 minuti per simulazione
- Passing score: 70% (28/40)
- Segna le risposte su carta, poi verifica

═══════════════════════════════════════════════════════════════════════════════
"""

# ══════════════════════════════════════════════════════════════════════════════
#                    SIMULATION 1 (40 Questions)
# ══════════════════════════════════════════════════════════════════════════════

SIMULATION_1 = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                         PCEP SIMULATION EXAM 1                               ║
║                      40 Questions - 45 Minutes                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

SECTION 1: Computer Programming & Python Fundamentals (7 questions)
═══════════════════════════════════════════════════════════════════════════════

Q1. What is the output of the following code?
    
    print(type(1/1))
    
    A) <class 'int'>
    B) <class 'float'>
    C) <class 'str'>
    D) <class 'bool'>

Q2. Which of the following is NOT a valid Python keyword?
    
    A) elif
    B) lambda
    C) switch
    D) finally

Q3. What does the interpreter do with Python source code?
    
    A) Compiles it directly to machine code
    B) Compiles it to bytecode, then interprets it
    C) Only interprets it line by line
    D) Converts it to C code first

Q4. Which of the following is a valid variable name in Python?
    
    A) 2nd_place
    B) second-place
    C) second_place
    D) class

Q5. What will be the output?
    
    x = 10
    print(x, end='')
    print(x)
    
    A) 10 10
    B) 1010
    C) 10
       10
    D) Error

Q6. What is the result of: print("Hello" + "World")
    
    A) Hello World
    B) HelloWorld
    C) Hello+World
    D) Error

Q7. Which implementation of Python is written in Java?
    
    A) CPython
    B) PyPy
    C) Jython
    D) IronPython


SECTION 2: Data Types, Variables, Operators, I/O (12 questions)
═══════════════════════════════════════════════════════════════════════════════

Q8. What is the output?
    
    print(17 // 4)
    
    A) 4.25
    B) 4
    C) 5
    D) 4.0

Q9. What is the output?
    
    print(17 % 4)
    
    A) 4.25
    B) 4
    C) 1
    D) 0

Q10. What is the result of: print(-7 // 2)
    
    A) -3
    B) -4
    C) -3.5
    D) 3

Q11. What is the output?
    
    print(2 ** 3 ** 2)
    
    A) 64
    B) 512
    C) 36
    D) 8

Q12. What is the output?
    
    x = 5
    y = 2
    print(x / y)
    
    A) 2
    B) 2.5
    C) 2.0
    D) Error

Q13. What is the output?
    
    print(bool(""))
    
    A) True
    B) False
    C) ""
    D) Error

Q14. What is the output?
    
    print(bool(0.0))
    
    A) True
    B) False
    C) 0.0
    D) Error

Q15. What is the output?
    
    print(int(3.9))
    
    A) 3
    B) 4
    C) 3.9
    D) Error

Q16. What is the output?
    
    print(int("3.9"))
    
    A) 3
    B) 4
    C) 3.9
    D) Error

Q17. What is the output?
    
    a = "Python"
    print(a * 2)
    
    A) Python2
    B) PythonPython
    C) Python Python
    D) Error

Q18. What is the result of: 5 + 3 * 2
    
    A) 16
    B) 11
    C) 13
    D) 10

Q19. What is the output?
    
    x = 10
    x += 5
    x *= 2
    print(x)
    
    A) 30
    B) 25
    C) 20
    D) 35


SECTION 3: Boolean, Conditionals, Loops, Lists (10 questions)
═══════════════════════════════════════════════════════════════════════════════

Q20. What is the output?
    
    print(True and False or True)
    
    A) True
    B) False
    C) None
    D) Error

Q21. What is the output?
    
    print(not True or True and False)
    
    A) True
    B) False
    C) None
    D) Error

Q22. What is the output?
    
    x = 5
    print(x > 3 and x < 10)
    
    A) True
    B) False
    C) 5
    D) Error

Q23. What is the output?
    
    for i in range(3):
        print(i, end=' ')
    
    A) 1 2 3
    B) 0 1 2
    C) 0 1 2 3
    D) 1 2

Q24. What is the output?
    
    for i in range(1, 5, 2):
        print(i, end=' ')
    
    A) 1 2 3 4
    B) 1 3
    C) 1 3 5
    D) 2 4

Q25. What is the output?
    
    x = 0
    while x < 3:
        x += 1
        print(x, end=' ')
    
    A) 0 1 2
    B) 1 2 3
    C) 0 1 2 3
    D) 1 2 3 4

Q26. What is the output?
    
    for i in range(5):
        if i == 3:
            break
        print(i, end=' ')
    
    A) 0 1 2
    B) 0 1 2 3
    C) 0 1 2 4
    D) 1 2

Q27. What is the output?
    
    for i in range(5):
        if i == 3:
            continue
        print(i, end=' ')
    
    A) 0 1 2
    B) 0 1 2 3 4
    C) 0 1 2 4
    D) 1 2 4

Q28. What is the output?
    
    my_list = [1, 2, 3, 4, 5]
    print(my_list[1:4])
    
    A) [1, 2, 3, 4]
    B) [2, 3, 4]
    C) [2, 3, 4, 5]
    D) [1, 2, 3]

Q29. What is the output?
    
    my_list = [1, 2, 3, 4, 5]
    print(my_list[-2])
    
    A) 4
    B) 5
    C) 3
    D) Error


SECTION 4: Functions, Tuples, Dicts, Exceptions (11 questions)
═══════════════════════════════════════════════════════════════════════════════

Q30. What is the output?
    
    def func(a, b=5):
        return a + b
    
    print(func(3))
    
    A) 3
    B) 5
    C) 8
    D) Error

Q31. What is the output?
    
    def func(a, b=5):
        return a + b
    
    print(func(b=10, a=2))
    
    A) 12
    B) 7
    C) Error
    D) 15

Q32. What is the output?
    
    def func():
        pass
    
    print(func())
    
    A) pass
    B) None
    C) Error
    D) 0

Q33. What is the output?
    
    x = 10
    
    def func():
        x = 20
        return x
    
    func()
    print(x)
    
    A) 10
    B) 20
    C) Error
    D) None

Q34. What is the output?
    
    x = 10
    
    def func():
        global x
        x = 20
    
    func()
    print(x)
    
    A) 10
    B) 20
    C) Error
    D) None

Q35. What is the output?
    
    t = (1, 2, 3)
    print(t[0])
    
    A) (1)
    B) 1
    C) [1]
    D) Error

Q36. What is the output?
    
    t = (1)
    print(type(t))
    
    A) <class 'tuple'>
    B) <class 'int'>
    C) <class 'list'>
    D) Error

Q37. What is the output?
    
    d = {'a': 1, 'b': 2}
    print(d.get('c', 0))
    
    A) None
    B) 0
    C) Error
    D) 'c'

Q38. What is the output?
    
    d = {'a': 1, 'b': 2}
    print('a' in d)
    
    A) True
    B) False
    C) 1
    D) Error

Q39. What is the output?
    
    try:
        x = 1 / 0
    except ZeroDivisionError:
        print("A", end=' ')
    except:
        print("B", end=' ')
    finally:
        print("C")
    
    A) A C
    B) B C
    C) A B C
    D) C

Q40. What is the output?
    
    try:
        x = int("hello")
    except ValueError:
        print("A", end=' ')
    except TypeError:
        print("B", end=' ')
    else:
        print("C", end=' ')
    finally:
        print("D")
    
    A) A D
    B) B D
    C) C D
    D) A B D

══════════════════════════════════════════════════════════════════════════════
                              END OF SIMULATION 1
══════════════════════════════════════════════════════════════════════════════
"""

SIMULATION_1_ANSWERS = """
══════════════════════════════════════════════════════════════════════════════
                         SIMULATION 1 - ANSWERS
══════════════════════════════════════════════════════════════════════════════

Q1:  B) <class 'float'>        # / sempre float in Python 3
Q2:  C) switch                 # Python non ha switch (usa match in 3.10+)
Q3:  B) Compiles to bytecode   # .pyc files
Q4:  C) second_place           # valido, gli altri no
Q5:  B) 1010                   # end='' rimuove newline
Q6:  B) HelloWorld             # + concatena senza spazi
Q7:  C) Jython                 # Jython = Java, IronPython = .NET

Q8:  B) 4                      # floor division
Q9:  C) 1                      # modulo: 17 = 4*4 + 1
Q10: B) -4                     # floor verso -∞, non verso zero!
Q11: B) 512                    # ** destro-associativo: 2^(3^2) = 2^9 = 512
Q12: B) 2.5                    # / sempre float
Q13: B) False                  # stringa vuota è falsy
Q14: B) False                  # 0.0 è falsy
Q15: A) 3                      # int() tronca verso zero
Q16: D) Error                  # int() non converte float string!
Q17: B) PythonPython           # * ripete stringhe
Q18: B) 11                     # * prima di +: 5 + 6 = 11
Q19: A) 30                     # 10+5=15, 15*2=30

Q20: A) True                   # (T and F) or T = F or T = T
Q21: B) False                  # not T = F, T and F = F, F or F = F
Q22: A) True                   # 5>3=T, 5<10=T, T and T = T
Q23: B) 0 1 2                  # range(3) = 0,1,2
Q24: B) 1 3                    # range(1,5,2) = 1,3
Q25: B) 1 2 3                  # x incrementa PRIMA del print
Q26: A) 0 1 2                  # break a i=3, non stampa 3
Q27: C) 0 1 2 4                # continue salta solo i=3
Q28: B) [2, 3, 4]              # slice [1:4] = indici 1,2,3
Q29: A) 4                      # -2 = penultimo elemento

Q30: C) 8                      # 3 + 5(default) = 8
Q31: A) 12                     # keyword args: 2 + 10 = 12
Q32: B) None                   # funzione senza return restituisce None
Q33: A) 10                     # x locale in func, non modifica globale
Q34: B) 20                     # global x modifica la variabile globale
Q35: B) 1                      # accesso elemento tupla
Q36: B) <class 'int'>          # (1) è int, (1,) sarebbe tuple!
Q37: B) 0                      # get() con default
Q38: A) True                   # 'in' controlla le chiavi
Q39: A) A C                    # ZeroDivisionError catturato, finally sempre
Q40: A) A D                    # ValueError catturato, else NON eseguito, finally sempre

SCORE: ___/40  (70% = 28 per passare)
══════════════════════════════════════════════════════════════════════════════
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SIMULATION 2 (40 Questions)
# ══════════════════════════════════════════════════════════════════════════════

SIMULATION_2 = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                         PCEP SIMULATION EXAM 2                               ║
║                      40 Questions - 45 Minutes                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Q1. What is the output?
    
    print(10 + 20 * 30)
    
    A) 900
    B) 610
    C) 70
    D) Error

Q2. What is the output?
    
    print((10 + 20) * 30)
    
    A) 900
    B) 610
    C) 70
    D) Error

Q3. What is the output?
    
    print(5 == 5.0)
    
    A) True
    B) False
    C) Error
    D) None

Q4. What is the output?
    
    print(5 is 5.0)
    
    A) True
    B) False
    C) Error
    D) None

Q5. What is the output?
    
    x = [1, 2, 3]
    y = x
    y.append(4)
    print(x)
    
    A) [1, 2, 3]
    B) [1, 2, 3, 4]
    C) [4]
    D) Error

Q6. What is the output?
    
    x = [1, 2, 3]
    y = x[:]
    y.append(4)
    print(x)
    
    A) [1, 2, 3]
    B) [1, 2, 3, 4]
    C) [4]
    D) Error

Q7. What is the output?
    
    print([1, 2] + [3, 4])
    
    A) [1, 2, 3, 4]
    B) [4, 6]
    C) [[1, 2], [3, 4]]
    D) Error

Q8. What is the output?
    
    print([1, 2] * 3)
    
    A) [3, 6]
    B) [1, 2, 1, 2, 1, 2]
    C) [1, 2, 3]
    D) Error

Q9. What is the output?
    
    my_list = [1, 2, 3]
    my_list.insert(1, 10)
    print(my_list)
    
    A) [1, 10, 2, 3]
    B) [10, 1, 2, 3]
    C) [1, 2, 10, 3]
    D) [1, 2, 3, 10]

Q10. What is the output?
    
    my_list = [3, 1, 4, 1, 5]
    my_list.sort()
    print(my_list)
    
    A) [1, 1, 3, 4, 5]
    B) [5, 4, 3, 1, 1]
    C) [3, 1, 4, 1, 5]
    D) None

Q11. What is the output?
    
    my_list = [3, 1, 4, 1, 5]
    print(sorted(my_list))
    print(my_list)
    
    A) [1, 1, 3, 4, 5] then [1, 1, 3, 4, 5]
    B) [1, 1, 3, 4, 5] then [3, 1, 4, 1, 5]
    C) [3, 1, 4, 1, 5] then [1, 1, 3, 4, 5]
    D) Error

Q12. What is the output?
    
    print("hello".upper())
    
    A) Hello
    B) HELLO
    C) hello
    D) Error

Q13. What is the output?
    
    print("  hello  ".strip())
    
    A) "hello"
    B) hello
    C) "  hello  "
    D) Error

Q14. What is the output?
    
    print("hello world".split())
    
    A) ['hello world']
    B) ['hello', 'world']
    C) ['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']
    D) Error

Q15. What is the output?
    
    print("-".join(['a', 'b', 'c']))
    
    A) a-b-c
    B) -a-b-c-
    C) abc
    D) ['a', '-', 'b', '-', 'c']

Q16. What is the output?
    
    x = 10
    if x > 5:
        print("A", end=' ')
    if x > 8:
        print("B", end=' ')
    if x > 12:
        print("C")
    
    A) A
    B) A B
    C) A B C
    D) B

Q17. What is the output?
    
    x = 10
    if x > 5:
        print("A", end=' ')
    elif x > 8:
        print("B", end=' ')
    elif x > 12:
        print("C")
    
    A) A
    B) A B
    C) A B C
    D) B

Q18. What is the output?
    
    for i in range(3):
        pass
    print(i)
    
    A) 2
    B) 3
    C) Error
    D) None

Q19. What is the output?
    
    for i in range(3):
        print(i)
    else:
        print("done")
    
    A) 0 1 2
    B) 0 1 2 done
    C) done
    D) Error

Q20. What is the output?
    
    for i in range(3):
        if i == 1:
            break
    else:
        print("done")
    print("end")
    
    A) done end
    B) end
    C) done
    D) Error

Q21. What is the output?
    
    def func(x, y, z=3):
        return x + y + z
    
    print(func(1, 2))
    
    A) 6
    B) 3
    C) Error
    D) None

Q22. What is the output?
    
    def func(*args):
        return sum(args)
    
    print(func(1, 2, 3, 4))
    
    A) 10
    B) (1, 2, 3, 4)
    C) [1, 2, 3, 4]
    D) Error

Q23. What is the output?
    
    def func(**kwargs):
        return len(kwargs)
    
    print(func(a=1, b=2, c=3))
    
    A) 3
    B) 6
    C) {'a': 1, 'b': 2, 'c': 3}
    D) Error

Q24. What is the output?
    
    def outer():
        x = 10
        def inner():
            return x
        return inner()
    
    print(outer())
    
    A) 10
    B) Error
    C) None
    D) inner

Q25. What is the output?
    
    nums = [1, 2, 3, 4, 5]
    result = [x * 2 for x in nums if x % 2 == 0]
    print(result)
    
    A) [2, 4, 6, 8, 10]
    B) [4, 8]
    C) [2, 4]
    D) [1, 2, 3, 4, 5]

Q26. What is the output?
    
    d = {'a': 1, 'b': 2, 'c': 3}
    print(list(d.keys()))
    
    A) ['a', 'b', 'c']
    B) [1, 2, 3]
    C) [('a', 1), ('b', 2), ('c', 3)]
    D) {'a', 'b', 'c'}

Q27. What is the output?
    
    d = {'a': 1, 'b': 2, 'c': 3}
    print(list(d.values()))
    
    A) ['a', 'b', 'c']
    B) [1, 2, 3]
    C) [('a', 1), ('b', 2), ('c', 3)]
    D) {1, 2, 3}

Q28. What is the output?
    
    d = {'a': 1, 'b': 2}
    d['c'] = 3
    d['a'] = 10
    print(d)
    
    A) {'a': 1, 'b': 2, 'c': 3}
    B) {'a': 10, 'b': 2, 'c': 3}
    C) {'a': 1, 'b': 2, 'c': 3, 'a': 10}
    D) Error

Q29. What is the output?
    
    t = (1, 2, [3, 4])
    t[2].append(5)
    print(t)
    
    A) (1, 2, [3, 4, 5])
    B) Error
    C) (1, 2, [3, 4])
    D) (1, 2, [3, 4], 5)

Q30. What is the output?
    
    t = (1, 2, 3)
    t[0] = 10
    print(t)
    
    A) (10, 2, 3)
    B) Error
    C) (1, 2, 3)
    D) [10, 2, 3]

Q31. What is the output?
    
    print(len({1, 2, 2, 3, 3, 3}))
    
    A) 6
    B) 3
    C) 1
    D) Error

Q32. What is the output?
    
    print({1, 2, 3} & {2, 3, 4})
    
    A) {1, 2, 3, 4}
    B) {2, 3}
    C) {1, 4}
    D) Error

Q33. What is the output?
    
    print({1, 2, 3} | {2, 3, 4})
    
    A) {1, 2, 3, 4}
    B) {2, 3}
    C) {1, 4}
    D) {1, 2, 2, 3, 3, 4}

Q34. What is the output?
    
    try:
        print(1/0)
    except:
        print("error")
    
    A) 0
    B) error
    C) ZeroDivisionError
    D) inf

Q35. What is the output?
    
    x = None
    print(x is None)
    
    A) True
    B) False
    C) None
    D) Error

Q36. What is the output?
    
    print(0 or 5)
    
    A) True
    B) False
    C) 0
    D) 5

Q37. What is the output?
    
    print(5 or 0)
    
    A) True
    B) False
    C) 5
    D) 0

Q38. What is the output?
    
    print(5 and 0)
    
    A) True
    B) False
    C) 5
    D) 0

Q39. What is the output?
    
    print(0 and 5)
    
    A) True
    B) False
    C) 5
    D) 0

Q40. What is the output?
    
    x = [1, 2, 3]
    print(x.pop())
    print(x)
    
    A) 1 then [2, 3]
    B) 3 then [1, 2]
    C) 3 then [1, 2, 3]
    D) Error

══════════════════════════════════════════════════════════════════════════════
                              END OF SIMULATION 2
══════════════════════════════════════════════════════════════════════════════
"""

SIMULATION_2_ANSWERS = """
══════════════════════════════════════════════════════════════════════════════
                         SIMULATION 2 - ANSWERS
══════════════════════════════════════════════════════════════════════════════

Q1:  B) 610                    # 20*30=600, 10+600=610
Q2:  A) 900                    # (30)*30=900
Q3:  A) True                   # == confronta valori
Q4:  B) False                  # is confronta identità (oggetti diversi)
Q5:  B) [1, 2, 3, 4]           # y è alias di x, stessa lista!
Q6:  A) [1, 2, 3]              # y è COPIA, x non cambia
Q7:  A) [1, 2, 3, 4]           # + concatena liste
Q8:  B) [1, 2, 1, 2, 1, 2]     # * ripete lista
Q9:  A) [1, 10, 2, 3]          # insert(1, 10) inserisce 10 all'indice 1
Q10: A) [1, 1, 3, 4, 5]        # sort() modifica in-place
Q11: B) [1,1,3,4,5] then [3,1,4,1,5]  # sorted() restituisce nuova lista
Q12: B) HELLO                  # upper() tutto maiuscolo
Q13: B) hello                  # strip() rimuove spazi (no quotes nell'output)
Q14: B) ['hello', 'world']     # split() di default su whitespace
Q15: A) a-b-c                  # join unisce con separatore

Q16: B) A B                    # if separati, entrambi veri (no C perché 10<12)
Q17: A) A                      # elif: solo primo branch eseguito
Q18: A) 2                      # i esiste dopo il loop, ultimo valore
Q19: B) 0 1 2 done             # else eseguito se NO break
Q20: B) end                    # break salta else, ma end sempre eseguito
Q21: A) 6                      # 1+2+3=6
Q22: A) 10                     # *args tupla, sum() somma
Q23: A) 3                      # **kwargs dict, len() = 3 chiavi
Q24: A) 10                     # closure, inner accede a x di outer
Q25: B) [4, 8]                 # solo pari (2,4), moltiplicati per 2

Q26: A) ['a', 'b', 'c']        # keys() restituisce chiavi
Q27: B) [1, 2, 3]              # values() restituisce valori
Q28: B) {'a': 10, 'b': 2, 'c': 3}  # 'a' sovrascritta
Q29: A) (1, 2, [3, 4, 5])      # lista DENTRO tupla è mutabile!
Q30: B) Error                  # tupla immutabile
Q31: B) 3                      # set rimuove duplicati
Q32: B) {2, 3}                 # & = intersezione
Q33: A) {1, 2, 3, 4}           # | = unione

Q34: B) error                  # except cattura ZeroDivisionError
Q35: A) True                   # is None corretto per confronto con None
Q36: D) 5                      # or restituisce primo truthy (0 falsy, 5 truthy)
Q37: C) 5                      # or restituisce primo truthy
Q38: D) 0                      # and restituisce primo falsy
Q39: D) 0                      # and restituisce primo falsy
Q40: B) 3 then [1, 2]          # pop() rimuove e restituisce ultimo

SCORE: ___/40  (70% = 28 per passare)
══════════════════════════════════════════════════════════════════════════════
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SIMULATION 3 (40 Questions) - EDGE CASES
# ══════════════════════════════════════════════════════════════════════════════

SIMULATION_3 = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                         PCEP SIMULATION EXAM 3                               ║
║                    40 Questions - 45 Minutes                                 ║
║                         ⚠️ EDGE CASES FOCUS ⚠️                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Q1. What is the output?
    
    print(-11 % 3)
    
    A) -2
    B) 1
    C) -1
    D) 2

Q2. What is the output?
    
    print(11 % -3)
    
    A) 2
    B) -1
    C) -2
    D) 1

Q3. What is the output?
    
    print(-11 // 3)
    
    A) -3
    B) -4
    C) -3.67
    D) 3

Q4. What is the output?
    
    print(11 // -3)
    
    A) -3
    B) -4
    C) -3.67
    D) 3

Q5. What is the output?
    
    print(0.1 + 0.2 == 0.3)
    
    A) True
    B) False
    C) Error
    D) None

Q6. What is the output?
    
    x = []
    if x:
        print("A")
    else:
        print("B")
    
    A) A
    B) B
    C) Error
    D) None

Q7. What is the output?
    
    x = {}
    if x:
        print("A")
    else:
        print("B")
    
    A) A
    B) B
    C) Error
    D) None

Q8. What is the output?
    
    print(bool([0]))
    
    A) True
    B) False
    C) 0
    D) Error

Q9. What is the output?
    
    print(bool([]))
    
    A) True
    B) False
    C) []
    D) Error

Q10. What is the output?
    
    print(bool("False"))
    
    A) True
    B) False
    C) "False"
    D) Error

Q11. What is the output?
    
    print(bool(" "))
    
    A) True
    B) False
    C) " "
    D) Error

Q12. What is the output?
    
    x = [1, 2, 3]
    y = [1, 2, 3]
    print(x == y)
    print(x is y)
    
    A) True True
    B) True False
    C) False True
    D) False False

Q13. What is the output?
    
    a = "hello"
    b = "hello"
    print(a is b)
    
    A) True
    B) False
    C) Error
    D) Depends on implementation

Q14. What is the output?
    
    print(range(5)[-1])
    
    A) 5
    B) 4
    C) Error
    D) -1

Q15. What is the output?
    
    print(list(range(0)))
    
    A) [0]
    B) []
    C) Error
    D) None

Q16. What is the output?
    
    print(list(range(5, 2)))
    
    A) [5, 4, 3]
    B) [5, 4, 3, 2]
    C) []
    D) Error

Q17. What is the output?
    
    print(list(range(5, 2, -1)))
    
    A) [5, 4, 3]
    B) [5, 4, 3, 2]
    C) []
    D) Error

Q18. What is the output?
    
    x = [1, 2, 3]
    x[10:] = [4, 5]
    print(x)
    
    A) [1, 2, 3, 4, 5]
    B) Error (index out of range)
    C) [1, 2, 3]
    D) [4, 5]

Q19. What is the output?
    
    x = "hello"
    print(x[100:])
    
    A) Error
    B) ""
    C) "hello"
    D) None

Q20. What is the output?
    
    def f(a=[]):
        a.append(1)
        return a
    
    print(f())
    print(f())
    
    A) [1] [1]
    B) [1] [1, 1]
    C) [] []
    D) Error

Q21. What is the output?
    
    def f(x):
        x = x + [4]
    
    my_list = [1, 2, 3]
    f(my_list)
    print(my_list)
    
    A) [1, 2, 3, 4]
    B) [1, 2, 3]
    C) [4]
    D) Error

Q22. What is the output?
    
    def f(x):
        x += [4]
    
    my_list = [1, 2, 3]
    f(my_list)
    print(my_list)
    
    A) [1, 2, 3, 4]
    B) [1, 2, 3]
    C) [4]
    D) Error

Q23. What is the output?
    
    print("" or "default")
    
    A) ""
    B) "default"
    C) True
    D) False

Q24. What is the output?
    
    print("value" or "default")
    
    A) "value"
    B) "default"
    C) True
    D) False

Q25. What is the output?
    
    print(None or "default")
    
    A) None
    B) "default"
    C) True
    D) False

Q26. What is the output?
    
    d = {}
    d[1] = "a"
    d[1.0] = "b"
    print(d)
    
    A) {1: 'a', 1.0: 'b'}
    B) {1: 'b'}
    C) {1.0: 'b'}
    D) Error

Q27. What is the output?
    
    d = {}
    d[True] = "a"
    d[1] = "b"
    print(d)
    
    A) {True: 'a', 1: 'b'}
    B) {True: 'b'}
    C) {1: 'b'}
    D) Error

Q28. What is the output?
    
    try:
        raise ValueError("oops")
    except Exception as e:
        print(type(e).__name__)
    
    A) Exception
    B) ValueError
    C) oops
    D) Error

Q29. What is the output?
    
    x = 5
    def f():
        print(x)
        x = 10
    f()
    
    A) 5
    B) 10
    C) Error (UnboundLocalError)
    D) None

Q30. What is the output?
    
    x = 5
    def f():
        global x
        print(x)
        x = 10
    f()
    print(x)
    
    A) 5 5
    B) 5 10
    C) 10 10
    D) Error

Q31. What is the output?
    
    print([1, 2, 3][::-1])
    
    A) [1, 2, 3]
    B) [3, 2, 1]
    C) Error
    D) []

Q32. What is the output?
    
    print("hello"[::-1])
    
    A) "hello"
    B) "olleh"
    C) Error
    D) ['o', 'l', 'l', 'e', 'h']

Q33. What is the output?
    
    x = [1, 2, 3]
    y = x
    x = x + [4]
    print(y)
    
    A) [1, 2, 3]
    B) [1, 2, 3, 4]
    C) Error
    D) [4]

Q34. What is the output?
    
    x = [1, 2, 3]
    y = x
    x += [4]
    print(y)
    
    A) [1, 2, 3]
    B) [1, 2, 3, 4]
    C) Error
    D) [4]

Q35. What is the output?
    
    print(1 < 2 < 3)
    
    A) True
    B) False
    C) Error
    D) 1

Q36. What is the output?
    
    print(1 < 2 > 0)
    
    A) True
    B) False
    C) Error
    D) 2

Q37. What is the output?
    
    print(type(lambda: None))
    
    A) <class 'function'>
    B) <class 'lambda'>
    C) <class 'NoneType'>
    D) Error

Q38. What is the output?
    
    f = lambda x, y=10: x + y
    print(f(5))
    
    A) 5
    B) 15
    C) Error
    D) None

Q39. What is the output?
    
    x = {'a': 1, 'b': 2}
    y = {'b': 3, 'c': 4}
    x.update(y)
    print(x)
    
    A) {'a': 1, 'b': 2, 'c': 4}
    B) {'a': 1, 'b': 3, 'c': 4}
    C) {'a': 1, 'b': 2, 'b': 3, 'c': 4}
    D) Error

Q40. What is the output?
    
    print(min([3, 1, 4, 1, 5]))
    print(max([3, 1, 4, 1, 5]))
    
    A) 1 5
    B) 3 5
    C) 1 4
    D) Error

══════════════════════════════════════════════════════════════════════════════
                              END OF SIMULATION 3
══════════════════════════════════════════════════════════════════════════════
"""

SIMULATION_3_ANSWERS = """
══════════════════════════════════════════════════════════════════════════════
                         SIMULATION 3 - ANSWERS
══════════════════════════════════════════════════════════════════════════════

Q1:  B) 1          # -11 = -4*3 + 1, segno del divisore (3)
Q2:  C) -2         # 11 = -4*(-3) + (-2), segno del divisore (-3)
Q3:  B) -4         # floor division verso -∞
Q4:  B) -4         # floor division verso -∞
Q5:  B) False      # floating point imprecision!
Q6:  B) B          # lista vuota è falsy
Q7:  B) B          # dict vuoto è falsy
Q8:  A) True       # lista con elementi è truthy (anche se contiene 0)
Q9:  B) False      # lista vuota è falsy
Q10: A) True       # stringa non vuota è truthy (anche se dice "False"!)
Q11: A) True       # spazio è un carattere, stringa non vuota

Q12: B) True False # == confronta valori, is confronta identità
Q13: A) True       # string interning (ottimizzazione CPython)
Q14: B) 4          # range supporta indici negativi
Q15: B) []         # range(0) è vuoto
Q16: C) []         # step default +1, non raggiunge 2 da 5
Q17: A) [5, 4, 3]  # step -1, non include 2
Q18: A) [1, 2, 3, 4, 5]  # slice assignment estende la lista!
Q19: B) ""         # slice oltre la lunghezza restituisce vuoto, no errore

Q20: B) [1] [1, 1] # ⚠️ TRAPPOLA: default mutabile condiviso!
Q21: B) [1, 2, 3]  # x + [4] crea NUOVA lista, non modifica originale
Q22: A) [1, 2, 3, 4]  # += modifica IN-PLACE per liste!
Q23: B) "default"  # "" è falsy, restituisce secondo operando
Q24: A) "value"    # "value" è truthy, restituisce primo operando
Q25: B) "default"  # None è falsy

Q26: B) {1: 'b'}   # 1 == 1.0, stessa chiave! Ultimo valore vince
Q27: B) {True: 'b'}  # True == 1, stessa chiave! (bool sottoclasse di int)
Q28: B) ValueError # type(e).__name__ dà il nome della classe
Q29: C) Error      # assegnamento rende x locale, ma print(x) prima!
Q30: B) 5 10       # global permette lettura e scrittura

Q31: B) [3, 2, 1]  # [::-1] inverte
Q32: B) "olleh"    # [::-1] su stringa inverte
Q33: A) [1, 2, 3]  # x + [4] crea NUOVO oggetto, y punta ancora al vecchio
Q34: B) [1, 2, 3, 4]  # += modifica IN-PLACE, y punta allo stesso oggetto!
Q35: A) True       # chained comparison: (1<2) and (2<3)
Q36: A) True       # chained: (1<2) and (2>0)

Q37: A) <class 'function'>  # lambda crea oggetto function
Q38: B) 15         # 5 + 10 (default)
Q39: B) {'a': 1, 'b': 3, 'c': 4}  # update sovrascrive chiavi esistenti
Q40: A) 1 5        # min e max standard

SCORE: ___/40  (70% = 28 per passare)

⚠️ TRAPPOLE CRITICHE IN QUESTO EXAM:
- Q5: float comparison
- Q20: mutable default argument
- Q21 vs Q22: = vs += su liste
- Q26-Q27: hash equality (1==1.0==True)
- Q29: UnboundLocalError
- Q33 vs Q34: = vs += con aliasing
══════════════════════════════════════════════════════════════════════════════
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SIMULATION 4 (40 Questions)
# ══════════════════════════════════════════════════════════════════════════════

SIMULATION_4 = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                         PCEP SIMULATION EXAM 4                               ║
║                      40 Questions - 45 Minutes                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Q1. What is the output?
    print(2 * 3 ** 2)
    
    A) 36    B) 18    C) 12    D) 64

Q2. What is the output?
    print(10 // 3 * 3 + 10 % 3)
    
    A) 10    B) 9    C) 11    D) 12

Q3. What is the output?
    x = 5
    print(x := x + 1)
    
    A) 5    B) 6    C) Error    D) None

Q4. What is the output?
    print("Python"[1:4])
    
    A) "Pyt"    B) "yth"    C) "ytho"    D) "ython"

Q5. What is the output?
    print("Python"[::2])
    
    A) "Pto"    B) "yhn"    C) "Pyt"    D) "Python"

Q6. What is the output?
    x = [1, 2, 3, 4, 5]
    del x[1:3]
    print(x)
    
    A) [1, 4, 5]    B) [2, 3]    C) [1, 2, 5]    D) Error

Q7. What is the output?
    print([1, 2, 3].index(2))
    
    A) 1    B) 2    C) True    D) [2]

Q8. What is the output?
    print([1, 2, 3, 2, 1].count(2))
    
    A) 1    B) 2    C) [2, 2]    D) 4

Q9. What is the output?
    x = [1, 2, 3]
    x.reverse()
    print(x)
    
    A) None    B) [3, 2, 1]    C) [1, 2, 3]    D) Error

Q10. What is the output?
     x = [3, 1, 2]
     print(x.sort())
     
     A) [1, 2, 3]    B) None    C) [3, 1, 2]    D) Error

Q11. What is the output?
     x = (1, 2, 3)
     y = x + (4,)
     print(y)
     
     A) (1, 2, 3, 4)    B) Error    C) (1, 2, 3, (4,))    D) [1, 2, 3, 4]

Q12. What is the output?
     print(tuple("abc"))
     
     A) ("abc",)    B) ('a', 'b', 'c')    C) "abc"    D) Error

Q13. What is the output?
     d = dict(a=1, b=2)
     print(d)
     
     A) {'a': 1, 'b': 2}    B) {a: 1, b: 2}    C) Error    D) [('a', 1), ('b', 2)]

Q14. What is the output?
     d = {1: 'a', 2: 'b'}
     print(d.pop(3, 'c'))
     
     A) Error    B) None    C) 'c'    D) 3

Q15. What is the output?
     for i, v in enumerate(['a', 'b', 'c']):
         print(i, v, end=' ')
     
     A) a 0 b 1 c 2    B) 0 a 1 b 2 c    C) 1 a 2 b 3 c    D) Error

Q16. What is the output?
     for k, v in {'x': 1, 'y': 2}.items():
         print(k, end='')
     
     A) xy    B) 12    C) x1y2    D) Error

Q17. What is the output?
     print(all([True, 1, "hello"]))
     
     A) True    B) False    C) "hello"    D) Error

Q18. What is the output?
     print(any([False, 0, ""]))
     
     A) True    B) False    C) ""    D) Error

Q19. What is the output?
     print(any([False, 0, "", None, [], 1]))
     
     A) True    B) False    C) 1    D) Error

Q20. What is the output?
     nums = [1, 2, 3]
     print(list(map(lambda x: x*2, nums)))
     
     A) [2, 4, 6]    B) [1, 2, 3, 1, 2, 3]    C) 12    D) Error

Q21. What is the output?
     nums = [1, 2, 3, 4, 5]
     print(list(filter(lambda x: x%2==0, nums)))
     
     A) [1, 3, 5]    B) [2, 4]    C) [False, True, False, True, False]    D) Error

Q22. What is the output?
     def func(a, b, /, c, d, *, e, f):
         return a + b + c + d + e + f
     
     print(func(1, 2, 3, d=4, e=5, f=6))
     
     A) 21    B) Error    C) 15    D) (1, 2, 3, 4, 5, 6)

Q23. What is the output?
     print(abs(-5))
     print(abs(5))
     
     A) -5 5    B) 5 5    C) 5 -5    D) Error

Q24. What is the output?
     print(round(2.5))
     print(round(3.5))
     
     A) 2 4    B) 3 4    C) 2 3    D) 3 3

Q25. What is the output?
     print(pow(2, 3))
     print(pow(2, 3, 5))
     
     A) 8 8    B) 8 3    C) 6 3    D) 8 Error

Q26. What is the output?
     x = "hello"
     print(x.replace("l", "L", 1))
     
     A) "heLLo"    B) "heLlo"    C) "heL"    D) Error

Q27. What is the output?
     print("hello".find("x"))
     
     A) -1    B) Error    C) None    D) False

Q28. What is the output?
     print("hello".index("x"))
     
     A) -1    B) Error    C) None    D) False

Q29. What is the output?
     print("hello world".title())
     
     A) "Hello world"    B) "Hello World"    C) "HELLO WORLD"    D) Error

Q30. What is the output?
     print("123".isdigit())
     print("12.3".isdigit())
     
     A) True True    B) True False    C) False True    D) False False

Q31. What is the output?
     try:
         x = 1 / 0
     except ArithmeticError:
         print("A", end=' ')
     except ZeroDivisionError:
         print("B", end=' ')
     
     A) A    B) B    C) A B    D) Error

Q32. What is the output?
     try:
         x = 1 / 0
     except ZeroDivisionError:
         print("A", end=' ')
     except ArithmeticError:
         print("B", end=' ')
     
     A) A    B) B    C) A B    D) Error

Q33. What is the output?
     class MyError(Exception):
         pass
     
     try:
         raise MyError("oops")
     except Exception:
         print("caught")
     
     A) caught    B) Error    C) MyError    D) oops

Q34. What is the output?
     def gen():
         yield 1
         yield 2
         yield 3
     
     g = gen()
     print(next(g))
     print(next(g))
     
     A) 1 2    B) 1 1    C) Error    D) None None

Q35. What is the output?
     print([x*2 for x in range(3)])
     
     A) [0, 2, 4]    B) [2, 4, 6]    C) [0, 1, 2]    D) Error

Q36. What is the output?
     print({x: x**2 for x in range(3)})
     
     A) {0: 0, 1: 1, 2: 4}    B) {1: 1, 2: 4, 3: 9}    C) [0, 1, 4]    D) Error

Q37. What is the output?
     print({x for x in [1, 2, 2, 3, 3, 3]})
     
     A) {1, 2, 3}    B) {1, 2, 2, 3, 3, 3}    C) [1, 2, 3]    D) Error

Q38. What is the output?
     x = [i for i in range(5) if i % 2 == 0]
     print(x)
     
     A) [0, 2, 4]    B) [1, 3]    C) [True, False, True, False, True]    D) Error

Q39. What is the output?
     print(eval("2 + 3 * 4"))
     
     A) 20    B) 14    C) "2 + 3 * 4"    D) Error

Q40. What is the output?
     print(ord('A'))
     print(chr(66))
     
     A) 65 B    B) A 66    C) 65 66    D) Error

══════════════════════════════════════════════════════════════════════════════
                              END OF SIMULATION 4
══════════════════════════════════════════════════════════════════════════════
"""

SIMULATION_4_ANSWERS = """
══════════════════════════════════════════════════════════════════════════════
                         SIMULATION 4 - ANSWERS
══════════════════════════════════════════════════════════════════════════════

Q1:  B) 18         # ** prima di *: 3**2=9, 2*9=18
Q2:  A) 10         # 10//3=3, 3*3=9, 10%3=1, 9+1=10
Q3:  B) 6          # walrus operator := assegna E restituisce
Q4:  B) "yth"      # indici 1,2,3
Q5:  A) "Pto"      # ogni 2 caratteri: P(0), t(2), o(4)
Q6:  A) [1, 4, 5]  # del rimuove slice [1:3] = elementi 2,3
Q7:  A) 1          # index() restituisce indice (0-based)
Q8:  B) 2          # count() conta occorrenze
Q9:  B) [3, 2, 1]  # reverse() modifica in-place, restituisce None ma lista cambiata
Q10: B) None       # sort() modifica in-place, restituisce None

Q11: A) (1, 2, 3, 4)  # concatenazione tuple
Q12: B) ('a', 'b', 'c')  # tuple() da iterabile
Q13: A) {'a': 1, 'b': 2}  # dict con keyword args
Q14: C) 'c'        # pop() con default restituisce default se chiave non esiste
Q15: B) 0 a 1 b 2 c  # enumerate restituisce (index, value)
Q16: A) xy         # .items() restituisce (key, value), stampiamo solo key
Q17: A) True       # all() True se TUTTI truthy
Q18: B) False      # any() True se ALMENO UNO truthy (nessuno lo è)
Q19: A) True       # 1 è truthy
Q20: A) [2, 4, 6]  # map applica funzione

Q21: B) [2, 4]     # filter mantiene dove condizione True
Q22: A) 21         # a,b positional-only; c,d positional or keyword; e,f keyword-only
Q23: B) 5 5        # abs() valore assoluto
Q24: A) 2 4        # banker's rounding: .5 arrotonda al pari più vicino!
Q25: B) 8 3        # pow(2,3)=8, pow(2,3,5)=8%5=3
Q26: B) "heLlo"    # replace con count=1, solo prima occorrenza
Q27: A) -1         # find() restituisce -1 se non trovato
Q28: B) Error      # index() solleva ValueError se non trovato!
Q29: B) "Hello World"  # title() capitalizza ogni parola
Q30: B) True False # isdigit() False con il punto

Q31: A) A          # ZeroDivisionError È ArithmeticError, primo except vince
Q32: A) A          # ordine: primo matching except eseguito
Q33: A) caught     # MyError eredita da Exception
Q34: A) 1 2        # generator restituisce valori con next()
Q35: A) [0, 2, 4]  # list comprehension
Q36: A) {0: 0, 1: 1, 2: 4}  # dict comprehension
Q37: A) {1, 2, 3}  # set comprehension (rimuove duplicati)
Q38: A) [0, 2, 4]  # list comprehension con condizione
Q39: B) 14         # eval() esegue stringa come codice Python
Q40: A) 65 B       # ord('A')=65, chr(66)='B'

SCORE: ___/40  (70% = 28 per passare)
══════════════════════════════════════════════════════════════════════════════
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SIMULATION 5 (40 Questions) - FINAL REVIEW
# ══════════════════════════════════════════════════════════════════════════════

SIMULATION_5 = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                         PCEP SIMULATION EXAM 5                               ║
║                      40 Questions - 45 Minutes                               ║
║                         🎯 FINAL REVIEW 🎯                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Q1. What is the output?
    print(0b1010)
    
    A) 1010    B) 10    C) 0b1010    D) Error

Q2. What is the output?
    print(0o17)
    
    A) 17    B) 15    C) 0o17    D) Error

Q3. What is the output?
    print(0xFF)
    
    A) 255    B) FF    C) 0xFF    D) Error

Q4. What is the output?
    print(1e3)
    
    A) 1000    B) 1000.0    C) 1e3    D) Error

Q5. What is the output?
    x = """hello
    world"""
    print(len(x.split()))
    
    A) 2    B) 11    C) 1    D) Error

Q6. What is the output?
    print(f"{10:05d}")
    
    A) 00010    B) 10000    C) 10    D) Error

Q7. What is the output?
    print(f"{3.14159:.2f}")
    
    A) 3.14    B) 3.14159    C) 3.15    D) Error

Q8. What is the output?
    print("{}{}{}".format(1, 2, 3))
    
    A) 123    B) 1 2 3    C) {1}{2}{3}    D) Error

Q9. What is the output?
    print("%d + %d = %d" % (2, 3, 5))
    
    A) 2 + 3 = 5    B) %d + %d = %d    C) Error    D) 235

Q10. What is the output?
     a, b, *c = [1, 2, 3, 4, 5]
     print(c)
     
     A) [3, 4, 5]    B) 3    C) (3, 4, 5)    D) Error

Q11. What is the output?
     *a, b, c = [1, 2, 3, 4, 5]
     print(a)
     
     A) [1, 2, 3]    B) 1    C) (1, 2, 3)    D) Error

Q12. What is the output?
     def f(a, *args, **kwargs):
         return len(args) + len(kwargs)
     
     print(f(1, 2, 3, x=4, y=5))
     
     A) 4    B) 5    C) 6    D) Error

Q13. What is the output?
     print([1, 2, 3] == [1, 2, 3])
     print([1, 2, 3] is [1, 2, 3])
     
     A) True True    B) True False    C) False True    D) False False

Q14. What is the output?
     x = {1: 'a'}
     y = x.copy()
     y[2] = 'b'
     print(x)
     
     A) {1: 'a', 2: 'b'}    B) {1: 'a'}    C) Error    D) {2: 'b'}

Q15. What is the output?
     print("hello".startswith("he"))
     print("hello".endswith("o"))
     
     A) True True    B) True False    C) False True    D) False False

Q16. What is the output?
     print("a,b,c".split(","))
     print(",".join(["a", "b", "c"]))
     
     A) ['a', 'b', 'c'] a,b,c    B) a,b,c ['a', 'b', 'c']    C) Error    D) None

Q17. What is the output?
     x = [[1, 2], [3, 4]]
     y = x[:]
     y[0][0] = 99
     print(x[0][0])
     
     A) 1    B) 99    C) Error    D) [[99, 2], [3, 4]]

Q18. What is the output?
     import copy
     x = [[1, 2], [3, 4]]
     y = copy.deepcopy(x)
     y[0][0] = 99
     print(x[0][0])
     
     A) 1    B) 99    C) Error    D) [[1, 2], [3, 4]]

Q19. What is the output?
     print(isinstance(True, int))
     
     A) True    B) False    C) Error    D) None

Q20. What is the output?
     print(issubclass(bool, int))
     
     A) True    B) False    C) Error    D) None

Q21. What is the output?
     x = 0
     for i in range(1, 4):
         x += i
     else:
         x *= 2
     print(x)
     
     A) 6    B) 12    C) 3    D) Error

Q22. What is the output?
     x = 0
     for i in range(1, 4):
         if i == 2:
             break
         x += i
     else:
         x *= 2
     print(x)
     
     A) 1    B) 2    C) 6    D) Error

Q23. What is the output?
     def f():
         try:
             return 1
         finally:
             return 2
     
     print(f())
     
     A) 1    B) 2    C) Error    D) None

Q24. What is the output?
     x = None
     y = x or []
     y.append(1)
     print(x, y)
     
     A) None [1]    B) [1] [1]    C) None None    D) Error

Q25. What is the output?
     print(sorted([3, 1, 2], reverse=True))
     
     A) [1, 2, 3]    B) [3, 2, 1]    C) [2, 1, 3]    D) Error

Q26. What is the output?
     print(sorted("hello"))
     
     A) "ehllo"    B) ['e', 'h', 'l', 'l', 'o']    C) ['h', 'e', 'l', 'l', 'o']    D) Error

Q27. What is the output?
     d = {'b': 2, 'a': 1, 'c': 3}
     print(sorted(d))
     
     A) ['a', 'b', 'c']    B) [1, 2, 3]    C) [('a', 1), ('b', 2), ('c', 3)]    D) Error

Q28. What is the output?
     print(zip([1, 2], ['a', 'b']))
     
     A) [(1, 'a'), (2, 'b')]
     B) <zip object at ...>
     C) [[1, 'a'], [2, 'b']]
     D) Error

Q29. What is the output?
     print(list(zip([1, 2, 3], ['a', 'b'])))
     
     A) [(1, 'a'), (2, 'b'), (3, None)]
     B) [(1, 'a'), (2, 'b')]
     C) Error
     D) [(1, 'a'), (2, 'b'), (3,)]

Q30. What is the output?
     a, b = [1, 2]
     print(a, b)
     
     A) 1 2    B) [1] [2]    C) Error    D) (1, 2)

Q31. What is the output?
     a, b = b, a = 1, 2
     print(a, b)
     
     A) 1 2    B) 2 1    C) Error    D) 1 1

Q32. What is the output?
     x = [0, 1, 2, 3, 4]
     print(x[1:10])
     
     A) Error    B) [1, 2, 3, 4]    C) [1]    D) []

Q33. What is the output?
     x = [0, 1, 2, 3, 4]
     print(x[-10:2])
     
     A) Error    B) [0, 1]    C) [0, 1, 2]    D) []

Q34. What is the output?
     x = "hello"
     x = x[:2] + "X" + x[3:]
     print(x)
     
     A) heXlo    B) helXlo    C) Error    D) heXo

Q35. What is the output?
     print(3 in [1, 2, 3])
     print(3 in {1: 'a', 2: 'b', 3: 'c'})
     
     A) True True    B) True False    C) False True    D) False False

Q36. What is the output?
     print('a' in {1: 'a', 2: 'b'})
     print('a' in {1: 'a', 2: 'b'}.values())
     
     A) True True    B) False True    C) True False    D) False False

Q37. What is the output?
     x = [1, 2, 3]
     x.clear()
     print(x)
     
     A) None    B) []    C) Error    D) [1, 2, 3]

Q38. What is the output?
     d = {'a': 1, 'b': 2}
     d.clear()
     print(d)
     
     A) None    B) {}    C) Error    D) {'a': 1, 'b': 2}

Q39. What is the output?
     print(type(range(5)))
     
     A) <class 'list'>    B) <class 'range'>    C) <class 'tuple'>    D) Error

Q40. What is the output?
     print(sum(range(1, 11)))
     
     A) 55    B) 45    C) 10    D) Error

══════════════════════════════════════════════════════════════════════════════
                              END OF SIMULATION 5
══════════════════════════════════════════════════════════════════════════════
"""

SIMULATION_5_ANSWERS = """
══════════════════════════════════════════════════════════════════════════════
                         SIMULATION 5 - ANSWERS
══════════════════════════════════════════════════════════════════════════════

Q1:  B) 10         # 0b1010 binario = 1*8+0*4+1*2+0 = 10
Q2:  B) 15         # 0o17 ottale = 1*8+7 = 15
Q3:  A) 255        # 0xFF esadecimale = 15*16+15 = 255
Q4:  B) 1000.0     # notazione scientifica, sempre float
Q5:  A) 2          # split() default whitespace, "hello" "world"
Q6:  A) 00010      # :05d = 5 cifre, zero-padded
Q7:  A) 3.14       # :.2f = 2 decimali
Q8:  A) 123        # format() sostituisce {}
Q9:  A) 2 + 3 = 5  # printf-style formatting
Q10: A) [3, 4, 5]  # * cattura il resto

Q11: A) [1, 2, 3]  # * all'inizio cattura primi elementi
Q12: A) 4          # args=(2,3), kwargs={x:4, y:5}, 2+2=4
Q13: B) True False # == valore, is identità
Q14: B) {1: 'a'}   # copy() shallow, modifica a y non tocca x
Q15: A) True True  # metodi stringa
Q16: A) ['a','b','c'] a,b,c  # split/join
Q17: B) 99         # shallow copy! liste interne condivise
Q18: A) 1          # deepcopy copia tutto ricorsivamente
Q19: A) True       # bool È sottoclasse di int
Q20: A) True       # bool eredita da int

Q21: B) 12         # 1+2+3=6, else eseguito, 6*2=12
Q22: A) 1          # break salta else, x=1
Q23: B) 2          # finally override return!
Q24: A) None [1]   # x resta None, y diventa lista nuova
Q25: B) [3, 2, 1]  # sorted con reverse=True
Q26: B) ['e','h','l','l','o']  # sorted su stringa restituisce lista
Q27: A) ['a','b','c']  # sorted su dict ordina chiavi
Q28: B) <zip object>   # zip restituisce iterator, non lista
Q29: B) [(1,'a'),(2,'b')]  # zip si ferma al più corto
Q30: A) 1 2        # unpacking lista

Q31: B) 2 1        # assegnamento multiplo: a=2, b=1
Q32: B) [1, 2, 3, 4]  # slice oltre fine = fino alla fine
Q33: B) [0, 1]     # indice negativo oltre inizio = dall'inizio
Q34: A) heXlo      # concatenazione stringhe (immutabili)
Q35: A) True True  # in controlla elementi lista, CHIAVI dict
Q36: B) False True # 'a' non è chiave, ma è in values()
Q37: B) []         # clear() svuota lista
Q38: B) {}         # clear() svuota dict
Q39: B) <class 'range'>  # range è il suo tipo
Q40: A) 55         # sum(1..10) = 55 (Gauss: n*(n+1)/2)

SCORE: ___/40  (70% = 28 per passare)

══════════════════════════════════════════════════════════════════════════════
                    🎉 COMPLETATO IL TRAINING PCEP! 🎉
══════════════════════════════════════════════════════════════════════════════
"""


if __name__ == "__main__":
    print("=" * 70)
    print("PCEP-30-02 EXAM SIMULATIONS")
    print("5 Complete Exams - 200 Questions Total")
    print("=" * 70)
    print("""
    Comandi:
    print(SIMULATION_1)          # Exam 1
    print(SIMULATION_1_ANSWERS)  # Answers 1
    print(SIMULATION_2)          # Exam 2
    print(SIMULATION_2_ANSWERS)  # Answers 2
    print(SIMULATION_3)          # Exam 3 (Edge Cases)
    print(SIMULATION_3_ANSWERS)  # Answers 3
    print(SIMULATION_4)          # Exam 4
    print(SIMULATION_4_ANSWERS)  # Answers 4
    print(SIMULATION_5)          # Exam 5 (Final Review)
    print(SIMULATION_5_ANSWERS)  # Answers 5
    """)
