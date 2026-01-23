"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    PCPP1-32-10x EXAM SIMULATION                              ║
║                                                                              ║
║                    Certified Professional in Python 1                        ║
║                    45 Domande | 65 Minuti | 70% Pass                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

PESO SEZIONI:
- Advanced OOP: 25%
- Best Practices & PEP: 20%  
- GUI Programming: 10%
- Network Programming: 20%
- File Processing: 25%

═══════════════════════════════════════════════════════════════════════════════
"""

PCPP1_EXAM_1 = """
══════════════════════════════════════════════════════════════════════════════
                         PCPP1 EXAM SIMULATION #1
                              START YOUR TIMER: 65:00
══════════════════════════════════════════════════════════════════════════════

SECTION 1: ADVANCED OOP (25%)
───────────────────────────────────────────────────────────────────────────────

Q1. What is the output?

    class Meta(type):
        def __new__(mcs, name, bases, attrs):
            attrs['x'] = 100
            return super().__new__(mcs, name, bases, attrs)
    
    class A(metaclass=Meta):
        pass
    
    print(A.x)

    A) AttributeError
    B) 100
    C) None
    D) Meta

───────────────────────────────────────────────────────────────────────────────

Q2. What is the output?

    from abc import ABC, abstractmethod
    
    class Base(ABC):
        @abstractmethod
        def method(self):
            pass
    
    class Child(Base):
        pass
    
    c = Child()

    A) None
    B) TypeError (can't instantiate)
    C) Creates instance successfully
    D) AttributeError

───────────────────────────────────────────────────────────────────────────────

Q3. What is the output?

    class A:
        def __init__(self):
            print("A", end=" ")
    
    class B(A):
        def __init__(self):
            print("B", end=" ")
            super().__init__()
    
    class C(A):
        def __init__(self):
            print("C", end=" ")
            super().__init__()
    
    class D(B, C):
        def __init__(self):
            print("D", end=" ")
            super().__init__()
    
    d = D()

    A) D B C A
    B) D B A C A
    C) D B A
    D) D A B C

───────────────────────────────────────────────────────────────────────────────

Q4. What is the output?

    def decorator(func):
        def wrapper(*args, **kwargs):
            print("Before", end=" ")
            result = func(*args, **kwargs)
            print("After", end=" ")
            return result
        return wrapper
    
    @decorator
    def greet():
        print("Hello", end=" ")
    
    greet()

    A) Hello
    B) Before Hello After
    C) Before After Hello
    D) Hello Before After

───────────────────────────────────────────────────────────────────────────────

Q5. What is the output?

    class MyClass:
        _instance = None
        
        def __new__(cls):
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    a = MyClass()
    b = MyClass()
    print(a is b)

    A) True
    B) False
    C) Error
    D) None

───────────────────────────────────────────────────────────────────────────────

Q6. What does __slots__ do?

    A) Limits which methods can be called
    B) Limits which attributes an instance can have
    C) Limits inheritance depth
    D) Limits method resolution order

───────────────────────────────────────────────────────────────────────────────

Q7. What is the output?

    class A:
        @property
        def value(self):
            return self._value
        
        @value.setter
        def value(self, val):
            self._value = val * 2
    
    a = A()
    a.value = 5
    print(a.value)

    A) 5
    B) 10
    C) AttributeError
    D) None

───────────────────────────────────────────────────────────────────────────────

Q8. What is the MRO of class D?

    class A: pass
    class B(A): pass
    class C(A): pass
    class D(B, C): pass

    A) [D, B, C, A, object]
    B) [D, B, A, C, object]
    C) [D, C, B, A, object]
    D) [D, A, B, C, object]

───────────────────────────────────────────────────────────────────────────────

Q9. What is the output?

    class A:
        def __init__(self, x):
            self.x = x
        
        def __eq__(self, other):
            return self.x == other.x
        
        def __hash__(self):
            return hash(self.x)
    
    a1 = A(1)
    a2 = A(1)
    s = {a1, a2}
    print(len(s))

    A) 1
    B) 2
    C) Error
    D) 0

───────────────────────────────────────────────────────────────────────────────

Q10. Which method is called when accessing a non-existent attribute?

    A) __getattr__
    B) __getattribute__
    C) __setattr__
    D) __delattr__

───────────────────────────────────────────────────────────────────────────────

SECTION 2: BEST PRACTICES & PEP (20%)
───────────────────────────────────────────────────────────────────────────────

Q11. According to PEP 8, what should the maximum line length be?

    A) 72 characters
    B) 79 characters
    C) 99 characters
    D) 120 characters

───────────────────────────────────────────────────────────────────────────────

Q12. Which import style is recommended by PEP 8?

    A) from module import *
    B) import module
    C) from module import func1, func2, func3, func4, func5
    D) Both B and specific imports are acceptable

───────────────────────────────────────────────────────────────────────────────

Q13. What is the Zen of Python principle that says "Simple is better than complex"?

    A) PEP 8
    B) PEP 20
    C) PEP 257
    D) PEP 484

───────────────────────────────────────────────────────────────────────────────

Q14. What is the correct type hint for a function returning either int or None?

    A) def func() -> int | None:
    B) def func() -> Optional[int]:
    C) def func() -> Union[int, None]:
    D) All of the above are valid in Python 3.10+

───────────────────────────────────────────────────────────────────────────────

Q15. According to SOLID principles, what does the 'S' stand for?

    A) Simple Responsibility
    B) Single Responsibility
    C) Strict Responsibility
    D) Separate Responsibility

───────────────────────────────────────────────────────────────────────────────

Q16. What is the purpose of __all__ in a module?

    A) Lists all classes in the module
    B) Controls what is exported with 'from module import *'
    C) Lists all functions in the module
    D) Prevents importing the module

───────────────────────────────────────────────────────────────────────────────

Q17. Which naming convention is correct for constants according to PEP 8?

    A) myConstant
    B) my_constant
    C) MY_CONSTANT
    D) MyConstant

───────────────────────────────────────────────────────────────────────────────

Q18. What tool checks code against PEP 8?

    A) pylint
    B) flake8
    C) pycodestyle
    D) All of the above

───────────────────────────────────────────────────────────────────────────────

SECTION 3: NETWORK PROGRAMMING (20%)
───────────────────────────────────────────────────────────────────────────────

Q19. What is the output?

    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(type(s).__name__)

    A) socket
    B) Socket
    C) connection
    D) stream

───────────────────────────────────────────────────────────────────────────────

Q20. What is the difference between AF_INET and AF_INET6?

    A) AF_INET is for TCP, AF_INET6 is for UDP
    B) AF_INET is for IPv4, AF_INET6 is for IPv6
    C) AF_INET is for client, AF_INET6 is for server
    D) No difference

───────────────────────────────────────────────────────────────────────────────

Q21. What is the difference between SOCK_STREAM and SOCK_DGRAM?

    A) SOCK_STREAM is TCP, SOCK_DGRAM is UDP
    B) SOCK_STREAM is UDP, SOCK_DGRAM is TCP
    C) SOCK_STREAM is IPv4, SOCK_DGRAM is IPv6
    D) No difference

───────────────────────────────────────────────────────────────────────────────

Q22. What does socket.bind() do?

    A) Connects to a remote server
    B) Associates the socket with a specific address and port
    C) Accepts incoming connections
    D) Sends data to the server

───────────────────────────────────────────────────────────────────────────────

Q23. What is the correct order for a TCP server?

    A) socket() → bind() → listen() → accept() → recv()/send()
    B) socket() → listen() → bind() → accept() → recv()/send()
    C) socket() → accept() → bind() → listen() → recv()/send()
    D) socket() → bind() → accept() → listen() → recv()/send()

───────────────────────────────────────────────────────────────────────────────

Q24. What HTTP status code indicates success?

    A) 100
    B) 200
    C) 300
    D) 400

───────────────────────────────────────────────────────────────────────────────

Q25. What is the output?

    import json
    data = {'name': 'John', 'age': 30}
    result = json.dumps(data)
    print(type(result).__name__)

    A) dict
    B) str
    C) bytes
    D) JSON

───────────────────────────────────────────────────────────────────────────────

Q26. Which library is commonly used for HTTP requests in Python?

    A) http
    B) urllib
    C) requests
    D) All of the above

───────────────────────────────────────────────────────────────────────────────

Q27. What does REST stand for?

    A) Remote Execution Standard Transfer
    B) Representational State Transfer
    C) Resource Exchange Simple Transfer
    D) Request Execute Send Transfer

───────────────────────────────────────────────────────────────────────────────

SECTION 4: FILE PROCESSING (25%)
───────────────────────────────────────────────────────────────────────────────

Q28. What is the output?

    import csv
    
    data = [['Name', 'Age'], ['John', 30], ['Jane', 25]]
    
    with open('test.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)
    
    with open('test.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            print(row[0], end=" ")

    A) Name John Jane
    B) ['Name', 'Age'] ['John', 30] ['Jane', 25]
    C) Name Age John 30 Jane 25
    D) Error

───────────────────────────────────────────────────────────────────────────────

Q29. What is the purpose of newline='' when opening CSV files?

    A) To preserve original line endings
    B) To prevent extra blank rows
    C) To improve performance
    D) It's optional and has no effect

───────────────────────────────────────────────────────────────────────────────

Q30. What is the output?

    import xml.etree.ElementTree as ET
    
    xml_str = '<root><item>Hello</item></root>'
    root = ET.fromstring(xml_str)
    print(root.find('item').text)

    A) <item>Hello</item>
    B) Hello
    C) item
    D) root

───────────────────────────────────────────────────────────────────────────────

Q31. What SQLite function connects to a database?

    A) sqlite3.open()
    B) sqlite3.connect()
    C) sqlite3.database()
    D) sqlite3.create()

───────────────────────────────────────────────────────────────────────────────

Q32. What is the output?

    import sqlite3
    
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE test (id INTEGER, name TEXT)')
    cursor.execute('INSERT INTO test VALUES (1, "John")')
    cursor.execute('SELECT * FROM test')
    print(cursor.fetchone())

    A) (1, 'John')
    B) ['1', 'John']
    C) {'id': 1, 'name': 'John'}
    D) 1, John

───────────────────────────────────────────────────────────────────────────────

Q33. What is the purpose of conn.commit()?

    A) To close the connection
    B) To save changes to the database
    C) To execute a query
    D) To create a cursor

───────────────────────────────────────────────────────────────────────────────

Q34. What is the correct way to use parameters in SQLite?

    A) cursor.execute(f"SELECT * FROM users WHERE id={user_id}")
    B) cursor.execute("SELECT * FROM users WHERE id=" + user_id)
    C) cursor.execute("SELECT * FROM users WHERE id=?", (user_id,))
    D) cursor.execute("SELECT * FROM users WHERE id=%s", user_id)

───────────────────────────────────────────────────────────────────────────────

Q35. What is the output?

    import logging
    
    logging.basicConfig(level=logging.WARNING)
    logging.info("Info message")
    logging.warning("Warning message")

    A) Info message and Warning message
    B) Warning message only
    C) Info message only
    D) Nothing

───────────────────────────────────────────────────────────────────────────────

Q36. What logging level is higher: ERROR or WARNING?

    A) WARNING
    B) ERROR
    C) They are equal
    D) Depends on configuration

───────────────────────────────────────────────────────────────────────────────

Q37. What is the output?

    from pathlib import Path
    
    p = Path('/home/user/file.txt')
    print(p.suffix)

    A) .txt
    B) txt
    C) file.txt
    D) file

───────────────────────────────────────────────────────────────────────────────

Q38. What is the output?

    from pathlib import Path
    
    p = Path('/home/user/file.tar.gz')
    print(p.stem)

    A) file
    B) file.tar
    C) tar.gz
    D) .gz

───────────────────────────────────────────────────────────────────────────────

Q39. What method reads an entire file as a string?

    A) file.read()
    B) file.readline()
    C) file.readlines()
    D) file.readall()

───────────────────────────────────────────────────────────────────────────────

Q40. What is the output?

    import configparser
    
    config = configparser.ConfigParser()
    config['DEFAULT'] = {'debug': 'true'}
    config['production'] = {'debug': 'false'}
    
    print(config.getboolean('production', 'debug'))

    A) True
    B) False
    C) 'false'
    D) Error

───────────────────────────────────────────────────────────────────────────────

SECTION 5: GUI PROGRAMMING (10%)
───────────────────────────────────────────────────────────────────────────────

Q41. Which module is Python's standard GUI library?

    A) PyQt
    B) wxPython
    C) tkinter
    D) Kivy

───────────────────────────────────────────────────────────────────────────────

Q42. What is the main event loop function in tkinter?

    A) root.run()
    B) root.mainloop()
    C) root.start()
    D) root.execute()

───────────────────────────────────────────────────────────────────────────────

Q43. What method places widgets using grid layout?

    A) widget.pack()
    B) widget.place()
    C) widget.grid()
    D) widget.layout()

───────────────────────────────────────────────────────────────────────────────

Q44. What is the output?

    import tkinter as tk
    
    root = tk.Tk()
    label = tk.Label(root, text="Hello")
    print(type(label).__name__)

    A) Label
    B) Widget
    C) tkinter.Label
    D) Tk

───────────────────────────────────────────────────────────────────────────────

Q45. What is a callback in GUI programming?

    A) A function that returns a value
    B) A function called when an event occurs
    C) A function that creates widgets
    D) A function that closes the window

───────────────────────────────────────────────────────────────────────────────

                              END OF EXAM
══════════════════════════════════════════════════════════════════════════════
"""

PCPP1_EXAM_1_ANSWERS = """
══════════════════════════════════════════════════════════════════════════════
                    PCPP1 EXAM SIMULATION #1 - ANSWERS
══════════════════════════════════════════════════════════════════════════════

SECTION 1: ADVANCED OOP
────────────────────────
Q1:  B) 100
     → Metaclass __new__ aggiunge attributo 'x' alla classe

Q2:  B) TypeError (can't instantiate)
     → Non si può istanziare classe con metodi astratti non implementati

Q3:  A) D B C A
     → MRO: D → B → C → A, super() segue MRO

Q4:  B) Before Hello After
     → Decorator wrappa la funzione

Q5:  A) True
     → Singleton pattern: stessa istanza restituita

Q6:  B) Limits which attributes an instance can have
     → __slots__ ottimizza memoria e limita attributi

Q7:  B) 10
     → Setter moltiplica per 2: 5 * 2 = 10

Q8:  A) [D, B, C, A, object]
     → C3 linearization

Q9:  A) 1
     → __eq__ e __hash__ consistenti: a1 == a2 e hash uguali

Q10: A) __getattr__
     → __getattr__ chiamato solo se attributo non trovato

SECTION 2: BEST PRACTICES & PEP
────────────────────────────────
Q11: B) 79 characters
     → PEP 8 raccomanda 79 caratteri

Q12: D) Both B and specific imports are acceptable
     → import module o from module import specific

Q13: B) PEP 20
     → The Zen of Python

Q14: D) All of the above are valid in Python 3.10+
     → int | None (3.10+), Optional[int], Union[int, None]

Q15: B) Single Responsibility
     → Una classe, una responsabilità

Q16: B) Controls what is exported with 'from module import *'
     → __all__ definisce API pubblica

Q17: C) MY_CONSTANT
     → SCREAMING_SNAKE_CASE per costanti

Q18: D) All of the above
     → pylint, flake8, pycodestyle tutti controllano PEP 8

SECTION 3: NETWORK PROGRAMMING
───────────────────────────────
Q19: A) socket
     → Il tipo è 'socket'

Q20: B) AF_INET is for IPv4, AF_INET6 is for IPv6
     → Address Family

Q21: A) SOCK_STREAM is TCP, SOCK_DGRAM is UDP
     → Stream = TCP (connessione), Datagram = UDP (pacchetti)

Q22: B) Associates the socket with a specific address and port
     → bind() associa indirizzo

Q23: A) socket() → bind() → listen() → accept() → recv()/send()
     → Ordine corretto TCP server

Q24: B) 200
     → 200 OK = successo

Q25: B) str
     → json.dumps() restituisce stringa JSON

Q26: D) All of the above
     → http, urllib (standard), requests (third-party)

Q27: B) Representational State Transfer
     → Architettura REST

SECTION 4: FILE PROCESSING
───────────────────────────
Q28: A) Name John Jane
     → Stampa primo elemento di ogni riga

Q29: B) To prevent extra blank rows
     → newline='' evita righe vuote su Windows

Q30: B) Hello
     → .text restituisce contenuto testuale

Q31: B) sqlite3.connect()
     → Connette/crea database

Q32: A) (1, 'John')
     → fetchone() restituisce tupla

Q33: B) To save changes to the database
     → commit() salva transazioni

Q34: C) cursor.execute("SELECT * FROM users WHERE id=?", (user_id,))
     → Parametrizzazione previene SQL injection

Q35: B) Warning message only
     → Level WARNING: solo WARNING e superiori

Q36: B) ERROR
     → DEBUG < INFO < WARNING < ERROR < CRITICAL

Q37: A) .txt
     → suffix include il punto

Q38: B) file.tar
     → stem esclude solo ULTIMA estensione

Q39: A) file.read()
     → read() legge tutto come stringa

Q40: B) False
     → getboolean() converte 'false' in False

SECTION 5: GUI PROGRAMMING
───────────────────────────
Q41: C) tkinter
     → Standard library GUI

Q42: B) root.mainloop()
     → Event loop principale

Q43: C) widget.grid()
     → Grid geometry manager

Q44: A) Label
     → type().__name__ = 'Label'

Q45: B) A function called when an event occurs
     → Callback = event handler

══════════════════════════════════════════════════════════════════════════════
                              SCORE CALCULATION
══════════════════════════════════════════════════════════════════════════════

PASS THRESHOLD: 32/45 (70%)

Per sezione:
- Advanced OOP: 10 domande (25%)
- Best Practices: 8 domande (20%)
- Network: 9 domande (20%)
- File Processing: 13 domande (25%)
- GUI: 5 domande (10%)

══════════════════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print("PCPP1 EXAM SIMULATION")
    print("=" * 50)
    print("print(PCPP1_EXAM_1) - Start exam")
    print("print(PCPP1_EXAM_1_ANSWERS) - Check answers")
