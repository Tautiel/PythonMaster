"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    PYTHON ESSENTIALS 2 - MODULE 3                            ║
║           Object-Oriented Programming (OOP)                                  ║
║                                                                              ║
║                     Allineato al Syllabus PCAP-31-03                         ║
║                         OpenEDG Python Institute                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

PCAP-31-03 Exam Block 3: OOP (34%) - IL PIÙ IMPORTANTE!

STRUTTURA MODULO:
├── Section 3.1: Classes and Objects
├── Section 3.2: Instance vs Class Variables
├── Section 3.3: Methods (instance, class, static)
├── Section 3.4: Inheritance
├── Section 3.5: Polymorphism and Encapsulation
├── Section 3.6: Special Methods (__str__, __repr__, etc.)
├── Labs (15 esercizi)
└── Module 3 Quiz (40 domande)

TEMPO STIMATO: 10-12 ore (modulo più complesso!)

═══════════════════════════════════════════════════════════════════════════════
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 3.1: CLASSES AND OBJECTS
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    3.1 TEORIA: CLASSI E OGGETTI                              │
└──────────────────────────────────────────────────────────────────────────────┘

CLASSE = Blueprint/Stampo
OGGETTO = Istanza della classe
"""

class Dog:
    pass

# Creare un'istanza (oggetto)
my_dog = Dog()
print(type(my_dog))  # <class '__main__.Dog'>


"""
__init__ - Costruttore:
───────────────────────
Viene chiamato automaticamente quando si crea un'istanza
"""

class Dog:
    def __init__(self, name, age):
        self.name = name    # Attributo di istanza
        self.age = age      # Attributo di istanza

# Creare istanze
dog1 = Dog("Buddy", 3)
dog2 = Dog("Max", 5)

print(dog1.name)  # 'Buddy'
print(dog2.name)  # 'Max'


"""
self:
─────
- Primo parametro di OGNI metodo di istanza
- Riferimento all'istanza corrente
- Python lo passa automaticamente!
"""

class Cat:
    def __init__(self, name):
        self.name = name
    
    def speak(self):  # self è obbligatorio!
        return f"{self.name} says meow"

cat = Cat("Whiskers")
print(cat.speak())  # 'Whiskers says meow'

# Equivalente a:
print(Cat.speak(cat))  # Esplicito, ma non usato normalmente


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 3.2: INSTANCE VS CLASS VARIABLES
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    3.2 TEORIA: VARIABILI                                     │
└──────────────────────────────────────────────────────────────────────────────┘

INSTANCE VARIABLES: Uniche per ogni istanza (self.x)
CLASS VARIABLES: Condivise tra TUTTE le istanze
"""

class Counter:
    count = 0  # CLASS variable - condivisa!
    
    def __init__(self, name):
        self.name = name       # INSTANCE variable - unica
        Counter.count += 1     # Incrementa la class variable

c1 = Counter("A")
c2 = Counter("B")
c3 = Counter("C")

print(Counter.count)  # 3 (condivisa)
print(c1.name)        # 'A' (unica)
print(c2.name)        # 'B' (unica)


"""
⚠️ TRAPPOLA: Accesso tramite self vs Class
──────────────────────────────────────────
"""

class Example:
    data = []  # Class variable MUTABILE - pericoloso!
    
    def add(self, item):
        self.data.append(item)  # Modifica la class variable!

e1 = Example()
e2 = Example()

e1.add(1)
e2.add(2)

print(e1.data)  # [1, 2] - Condivisa!
print(e2.data)  # [1, 2] - Stessa lista!


"""
⚠️ TRAPPOLA: Shadowing
──────────────────────
Assegnare tramite self CREA una instance variable che "nasconde" la class variable
"""

class Shadow:
    x = 10  # Class variable
    
    def change(self):
        self.x = 20  # Crea una NUOVA instance variable!

s = Shadow()
print(s.x)        # 10 (legge class variable)
s.change()
print(s.x)        # 20 (legge instance variable)
print(Shadow.x)   # 10 (class variable invariata!)


# ══════════════════════════════════════════════════════════════════════════════
#                     SECTION 3.1-3.2 - QUIZ
# ══════════════════════════════════════════════════════════════════════════════

QUIZ_3_1_3_2 = """
══════════════════════════════════════════════════════════════════════════════
QUIZ 3.1.1
══════════════════════════════════════════════════════════════════════════════
class A:
    pass

print(type(A()))

Stampa:
A) <class 'type'>
B) <class '__main__.A'>
C) <class 'A'>
D) A

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.1.2
══════════════════════════════════════════════════════════════════════════════
class Dog:
    def __init__(self, name):
        self.name = name

d = Dog()
print(d.name)

Cosa succede?
A) Stampa ''
B) Stampa None
C) TypeError
D) AttributeError

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.2.1 - CRITICO!
══════════════════════════════════════════════════════════════════════════════
class Counter:
    count = 0
    def __init__(self):
        Counter.count += 1

a = Counter()
b = Counter()
print(a.count, b.count, Counter.count)

Stampa:
A) 1 2 2
B) 2 2 2
C) 1 1 2
D) 0 0 2

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.2.2 - CRITICO!
══════════════════════════════════════════════════════════════════════════════
class Test:
    x = 10
    def change(self):
        self.x = 20

t = Test()
t.change()
print(t.x, Test.x)

Stampa:
A) 20 20
B) 20 10
C) 10 20
D) 10 10

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.2.3 - TRAPPOLA!
══════════════════════════════════════════════════════════════════════════════
class Shared:
    data = []
    def add(self, item):
        self.data.append(item)

a = Shared()
b = Shared()
a.add(1)
b.add(2)
print(a.data)

Stampa:
A) [1]
B) [2]
C) [1, 2]
D) []

Tua risposta: ___
"""


RISPOSTE_3_1_3_2 = """
══════════════════════════════════════════════════════════════════════════════
RISPOSTE QUIZ 3.1-3.2
══════════════════════════════════════════════════════════════════════════════

3.1.1: B) <class '__main__.A'>
       type() di un'istanza mostra la classe.

3.1.2: C) TypeError
       __init__ richiede 'name' ma non è stato passato!
       TypeError: __init__() missing 1 required positional argument: 'name'

3.2.1: B) 2 2 2
       count è una class variable. a.count e b.count leggono la stessa variabile.

3.2.2: B) 20 10
       self.x = 20 CREA una instance variable che "nasconde" la class variable.
       Test.x resta 10.

3.2.3: C) [1, 2]
       data è una class variable MUTABILE. append() modifica la stessa lista
       per tutte le istanze!
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 3.3: METHODS
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    3.3 TEORIA: TIPI DI METODI                                │
└──────────────────────────────────────────────────────────────────────────────┘

1. INSTANCE METHOD - Accede a self (istanza)
2. CLASS METHOD - Accede a cls (classe), decorato con @classmethod
3. STATIC METHOD - Non accede né a self né a cls, decorato con @staticmethod
"""

class MyClass:
    class_var = 0
    
    def __init__(self, value):
        self.instance_var = value
    
    # INSTANCE METHOD
    def instance_method(self):
        return f"Instance: {self.instance_var}"
    
    # CLASS METHOD
    @classmethod
    def class_method(cls):
        return f"Class var: {cls.class_var}"
    
    # STATIC METHOD
    @staticmethod
    def static_method(x, y):
        return x + y  # Non usa self né cls

obj = MyClass(10)

# Chiamate
print(obj.instance_method())     # Instance: 10
print(MyClass.class_method())    # Class var: 0
print(obj.class_method())        # Class var: 0 (funziona anche su istanza)
print(MyClass.static_method(2, 3))  # 5
print(obj.static_method(2, 3))   # 5 (funziona anche su istanza)


"""
QUANDO USARE COSA:
──────────────────
- Instance method: Quando serve accesso all'istanza (self)
- Class method: Factory methods, accesso a class variables
- Static method: Utility functions che non usano dati della classe
"""

class Date:
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day
    
    @classmethod
    def from_string(cls, date_string):
        """Factory method: crea Date da stringa"""
        year, month, day = map(int, date_string.split('-'))
        return cls(year, month, day)  # Crea nuova istanza

# Uso factory method
d = Date.from_string("2024-06-15")
print(d.year)  # 2024


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 3.4: INHERITANCE
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    3.4 TEORIA: EREDITARIETÀ                                  │
└──────────────────────────────────────────────────────────────────────────────┘
"""

class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        return "Some sound"

class Dog(Animal):  # Dog eredita da Animal
    def speak(self):  # Override del metodo
        return f"{self.name} says Woof!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"

dog = Dog("Buddy")
cat = Cat("Whiskers")

print(dog.speak())  # Buddy says Woof!
print(cat.speak())  # Whiskers says Meow!


"""
super() - Chiamare metodi della classe parent:
──────────────────────────────────────────────
"""

class Animal:
    def __init__(self, name):
        self.name = name

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)  # Chiama __init__ di Animal
        self.breed = breed

d = Dog("Buddy", "Labrador")
print(d.name)   # Buddy
print(d.breed)  # Labrador


"""
isinstance() e issubclass():
────────────────────────────
"""
print(isinstance(dog, Dog))      # True
print(isinstance(dog, Animal))   # True (Dog eredita da Animal)
print(isinstance(dog, Cat))      # False

print(issubclass(Dog, Animal))   # True
print(issubclass(Animal, Dog))   # False


"""
MRO - Method Resolution Order:
──────────────────────────────
Ordine in cui Python cerca i metodi
"""

class A:
    def method(self):
        return "A"

class B(A):
    def method(self):
        return "B"

class C(A):
    def method(self):
        return "C"

class D(B, C):  # Multiple inheritance
    pass

d = D()
print(d.method())  # 'B' - segue MRO
print(D.__mro__)   # (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)


# ══════════════════════════════════════════════════════════════════════════════
#                     SECTION 3.3-3.4 - QUIZ
# ══════════════════════════════════════════════════════════════════════════════

QUIZ_3_3_3_4 = """
══════════════════════════════════════════════════════════════════════════════
QUIZ 3.3.1
══════════════════════════════════════════════════════════════════════════════
class A:
    @staticmethod
    def method():
        return "static"

print(A.method())
print(A().method())

Stampa:
A) static \\n Error
B) static \\n static
C) Error \\n static
D) Error \\n Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.3.2
══════════════════════════════════════════════════════════════════════════════
class A:
    x = 10
    @classmethod
    def get_x(cls):
        return cls.x

print(A.get_x())

Stampa:
A) 10
B) Error (cls not defined)
C) None
D) x

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.4.1
══════════════════════════════════════════════════════════════════════════════
class Parent:
    def greet(self):
        return "Hi"

class Child(Parent):
    pass

c = Child()
print(c.greet())

Stampa:
A) Error
B) None
C) 'Hi'
D) 'Child'

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.4.2
══════════════════════════════════════════════════════════════════════════════
class A:
    def f(self):
        return "A"

class B(A):
    def f(self):
        return "B"

b = B()
print(isinstance(b, A))

Stampa:
A) True
B) False
C) Error
D) None

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.4.3
══════════════════════════════════════════════════════════════════════════════
print(issubclass(bool, int))

Stampa:
A) True
B) False
C) Error
D) None

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.4.4
══════════════════════════════════════════════════════════════════════════════
class A:
    def __init__(self):
        self.x = 1

class B(A):
    def __init__(self):
        self.y = 2

b = B()
print(hasattr(b, 'x'))

Stampa:
A) True
B) False
C) Error
D) None

Tua risposta: ___
"""


RISPOSTE_3_3_3_4 = """
══════════════════════════════════════════════════════════════════════════════
RISPOSTE QUIZ 3.3-3.4
══════════════════════════════════════════════════════════════════════════════

3.3.1: B) static \\n static
       Static methods funzionano sia sulla classe che sull'istanza.

3.3.2: A) 10
       @classmethod riceve cls automaticamente.

3.4.1: C) 'Hi'
       Child eredita greet() da Parent.

3.4.2: A) True
       b è istanza di B, e B eredita da A, quindi b è anche istanza di A.

3.4.3: A) True
       bool è una sottoclasse di int in Python!
       True == 1, False == 0

3.4.4: B) False
       B.__init__ NON chiama super().__init__(), quindi x non viene creato.
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 3.5: ENCAPSULATION
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    3.5 TEORIA: INCAPSULAMENTO                                │
└──────────────────────────────────────────────────────────────────────────────┘

Python NON ha vero private/protected, ma usa convenzioni:
- public:    name
- protected: _name (convenzione, non enforced)
- private:   __name (name mangling)
"""

class MyClass:
    def __init__(self):
        self.public = "public"
        self._protected = "protected"   # Convenzione: uso interno
        self.__private = "private"      # Name mangling!

obj = MyClass()
print(obj.public)        # 'public'
print(obj._protected)    # 'protected' (accessibile, ma sconsigliato)
# print(obj.__private)   # AttributeError!
print(obj._MyClass__private)  # 'private' - name mangling!


"""
NAME MANGLING:
──────────────
__name diventa _ClassName__name
Serve per evitare conflitti in sottoclassi, NON per sicurezza!
"""


"""
@property - Getter/Setter:
──────────────────────────
"""

class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        """Getter"""
        return self._radius
    
    @radius.setter
    def radius(self, value):
        """Setter con validazione"""
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value
    
    @property
    def area(self):
        """Read-only property"""
        return 3.14159 * self._radius ** 2

c = Circle(5)
print(c.radius)      # 5 (usa getter)
c.radius = 10        # Usa setter
print(c.area)        # 314.159
# c.area = 100       # AttributeError (no setter!)


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 3.6: SPECIAL METHODS
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    3.6 TEORIA: METODI SPECIALI (DUNDER)                      │
└──────────────────────────────────────────────────────────────────────────────┘

I metodi __name__ (dunder = double underscore) hanno significati speciali.
"""

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        """Chiamato da print() e str()"""
        return f"Point({self.x}, {self.y})"
    
    def __repr__(self):
        """Rappresentazione "ufficiale" - usato in console interattiva"""
        return f"Point({self.x}, {self.y})"
    
    def __eq__(self, other):
        """Chiamato da =="""
        return self.x == other.x and self.y == other.y
    
    def __add__(self, other):
        """Chiamato da +"""
        return Point(self.x + other.x, self.y + other.y)
    
    def __len__(self):
        """Chiamato da len()"""
        return 2  # Un punto ha 2 coordinate
    
    def __bool__(self):
        """Chiamato da bool() e in contesto booleano"""
        return self.x != 0 or self.y != 0

p1 = Point(1, 2)
p2 = Point(3, 4)

print(p1)           # Point(1, 2) - usa __str__
print(p1 == p2)     # False - usa __eq__
print(p1 + p2)      # Point(4, 6) - usa __add__
print(len(p1))      # 2 - usa __len__
print(bool(Point(0, 0)))  # False - usa __bool__


"""
ALTRI METODI SPECIALI COMUNI:
─────────────────────────────
__sub__(self, other)    → -
__mul__(self, other)    → *
__truediv__(self, other) → /
__lt__(self, other)     → <
__le__(self, other)     → <=
__gt__(self, other)     → >
__ge__(self, other)     → >=
__ne__(self, other)     → !=
__getitem__(self, key)  → obj[key]
__setitem__(self, key, value) → obj[key] = value
__contains__(self, item) → in
__call__(self, ...)     → obj()
"""


# ══════════════════════════════════════════════════════════════════════════════
#                     SECTION 3.5-3.6 - QUIZ
# ══════════════════════════════════════════════════════════════════════════════

QUIZ_3_5_3_6 = """
══════════════════════════════════════════════════════════════════════════════
QUIZ 3.5.1
══════════════════════════════════════════════════════════════════════════════
class A:
    def __init__(self):
        self.__x = 10

a = A()
print(a.__x)

Cosa succede?
A) 10
B) AttributeError
C) None
D) __x

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.5.2
══════════════════════════════════════════════════════════════════════════════
class A:
    def __init__(self):
        self.__x = 10

a = A()
print(a._A__x)

Stampa:
A) 10
B) AttributeError
C) None
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.6.1
══════════════════════════════════════════════════════════════════════════════
class A:
    def __str__(self):
        return "str"
    def __repr__(self):
        return "repr"

print(A())

Stampa:
A) str
B) repr
C) <__main__.A object>
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.6.2
══════════════════════════════════════════════════════════════════════════════
class A:
    def __add__(self, other):
        return "added"

a = A()
print(a + a)

Stampa:
A) Error
B) 'added'
C) 'addedadded'
D) A()

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.6.3
══════════════════════════════════════════════════════════════════════════════
class A:
    def __len__(self):
        return 5

print(len(A()))

Stampa:
A) Error
B) 5
C) None
D) <A>

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.6.4
══════════════════════════════════════════════════════════════════════════════
Quale metodo viene chiamato quando usi obj[key]?

A) __getitem__
B) __getattr__
C) __get__
D) __index__

Tua risposta: ___
"""


RISPOSTE_3_5_3_6 = """
══════════════════════════════════════════════════════════════════════════════
RISPOSTE QUIZ 3.5-3.6
══════════════════════════════════════════════════════════════════════════════

3.5.1: B) AttributeError
       __x è "mangled" in _A__x, quindi __x non esiste.

3.5.2: A) 10
       _A__x è il nome reale dopo il name mangling.

3.6.1: A) str
       print() chiama __str__() se disponibile.

3.6.2: B) 'added'
       a + a chiama a.__add__(a), che restituisce "added".

3.6.3: B) 5
       len() chiama __len__().

3.6.4: A) __getitem__
       obj[key] chiama obj.__getitem__(key).
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    MODULE 3 - TEST FINALE
# ══════════════════════════════════════════════════════════════════════════════

MODULE_3_TEST = """
══════════════════════════════════════════════════════════════════════════════
                     PE2 MODULE 3 - TEST FINALE
                    40 domande - Target: 70% (28/40)
                    MODULO PIÙ IMPORTANTE PER PCAP!
══════════════════════════════════════════════════════════════════════════════

Q1. class A: pass; print(type(A())) → ?
    A) type    B) <class '__main__.A'>    C) A    D) object

Q2. class A: x=5; print(A.x, A().x) → ?
    A) 5 5    B) Error Error    C) 5 Error    D) Error 5

Q3. class A: x=5
    a=A(); a.x=10; print(A.x) → ?
    A) 10    B) 5    C) Error    D) None

Q4. class A:
        data=[]
        def add(self,x): self.data.append(x)
    a,b=A(),A(); a.add(1); b.add(2); print(a.data) → ?
    A) [1]    B) [2]    C) [1,2]    D) Error

Q5. class A:
        def __init__(self,x): self.x=x
    A() → ?
    A) Crea A con x=None    B) TypeError    C) A con x=0    D) AttributeError

Q6. class A:
        @staticmethod
        def f(): return 1
    print(A.f(), A().f()) → ?
    A) 1 1    B) Error Error    C) 1 Error    D) Error 1

Q7. class A:
        @classmethod
        def f(cls): return cls.__name__
    print(A.f()) → ?
    A) 'A'    B) 'f'    C) Error    D) None

Q8. class P: pass
    class C(P): pass
    print(isinstance(C(), P)) → ?
    A) True    B) False    C) Error    D) C

Q9. print(issubclass(bool, int)) → ?
    A) True    B) False    C) Error    D) None

Q10. class A:
         def f(self): return "A"
     class B(A):
         def f(self): return "B"
     print(B().f()) → ?
     A) "A"    B) "B"    C) "AB"    D) Error

Q11. class A:
         def __init__(self): self.x=1
     class B(A):
         def __init__(self): self.y=2
     b=B(); print(hasattr(b,'x')) → ?
     A) True    B) False    C) Error    D) 1

Q12. class A:
         def __init__(self): self.x=1
     class B(A):
         def __init__(self): super().__init__(); self.y=2
     b=B(); print(b.x, b.y) → ?
     A) 1 2    B) Error    C) None 2    D) 1 None

Q13. class A:
         def __init__(self): self.__x=10
     a=A(); print(a._A__x) → ?
     A) Error    B) 10    C) None    D) __x

Q14. class A:
         def __str__(self): return "S"
         def __repr__(self): return "R"
     print(str(A())) → ?
     A) S    B) R    C) SR    D) Error

Q15. class A:
         def __add__(self,o): return 42
     print(A()+A()) → ?
     A) Error    B) 42    C) A()A()    D) 84

Q16. class A:
         def __len__(self): return 3
     print(len(A())) → ?
     A) Error    B) 3    C) None    D) 0

Q17. class A:
         def __bool__(self): return False
     print(bool(A())) → ?
     A) True    B) False    C) Error    D) None

Q18. class A:
         def __eq__(self,o): return True
     print(A()==A()) → ?
     A) True    B) False    C) Error    D) None

Q19. Quale decoratore crea un class method?
     A) @static    B) @classmethod    C) @class    D) @method

Q20. self in un metodo si riferisce a:
     A) La classe    B) L'istanza corrente    C) Il metodo    D) Il modulo

Q21. cls in un class method si riferisce a:
     A) L'istanza    B) La classe    C) Il modulo    D) self

Q22. class A: x=[]; class B(A): pass
     A.x.append(1); print(B.x) → ?
     A) []    B) [1]    C) Error    D) None

Q23. class A:
         @property
         def x(self): return 5
     a=A(); a.x=10 → ?
     A) Imposta x=10    B) AttributeError    C) a.x diventa 10    D) Niente

Q24. MRO sta per:
     A) Method Resolution Order    B) Multiple Return Object
     C) Module Resource Object    D) Main Runtime Operation

Q25. class A: pass; class B: pass; class C(A,B): pass
     C.__mro__[1] è:
     A) object    B) A    C) B    D) C

Q26. __init__ è chiamato:
     A) Prima della creazione    B) Durante la creazione
     C) Dopo che l'oggetto è creato    D) Mai automaticamente

Q27. Quale metodo è chiamato da print(obj)?
     A) __print__    B) __str__    C) __repr__    D) __display__

Q28. class A:
         def __getitem__(self,k): return k*2
     print(A()[5]) → ?
     A) 5    B) 10    C) Error    D) [5]

Q29. _name indica (per convenzione):
     A) Privato    B) Protetto/interno    C) Pubblico    D) Costante

Q30. __name__ (con __ sia prima che dopo) indica:
     A) Privato    B) Metodo speciale    C) Costante    D) Errore

Q31. class A:
         count=0
         def __init__(self): A.count+=1; self.id=A.count
     x,y,z=A(),A(),A(); print(y.id) → ?
     A) 1    B) 2    C) 3    D) 0

Q32. class A:
         def f(self): return self.g()
         def g(self): return 1
     class B(A):
         def g(self): return 2
     print(B().f()) → ?
     A) 1    B) 2    C) Error    D) None

Q33. Cosa restituisce __init__?
     A) self    B) L'istanza    C) None    D) La classe

Q34. class A:
         def __contains__(self,x): return True
     print(99 in A()) → ?
     A) True    B) False    C) Error    D) 99

Q35. class A: pass
     a=A(); a.x=5; print(a.x) → ?
     A) Error    B) 5    C) None    D) x

Q36. class A:
         def __call__(self): return "called"
     print(A()()) → ?
     A) Error    B) "called"    C) A()    D) None

Q37. super() in una sottoclasse restituisce:
     A) self    B) Proxy alla classe parent    C) None    D) La classe

Q38. class A:
         def __init__(s,x): s.x=x
     a=A(5); print(a.x) → ?
     A) Error    B) 5    C) s    D) None

Q39. Quale attributo mostra i parent di una classe?
     A) __parent__    B) __bases__    C) __super__    D) __mro__

Q40. class A: x=1
     class B(A): x=2
     class C(B): pass
     print(C.x) → ?
     A) 1    B) 2    C) None    D) Error


══════════════════════════════════════════════════════════════════════════════
                              FINE TEST
══════════════════════════════════════════════════════════════════════════════
"""


MODULE_3_TEST_ANSWERS = """
══════════════════════════════════════════════════════════════════════════════
                    PE2 MODULE 3 - RISPOSTE
══════════════════════════════════════════════════════════════════════════════

Q1:  B) <class '__main__.A'>
Q2:  A) 5 5 (class var accessibile da entrambi)
Q3:  B) 5 (a.x=10 crea instance var, A.x resta 5)
Q4:  C) [1,2] (class var mutabile condivisa!)
Q5:  B) TypeError (x richiesto)
Q6:  A) 1 1 (static funziona su entrambi)
Q7:  A) 'A'
Q8:  A) True
Q9:  A) True (bool è sottoclasse di int!)
Q10: B) "B" (override)
Q11: B) False (super().__init__ non chiamato)
Q12: A) 1 2 (super().__init__ chiamato)
Q13: B) 10 (name mangling)
Q14: A) S (str() usa __str__)
Q15: B) 42
Q16: B) 3
Q17: B) False
Q18: A) True
Q19: B) @classmethod
Q20: B) L'istanza corrente
Q21: B) La classe
Q22: B) [1] (class var condivisa)
Q23: B) AttributeError (no setter)
Q24: A) Method Resolution Order
Q25: B) A (primo parent)
Q26: C) Dopo che l'oggetto è creato
Q27: B) __str__
Q28: B) 10
Q29: B) Protetto/interno
Q30: B) Metodo speciale
Q31: B) 2
Q32: B) 2 (self.g() chiama B.g per istanza B)
Q33: C) None (sempre)
Q34: A) True
Q35: B) 5 (puoi aggiungere attributi dinamicamente)
Q36: B) "called"
Q37: B) Proxy alla classe parent
Q38: B) 5 (s è solo il nome del parametro, come self)
Q39: B) __bases__
Q40: B) 2 (ereditato da B)

PUNTEGGIO: ___/40
Target: 28/40 (70%)
"""


if __name__ == "__main__":
    print("=" * 78)
    print("PYTHON ESSENTIALS 2 - MODULE 3")
    print("Object-Oriented Programming")
    print("=" * 78)
    print("""
    QUESTO È IL MODULO PIÙ IMPORTANTE PER PCAP (34%)!
    
    CONTENUTO:
    - Classes and Objects
    - Instance vs Class Variables
    - Instance/Class/Static Methods
    - Inheritance and super()
    - Encapsulation and Name Mangling
    - Special Methods (__str__, __add__, etc.)
    
    TRAPPOLE CRITICHE:
    ⚠️  Class variable MUTABILE condivisa tra istanze
    ⚠️  self.x = y CREA instance var, non modifica class var
    ⚠️  __name mangling: __x diventa _Class__x
    ⚠️  super().__init__() va chiamato esplicitamente!
    ⚠️  bool è sottoclasse di int
    """)
