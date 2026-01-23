"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║            PROFESSIONAL PYTHON PROGRAMMER - MODULE 2                         ║
║                          Design Patterns                                     ║
║                                                                              ║
║                     Allineato al Syllabus PCPP2                              ║
║                         OpenEDG Python Institute                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

PCPP2 Exam Section: Design Patterns (~25%)

STRUTTURA MODULO:
├── Section 2.1: Creational Patterns (Singleton, Factory, Builder)
├── Section 2.2: Structural Patterns (Adapter, Decorator, Facade)
├── Section 2.3: Behavioral Patterns (Observer, Strategy, Command)
└── Module 2 Test (20 domande)

TEMPO STIMATO: 5-6 ore

═══════════════════════════════════════════════════════════════════════════════
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Callable, Any

# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 2.1: CREATIONAL PATTERNS
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    2.1 TEORIA: CREATIONAL PATTERNS                           │
└──────────────────────────────────────────────────────────────────────────────┘

Creational Patterns: Gestiscono la creazione di oggetti.
Obiettivo: Rendere il sistema indipendente da come gli oggetti sono creati.
"""

# ═══════════════════════════════════════════════════════════════════════════
# SINGLETON: Una sola istanza
# ═══════════════════════════════════════════════════════════════════════════
"""
SINGLETON:
Garantisce che una classe abbia una sola istanza.
Usato per: Logger, Database connection, Configuration.
"""

# Metodo 1: Metaclass
class SingletonMeta(type):
    """Metaclass per implementare Singleton."""
    _instances: Dict[type, Any] = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class DatabaseConnection(metaclass=SingletonMeta):
    """Singleton: una sola connessione al database."""
    
    def __init__(self, host: str = "localhost"):
        self.host = host
        self.connected = False
        print(f"Creating connection to {host}")
    
    def connect(self):
        self.connected = True
        print(f"Connected to {self.host}")


# Test Singleton
# db1 = DatabaseConnection("server1")
# db2 = DatabaseConnection("server2")  # Restituisce stessa istanza!
# print(db1 is db2)  # True


# Metodo 2: Decorator
def singleton(cls):
    """Decorator per implementare Singleton."""
    instances = {}
    
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance


@singleton
class Logger:
    """Singleton Logger."""
    
    def __init__(self):
        self.log_file = "app.log"
    
    def log(self, message: str):
        print(f"[LOG] {message}")


# ═══════════════════════════════════════════════════════════════════════════
# FACTORY METHOD: Delega creazione alle sottoclassi
# ═══════════════════════════════════════════════════════════════════════════
"""
FACTORY METHOD:
Definisce un'interfaccia per creare oggetti, ma le sottoclassi
decidono quale classe istanziare.
"""

class Document(ABC):
    """Interfaccia prodotto."""
    
    @abstractmethod
    def create(self) -> str:
        pass


class PDFDocument(Document):
    def create(self) -> str:
        return "PDF Document created"


class WordDocument(Document):
    def create(self) -> str:
        return "Word Document created"


class DocumentCreator(ABC):
    """Creator astratto con factory method."""
    
    @abstractmethod
    def create_document(self) -> Document:
        """Factory method - implementato dalle sottoclassi."""
        pass
    
    def process(self) -> str:
        """Logica comune che usa il factory method."""
        doc = self.create_document()
        return doc.create()


class PDFCreator(DocumentCreator):
    def create_document(self) -> Document:
        return PDFDocument()


class WordCreator(DocumentCreator):
    def create_document(self) -> Document:
        return WordDocument()


# ═══════════════════════════════════════════════════════════════════════════
# ABSTRACT FACTORY: Famiglie di oggetti correlati
# ═══════════════════════════════════════════════════════════════════════════
"""
ABSTRACT FACTORY:
Crea famiglie di oggetti correlati senza specificare le classi concrete.
"""

class Button(ABC):
    @abstractmethod
    def render(self) -> str:
        pass


class Checkbox(ABC):
    @abstractmethod
    def render(self) -> str:
        pass


# Famiglia Windows
class WindowsButton(Button):
    def render(self) -> str:
        return "Windows Button"


class WindowsCheckbox(Checkbox):
    def render(self) -> str:
        return "Windows Checkbox"


# Famiglia Mac
class MacButton(Button):
    def render(self) -> str:
        return "Mac Button"


class MacCheckbox(Checkbox):
    def render(self) -> str:
        return "Mac Checkbox"


class GUIFactory(ABC):
    """Abstract Factory."""
    
    @abstractmethod
    def create_button(self) -> Button:
        pass
    
    @abstractmethod
    def create_checkbox(self) -> Checkbox:
        pass


class WindowsFactory(GUIFactory):
    def create_button(self) -> Button:
        return WindowsButton()
    
    def create_checkbox(self) -> Checkbox:
        return WindowsCheckbox()


class MacFactory(GUIFactory):
    def create_button(self) -> Button:
        return MacButton()
    
    def create_checkbox(self) -> Checkbox:
        return MacCheckbox()


# ═══════════════════════════════════════════════════════════════════════════
# BUILDER: Costruzione step-by-step
# ═══════════════════════════════════════════════════════════════════════════
"""
BUILDER:
Separa la costruzione di un oggetto complesso dalla sua rappresentazione.
"""

class Pizza:
    """Prodotto complesso."""
    
    def __init__(self):
        self.size: str = ""
        self.cheese: bool = False
        self.pepperoni: bool = False
        self.mushrooms: bool = False
        self.olives: bool = False
    
    def __str__(self) -> str:
        toppings = []
        if self.cheese: toppings.append("cheese")
        if self.pepperoni: toppings.append("pepperoni")
        if self.mushrooms: toppings.append("mushrooms")
        if self.olives: toppings.append("olives")
        return f"{self.size} pizza with {', '.join(toppings) or 'nothing'}"


class PizzaBuilder:
    """Builder con fluent interface."""
    
    def __init__(self):
        self._pizza = Pizza()
    
    def set_size(self, size: str) -> 'PizzaBuilder':
        self._pizza.size = size
        return self
    
    def add_cheese(self) -> 'PizzaBuilder':
        self._pizza.cheese = True
        return self
    
    def add_pepperoni(self) -> 'PizzaBuilder':
        self._pizza.pepperoni = True
        return self
    
    def add_mushrooms(self) -> 'PizzaBuilder':
        self._pizza.mushrooms = True
        return self
    
    def add_olives(self) -> 'PizzaBuilder':
        self._pizza.olives = True
        return self
    
    def build(self) -> Pizza:
        return self._pizza


# Uso fluent
pizza = (PizzaBuilder()
         .set_size("Large")
         .add_cheese()
         .add_pepperoni()
         .add_mushrooms()
         .build())
print(pizza)  # Large pizza with cheese, pepperoni, mushrooms


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 2.2: STRUCTURAL PATTERNS
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    2.2 TEORIA: STRUCTURAL PATTERNS                           │
└──────────────────────────────────────────────────────────────────────────────┘

Structural Patterns: Gestiscono la composizione di classi e oggetti.
"""

# ═══════════════════════════════════════════════════════════════════════════
# ADAPTER: Converte interfacce incompatibili
# ═══════════════════════════════════════════════════════════════════════════
"""
ADAPTER:
Converte l'interfaccia di una classe in un'altra che il client si aspetta.
"""

# Sistema esistente (legacy)
class LegacyPrinter:
    """Sistema legacy con interfaccia diversa."""
    
    def print_document(self, text: str):
        return f"[LEGACY] Printing: {text}"


# Interfaccia che il client si aspetta
class ModernPrinter(ABC):
    @abstractmethod
    def print(self, content: str) -> str:
        pass


# Adapter
class PrinterAdapter(ModernPrinter):
    """Adapter: adatta LegacyPrinter a ModernPrinter."""
    
    def __init__(self, legacy_printer: LegacyPrinter):
        self._legacy = legacy_printer
    
    def print(self, content: str) -> str:
        # Converte la chiamata moderna alla legacy
        return self._legacy.print_document(content)


# Uso
legacy = LegacyPrinter()
adapter = PrinterAdapter(legacy)
print(adapter.print("Hello"))  # Usa interfaccia moderna


# ═══════════════════════════════════════════════════════════════════════════
# DECORATOR: Aggiunge funzionalità dinamicamente
# ═══════════════════════════════════════════════════════════════════════════
"""
DECORATOR (Pattern, non Python decorator):
Aggiunge responsabilità a un oggetto dinamicamente.
"""

class Coffee(ABC):
    """Component interface."""
    
    @abstractmethod
    def cost(self) -> float:
        pass
    
    @abstractmethod
    def description(self) -> str:
        pass


class SimpleCoffee(Coffee):
    """Concrete component."""
    
    def cost(self) -> float:
        return 2.0
    
    def description(self) -> str:
        return "Simple Coffee"


class CoffeeDecorator(Coffee):
    """Base decorator."""
    
    def __init__(self, coffee: Coffee):
        self._coffee = coffee
    
    def cost(self) -> float:
        return self._coffee.cost()
    
    def description(self) -> str:
        return self._coffee.description()


class MilkDecorator(CoffeeDecorator):
    """Concrete decorator - aggiunge latte."""
    
    def cost(self) -> float:
        return self._coffee.cost() + 0.5
    
    def description(self) -> str:
        return f"{self._coffee.description()}, Milk"


class SugarDecorator(CoffeeDecorator):
    """Concrete decorator - aggiunge zucchero."""
    
    def cost(self) -> float:
        return self._coffee.cost() + 0.2
    
    def description(self) -> str:
        return f"{self._coffee.description()}, Sugar"


# Uso - decoratori impilabili
coffee = SimpleCoffee()
coffee = MilkDecorator(coffee)
coffee = SugarDecorator(coffee)
print(f"{coffee.description()}: ${coffee.cost()}")
# Simple Coffee, Milk, Sugar: $2.7


# ═══════════════════════════════════════════════════════════════════════════
# FACADE: Interfaccia semplificata
# ═══════════════════════════════════════════════════════════════════════════
"""
FACADE:
Fornisce un'interfaccia semplificata a un sottosistema complesso.
"""

# Sottosistema complesso
class CPU:
    def freeze(self): return "CPU frozen"
    def jump(self, addr): return f"CPU jumping to {addr}"
    def execute(self): return "CPU executing"


class Memory:
    def load(self, pos, data): return f"Memory loaded {data} at {pos}"


class HardDrive:
    def read(self, sector, size): return f"Read {size}b from sector {sector}"


class ComputerFacade:
    """Facade - semplifica l'avvio del computer."""
    
    def __init__(self):
        self._cpu = CPU()
        self._memory = Memory()
        self._hd = HardDrive()
    
    def start(self) -> List[str]:
        """Interfaccia semplice per operazione complessa."""
        operations = []
        operations.append(self._cpu.freeze())
        operations.append(self._memory.load(0, self._hd.read(0, 1024)))
        operations.append(self._cpu.jump(0))
        operations.append(self._cpu.execute())
        return operations


# Uso semplice
computer = ComputerFacade()
print(computer.start())  # Una chiamata invece di molte


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 2.3: BEHAVIORAL PATTERNS
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    2.3 TEORIA: BEHAVIORAL PATTERNS                           │
└──────────────────────────────────────────────────────────────────────────────┘

Behavioral Patterns: Gestiscono comunicazione e responsabilità tra oggetti.
"""

# ═══════════════════════════════════════════════════════════════════════════
# OBSERVER: Notifiche di cambiamenti
# ═══════════════════════════════════════════════════════════════════════════
"""
OBSERVER:
Definisce una dipendenza uno-a-molti: quando un oggetto cambia stato,
tutti i dipendenti vengono notificati.

Usato per: Event systems, Data binding, Pub/Sub
"""

class Observer(ABC):
    """Observer interface."""
    
    @abstractmethod
    def update(self, message: str) -> None:
        pass


class Subject:
    """Subject (Observable)."""
    
    def __init__(self):
        self._observers: List[Observer] = []
    
    def attach(self, observer: Observer) -> None:
        self._observers.append(observer)
    
    def detach(self, observer: Observer) -> None:
        self._observers.remove(observer)
    
    def notify(self, message: str) -> None:
        for observer in self._observers:
            observer.update(message)


class PriceMonitor(Subject):
    """Subject concreto - monitora prezzo."""
    
    def __init__(self):
        super().__init__()
        self._price = 0.0
    
    @property
    def price(self) -> float:
        return self._price
    
    @price.setter
    def price(self, value: float) -> None:
        self._price = value
        self.notify(f"Price changed to {value}")


class EmailAlert(Observer):
    """Concrete observer - invia email."""
    
    def update(self, message: str) -> None:
        print(f"[EMAIL] {message}")


class SMSAlert(Observer):
    """Concrete observer - invia SMS."""
    
    def update(self, message: str) -> None:
        print(f"[SMS] {message}")


# Uso
monitor = PriceMonitor()
monitor.attach(EmailAlert())
monitor.attach(SMSAlert())
monitor.price = 100.0  # Notifica tutti gli observer


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY: Algoritmi intercambiabili
# ═══════════════════════════════════════════════════════════════════════════
"""
STRATEGY:
Definisce una famiglia di algoritmi intercambiabili.
"""

class PaymentStrategy(ABC):
    """Strategy interface."""
    
    @abstractmethod
    def pay(self, amount: float) -> str:
        pass


class CreditCardPayment(PaymentStrategy):
    def __init__(self, card_number: str):
        self.card_number = card_number
    
    def pay(self, amount: float) -> str:
        return f"Paid ${amount} with card {self.card_number[-4:]}"


class PayPalPayment(PaymentStrategy):
    def __init__(self, email: str):
        self.email = email
    
    def pay(self, amount: float) -> str:
        return f"Paid ${amount} via PayPal ({self.email})"


class CryptoPayment(PaymentStrategy):
    def __init__(self, wallet: str):
        self.wallet = wallet
    
    def pay(self, amount: float) -> str:
        return f"Paid ${amount} in crypto to {self.wallet[:8]}..."


class ShoppingCart:
    """Context che usa Strategy."""
    
    def __init__(self):
        self.items: List[float] = []
        self._payment_strategy: PaymentStrategy = None
    
    def add_item(self, price: float) -> None:
        self.items.append(price)
    
    def set_payment_method(self, strategy: PaymentStrategy) -> None:
        self._payment_strategy = strategy
    
    def checkout(self) -> str:
        total = sum(self.items)
        if not self._payment_strategy:
            raise ValueError("Payment method not set")
        return self._payment_strategy.pay(total)


# Uso
cart = ShoppingCart()
cart.add_item(29.99)
cart.add_item(59.99)

cart.set_payment_method(CreditCardPayment("1234-5678-9012-3456"))
print(cart.checkout())  # Paid $89.98 with card 3456

cart.set_payment_method(PayPalPayment("user@email.com"))
print(cart.checkout())  # Paid $89.98 via PayPal


# ═══════════════════════════════════════════════════════════════════════════
# COMMAND: Incapsula richieste come oggetti
# ═══════════════════════════════════════════════════════════════════════════
"""
COMMAND:
Incapsula una richiesta come oggetto, permettendo di parametrizzare
client con diverse richieste, accodare o loggare richieste, e supportare undo.
"""

class Command(ABC):
    """Command interface."""
    
    @abstractmethod
    def execute(self) -> None:
        pass
    
    @abstractmethod
    def undo(self) -> None:
        pass


class Light:
    """Receiver."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_on = False
    
    def turn_on(self) -> None:
        self.is_on = True
        print(f"{self.name} light is ON")
    
    def turn_off(self) -> None:
        self.is_on = False
        print(f"{self.name} light is OFF")


class LightOnCommand(Command):
    """Concrete command."""
    
    def __init__(self, light: Light):
        self._light = light
    
    def execute(self) -> None:
        self._light.turn_on()
    
    def undo(self) -> None:
        self._light.turn_off()


class LightOffCommand(Command):
    def __init__(self, light: Light):
        self._light = light
    
    def execute(self) -> None:
        self._light.turn_off()
    
    def undo(self) -> None:
        self._light.turn_on()


class RemoteControl:
    """Invoker con history per undo."""
    
    def __init__(self):
        self._history: List[Command] = []
    
    def execute(self, command: Command) -> None:
        command.execute()
        self._history.append(command)
    
    def undo(self) -> None:
        if self._history:
            command = self._history.pop()
            command.undo()


# Uso
living_room = Light("Living Room")
remote = RemoteControl()

remote.execute(LightOnCommand(living_room))   # ON
remote.execute(LightOffCommand(living_room))  # OFF
remote.undo()  # ON (undo ultimo comando)


# ══════════════════════════════════════════════════════════════════════════════
#                    MODULE 2 TEST
# ══════════════════════════════════════════════════════════════════════════════

MODULE_2_TEST = """
══════════════════════════════════════════════════════════════════════════════
                    PP MODULE 2 - TEST FINALE
                    20 domande - Target: 70%
══════════════════════════════════════════════════════════════════════════════

Q1. Singleton garantisce:
    A) Molte istanze    B) Una sola istanza    C) Zero istanze    D) Due istanze

Q2. Factory Method è un pattern:
    A) Structural    B) Behavioral    C) Creational    D) Architectural

Q3. Builder è utile per:
    A) Una sola istanza    B) Oggetti semplici    C) Oggetti complessi    D) Ereditarietà

Q4. Abstract Factory crea:
    A) Un oggetto    B) Famiglie di oggetti    C) Singleton    D) Builder

Q5. Adapter converte:
    A) Dati    B) Interfacce    C) Tipi    D) Classi

Q6. Decorator aggiunge:
    A) Classi    B) Metodi fissi    C) Funzionalità dinamiche    D) Interfacce

Q7. Facade fornisce:
    A) Complessità    B) Interfaccia semplificata    C) Più interfacce    D) Database

Q8. Observer implementa:
    A) Uno-a-uno    B) Uno-a-molti    C) Molti-a-uno    D) Molti-a-molti

Q9. Strategy permette di:
    A) Fixare algoritmi    B) Scambiare algoritmi    C) Eliminare algoritmi    D) Copiare

Q10. Command incapsula:
     A) Dati    B) Richieste    C) Classi    D) Moduli

Q11. Quale pattern supporta UNDO?
     A) Singleton    B) Factory    C) Command    D) Adapter

Q12. Quale pattern è usato per logging centralizzato?
     A) Observer    B) Singleton    C) Strategy    D) Facade

Q13. Fluent interface è tipica di:
     A) Singleton    B) Builder    C) Factory    D) Observer

Q14. Quale pattern NON è Creational?
     A) Singleton    B) Factory    C) Builder    D) Observer

Q15. Il metodo notify() è tipico di:
     A) Singleton    B) Factory    C) Observer    D) Command

Q16. Il metodo execute() è tipico di:
     A) Singleton    B) Factory    C) Observer    D) Command

Q17. Adapter usa tipicamente:
     A) Ereditarietà    B) Composizione    C) Entrambi    D) Nessuno

Q18. Decorator Pattern vs Python decorator:
     A) Identici    B) Diversi    C) Decorator non esiste    D) Solo Python

Q19. Il "Context" in Strategy è:
     A) L'algoritmo    B) Chi usa la strategy    C) L'interfaccia    D) Il risultato

Q20. Pub/Sub è un'implementazione di:
     A) Singleton    B) Factory    C) Observer    D) Command


══════════════════════════════════════════════════════════════════════════════
"""

MODULE_2_TEST_ANSWERS = """
══════════════════════════════════════════════════════════════════════════════
                         RISPOSTE TEST MODULE 2
══════════════════════════════════════════════════════════════════════════════

Q1:  B) Una sola istanza
Q2:  C) Creational
Q3:  C) Oggetti complessi (costruzione step-by-step)
Q4:  B) Famiglie di oggetti correlati
Q5:  B) Interfacce
Q6:  C) Funzionalità dinamiche
Q7:  B) Interfaccia semplificata
Q8:  B) Uno-a-molti (Subject notifica molti Observer)
Q9:  B) Scambiare algoritmi a runtime
Q10: B) Richieste (come oggetti)
Q11: C) Command (mantiene history)
Q12: B) Singleton (una sola istanza del logger)
Q13: B) Builder (method chaining)
Q14: D) Observer (è Behavioral)
Q15: C) Observer
Q16: D) Command
Q17: B) Composizione (wrappa l'oggetto da adattare)
Q18: B) Diversi (pattern OO vs language feature)
Q19: B) Chi usa la strategy (mantiene riferimento)
Q20: C) Observer

══════════════════════════════════════════════════════════════════════════════
"""


if __name__ == "__main__":
    print("=" * 70)
    print("PP MODULE 2: Design Patterns")
    print("=" * 70)
    print("""
    Comandi:
    print(MODULE_2_TEST)         # Test finale
    print(MODULE_2_TEST_ANSWERS) # Risposte
    """)
