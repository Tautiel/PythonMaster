"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    PCPP2-32-20x EXAM SIMULATION                              ║
║                                                                              ║
║                    Certified Professional in Python 2                        ║
║                    45 Domande | 65 Minuti | 70% Pass                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

PESO SEZIONI:
- Testing: 25%
- Design Patterns: 25%
- Concurrency: 25%
- Advanced Database: 25%

═══════════════════════════════════════════════════════════════════════════════
"""

PCPP2_EXAM_1 = """
══════════════════════════════════════════════════════════════════════════════
                         PCPP2 EXAM SIMULATION #1
                              START YOUR TIMER: 65:00
══════════════════════════════════════════════════════════════════════════════

SECTION 1: TESTING (25%)
───────────────────────────────────────────────────────────────────────────────

Q1. What is the output?

    import unittest
    
    class TestMath(unittest.TestCase):
        def test_add(self):
            self.assertEqual(1 + 1, 2)
        
        def test_multiply(self):
            self.assertEqual(2 * 3, 6)
    
    # How many tests will run?

    A) 1
    B) 2
    C) 3
    D) 0

───────────────────────────────────────────────────────────────────────────────

Q2. Which assertion checks that a function raises an exception?

    A) self.assertError
    B) self.assertRaises
    C) self.assertException
    D) self.assertThrows

───────────────────────────────────────────────────────────────────────────────

Q3. What is the purpose of setUp() in unittest?

    A) To clean up after tests
    B) To run code before EACH test method
    C) To run code before ALL tests
    D) To skip tests

───────────────────────────────────────────────────────────────────────────────

Q4. What is the output?

    from unittest.mock import Mock
    
    mock = Mock()
    mock.method.return_value = 42
    print(mock.method())

    A) None
    B) 42
    C) Mock
    D) Error

───────────────────────────────────────────────────────────────────────────────

Q5. What is the purpose of @patch decorator?

    A) To create test fixtures
    B) To temporarily replace objects
    C) To skip tests
    D) To run tests in parallel

───────────────────────────────────────────────────────────────────────────────

Q6. In pytest, how do you mark a test to be skipped?

    A) @unittest.skip
    B) @pytest.skip
    C) @pytest.mark.skip
    D) @skip

───────────────────────────────────────────────────────────────────────────────

Q7. What is a fixture in pytest?

    A) A broken test
    B) A reusable setup/teardown mechanism
    C) A test case
    D) A mock object

───────────────────────────────────────────────────────────────────────────────

Q8. What is TDD?

    A) Test Driven Design
    B) Test Driven Development
    C) Type Driven Development
    D) Test Document Design

───────────────────────────────────────────────────────────────────────────────

Q9. Which command runs pytest with coverage?

    A) pytest --coverage
    B) pytest --cov
    C) pytest -c
    D) pytest coverage

───────────────────────────────────────────────────────────────────────────────

Q10. What is the output?

    from unittest.mock import MagicMock
    
    mock = MagicMock()
    mock.__str__ = MagicMock(return_value="Hello")
    print(str(mock))

    A) <MagicMock>
    B) Hello
    C) None
    D) Error

───────────────────────────────────────────────────────────────────────────────

SECTION 2: DESIGN PATTERNS (25%)
───────────────────────────────────────────────────────────────────────────────

Q11. Which pattern ensures only one instance of a class exists?

    A) Factory
    B) Singleton
    C) Observer
    D) Strategy

───────────────────────────────────────────────────────────────────────────────

Q12. Which pattern provides an interface for creating families of objects?

    A) Singleton
    B) Builder
    C) Abstract Factory
    D) Prototype

───────────────────────────────────────────────────────────────────────────────

Q13. Which pattern allows an object to alter its behavior when state changes?

    A) Strategy
    B) State
    C) Observer
    D) Command

───────────────────────────────────────────────────────────────────────────────

Q14. Which pattern defines a one-to-many dependency between objects?

    A) Mediator
    B) Observer
    C) Chain of Responsibility
    D) Visitor

───────────────────────────────────────────────────────────────────────────────

Q15. Which pattern converts the interface of a class into another interface?

    A) Adapter
    B) Bridge
    C) Decorator
    D) Facade

───────────────────────────────────────────────────────────────────────────────

Q16. Which pattern is used to add behavior to objects dynamically?

    A) Adapter
    B) Decorator
    C) Proxy
    D) Bridge

───────────────────────────────────────────────────────────────────────────────

Q17. Which pattern provides a simplified interface to a complex system?

    A) Adapter
    B) Bridge
    C) Facade
    D) Proxy

───────────────────────────────────────────────────────────────────────────────

Q18. Which pattern encapsulates a request as an object?

    A) Strategy
    B) Command
    C) State
    D) Observer

───────────────────────────────────────────────────────────────────────────────

Q19. What type of pattern is Singleton?

    A) Creational
    B) Structural
    C) Behavioral
    D) Architectural

───────────────────────────────────────────────────────────────────────────────

Q20. What type of pattern is Observer?

    A) Creational
    B) Structural
    C) Behavioral
    D) Architectural

───────────────────────────────────────────────────────────────────────────────

SECTION 3: CONCURRENCY (25%)
───────────────────────────────────────────────────────────────────────────────

Q21. What is the GIL?

    A) Global Interpreter Lock
    B) General Import Library
    C) Global Instance Logger
    D) General Interface Layer

───────────────────────────────────────────────────────────────────────────────

Q22. What is the output?

    import threading
    
    counter = 0
    
    def increment():
        global counter
        for _ in range(100000):
            counter += 1
    
    t1 = threading.Thread(target=increment)
    t2 = threading.Thread(target=increment)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print(counter == 200000)

    A) Always True
    B) Always False
    C) Sometimes True, sometimes False
    D) Error

───────────────────────────────────────────────────────────────────────────────

Q23. Which synchronization primitive prevents race conditions?

    A) Thread
    B) Lock
    C) Process
    D) Queue

───────────────────────────────────────────────────────────────────────────────

Q24. What is the output?

    import asyncio
    
    async def hello():
        return "Hello"
    
    result = hello()
    print(type(result).__name__)

    A) str
    B) coroutine
    C) Task
    D) Future

───────────────────────────────────────────────────────────────────────────────

Q25. What keyword is used to wait for a coroutine?

    A) wait
    B) yield
    C) await
    D) async

───────────────────────────────────────────────────────────────────────────────

Q26. What is the difference between threading and multiprocessing?

    A) threading uses processes, multiprocessing uses threads
    B) threading shares memory, multiprocessing has separate memory
    C) No difference
    D) threading is faster than multiprocessing

───────────────────────────────────────────────────────────────────────────────

Q27. What is a deadlock?

    A) A thread that runs forever
    B) Two or more threads waiting for each other
    C) A thread that crashes
    D) A thread that completes successfully

───────────────────────────────────────────────────────────────────────────────

Q28. Which module is best for CPU-bound tasks?

    A) threading
    B) asyncio
    C) multiprocessing
    D) concurrent

───────────────────────────────────────────────────────────────────────────────

Q29. What is the output?

    import asyncio
    
    async def main():
        await asyncio.sleep(0)
        return 42
    
    result = asyncio.run(main())
    print(result)

    A) None
    B) 42
    C) coroutine
    D) Error

───────────────────────────────────────────────────────────────────────────────

Q30. What does thread.daemon = True mean?

    A) Thread runs in background
    B) Thread exits when main program exits
    C) Thread has higher priority
    D) Thread runs only once

───────────────────────────────────────────────────────────────────────────────

SECTION 4: ADVANCED DATABASE (25%)
───────────────────────────────────────────────────────────────────────────────

Q31. What is an ORM?

    A) Object Relational Mapping
    B) Object Resource Manager
    C) Ordered Record Model
    D) Open Relation Model

───────────────────────────────────────────────────────────────────────────────

Q32. In SQLAlchemy, what is a Session?

    A) A database connection
    B) A transaction manager
    C) A table definition
    D) A query result

───────────────────────────────────────────────────────────────────────────────

Q33. What is the purpose of engine in SQLAlchemy?

    A) To define tables
    B) To manage database connections
    C) To execute queries
    D) Both B and C

───────────────────────────────────────────────────────────────────────────────

Q34. What SQL statement does session.commit() execute?

    A) SELECT
    B) COMMIT
    C) ROLLBACK
    D) INSERT

───────────────────────────────────────────────────────────────────────────────

Q35. What is a foreign key constraint?

    A) A unique identifier
    B) A reference to another table's primary key
    C) An index
    D) A default value

───────────────────────────────────────────────────────────────────────────────

Q36. What is the N+1 query problem?

    A) Having too many tables
    B) Making one query for parent and N queries for children
    C) Having N+1 columns
    D) Making N+1 connections

───────────────────────────────────────────────────────────────────────────────

Q37. What SQLAlchemy feature solves N+1 problem?

    A) lazy loading
    B) eager loading (joinedload)
    C) lazy='dynamic'
    D) cascade

───────────────────────────────────────────────────────────────────────────────

Q38. What is a database transaction?

    A) A single SQL statement
    B) A group of operations treated as one unit
    C) A database table
    D) A connection pool

───────────────────────────────────────────────────────────────────────────────

Q39. What does ACID stand for?

    A) Atomic, Consistent, Isolated, Durable
    B) Abstract, Concrete, Interface, Design
    C) Async, Concurrent, Independent, Distributed
    D) Access, Create, Insert, Delete

───────────────────────────────────────────────────────────────────────────────

Q40. What is the Repository Pattern?

    A) A version control system
    B) An abstraction layer over data access
    C) A database connection pool
    D) A file storage system

───────────────────────────────────────────────────────────────────────────────

Q41. What is the output?

    from sqlalchemy import create_engine
    engine = create_engine('sqlite:///:memory:')
    print(type(engine).__name__)

    A) Connection
    B) Engine
    C) Session
    D) Database

───────────────────────────────────────────────────────────────────────────────

Q42. What does session.rollback() do?

    A) Commits changes
    B) Reverts uncommitted changes
    C) Closes the session
    D) Creates a new transaction

───────────────────────────────────────────────────────────────────────────────

Q43. In SQLAlchemy, what is relationship() used for?

    A) To define database indexes
    B) To define associations between models
    C) To define primary keys
    D) To define constraints

───────────────────────────────────────────────────────────────────────────────

Q44. What is connection pooling?

    A) Sharing database connections between requests
    B) Creating multiple databases
    C) Storing queries in memory
    D) Caching query results

───────────────────────────────────────────────────────────────────────────────

Q45. What is the purpose of migrations (Alembic)?

    A) To move data between databases
    B) To version control database schema changes
    C) To backup databases
    D) To optimize queries

───────────────────────────────────────────────────────────────────────────────

                              END OF EXAM
══════════════════════════════════════════════════════════════════════════════
"""

PCPP2_EXAM_1_ANSWERS = """
══════════════════════════════════════════════════════════════════════════════
                    PCPP2 EXAM SIMULATION #1 - ANSWERS
══════════════════════════════════════════════════════════════════════════════

SECTION 1: TESTING
──────────────────
Q1:  B) 2 - due metodi test_*
Q2:  B) self.assertRaises
Q3:  B) To run code before EACH test method
Q4:  B) 42 - return_value definisce output
Q5:  B) To temporarily replace objects
Q6:  C) @pytest.mark.skip
Q7:  B) A reusable setup/teardown mechanism
Q8:  B) Test Driven Development
Q9:  B) pytest --cov
Q10: B) Hello - MagicMock supporta __str__

SECTION 2: DESIGN PATTERNS
───────────────────────────
Q11: B) Singleton
Q12: C) Abstract Factory
Q13: B) State
Q14: B) Observer
Q15: A) Adapter
Q16: B) Decorator
Q17: C) Facade
Q18: B) Command
Q19: A) Creational
Q20: C) Behavioral

SECTION 3: CONCURRENCY
───────────────────────
Q21: A) Global Interpreter Lock
Q22: C) Sometimes True, sometimes False - race condition!
Q23: B) Lock
Q24: B) coroutine - non await = coroutine object
Q25: C) await
Q26: B) threading shares memory, multiprocessing has separate memory
Q27: B) Two or more threads waiting for each other
Q28: C) multiprocessing - bypassa GIL
Q29: B) 42 - asyncio.run() esegue coroutine
Q30: B) Thread exits when main program exits

SECTION 4: ADVANCED DATABASE
─────────────────────────────
Q31: A) Object Relational Mapping
Q32: B) A transaction manager
Q33: D) Both B and C - Engine gestisce connessioni e può eseguire
Q34: B) COMMIT
Q35: B) A reference to another table's primary key
Q36: B) Making one query for parent and N queries for children
Q37: B) eager loading (joinedload)
Q38: B) A group of operations treated as one unit
Q39: A) Atomic, Consistent, Isolated, Durable
Q40: B) An abstraction layer over data access
Q41: B) Engine
Q42: B) Reverts uncommitted changes
Q43: B) To define associations between models
Q44: A) Sharing database connections between requests
Q45: B) To version control database schema changes

══════════════════════════════════════════════════════════════════════════════
                              SCORE CALCULATION
══════════════════════════════════════════════════════════════════════════════

PASS THRESHOLD: 32/45 (70%)

Per sezione:
- Testing: 10 domande (25%)
- Design Patterns: 10 domande (25%)
- Concurrency: 10 domande (25%)
- Database: 15 domande (25%)

══════════════════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print("PCPP2 EXAM SIMULATION")
    print("=" * 50)
    print("print(PCPP2_EXAM_1) - Start exam")
    print("print(PCPP2_EXAM_1_ANSWERS) - Check answers")
