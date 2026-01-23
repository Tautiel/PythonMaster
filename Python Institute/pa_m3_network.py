"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║              PROFESSIONAL PYTHON ASSOCIATE - MODULE 3                        ║
║                       Network Programming                                    ║
║                                                                              ║
║                     Allineato al Syllabus PCPP1                              ║
║                         OpenEDG Python Institute                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

PCPP1 Exam Section: Network Programming (~25%)

STRUTTURA MODULO:
├── Section 3.1: Network Fundamentals (TCP/IP, Sockets)
├── Section 3.2: Socket Programming (Client/Server)
├── Section 3.3: HTTP and REST (requests, json)
├── Section 3.4: Working with APIs
└── Module 3 Test (20 domande)

TEMPO STIMATO: 4-5 ore

═══════════════════════════════════════════════════════════════════════════════
"""

# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 3.1: NETWORK FUNDAMENTALS
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    3.1 TEORIA: FONDAMENTI DI RETE                            │
└──────────────────────────────────────────────────────────────────────────────┘

MODELLO TCP/IP:
───────────────
Layer 4: Application (HTTP, FTP, SMTP, DNS)
Layer 3: Transport (TCP, UDP)
Layer 2: Internet (IP, ICMP)
Layer 1: Network Access (Ethernet, WiFi)


TCP vs UDP:
───────────
TCP (Transmission Control Protocol):
- Connection-oriented (handshake)
- Reliable delivery (acknowledgments)
- Ordered packets
- Error checking
- Usato per: HTTP, HTTPS, FTP, SMTP, SSH

UDP (User Datagram Protocol):
- Connectionless
- No guarantee of delivery
- No ordering
- Faster, less overhead
- Usato per: DNS, streaming, gaming, VoIP


SOCKET:
───────
Un socket è un endpoint di comunicazione.
Identificato da: (IP_address, port_number)

Porte comuni:
- 20, 21: FTP
- 22: SSH
- 23: Telnet
- 25: SMTP
- 53: DNS
- 80: HTTP
- 443: HTTPS
- 3306: MySQL
- 5432: PostgreSQL


IP ADDRESSES:
─────────────
IPv4: 192.168.1.1 (32 bit, ~4 miliardi indirizzi)
IPv6: 2001:0db8:85a3::8a2e:0370:7334 (128 bit)

Speciali:
- 127.0.0.1: localhost (loopback)
- 0.0.0.0: all interfaces
- 192.168.x.x: private network
- 10.x.x.x: private network
"""

import socket

# Ottenere info di rete
hostname = socket.gethostname()
print(f"Hostname: {hostname}")

# IP del host
ip_address = socket.gethostbyname(hostname)
print(f"IP Address: {ip_address}")

# Risolvere un dominio
try:
    ip = socket.gethostbyname("www.google.com")
    print(f"Google IP: {ip}")
except socket.gaierror:
    print("DNS lookup failed")

# Ottenere info complete
try:
    info = socket.getaddrinfo("www.google.com", 80)
    print(f"Address info: {info[0]}")
except socket.gaierror:
    pass


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 3.2: SOCKET PROGRAMMING
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    3.2 TEORIA: SOCKET PROGRAMMING                            │
└──────────────────────────────────────────────────────────────────────────────┘

FLUSSO TCP:
───────────

SERVER:                          CLIENT:
socket()                         socket()
bind()                           
listen()                         
accept() ←───────────────────── connect()
recv()   ←───────────────────── send()
send()   ─────────────────────→ recv()
close()                          close()
"""

# ═══════════════════════════════════════════════════════════════════════════
# TCP SERVER EXAMPLE
# ═══════════════════════════════════════════════════════════════════════════

def tcp_server_example():
    """
    Esempio di TCP Server.
    NON eseguire in produzione senza modifiche di sicurezza!
    """
    import socket
    
    HOST = '127.0.0.1'  # Localhost
    PORT = 65432        # Port > 1024 (non privilegiato)
    
    # Creare socket TCP
    # AF_INET = IPv4, SOCK_STREAM = TCP
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Permettere riuso dell'indirizzo
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Bind al host e porta
        s.bind((HOST, PORT))
        
        # Iniziare ad ascoltare (backlog=5)
        s.listen(5)
        print(f"Server listening on {HOST}:{PORT}")
        
        # Accettare connessioni (blocking)
        conn, addr = s.accept()
        
        with conn:
            print(f"Connected by {addr}")
            
            while True:
                # Ricevere dati (max 1024 bytes)
                data = conn.recv(1024)
                
                if not data:
                    break
                    
                print(f"Received: {data.decode()}")
                
                # Inviare risposta
                conn.sendall(b"Message received!")
    
    print("Server closed")


# ═══════════════════════════════════════════════════════════════════════════
# TCP CLIENT EXAMPLE
# ═══════════════════════════════════════════════════════════════════════════

def tcp_client_example():
    """Esempio di TCP Client."""
    import socket
    
    HOST = '127.0.0.1'
    PORT = 65432
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Connettersi al server
        s.connect((HOST, PORT))
        
        # Inviare dati
        s.sendall(b"Hello, Server!")
        
        # Ricevere risposta
        data = s.recv(1024)
        
    print(f"Received: {data.decode()}")


# ═══════════════════════════════════════════════════════════════════════════
# UDP EXAMPLE (Connectionless)
# ═══════════════════════════════════════════════════════════════════════════

def udp_server_example():
    """UDP Server - no connection needed."""
    import socket
    
    HOST = '127.0.0.1'
    PORT = 65433
    
    # SOCK_DGRAM = UDP
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind((HOST, PORT))
        print(f"UDP Server listening on {HOST}:{PORT}")
        
        while True:
            # recvfrom restituisce anche l'indirizzo
            data, addr = s.recvfrom(1024)
            print(f"Received from {addr}: {data.decode()}")
            
            # Rispondere
            s.sendto(b"ACK", addr)


def udp_client_example():
    """UDP Client - no connect needed."""
    import socket
    
    HOST = '127.0.0.1'
    PORT = 65433
    
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.sendto(b"Hello UDP!", (HOST, PORT))
        
        data, server = s.recvfrom(1024)
        print(f"Received: {data.decode()}")


# ═══════════════════════════════════════════════════════════════════════════
# SOCKET OPTIONS E TIMEOUTS
# ═══════════════════════════════════════════════════════════════════════════

def socket_options_example():
    """Opzioni comuni dei socket."""
    import socket
    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Timeout (in secondi)
    s.settimeout(5.0)  # 5 secondi
    # s.settimeout(None)  # Blocking (default)
    # s.settimeout(0)     # Non-blocking
    
    # Riuso indirizzo (utile per restart server)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # Keep-alive
    s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    
    # Buffer size
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8192)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8192)
    
    s.close()


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 3.3: HTTP AND REST
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    3.3 TEORIA: HTTP E REST                                   │
└──────────────────────────────────────────────────────────────────────────────┘

HTTP METHODS:
─────────────
GET     - Recuperare risorsa (idempotente, safe)
POST    - Creare risorsa (non idempotente)
PUT     - Aggiornare/creare risorsa (idempotente)
PATCH   - Aggiornamento parziale
DELETE  - Eliminare risorsa (idempotente)
HEAD    - Come GET ma solo headers
OPTIONS - Metodi supportati


HTTP STATUS CODES:
──────────────────
1xx: Informational
2xx: Success
    200 OK
    201 Created
    204 No Content
3xx: Redirection
    301 Moved Permanently
    302 Found (temporary redirect)
    304 Not Modified
4xx: Client Error
    400 Bad Request
    401 Unauthorized
    403 Forbidden
    404 Not Found
    405 Method Not Allowed
    429 Too Many Requests
5xx: Server Error
    500 Internal Server Error
    502 Bad Gateway
    503 Service Unavailable


REST PRINCIPLES:
────────────────
- Client-Server separation
- Stateless (no session)
- Cacheable
- Uniform interface (resources, HTTP verbs)
- Layered system
"""

# ═══════════════════════════════════════════════════════════════════════════
# REQUESTS LIBRARY
# ═══════════════════════════════════════════════════════════════════════════

# pip install requests

import json

def requests_examples():
    """Esempi con la libreria requests."""
    import requests
    
    # GET request
    response = requests.get('https://api.github.com')
    print(f"Status: {response.status_code}")  # 200
    print(f"Headers: {response.headers}")
    print(f"JSON: {response.json()}")
    
    # GET con parametri
    params = {'q': 'python', 'sort': 'stars'}
    response = requests.get(
        'https://api.github.com/search/repositories',
        params=params
    )
    
    # POST con JSON
    data = {'name': 'test', 'value': 123}
    response = requests.post(
        'https://httpbin.org/post',
        json=data  # Automaticamente serializza e setta Content-Type
    )
    
    # POST con form data
    response = requests.post(
        'https://httpbin.org/post',
        data={'field1': 'value1', 'field2': 'value2'}
    )
    
    # Headers personalizzati
    headers = {
        'Authorization': 'Bearer TOKEN_HERE',
        'User-Agent': 'MyApp/1.0'
    }
    response = requests.get('https://api.example.com', headers=headers)
    
    # Timeout (importante!)
    try:
        response = requests.get('https://api.example.com', timeout=5)
    except requests.Timeout:
        print("Request timed out")
    except requests.RequestException as e:
        print(f"Request failed: {e}")
    
    # Session (mantiene cookies, headers)
    session = requests.Session()
    session.headers.update({'Authorization': 'Bearer TOKEN'})
    
    response1 = session.get('https://api.example.com/endpoint1')
    response2 = session.get('https://api.example.com/endpoint2')
    
    session.close()


# ═══════════════════════════════════════════════════════════════════════════
# JSON HANDLING
# ═══════════════════════════════════════════════════════════════════════════

def json_examples():
    """Lavorare con JSON."""
    import json
    
    # Python dict → JSON string
    data = {
        'name': 'Marco',
        'age': 30,
        'skills': ['Python', 'Trading'],
        'active': True,
        'score': None
    }
    
    json_string = json.dumps(data)
    print(json_string)
    # {"name": "Marco", "age": 30, "skills": ["Python", "Trading"], "active": true, "score": null}
    
    # Pretty print
    json_pretty = json.dumps(data, indent=2, sort_keys=True)
    print(json_pretty)
    
    # JSON string → Python dict
    parsed = json.loads(json_string)
    print(parsed['name'])  # Marco
    
    # File operations
    with open('data.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    with open('data.json', 'r') as f:
        loaded = json.load(f)
    
    # Custom encoder per oggetti non-serializzabili
    from datetime import datetime
    
    class DateTimeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return super().default(obj)
    
    data_with_date = {'timestamp': datetime.now()}
    json.dumps(data_with_date, cls=DateTimeEncoder)


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 3.4: WORKING WITH APIs
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    3.4 TEORIA: LAVORARE CON API                              │
└──────────────────────────────────────────────────────────────────────────────┘
"""

# ═══════════════════════════════════════════════════════════════════════════
# API CLIENT PATTERN
# ═══════════════════════════════════════════════════════════════════════════

class APIClient:
    """
    Esempio di client API strutturato.
    Applicabile a qualsiasi REST API.
    """
    
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url.rstrip('/')
        self.session = None  # Inizializzato lazy
        self.api_key = api_key
    
    def _get_session(self):
        """Lazy initialization della sessione."""
        if self.session is None:
            import requests
            self.session = requests.Session()
            self.session.headers.update({
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            })
            if self.api_key:
                self.session.headers['Authorization'] = f'Bearer {self.api_key}'
        return self.session
    
    def _request(self, method: str, endpoint: str, **kwargs):
        """Metodo interno per tutte le richieste."""
        import requests
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = self._get_session().request(
                method=method,
                url=url,
                timeout=30,
                **kwargs
            )
            response.raise_for_status()  # Raise per 4xx/5xx
            return response.json()
        
        except requests.HTTPError as e:
            print(f"HTTP Error: {e.response.status_code}")
            raise
        except requests.Timeout:
            print("Request timed out")
            raise
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            raise
    
    def get(self, endpoint: str, params: dict = None):
        """GET request."""
        return self._request('GET', endpoint, params=params)
    
    def post(self, endpoint: str, data: dict = None):
        """POST request."""
        return self._request('POST', endpoint, json=data)
    
    def put(self, endpoint: str, data: dict = None):
        """PUT request."""
        return self._request('PUT', endpoint, json=data)
    
    def delete(self, endpoint: str):
        """DELETE request."""
        return self._request('DELETE', endpoint)
    
    def close(self):
        """Chiudi la sessione."""
        if self.session:
            self.session.close()
            self.session = None


# Esempio di uso:
# client = APIClient('https://api.example.com', api_key='your_key')
# users = client.get('/users')
# new_user = client.post('/users', {'name': 'Marco'})
# client.close()


# ═══════════════════════════════════════════════════════════════════════════
# RATE LIMITING E RETRY
# ═══════════════════════════════════════════════════════════════════════════

import time
from functools import wraps

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """
    Decorator per retry con exponential backoff.
    Utile per gestire rate limiting e errori temporanei.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    
                    delay = base_delay * (2 ** attempt)
                    print(f"Attempt {attempt + 1} failed: {e}")
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
            
            return None
        return wrapper
    return decorator


# Uso:
# @retry_with_backoff(max_retries=3)
# def fetch_data():
#     return api_client.get('/data')


# ══════════════════════════════════════════════════════════════════════════════
#                    MODULE 3 TEST
# ══════════════════════════════════════════════════════════════════════════════

MODULE_3_TEST = """
══════════════════════════════════════════════════════════════════════════════
                    PA MODULE 3 - TEST FINALE
                    20 domande - Target: 70%
══════════════════════════════════════════════════════════════════════════════

Q1. TCP è:
    A) Connectionless    B) Connection-oriented    C) Unreliable    D) Broadcast

Q2. UDP è usato per:
    A) HTTP    B) SMTP    C) Streaming/Gaming    D) SSH

Q3. Porta standard per HTTP?
    A) 21    B) 22    C) 80    D) 443

Q4. Porta standard per HTTPS?
    A) 21    B) 22    C) 80    D) 443

Q5. 127.0.0.1 è:
    A) Google    B) Localhost    C) Router    D) DNS

Q6. socket.AF_INET indica:
    A) UDP    B) TCP    C) IPv4    D) IPv6

Q7. socket.SOCK_STREAM indica:
    A) UDP    B) TCP    C) IPv4    D) Raw

Q8. socket.SOCK_DGRAM indica:
    A) UDP    B) TCP    C) IPv4    D) Raw

Q9. Status code 200 significa:
    A) Created    B) OK    C) Not Found    D) Error

Q10. Status code 404 significa:
     A) Created    B) OK    C) Not Found    D) Error

Q11. Status code 500 significa:
     A) Created    B) OK    C) Not Found    D) Server Error

Q12. HTTP GET è:
     A) Idempotente    B) Non-idempotente    C) Unsafe    D) Proibito

Q13. HTTP POST è:
     A) Idempotente    B) Non-idempotente    C) Safe    D) Readonly

Q14. json.dumps() converte:
     A) JSON → Python    B) Python → JSON string    C) File → JSON    D) Error

Q15. json.loads() converte:
     A) JSON string → Python    B) Python → JSON    C) File → JSON    D) Error

Q16. requests.get().json() restituisce:
     A) String    B) Bytes    C) Dict/List    D) Response

Q17. Per impostare headers in requests si usa:
     A) params    B) headers    C) data    D) json

Q18. Il timeout in requests è in:
     A) Millisecondi    B) Secondi    C) Minuti    D) Automatico

Q19. s.bind() serve per:
     A) Client    B) Server    C) Entrambi    D) Nessuno

Q20. s.connect() serve per:
     A) Client    B) Server    C) Entrambi    D) Nessuno


══════════════════════════════════════════════════════════════════════════════
"""

MODULE_3_TEST_ANSWERS = """
══════════════════════════════════════════════════════════════════════════════
                         RISPOSTE TEST MODULE 3
══════════════════════════════════════════════════════════════════════════════

Q1:  B) Connection-oriented
Q2:  C) Streaming/Gaming (bassa latenza)
Q3:  C) 80
Q4:  D) 443
Q5:  B) Localhost
Q6:  C) IPv4
Q7:  B) TCP
Q8:  A) UDP
Q9:  B) OK
Q10: C) Not Found
Q11: D) Server Error
Q12: A) Idempotente (e safe)
Q13: B) Non-idempotente
Q14: B) Python → JSON string
Q15: A) JSON string → Python
Q16: C) Dict/List (parsed JSON)
Q17: B) headers
Q18: B) Secondi
Q19: B) Server (bind to address)
Q20: A) Client (connect to server)

══════════════════════════════════════════════════════════════════════════════
"""


if __name__ == "__main__":
    print("=" * 70)
    print("PA MODULE 3: Network Programming")
    print("=" * 70)
    print("""
    Comandi:
    print(MODULE_3_TEST)         # Test finale
    print(MODULE_3_TEST_ANSWERS) # Risposte
    
    Esempi (non eseguire senza server):
    tcp_server_example()
    tcp_client_example()
    """)
