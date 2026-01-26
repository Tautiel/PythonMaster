#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PYTHON PROFESSIONAL 1 - MODULE 4                          ║
║                    NETWORK PROGRAMMING                                        ║
║                    PCPP1-32-101 Section 4: 18% (7 domande)                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

SYLLABUS:
├── PCPP1 4.1 - socket module (TCP/UDP sockets)
├── PCPP1 4.2 - requests library (HTTP)
├── PCPP1 4.3 - JSON and XML processing
└── PCPP1 4.4 - REST API concepts (CRUD)
"""

import socket
import json

# ══════════════════════════════════════════════════════════════════════════════
# 4.1 NETWORKING BASICS
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("4.1 NETWORKING BASICS")
print("=" * 70)

print("""
📋 NETWORK CONCEPTS:

IP ADDRESS
  - IPv4: 192.168.1.1 (32 bit)
  - IPv6: 2001:0db8:85a3::8a2e:0370:7334 (128 bit)

PORT
  - Numero 0-65535
  - Well-known: 0-1023 (HTTP=80, HTTPS=443, SSH=22)
  - Registered: 1024-49151
  - Dynamic: 49152-65535

SOCKET
  - Endpoint di comunicazione: (IP, Port)
  - TCP: Connection-oriented, reliable
  - UDP: Connectionless, fast

TCP vs UDP:
┌─────────────┬────────────────────────────────┐
│ TCP         │ UDP                            │
├─────────────┼────────────────────────────────┤
│ Reliable    │ Unreliable                     │
│ Ordered     │ Unordered                      │
│ Connected   │ Connectionless                 │
│ Slow        │ Fast                           │
│ HTTP, FTP   │ DNS, Gaming, Streaming         │
└─────────────┴────────────────────────────────┘
""")

# ══════════════════════════════════════════════════════════════════════════════
# 4.2 SOCKET MODULE (ESAME!)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("4.2 SOCKET MODULE (ESAME!)")
print("=" * 70)

print("""
📋 CREATING SOCKETS:

# TCP Socket
tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# UDP Socket
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

📋 PARAMETERS:
  AF_INET     = IPv4
  AF_INET6    = IPv6
  SOCK_STREAM = TCP
  SOCK_DGRAM  = UDP
""")

# TCP Server Example
print("""
═══════════════════════════════════════════════════════════════════════
TCP SERVER EXAMPLE:
═══════════════════════════════════════════════════════════════════════
import socket

# 1. Create socket
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 2. Bind to address
server.bind(('localhost', 5000))

# 3. Start listening
server.listen(5)  # max 5 queued connections

# 4. Accept connections (blocking)
client_socket, address = server.accept()
print(f"Connection from {address}")

# 5. Receive data
data = client_socket.recv(1024)  # max 1024 bytes
print(f"Received: {data.decode()}")

# 6. Send response
client_socket.send(b"Hello from server")

# 7. Close
client_socket.close()
server.close()
""")

# TCP Client Example
print("""
═══════════════════════════════════════════════════════════════════════
TCP CLIENT EXAMPLE:
═══════════════════════════════════════════════════════════════════════
import socket

# 1. Create socket
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 2. Connect to server
client.connect(('localhost', 5000))

# 3. Send data
client.send(b"Hello from client")

# 4. Receive response
response = client.recv(1024)
print(f"Server says: {response.decode()}")

# 5. Close
client.close()
""")

# Socket Methods Summary
print("""
📋 SOCKET METHODS SUMMARY:

SERVER METHODS:
  bind((host, port))     - Associa socket a indirizzo
  listen(backlog)        - Inizia ad accettare connessioni
  accept()              - Accetta connessione (blocking)
                          Returns: (new_socket, address)

CLIENT METHODS:
  connect((host, port))  - Connetti al server

COMMON METHODS:
  send(bytes)           - Invia dati
  recv(bufsize)         - Ricevi dati (blocking)
  close()               - Chiudi socket
  settimeout(seconds)   - Imposta timeout
  setblocking(bool)     - Blocking/non-blocking mode
""")

# UDP Example
print("""
═══════════════════════════════════════════════════════════════════════
UDP (CONNECTIONLESS):
═══════════════════════════════════════════════════════════════════════
# Server
server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server.bind(('localhost', 5000))
data, addr = server.recvfrom(1024)  # Nota: recvFROM
server.sendto(b"Response", addr)    # Nota: sendTO

# Client
client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client.sendto(b"Hello", ('localhost', 5000))
response, server = client.recvfrom(1024)
""")

# ══════════════════════════════════════════════════════════════════════════════
# 4.3 REQUESTS LIBRARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("4.3 REQUESTS LIBRARY")
print("=" * 70)

print("""
📋 HTTP METHODS (CRUD):
  GET     - Read (retrieve data)
  POST    - Create (submit data)
  PUT     - Update (replace data)
  PATCH   - Partial Update
  DELETE  - Delete

📋 REQUESTS USAGE:

import requests

# GET request
response = requests.get('https://api.example.com/users')
print(response.status_code)  # 200
print(response.json())       # Parse JSON response
print(response.text)         # Raw text

# GET with parameters
params = {'page': 1, 'limit': 10}
response = requests.get('https://api.example.com/users', params=params)

# POST request
data = {'name': 'Marco', 'email': 'marco@email.com'}
response = requests.post('https://api.example.com/users', json=data)

# Headers
headers = {'Authorization': 'Bearer token123'}
response = requests.get('https://api.example.com/me', headers=headers)

# PUT/DELETE
requests.put('https://api.example.com/users/1', json={'name': 'Updated'})
requests.delete('https://api.example.com/users/1')

📋 RESPONSE ATTRIBUTES:
  response.status_code   - HTTP status (200, 404, etc.)
  response.text          - Response body as text
  response.json()        - Parse JSON response
  response.headers       - Response headers
  response.ok            - True if status < 400
""")

# Status Codes
print("""
📋 HTTP STATUS CODES:
  1xx - Informational
  2xx - Success (200 OK, 201 Created)
  3xx - Redirect (301 Moved, 302 Found)
  4xx - Client Error (400 Bad Request, 404 Not Found)
  5xx - Server Error (500 Internal Error)
""")

# ══════════════════════════════════════════════════════════════════════════════
# 4.4 JSON PROCESSING (ESAME!)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("4.4 JSON PROCESSING (ESAME!)")
print("=" * 70)

print("""
📋 JSON MODULE:

json.dumps(obj)      - Python → JSON string
json.loads(string)   - JSON string → Python
json.dump(obj, file) - Python → JSON file
json.load(file)      - JSON file → Python
""")

# Example
data = {
    "name": "Marco",
    "age": 25,
    "scores": [95, 87, 92],
    "active": True,
    "address": None
}

# Python → JSON
json_string = json.dumps(data, indent=2)
print(f"json.dumps(data):\n{json_string}")

# JSON → Python
parsed = json.loads(json_string)
print(f"\njson.loads(json_string): {type(parsed)}")

# Type mapping
print("""
📋 JSON ↔ PYTHON TYPE MAPPING:
┌─────────────┬─────────────┐
│ JSON        │ Python      │
├─────────────┼─────────────┤
│ object      │ dict        │
│ array       │ list        │
│ string      │ str         │
│ number (int)│ int         │
│ number (real)│ float      │
│ true        │ True        │
│ false       │ False       │
│ null        │ None        │
└─────────────┴─────────────┘
""")

# ══════════════════════════════════════════════════════════════════════════════
# 4.5 XML PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("4.5 XML PROCESSING")
print("=" * 70)

print("""
📋 XML MODULES:

xml.etree.ElementTree - Standard library (recommended)
xml.dom.minidom       - DOM interface
xml.sax               - SAX parser (event-based)

📋 ElementTree USAGE:

import xml.etree.ElementTree as ET

# Parse XML
tree = ET.parse('data.xml')
root = tree.getroot()

# Or from string
root = ET.fromstring('<root><item>Hello</item></root>')

# Navigate
for child in root:
    print(child.tag, child.attrib, child.text)

# Find elements
item = root.find('item')          # First match
items = root.findall('.//item')   # All matches

# Attributes
element.get('attribute')          # Get attribute
element.set('attribute', 'value') # Set attribute

# Create XML
root = ET.Element('root')
child = ET.SubElement(root, 'child')
child.text = 'Hello'
tree = ET.ElementTree(root)
tree.write('output.xml')
""")

import xml.etree.ElementTree as ET

xml_string = """
<users>
    <user id="1">
        <name>Marco</name>
        <email>marco@email.com</email>
    </user>
    <user id="2">
        <name>Anna</name>
        <email>anna@email.com</email>
    </user>
</users>
"""

root = ET.fromstring(xml_string)
print("Parsing XML:")
for user in root.findall('user'):
    user_id = user.get('id')
    name = user.find('name').text
    email = user.find('email').text
    print(f"  ID={user_id}, Name={name}, Email={email}")

# ══════════════════════════════════════════════════════════════════════════════
# 4.6 REST API CONCEPTS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("4.6 REST API CONCEPTS")
print("=" * 70)

print("""
📋 REST (Representational State Transfer):

PRINCIPLES:
  - Stateless: ogni request è indipendente
  - Client-Server: separazione responsabilità
  - Cacheable: response possono essere cached
  - Uniform Interface: URL consistenti

CRUD MAPPING:
┌──────────┬────────────┬──────────────────────┐
│ CRUD     │ HTTP       │ Example              │
├──────────┼────────────┼──────────────────────┤
│ Create   │ POST       │ POST /users          │
│ Read     │ GET        │ GET /users/1         │
│ Update   │ PUT/PATCH  │ PUT /users/1         │
│ Delete   │ DELETE     │ DELETE /users/1      │
└──────────┴────────────┴──────────────────────┘

URL PATTERNS:
  GET    /users         - List all users
  GET    /users/1       - Get user 1
  POST   /users         - Create new user
  PUT    /users/1       - Update user 1
  DELETE /users/1       - Delete user 1
  GET    /users/1/posts - Get posts of user 1
""")

# ══════════════════════════════════════════════════════════════════════════════
# QUIZ
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("QUIZ DI VERIFICA")
print("=" * 70)

print("""
Q1. Quale socket type per TCP?
    A) SOCK_DGRAM  B) SOCK_STREAM  C) SOCK_RAW  D) SOCK_TCP
    → RISPOSTA: B

Q2. Quale metodo accetta connessioni TCP?
    A) listen()  B) connect()  C) accept()  D) bind()
    → RISPOSTA: C

Q3. json.dumps() fa cosa?
    A) JSON → Python  B) Python → JSON  C) File → Python  D) Python → File
    → RISPOSTA: B

Q4. HTTP 404 significa?
    A) OK  B) Created  C) Not Found  D) Server Error
    → RISPOSTA: C

Q5. Quale metodo HTTP per creare risorse?
    A) GET  B) POST  C) PUT  D) DELETE
    → RISPOSTA: B

Q6. AF_INET indica?
    A) IPv4  B) IPv6  C) TCP  D) UDP
    → RISPOSTA: A

Q7. requests.get().json() restituisce?
    A) String  B) Dict/List  C) Bytes  D) Response
    → RISPOSTA: B

Q8. socket.recv(1024) - cosa significa 1024?
    A) Port  B) Timeout  C) Max bytes  D) Min bytes
    → RISPOSTA: C
""")

print("\n" + "=" * 70)
print("NETWORK MODULE COMPLETATO!")
print("=" * 70)
