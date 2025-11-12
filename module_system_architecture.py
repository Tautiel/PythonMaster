#!/usr/bin/env python3
"""
🏗️ SYSTEM ARCHITECTURE MODULE
Understanding How Systems Really Work

Duration: 1 Week
Level: From Components to Distributed Systems
"""

import json
import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import hashlib
import random
from collections import deque
from datetime import datetime

# ============================================================================
# PART 1: HOW THE WEB REALLY WORKS
# ============================================================================

class WebRequestFlow:
    """Come funziona DAVVERO una richiesta web"""
    
    def __init__(self):
        self.components = []
        
    def complete_request_journey(self):
        """Il viaggio completo di una HTTP request"""
        
        print("\n🌐 THE COMPLETE WEB REQUEST JOURNEY")
        print("=" * 60)
        
        journey = """
        USER TYPES: https://trading-bot.com/api/prices
        
        1️⃣ DNS RESOLUTION (0-50ms)
           Browser → Local DNS Cache → Router DNS → ISP DNS → Root DNS
           → .com DNS → trading-bot.com DNS → IP: 104.21.45.123
        
        2️⃣ TCP HANDSHAKE (10-100ms)
           SYN → SYN-ACK → ACK (3-way handshake)
           Establish connection on port 443 (HTTPS)
        
        3️⃣ TLS HANDSHAKE (10-150ms)
           Client Hello → Server Hello → Certificate
           → Key Exchange → Change Cipher → Finished
        
        4️⃣ HTTP REQUEST
           GET /api/prices HTTP/2
           Host: trading-bot.com
           Authorization: Bearer token...
        
        5️⃣ LOAD BALANCER (1-5ms)
           → Health check servers
           → Choose server (round-robin/least-connections)
           → Forward to App Server 3
        
        6️⃣ REVERSE PROXY (Nginx) (1-3ms)
           → Check rate limits
           → Validate headers
           → Forward to application
        
        7️⃣ APPLICATION SERVER (10-500ms)
           → Parse request
           → Authenticate user
           → Check permissions
           → Business logic
        
        8️⃣ CACHE CHECK (Redis) (0.5-5ms)
           → Check if data cached
           → If hit: return cached
           → If miss: continue
        
        9️⃣ DATABASE QUERY (5-100ms)
           → Connection pool
           → Query execution
           → Index lookup
           → Return results
        
        🔟 RESPONSE JOURNEY BACK
           → Serialize to JSON
           → Compress (gzip)
           → Add headers
           → Send through reverse path
        
        TOTAL TIME: 50ms - 1s (typically)
        """
        
        print(journey)
        
        # Simulazione dettagliata
        class RequestSimulator:
            def __init__(self):
                self.latencies = {}
                
            async def simulate_request(self, url: str):
                """Simula request con timing reale"""
                steps = [
                    ("DNS Resolution", 20),
                    ("TCP Handshake", 30),
                    ("TLS Handshake", 40),
                    ("Load Balancer", 2),
                    ("App Server", 150),
                    ("Cache Check", 3),
                    ("Database", 50),
                    ("Response", 20)
                ]
                
                total_time = 0
                for step, duration in steps:
                    print(f"⏱️ {step}: {duration}ms")
                    await asyncio.sleep(duration / 1000)
                    total_time += duration
                
                print(f"\n✅ Total: {total_time}ms")
                return total_time
        
        return RequestSimulator()
    
    def load_balancing_algorithms(self):
        """Algoritmi di Load Balancing"""
        
        print("\n⚖️ LOAD BALANCING ALGORITHMS")
        print("=" * 60)
        
        class LoadBalancer:
            def __init__(self, servers: List[str]):
                self.servers = servers
                self.current = 0
                self.connections = {s: 0 for s in servers}
                self.weights = {s: 1 for s in servers}
                self.health = {s: True for s in servers}
                
            def round_robin(self) -> str:
                """Round Robin - Semplice rotazione"""
                server = self.servers[self.current]
                self.current = (self.current + 1) % len(self.servers)
                return server
            
            def least_connections(self) -> str:
                """Least Connections - Meno connessioni"""
                active_servers = [s for s in self.servers if self.health[s]]
                if not active_servers:
                    raise Exception("No healthy servers")
                
                return min(active_servers, key=lambda s: self.connections[s])
            
            def weighted_round_robin(self) -> str:
                """Weighted - Server più potenti ricevono più traffico"""
                # Server con weight 2 riceve 2x requests
                weighted_list = []
                for server in self.servers:
                    weighted_list.extend([server] * self.weights[server])
                
                server = weighted_list[self.current % len(weighted_list)]
                self.current += 1
                return server
            
            def ip_hash(self, client_ip: str) -> str:
                """IP Hash - Stesso client → stesso server"""
                hash_value = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
                index = hash_value % len(self.servers)
                return self.servers[index]
            
            def health_check(self):
                """Health check servers"""
                for server in self.servers:
                    # Simulate health check
                    self.health[server] = random.random() > 0.1  # 90% healthy
        
        # Demo
        servers = ["server1", "server2", "server3"]
        lb = LoadBalancer(servers)
        
        print("Algorithms comparison:")
        print(f"Round Robin: {[lb.round_robin() for _ in range(6)]}")
        print(f"Least Conn: {lb.least_connections()}")
        print(f"IP Hash: {lb.ip_hash('192.168.1.1')}")
        
        return lb

# ============================================================================
# PART 2: MICROSERVICES ARCHITECTURE
# ============================================================================

class MicroservicesArchitecture:
    """Architettura a Microservizi"""
    
    def monolith_vs_microservices(self):
        """Monolith vs Microservices comparison"""
        
        print("\n🏢 MONOLITH vs MICROSERVICES")
        print("=" * 60)
        
        comparison = """
        MONOLITH                    |  MICROSERVICES
        ----------------------------|---------------------------
        ✅ Simple deployment        | ❌ Complex deployment
        ✅ Easy debugging          | ❌ Distributed debugging
        ✅ Fast communication      | ❌ Network latency
        ✅ ACID transactions       | ❌ Eventual consistency
        ✅ One tech stack          | ✅ Multiple tech stacks
        ❌ Scaling all or nothing  | ✅ Scale individually
        ❌ Single point of failure | ✅ Fault isolation
        ❌ Long release cycles     | ✅ Independent deploys
        ❌ Tech lock-in            | ✅ Tech flexibility
        """
        
        print(comparison)
        
        # Esempio pratico: Trading System
        print("\n📊 TRADING SYSTEM ARCHITECTURE")
        
        class TradingSystemArchitecture:
            def monolithic_design(self):
                """Design monolitico"""
                return {
                    "application": "TradingBot",
                    "components": [
                        "User Management",
                        "Authentication",
                        "Order Management",
                        "Price Service",
                        "Risk Management",
                        "Reporting",
                        "Notifications"
                    ],
                    "database": "Single PostgreSQL",
                    "deployment": "Single server/container",
                    "scaling": "Vertical (bigger server)"
                }
            
            def microservices_design(self):
                """Design a microservizi"""
                return {
                    "services": {
                        "auth-service": {
                            "tech": "Python/FastAPI",
                            "db": "Redis",
                            "port": 3001
                        },
                        "user-service": {
                            "tech": "Python/Django",
                            "db": "PostgreSQL",
                            "port": 3002
                        },
                        "order-service": {
                            "tech": "Go",
                            "db": "MongoDB",
                            "port": 3003
                        },
                        "price-service": {
                            "tech": "Python/AsyncIO",
                            "db": "TimescaleDB",
                            "port": 3004
                        },
                        "risk-service": {
                            "tech": "Python/NumPy",
                            "db": "Redis",
                            "port": 3005
                        },
                        "notification-service": {
                            "tech": "Node.js",
                            "db": "None",
                            "port": 3006
                        }
                    },
                    "communication": "REST + gRPC + Message Queue",
                    "api_gateway": "Kong/Nginx",
                    "service_mesh": "Istio",
                    "orchestration": "Kubernetes"
                }
        
        arch = TradingSystemArchitecture()
        print("\nMonolithic:")
        print(json.dumps(arch.monolithic_design(), indent=2))
        print("\nMicroservices:")
        print(json.dumps(arch.microservices_design(), indent=2))
        
        return arch
    
    def service_communication(self):
        """Comunicazione tra servizi"""
        
        print("\n📡 SERVICE COMMUNICATION PATTERNS")
        print("=" * 60)
        
        class ServiceCommunication:
            def synchronous_patterns(self):
                """Patterns sincroni"""
                return {
                    "REST": {
                        "protocol": "HTTP/HTTPS",
                        "format": "JSON",
                        "pros": "Simple, standard",
                        "cons": "Latency, coupling"
                    },
                    "gRPC": {
                        "protocol": "HTTP/2",
                        "format": "Protocol Buffers",
                        "pros": "Fast, typed, streaming",
                        "cons": "Complex, less tooling"
                    },
                    "GraphQL": {
                        "protocol": "HTTP",
                        "format": "JSON",
                        "pros": "Flexible queries",
                        "cons": "Complex caching"
                    }
                }
            
            def asynchronous_patterns(self):
                """Patterns asincroni"""
                return {
                    "Message Queue": {
                        "tools": ["RabbitMQ", "AWS SQS"],
                        "pattern": "Producer-Consumer",
                        "pros": "Decoupling, buffering",
                        "cons": "Complexity, ordering"
                    },
                    "Event Streaming": {
                        "tools": ["Kafka", "Redis Streams"],
                        "pattern": "Pub-Sub",
                        "pros": "Real-time, replay",
                        "cons": "Complexity, storage"
                    },
                    "Webhooks": {
                        "tools": ["Custom", "Zapier"],
                        "pattern": "HTTP callbacks",
                        "pros": "Simple, standard",
                        "cons": "Reliability, security"
                    }
                }
            
            def circuit_breaker(self):
                """Circuit Breaker pattern"""
                class CircuitBreaker:
                    def __init__(self, failure_threshold=5, timeout=60):
                        self.failure_threshold = failure_threshold
                        self.timeout = timeout
                        self.failures = 0
                        self.last_failure_time = None
                        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
                    
                    def call(self, func, *args, **kwargs):
                        if self.state == "OPEN":
                            if time.time() - self.last_failure_time > self.timeout:
                                self.state = "HALF_OPEN"
                            else:
                                raise Exception("Circuit breaker is OPEN")
                        
                        try:
                            result = func(*args, **kwargs)
                            if self.state == "HALF_OPEN":
                                self.state = "CLOSED"
                                self.failures = 0
                            return result
                        except Exception as e:
                            self.failures += 1
                            self.last_failure_time = time.time()
                            
                            if self.failures >= self.failure_threshold:
                                self.state = "OPEN"
                            
                            raise e
                
                return CircuitBreaker()
        
        comm = ServiceCommunication()
        print("Synchronous:")
        print(json.dumps(comm.synchronous_patterns(), indent=2))
        print("\nAsynchronous:")
        print(json.dumps(comm.asynchronous_patterns(), indent=2))
        
        return comm

# ============================================================================
# PART 3: DISTRIBUTED SYSTEMS
# ============================================================================

class DistributedSystems:
    """Sistemi Distribuiti e patterns"""
    
    def cap_theorem(self):
        """CAP Theorem explanation"""
        
        print("\n🔺 CAP THEOREM")
        print("=" * 60)
        
        print("""
        You can only guarantee 2 out of 3:
        
        C - Consistency: All nodes see same data
        A - Availability: System remains operational
        P - Partition Tolerance: Handles network failures
        
        Real-world choices:
        """)
        
        systems = {
            "CP Systems (Consistency + Partition)": {
                "Examples": ["MongoDB", "Redis", "HBase"],
                "Use Case": "Financial transactions",
                "Trade-off": "May be unavailable during partition"
            },
            "AP Systems (Availability + Partition)": {
                "Examples": ["Cassandra", "DynamoDB", "CouchDB"],
                "Use Case": "Social media feeds",
                "Trade-off": "May show stale data"
            },
            "CA Systems (Consistency + Availability)": {
                "Examples": ["PostgreSQL", "MySQL"],
                "Use Case": "Single datacenter",
                "Trade-off": "Can't handle network partition"
            }
        }
        
        for category, details in systems.items():
            print(f"\n{category}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
    
    def consistency_models(self):
        """Modelli di consistenza"""
        
        print("\n🔄 CONSISTENCY MODELS")
        print("=" * 60)
        
        models = {
            "Strong Consistency": {
                "guarantee": "All reads return most recent write",
                "example": "Bank account balance",
                "implementation": "2PC, Paxos, Raft"
            },
            "Eventual Consistency": {
                "guarantee": "Eventually all nodes converge",
                "example": "Social media likes count",
                "implementation": "Gossip protocol"
            },
            "Weak Consistency": {
                "guarantee": "No guarantees after write",
                "example": "Live video streaming",
                "implementation": "Best effort"
            },
            "Read-after-Write": {
                "guarantee": "User sees own writes",
                "example": "User profile updates",
                "implementation": "Sticky sessions"
            }
        }
        
        for model, details in models.items():
            print(f"\n{model}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
    
    def distributed_consensus(self):
        """Algoritmi di consenso distribuito"""
        
        print("\n🤝 DISTRIBUTED CONSENSUS")
        print("=" * 60)
        
        # Simplified Raft implementation
        class RaftNode:
            """Simplified Raft consensus"""
            
            def __init__(self, node_id: int, peers: List[int]):
                self.node_id = node_id
                self.peers = peers
                self.state = "FOLLOWER"  # FOLLOWER, CANDIDATE, LEADER
                self.term = 0
                self.voted_for = None
                self.log = []
                
            def start_election(self):
                """Diventa candidato e richiedi voti"""
                self.state = "CANDIDATE"
                self.term += 1
                self.voted_for = self.node_id
                
                votes = 1  # Vote for self
                
                # Request votes from peers
                for peer in self.peers:
                    # Simulate vote request
                    if random.random() > 0.3:  # 70% chance of vote
                        votes += 1
                
                # Become leader if majority
                if votes > len(self.peers) / 2:
                    self.state = "LEADER"
                    return True
                else:
                    self.state = "FOLLOWER"
                    return False
            
            def append_entry(self, entry: str):
                """Leader appends entry to log"""
                if self.state != "LEADER":
                    raise Exception("Only leader can append")
                
                self.log.append({
                    'term': self.term,
                    'entry': entry,
                    'committed': False
                })
                
                # Replicate to followers
                confirmations = 1  # Self
                for peer in self.peers:
                    # Simulate replication
                    if random.random() > 0.2:  # 80% success
                        confirmations += 1
                
                # Commit if majority confirmed
                if confirmations > len(self.peers) / 2:
                    self.log[-1]['committed'] = True
                    return True
                return False
        
        # Demo
        node = RaftNode(1, [2, 3, 4, 5])
        if node.start_election():
            print("✅ Became leader!")
            if node.append_entry("SET price=100"):
                print("✅ Entry committed!")
        
        return RaftNode

# ============================================================================
# PART 4: CACHING STRATEGIES
# ============================================================================

class CachingStrategies:
    """Strategie di caching"""
    
    def caching_patterns(self):
        """Pattern di caching"""
        
        print("\n💾 CACHING PATTERNS")
        print("=" * 60)
        
        class CachePatterns:
            def __init__(self):
                self.cache = {}
                self.db = {"user1": "data1", "user2": "data2"}
            
            def cache_aside(self, key: str) -> str:
                """Cache-Aside (Lazy Loading)"""
                # 1. Check cache
                if key in self.cache:
                    print(f"Cache HIT: {key}")
                    return self.cache[key]
                
                # 2. Cache miss - load from DB
                print(f"Cache MISS: {key}")
                data = self.db.get(key)
                
                # 3. Update cache
                if data:
                    self.cache[key] = data
                
                return data
            
            def write_through(self, key: str, value: str):
                """Write-Through"""
                # 1. Write to cache
                self.cache[key] = value
                
                # 2. Write to DB (synchronous)
                self.db[key] = value
                print(f"Written to both cache and DB: {key}")
            
            def write_behind(self, key: str, value: str):
                """Write-Behind (Write-Back)"""
                # 1. Write to cache immediately
                self.cache[key] = value
                print(f"Written to cache: {key}")
                
                # 2. Write to DB later (async)
                # In real implementation: queue for batch write
                # self.write_queue.append((key, value))
            
            def refresh_ahead(self, key: str, ttl: int = 60):
                """Refresh-Ahead"""
                # Refresh cache before expiration
                # if time_until_expiry < ttl * 0.2:
                #     self.refresh_cache(key)
                pass
        
        # Cache eviction policies
        print("\n🗑️ EVICTION POLICIES:")
        
        eviction = {
            "LRU": "Least Recently Used - Rimuovi meno usato",
            "LFU": "Least Frequently Used - Rimuovi meno frequente",
            "FIFO": "First In First Out - Rimuovi più vecchio",
            "Random": "Random Replacement - Rimuovi casuale",
            "TTL": "Time To Live - Scade dopo X secondi"
        }
        
        for policy, description in eviction.items():
            print(f"  {policy:8} → {description}")
        
        # LRU Cache implementation
        class LRUCache:
            def __init__(self, capacity: int):
                self.capacity = capacity
                self.cache = {}
                self.order = deque()
            
            def get(self, key: str) -> Optional[Any]:
                if key in self.cache:
                    # Move to end (most recent)
                    self.order.remove(key)
                    self.order.append(key)
                    return self.cache[key]
                return None
            
            def put(self, key: str, value: Any):
                if key in self.cache:
                    self.order.remove(key)
                elif len(self.cache) >= self.capacity:
                    # Remove least recent
                    oldest = self.order.popleft()
                    del self.cache[oldest]
                
                self.cache[key] = value
                self.order.append(key)
        
        return CachePatterns(), LRUCache(100)

# ============================================================================
# PART 5: DATABASE PATTERNS
# ============================================================================

class DatabasePatterns:
    """Pattern per database"""
    
    def database_sharding(self):
        """Database sharding strategies"""
        
        print("\n🔪 DATABASE SHARDING")
        print("=" * 60)
        
        class ShardingStrategies:
            def range_sharding(self, key: int, num_shards: int) -> int:
                """Range-based sharding"""
                # Example: User IDs 0-999 → Shard 0
                #          User IDs 1000-1999 → Shard 1
                range_size = 1000
                return key // range_size % num_shards
            
            def hash_sharding(self, key: str, num_shards: int) -> int:
                """Hash-based sharding"""
                # Consistent distribution
                hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
                return hash_value % num_shards
            
            def geographic_sharding(self, location: str) -> str:
                """Geographic sharding"""
                regions = {
                    'US': 'db-us-east',
                    'EU': 'db-eu-west',
                    'ASIA': 'db-asia-south'
                }
                # Determine region from location
                return regions.get(location, 'db-global')
            
            def consistent_hashing(self):
                """Consistent hashing for dynamic shards"""
                class ConsistentHash:
                    def __init__(self, nodes: List[str], virtual_nodes: int = 150):
                        self.nodes = nodes
                        self.virtual_nodes = virtual_nodes
                        self.ring = {}
                        self._build_ring()
                    
                    def _hash(self, key: str) -> int:
                        return int(hashlib.md5(key.encode()).hexdigest(), 16)
                    
                    def _build_ring(self):
                        for node in self.nodes:
                            for i in range(self.virtual_nodes):
                                virtual_key = f"{node}:{i}"
                                hash_value = self._hash(virtual_key)
                                self.ring[hash_value] = node
                    
                    def get_node(self, key: str) -> str:
                        if not self.ring:
                            return None
                        
                        hash_value = self._hash(key)
                        
                        # Find first node clockwise
                        for node_hash in sorted(self.ring.keys()):
                            if node_hash >= hash_value:
                                return self.ring[node_hash]
                        
                        # Wrap around
                        return self.ring[min(self.ring.keys())]
                    
                    def add_node(self, node: str):
                        self.nodes.append(node)
                        for i in range(self.virtual_nodes):
                            virtual_key = f"{node}:{i}"
                            hash_value = self._hash(virtual_key)
                            self.ring[hash_value] = node
                    
                    def remove_node(self, node: str):
                        self.nodes.remove(node)
                        to_remove = []
                        for hash_value, n in self.ring.items():
                            if n == node:
                                to_remove.append(hash_value)
                        for hash_value in to_remove:
                            del self.ring[hash_value]
                
                return ConsistentHash(['shard1', 'shard2', 'shard3'])
        
        sharding = ShardingStrategies()
        
        # Demo
        print("Sharding Examples:")
        print(f"Range: User 1234 → Shard {sharding.range_sharding(1234, 3)}")
        print(f"Hash: 'user@email.com' → Shard {sharding.hash_sharding('user@email.com', 3)}")
        print(f"Geographic: 'EU' → {sharding.geographic_sharding('EU')}")
        
        return sharding

# ============================================================================
# PART 6: ARCHITECTURE PROJECTS
# ============================================================================

class ArchitectureProjects:
    """Progetti di architettura"""
    
    def design_url_shortener(self):
        """Design URL shortener system"""
        
        print("\n🔗 PROJECT: URL SHORTENER SYSTEM DESIGN")
        print("=" * 60)
        
        design = {
            "Requirements": {
                "Functional": [
                    "Shorten long URLs",
                    "Redirect to original",
                    "Custom aliases",
                    "Analytics"
                ],
                "Non-Functional": [
                    "100M URLs/day",
                    "<100ms latency",
                    "99.9% uptime"
                ]
            },
            
            "Capacity": {
                "Storage": "100M * 365 * 5 years * 500 bytes = 91TB",
                "Bandwidth": "100M * 500 bytes = 50GB/day",
                "QPS": "100M / 86400 = 1160 requests/sec"
            },
            
            "Architecture": {
                "API Gateway": "Rate limiting, authentication",
                "App Servers": "Stateless, horizontally scalable",
                "Cache": "Redis for hot URLs",
                "Database": "Cassandra for scale",
                "CDN": "Global distribution"
            },
            
            "Algorithm": {
                "Base62": "a-z, A-Z, 0-9 = 62 chars",
                "Length": "7 chars = 62^7 = 3.5 trillion URLs",
                "Generation": "Counter or hash-based"
            }
        }
        
        for section, content in design.items():
            print(f"\n{section}:")
            if isinstance(content, dict):
                for key, value in content.items():
                    print(f"  {key}: {value}")
            else:
                for item in content:
                    print(f"  • {item}")
        
        # Implementation
        class URLShortener:
            def __init__(self):
                self.counter = 1000000  # Start from 1M
                self.url_db = {}
                self.short_db = {}
                
            def encode(self, num: int) -> str:
                """Convert number to base62"""
                chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                base = len(chars)
                
                if num == 0:
                    return chars[0]
                
                result = []
                while num:
                    result.append(chars[num % base])
                    num //= base
                
                return ''.join(reversed(result))
            
            def shorten(self, long_url: str) -> str:
                """Shorten URL"""
                if long_url in self.url_db:
                    return self.url_db[long_url]
                
                short = self.encode(self.counter)
                self.counter += 1
                
                self.url_db[long_url] = short
                self.short_db[short] = long_url
                
                return f"short.ly/{short}"
        
        return URLShortener()

# ============================================================================
# EXERCISES
# ============================================================================

def architecture_exercises():
    """40 architecture exercises"""
    
    print("\n🏗️ ARCHITECTURE EXERCISES")
    print("=" * 60)
    
    exercises = {
        "System Design (1-20)": [
            "Design Twitter-like system",
            "Design YouTube video platform",
            "Design Uber ride-sharing",
            "Design Netflix streaming",
            "Design WhatsApp messaging",
            "Design Instagram photo sharing",
            "Design Dropbox file storage",
            "Design Facebook newsfeed",
            "Design Google Maps",
            "Design Stripe payments",
            "Design Airbnb booking",
            "Design Amazon e-commerce",
            "Design LinkedIn jobs",
            "Design Zoom video calls",
            "Design Reddit forums",
            "Design Discord chat",
            "Design Spotify music",
            "Design GitHub code hosting",
            "Design Slack workspace",
            "Design Trading platform"
        ],
        
        "Scaling (21-40)": [
            "Scale database to 1B records",
            "Implement database sharding",
            "Design caching strategy",
            "Implement rate limiting",
            "Design CDN distribution",
            "Implement load balancer",
            "Design microservices split",
            "Implement service mesh",
            "Design event-driven system",
            "Implement CQRS pattern",
            "Design data pipeline",
            "Implement message queue",
            "Design monitoring system",
            "Implement circuit breaker",
            "Design backup strategy",
            "Implement blue-green deploy",
            "Design disaster recovery",
            "Implement auto-scaling",
            "Design multi-region setup",
            "Implement chaos engineering"
        ]
    }
    
    for category, items in exercises.items():
        print(f"\n{category}:")
        for i, exercise in enumerate(items, 1):
            print(f"  {i}. {exercise}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run system architecture module"""
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║              🏗️ SYSTEM ARCHITECTURE MODULE                  ║
    ║           From Components to Distributed Systems            ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    modules = {
        "1": ("Web Request Flow", WebRequestFlow),
        "2": ("Microservices", MicroservicesArchitecture),
        "3": ("Distributed Systems", DistributedSystems),
        "4": ("Caching", CachingStrategies),
        "5": ("Database Patterns", DatabasePatterns),
        "6": ("Projects", ArchitectureProjects),
        "7": ("Exercises", architecture_exercises)
    }
    
    while True:
        print("\n📚 SELECT MODULE:")
        for key, (name, _) in modules.items():
            print(f"  {key}. {name}")
        print("  Q. Quit")
        
        choice = input("\nChoice: ").strip().upper()
        
        if choice == 'Q':
            break
        elif choice == '7':
            architecture_exercises()
        else:
            # Run module
            pass

if __name__ == "__main__":
    main()
