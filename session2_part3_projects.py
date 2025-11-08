"""
🚀 SESSIONE 2 - PARTE 3: PROGETTI PRODUCTION-READY
==================================================
4 Progetti Avanzati: API, Chat, Scraper, Task Queue
Durata: 90 minuti di progetti reali
"""

import asyncio
import json
import sqlite3
import hashlib
import jwt
import websockets
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import re
from bs4 import BeautifulSoup
import pickle
import uuid
from collections import deque
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("="*80)
print("💻 SESSIONE 2 PARTE 3: 4 PROGETTI PRODUCTION-READY")
print("="*80)
print("\n1. REST API Server (FastAPI-like)")
print("2. Real-time Chat System")
print("3. Async Web Scraper")
print("4. Task Queue System")
print("="*80)

# ==============================================================================
# PROGETTO 1: REST API SERVER
# ==============================================================================

print("\n" + "="*60)
print("🌐 PROGETTO 1: REST API SERVER")
print("="*60)

class HTTPMethod(Enum):
    """HTTP Methods"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"

@dataclass
class Request:
    """HTTP Request"""
    method: HTTPMethod
    path: str
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[Dict] = None
    params: Dict[str, str] = field(default_factory=dict)
    user: Optional['User'] = None

@dataclass
class Response:
    """HTTP Response"""
    status_code: int
    body: Any
    headers: Dict[str, str] = field(default_factory=dict)
    
    def json(self):
        return json.dumps(self.body)

@dataclass
class User:
    """User model"""
    id: str
    username: str
    email: str
    password_hash: str
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password with salt"""
        salt = "app_salt_"  # In produzione usa salt random
        return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    
    def verify_password(self, password: str) -> bool:
        """Verifica password"""
        return self.password_hash == User.hash_password(password)

class Database:
    """Simple database wrapper"""
    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS posts (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        self.conn.commit()
    
    def create_user(self, user: User) -> User:
        """Create new user"""
        self.conn.execute("""
            INSERT INTO users (id, username, email, password_hash)
            VALUES (?, ?, ?, ?)
        """, (user.id, user.username, user.email, user.password_hash))
        self.conn.commit()
        return user
    
    def get_user(self, username: str) -> Optional[User]:
        """Get user by username"""
        row = self.conn.execute("""
            SELECT * FROM users WHERE username = ?
        """, (username,)).fetchone()
        
        if row:
            return User(
                id=row['id'],
                username=row['username'],
                email=row['email'],
                password_hash=row['password_hash']
            )
        return None
    
    def create_post(self, post: Dict) -> Dict:
        """Create new post"""
        post_id = str(uuid.uuid4())
        self.conn.execute("""
            INSERT INTO posts (id, user_id, title, content)
            VALUES (?, ?, ?, ?)
        """, (post_id, post['user_id'], post['title'], post['content']))
        self.conn.commit()
        post['id'] = post_id
        return post
    
    def get_posts(self, user_id: Optional[str] = None) -> List[Dict]:
        """Get posts, optionally filtered by user"""
        if user_id:
            rows = self.conn.execute("""
                SELECT * FROM posts WHERE user_id = ?
                ORDER BY created_at DESC
            """, (user_id,)).fetchall()
        else:
            rows = self.conn.execute("""
                SELECT * FROM posts ORDER BY created_at DESC
            """).fetchall()
        
        return [dict(row) for row in rows]

class JWTAuth:
    """JWT Authentication handler"""
    SECRET_KEY = "super_secret_key_change_in_production"
    ALGORITHM = "HS256"
    
    @classmethod
    def create_token(cls, user: User) -> str:
        """Create JWT token"""
        payload = {
            "user_id": user.id,
            "username": user.username,
            "exp": datetime.utcnow() + timedelta(hours=24)
        }
        
        return jwt.encode(payload, cls.SECRET_KEY, algorithm=cls.ALGORITHM)
    
    @classmethod
    def verify_token(cls, token: str) -> Optional[Dict]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, cls.SECRET_KEY, algorithms=[cls.ALGORITHM])
            return payload
        except jwt.InvalidTokenError:
            return None

class APIRouter:
    """Router for API endpoints"""
    def __init__(self):
        self.routes: Dict[str, Dict[HTTPMethod, callable]] = {}
        self.middleware: List[callable] = []
    
    def route(self, path: str, method: HTTPMethod):
        """Decorator to register routes"""
        def decorator(func):
            if path not in self.routes:
                self.routes[path] = {}
            self.routes[path][method] = func
            return func
        return decorator
    
    def add_middleware(self, middleware: callable):
        """Add middleware function"""
        self.middleware.append(middleware)
    
    async def handle_request(self, request: Request) -> Response:
        """Handle incoming request"""
        # Run middleware
        for mw in self.middleware:
            request = await mw(request)
            if isinstance(request, Response):
                return request
        
        # Find route
        handler = self.routes.get(request.path, {}).get(request.method)
        
        if not handler:
            return Response(404, {"error": "Not found"})
        
        try:
            # Call handler
            result = await handler(request)
            
            if isinstance(result, Response):
                return result
            
            # Auto-wrap in Response
            return Response(200, result)
            
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            return Response(500, {"error": "Internal server error"})

class APIServer:
    """Main API Server"""
    def __init__(self):
        self.router = APIRouter()
        self.db = Database()
        self._setup_routes()
        self._setup_middleware()
    
    def _setup_middleware(self):
        """Setup middleware"""
        
        async def auth_middleware(request: Request) -> Request:
            """Authentication middleware"""
            # Skip auth for public endpoints
            public_paths = ['/register', '/login', '/health']
            if request.path in public_paths:
                return request
            
            # Check token
            auth_header = request.headers.get('Authorization', '')
            if not auth_header.startswith('Bearer '):
                return Response(401, {"error": "Unauthorized"})
            
            token = auth_header.split(' ')[1]
            payload = JWTAuth.verify_token(token)
            
            if not payload:
                return Response(401, {"error": "Invalid token"})
            
            # Add user to request
            user = self.db.get_user(payload['username'])
            if not user:
                return Response(401, {"error": "User not found"})
            
            request.user = user
            return request
        
        async def logging_middleware(request: Request) -> Request:
            """Logging middleware"""
            logger.info(f"{request.method.value} {request.path}")
            return request
        
        self.router.add_middleware(logging_middleware)
        self.router.add_middleware(auth_middleware)
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.router.route('/health', HTTPMethod.GET)
        async def health_check(request: Request):
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        @self.router.route('/register', HTTPMethod.POST)
        async def register(request: Request):
            data = request.body
            
            # Validate
            if not all(k in data for k in ['username', 'email', 'password']):
                return Response(400, {"error": "Missing required fields"})
            
            # Create user
            user = User(
                id=str(uuid.uuid4()),
                username=data['username'],
                email=data['email'],
                password_hash=User.hash_password(data['password'])
            )
            
            try:
                self.db.create_user(user)
                token = JWTAuth.create_token(user)
                
                return {
                    "user": {
                        "id": user.id,
                        "username": user.username,
                        "email": user.email
                    },
                    "token": token
                }
            except sqlite3.IntegrityError:
                return Response(400, {"error": "User already exists"})
        
        @self.router.route('/login', HTTPMethod.POST)
        async def login(request: Request):
            data = request.body
            
            user = self.db.get_user(data.get('username', ''))
            if not user or not user.verify_password(data.get('password', '')):
                return Response(401, {"error": "Invalid credentials"})
            
            token = JWTAuth.create_token(user)
            return {"token": token}
        
        @self.router.route('/posts', HTTPMethod.GET)
        async def get_posts(request: Request):
            posts = self.db.get_posts()
            return {"posts": posts}
        
        @self.router.route('/posts', HTTPMethod.POST)
        async def create_post(request: Request):
            data = request.body
            data['user_id'] = request.user.id
            
            post = self.db.create_post(data)
            return {"post": post}
        
        @self.router.route('/me', HTTPMethod.GET)
        async def get_profile(request: Request):
            return {
                "user": {
                    "id": request.user.id,
                    "username": request.user.username,
                    "email": request.user.email
                }
            }
    
    async def test_api(self):
        """Test the API"""
        print("\n🧪 Testing API Server...")
        
        # Test health check
        req = Request(HTTPMethod.GET, '/health')
        res = await self.router.handle_request(req)
        print(f"  Health: {res.status_code} - {res.body}")
        
        # Test registration
        req = Request(
            HTTPMethod.POST, 
            '/register',
            body={
                "username": "testuser",
                "email": "test@example.com",
                "password": "password123"
            }
        )
        res = await self.router.handle_request(req)
        print(f"  Register: {res.status_code}")
        
        if res.status_code == 200:
            token = res.body['token']
            print(f"  Token: {token[:20]}...")
            
            # Test authenticated endpoint
            req = Request(
                HTTPMethod.GET,
                '/me',
                headers={'Authorization': f'Bearer {token}'}
            )
            res = await self.router.handle_request(req)
            print(f"  Profile: {res.body}")

# ==============================================================================
# PROGETTO 2: REAL-TIME CHAT SYSTEM
# ==============================================================================

print("\n" + "="*60)
print("💬 PROGETTO 2: REAL-TIME CHAT SYSTEM")
print("="*60)

@dataclass
class ChatMessage:
    """Chat message"""
    id: str
    user: str
    room: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "user": self.user,
            "room": self.room,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }

class ChatRoom:
    """Chat room manager"""
    def __init__(self, room_id: str):
        self.room_id = room_id
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.messages: deque = deque(maxlen=100)  # Keep last 100 messages
        self.created_at = datetime.now()
    
    async def add_client(self, websocket):
        """Add client to room"""
        self.clients.add(websocket)
        
        # Send recent messages
        for msg in self.messages:
            await websocket.send(json.dumps(msg.to_dict()))
        
        # Notify others
        await self.broadcast({
            "type": "user_joined",
            "user": f"User-{id(websocket)}",
            "room": self.room_id
        }, exclude=websocket)
    
    async def remove_client(self, websocket):
        """Remove client from room"""
        self.clients.discard(websocket)
        
        # Notify others
        await self.broadcast({
            "type": "user_left",
            "user": f"User-{id(websocket)}",
            "room": self.room_id
        })
    
    async def broadcast(self, message: Dict, exclude=None):
        """Broadcast message to all clients"""
        disconnected = set()
        
        for client in self.clients:
            if client != exclude:
                try:
                    await client.send(json.dumps(message))
                except websockets.ConnectionClosed:
                    disconnected.add(client)
        
        # Remove disconnected clients
        for client in disconnected:
            self.clients.discard(client)
    
    async def handle_message(self, websocket, data: Dict):
        """Handle incoming message"""
        msg = ChatMessage(
            id=str(uuid.uuid4()),
            user=data.get('user', f'User-{id(websocket)}'),
            room=self.room_id,
            content=data['content']
        )
        
        self.messages.append(msg)
        
        # Broadcast to all clients
        await self.broadcast(msg.to_dict())

class ChatServer:
    """WebSocket chat server"""
    def __init__(self):
        self.rooms: Dict[str, ChatRoom] = {}
    
    def get_room(self, room_id: str) -> ChatRoom:
        """Get or create room"""
        if room_id not in self.rooms:
            self.rooms[room_id] = ChatRoom(room_id)
            logger.info(f"Created room: {room_id}")
        return self.rooms[room_id]
    
    async def handle_client(self, websocket, path):
        """Handle WebSocket client connection"""
        # Extract room from path
        room_id = path.strip('/') or 'general'
        room = self.get_room(room_id)
        
        logger.info(f"Client connected to room: {room_id}")
        
        try:
            # Add client to room
            await room.add_client(websocket)
            
            # Handle messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    if data.get('type') == 'message':
                        await room.handle_message(websocket, data)
                    
                except json.JSONDecodeError:
                    logger.error("Invalid message format")
                    
        except websockets.ConnectionClosed:
            logger.info("Client disconnected")
            
        finally:
            await room.remove_client(websocket)
    
    def demo(self):
        """Demo chat system"""
        print("\n🧪 Chat Server Demo...")
        print("  WebSocket server would run on ws://localhost:8765")
        print("  Rooms: /general, /tech, /random")
        print("  Features:")
        print("    • Real-time messaging")
        print("    • Multiple rooms")
        print("    • Message history")
        print("    • User join/leave notifications")
        
        # Simulate some activity
        room = self.get_room("general")
        
        async def simulate():
            # Simulate messages
            for i in range(3):
                msg = ChatMessage(
                    id=str(uuid.uuid4()),
                    user=f"User{i}",
                    room="general",
                    content=f"Hello from User{i}!"
                )
                room.messages.append(msg)
                print(f"    Message: {msg.user}: {msg.content}")
        
        asyncio.run(simulate())

# ==============================================================================
# PROGETTO 3: ASYNC WEB SCRAPER
# ==============================================================================

print("\n" + "="*60)
print("🕷️ PROGETTO 3: ASYNC WEB SCRAPER")
print("="*60)

@dataclass
class ScrapedData:
    """Scraped data container"""
    url: str
    title: Optional[str] = None
    content: Optional[str] = None
    links: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    meta: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

class RateLimiter:
    """Rate limiter for requests"""
    def __init__(self, max_requests: int = 10, time_window: float = 1.0):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
    
    async def acquire(self):
        """Wait if necessary to respect rate limit"""
        now = time.time()
        
        # Remove old requests
        while self.requests and self.requests[0] < now - self.time_window:
            self.requests.popleft()
        
        # Wait if at limit
        if len(self.requests) >= self.max_requests:
            sleep_time = self.time_window - (now - self.requests[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self.requests.append(now)

class URLFilter:
    """URL filter and normalizer"""
    def __init__(self, allowed_domains: List[str] = None):
        self.allowed_domains = allowed_domains or []
        self.visited = set()
    
    def normalize_url(self, url: str, base_url: str = None) -> str:
        """Normalize URL"""
        # Simple normalization
        if url.startswith('//'):
            url = 'https:' + url
        elif url.startswith('/') and base_url:
            from urllib.parse import urljoin
            url = urljoin(base_url, url)
        
        # Remove fragment
        url = url.split('#')[0]
        
        # Remove trailing slash
        if url.endswith('/'):
            url = url[:-1]
        
        return url
    
    def should_scrape(self, url: str) -> bool:
        """Check if URL should be scraped"""
        # Check if visited
        if url in self.visited:
            return False
        
        # Check domain
        if self.allowed_domains:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            if not any(allowed in domain for allowed in self.allowed_domains):
                return False
        
        # Check file extension
        excluded_extensions = ['.pdf', '.zip', '.exe', '.mp4', '.jpg', '.png']
        if any(url.lower().endswith(ext) for ext in excluded_extensions):
            return False
        
        return True
    
    def mark_visited(self, url: str):
        """Mark URL as visited"""
        self.visited.add(url)

class AsyncWebScraper:
    """Async web scraper with multiple features"""
    def __init__(self, max_depth: int = 2, max_workers: int = 5):
        self.max_depth = max_depth
        self.max_workers = max_workers
        self.rate_limiter = RateLimiter(max_requests=10, time_window=1)
        self.url_filter = URLFilter()
        self.scraped_data: List[ScrapedData] = []
    
    async def fetch_page(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Fetch page content"""
        try:
            await self.rate_limiter.acquire()
            
            # In real implementation, use aiohttp
            # async with session.get(url) as response:
            #     return await response.text()
            
            # Simulated response
            await asyncio.sleep(0.5)
            return f"""
            <html>
                <head><title>Page Title</title></head>
                <body>
                    <h1>Sample Page</h1>
                    <p>Content here</p>
                    <a href="/page1">Link 1</a>
                    <a href="/page2">Link 2</a>
                    <img src="/image.jpg" />
                </body>
            </html>
            """
            
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def parse_html(self, html: str, base_url: str) -> ScrapedData:
        """Parse HTML and extract data"""
        soup = BeautifulSoup(html, 'html.parser')
        
        data = ScrapedData(url=base_url)
        
        # Extract title
        title_tag = soup.find('title')
        data.title = title_tag.text if title_tag else None
        
        # Extract text content
        for script in soup(['script', 'style']):
            script.decompose()
        data.content = soup.get_text().strip()
        
        # Extract links
        for link in soup.find_all('a', href=True):
            url = self.url_filter.normalize_url(link['href'], base_url)
            if url.startswith('http'):
                data.links.append(url)
        
        # Extract images
        for img in soup.find_all('img', src=True):
            img_url = self.url_filter.normalize_url(img['src'], base_url)
            data.images.append(img_url)
        
        # Extract meta tags
        for meta in soup.find_all('meta'):
            if meta.get('name'):
                data.meta[meta['name']] = meta.get('content', '')
        
        return data
    
    async def scrape_url(self, session: aiohttp.ClientSession, 
                        url: str, depth: int = 0) -> Optional[ScrapedData]:
        """Scrape single URL"""
        if depth > self.max_depth:
            return None
        
        if not self.url_filter.should_scrape(url):
            return None
        
        self.url_filter.mark_visited(url)
        
        logger.info(f"Scraping: {url} (depth: {depth})")
        
        html = await self.fetch_page(session, url)
        if not html:
            return None
        
        data = self.parse_html(html, url)
        self.scraped_data.append(data)
        
        # Scrape linked pages
        if depth < self.max_depth:
            tasks = []
            for link in data.links[:5]:  # Limit to 5 links per page
                if self.url_filter.should_scrape(link):
                    tasks.append(self.scrape_url(session, link, depth + 1))
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        
        return data
    
    async def scrape(self, start_urls: List[str]):
        """Main scraping method"""
        # In real implementation, create session
        # async with aiohttp.ClientSession() as session:
        session = None  # Placeholder
        
        tasks = [self.scrape_url(session, url, 0) for url in start_urls]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = [r for r in results if isinstance(r, ScrapedData)]
        
        print(f"\n📊 Scraping Results:")
        print(f"  Total pages scraped: {len(self.scraped_data)}")
        print(f"  Successful: {len(successful)}")
        print(f"  URLs visited: {len(self.url_filter.visited)}")
        
        return self.scraped_data
    
    def demo(self):
        """Demo scraper"""
        print("\n🧪 Web Scraper Demo...")
        
        async def run_demo():
            start_urls = [
                "https://example.com",
                "https://example.org"
            ]
            
            print(f"  Starting scrape of {len(start_urls)} URLs")
            print(f"  Max depth: {self.max_depth}")
            print(f"  Rate limit: 10 req/sec")
            
            await self.scrape(start_urls)
            
            # Show sample data
            if self.scraped_data:
                sample = self.scraped_data[0]
                print(f"\n  Sample scraped data:")
                print(f"    URL: {sample.url}")
                print(f"    Title: {sample.title}")
                print(f"    Links found: {len(sample.links)}")
                print(f"    Images found: {len(sample.images)}")
        
        asyncio.run(run_demo())

# ==============================================================================
# PROGETTO 4: TASK QUEUE SYSTEM
# ==============================================================================

print("\n" + "="*60)
print("📋 PROGETTO 4: TASK QUEUE SYSTEM")
print("="*60)

class TaskStatus(Enum):
    """Task status"""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    RETRY = "RETRY"

@dataclass
class Task:
    """Task definition"""
    id: str
    name: str
    func: callable
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "result": str(self.result) if self.result else None,
            "error": self.error,
            "retries": self.retries,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }

class TaskQueue:
    """Async task queue"""
    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers
        self.pending_queue: asyncio.Queue = asyncio.Queue()
        self.tasks: Dict[str, Task] = {}
        self.workers: List[asyncio.Task] = []
        self.running = False
    
    def add_task(self, name: str, func: callable, *args, **kwargs) -> Task:
        """Add task to queue"""
        task = Task(
            id=str(uuid.uuid4()),
            name=name,
            func=func,
            args=args,
            kwargs=kwargs
        )
        
        self.tasks[task.id] = task
        asyncio.create_task(self.pending_queue.put(task))
        
        logger.info(f"Task added: {task.id} - {task.name}")
        return task
    
    async def worker(self, worker_id: int):
        """Worker coroutine"""
        logger.info(f"Worker {worker_id} started")
        
        while self.running:
            try:
                # Get task with timeout
                task = await asyncio.wait_for(
                    self.pending_queue.get(),
                    timeout=1.0
                )
                
                await self.execute_task(task, worker_id)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
        
        logger.info(f"Worker {worker_id} stopped")
    
    async def execute_task(self, task: Task, worker_id: int):
        """Execute single task"""
        logger.info(f"Worker {worker_id} executing: {task.name}")
        
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        
        try:
            # Execute task
            if asyncio.iscoroutinefunction(task.func):
                result = await task.func(*task.args, **task.kwargs)
            else:
                result = await asyncio.to_thread(
                    task.func, *task.args, **task.kwargs
                )
            
            task.result = result
            task.status = TaskStatus.SUCCESS
            task.completed_at = datetime.now()
            
            logger.info(f"Task {task.id} completed successfully")
            
        except Exception as e:
            task.error = str(e)
            task.retries += 1
            
            if task.retries < task.max_retries:
                task.status = TaskStatus.RETRY
                logger.warning(f"Task {task.id} failed, retrying ({task.retries}/{task.max_retries})")
                
                # Re-queue for retry
                await asyncio.sleep(2 ** task.retries)  # Exponential backoff
                await self.pending_queue.put(task)
                
            else:
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
                logger.error(f"Task {task.id} failed after {task.retries} retries: {e}")
    
    async def start(self):
        """Start task queue"""
        if self.running:
            return
        
        self.running = True
        
        # Start workers
        for i in range(self.max_workers):
            worker = asyncio.create_task(self.worker(i))
            self.workers.append(worker)
        
        logger.info(f"Task queue started with {self.max_workers} workers")
    
    async def stop(self):
        """Stop task queue"""
        self.running = False
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        logger.info("Task queue stopped")
    
    def get_stats(self) -> Dict:
        """Get queue statistics"""
        stats = {
            "total": len(self.tasks),
            "pending": 0,
            "running": 0,
            "success": 0,
            "failed": 0,
            "retry": 0
        }
        
        for task in self.tasks.values():
            stats[task.status.value.lower()] += 1
        
        return stats
    
    async def demo(self):
        """Demo task queue"""
        print("\n🧪 Task Queue Demo...")
        
        # Define sample tasks
        async def async_task(name: str, duration: float):
            await asyncio.sleep(duration)
            return f"Async {name} completed"
        
        def sync_task(name: str, value: int):
            time.sleep(0.5)
            return f"Sync {name}: {value * 2}"
        
        def failing_task():
            raise ValueError("Task failed!")
        
        # Start queue
        await self.start()
        
        # Add tasks
        tasks = []
        for i in range(5):
            task = self.add_task(
                f"async_task_{i}",
                async_task,
                f"Task{i}",
                random.uniform(0.5, 1.5)
            )
            tasks.append(task)
        
        for i in range(3):
            task = self.add_task(
                f"sync_task_{i}",
                sync_task,
                f"Task{i}",
                i * 10
            )
            tasks.append(task)
        
        # Add failing task
        task = self.add_task("failing_task", failing_task)
        tasks.append(task)
        
        # Wait a bit
        await asyncio.sleep(5)
        
        # Show stats
        stats = self.get_stats()
        print(f"\n📊 Queue Statistics:")
        for status, count in stats.items():
            print(f"  {status}: {count}")
        
        # Show some task details
        print(f"\n📋 Sample Task Details:")
        for task in tasks[:3]:
            t = self.tasks[task.id]
            print(f"  {t.name}: {t.status.value}")
            if t.result:
                print(f"    Result: {t.result}")
        
        # Stop queue
        await self.stop()

# ==============================================================================
# MAIN - Demo di tutti i progetti
# ==============================================================================

async def main():
    """Main demo function"""
    
    print("\n" + "="*60)
    print("🚀 DEMO DEI 4 PROGETTI")
    print("="*60)
    
    # Progetto 1: API Server
    print("\n1️⃣ REST API SERVER")
    api_server = APIServer()
    await api_server.test_api()
    
    # Progetto 2: Chat System
    print("\n2️⃣ CHAT SYSTEM")
    chat_server = ChatServer()
    chat_server.demo()
    
    # Progetto 3: Web Scraper
    print("\n3️⃣ WEB SCRAPER")
    scraper = AsyncWebScraper(max_depth=2)
    scraper.demo()
    
    # Progetto 4: Task Queue
    print("\n4️⃣ TASK QUEUE")
    task_queue = TaskQueue(max_workers=3)
    await task_queue.demo()
    
    print("\n" + "="*60)
    print("✅ TUTTI I PROGETTI COMPLETATI!")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
