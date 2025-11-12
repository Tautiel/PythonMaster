#!/usr/bin/env python3
"""
🔒 SECURITY FUNDAMENTALS MODULE
Building Secure Python Applications

Duration: 1 Week Intensive + Ongoing
Level: From Vulnerable to Secure Code
"""

import hashlib
import secrets
import hmac
import base64
import json
import re
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
import sqlite3
import jwt
import bcrypt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import pyotp
import time

# ============================================================================
# PART 1: OWASP TOP 10 IMPLEMENTATION
# ============================================================================

class OWASPTop10:
    """Implementazione pratica OWASP Top 10"""
    
    def __init__(self):
        self.vulnerabilities = {
            "A01": "Broken Access Control",
            "A02": "Cryptographic Failures", 
            "A03": "Injection",
            "A04": "Insecure Design",
            "A05": "Security Misconfiguration",
            "A06": "Vulnerable Components",
            "A07": "Authentication Failures",
            "A08": "Data Integrity Failures",
            "A09": "Logging Failures",
            "A10": "Server-Side Request Forgery"
        }
    
    def a01_broken_access_control(self):
        """A01: Broken Access Control"""
        
        print("\n🚫 A01: BROKEN ACCESS CONTROL")
        print("=" * 60)
        
        # VULNERABILE
        class VulnerableAPI:
            def get_user_data(self, user_id):
                """❌ Non verifica se user può accedere a user_id"""
                return f"SELECT * FROM users WHERE id = {user_id}"
            
            def delete_post(self, post_id):
                """❌ Non verifica ownership"""
                return f"DELETE FROM posts WHERE id = {post_id}"
        
        # SICURO
        class SecureAPI:
            def __init__(self):
                self.current_user = None
                
            def get_user_data(self, user_id, requesting_user):
                """✅ Verifica autorizzazione"""
                # Check se admin o stesso user
                if requesting_user.role == 'admin' or requesting_user.id == user_id:
                    return self._fetch_user_data(user_id)
                else:
                    raise PermissionError("Access denied")
            
            def delete_post(self, post_id, requesting_user):
                """✅ Verifica ownership"""
                post = self._get_post(post_id)
                if post.author_id == requesting_user.id or requesting_user.role == 'admin':
                    return self._delete_post(post_id)
                else:
                    raise PermissionError("Not your post")
        
        # Best Practices
        print("Prevention:")
        prevention = [
            "1. Deny by default - tutto bloccato unless explicitly allowed",
            "2. Implement RBAC (Role-Based Access Control)",
            "3. Check ownership per ogni risorsa",
            "4. Log access attempts",
            "5. Rate limiting per prevenire brute force"
        ]
        for p in prevention:
            print(f"  {p}")
        
        # Esempio RBAC
        class RoleBasedAccessControl:
            def __init__(self):
                self.permissions = {
                    'admin': ['read', 'write', 'delete', 'admin'],
                    'user': ['read', 'write'],
                    'guest': ['read']
                }
            
            def check_permission(self, user_role, action):
                """Check se role può fare action"""
                allowed_actions = self.permissions.get(user_role, [])
                return action in allowed_actions
            
            def require_permission(self, action):
                """Decorator per proteggere functions"""
                def decorator(func):
                    def wrapper(self, *args, **kwargs):
                        user = kwargs.get('user') or self.current_user
                        if not self.check_permission(user.role, action):
                            raise PermissionError(f"Action '{action}' not allowed for role '{user.role}'")
                        return func(self, *args, **kwargs)
                    return wrapper
                return decorator
        
        return RoleBasedAccessControl()
    
    def a02_cryptographic_failures(self):
        """A02: Cryptographic Failures"""
        
        print("\n🔐 A02: CRYPTOGRAPHIC FAILURES")
        print("=" * 60)
        
        # VULNERABILE
        class VulnerableCrypto:
            def store_password(self, password):
                """❌ Plain text o weak hashing"""
                # return password  # NEVER!
                # return hashlib.md5(password.encode()).hexdigest()  # WEAK!
                return hashlib.sha256(password.encode()).hexdigest()  # Still vulnerable to rainbow tables!
            
            def encrypt_data(self, data):
                """❌ Hardcoded key"""
                key = "my-secret-key-123"  # NEVER hardcode!
                # Weak encryption implementation
                return data
        
        # SICURO
        class SecureCrypto:
            def __init__(self):
                # Generate salt for each password
                self.rounds = 12  # bcrypt rounds
                
            def hash_password(self, password: str) -> str:
                """✅ Secure password hashing with bcrypt"""
                # bcrypt automatically handles salt
                salt = bcrypt.gensalt(rounds=self.rounds)
                hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
                return hashed.decode('utf-8')
            
            def verify_password(self, password: str, hashed: str) -> bool:
                """✅ Secure password verification"""
                return bcrypt.checkpw(
                    password.encode('utf-8'), 
                    hashed.encode('utf-8')
                )
            
            def generate_key(self, password: str, salt: bytes = None) -> bytes:
                """✅ Derive key from password"""
                if salt is None:
                    salt = os.urandom(16)
                
                kdf = PBKDF2(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
                return key
            
            def encrypt_data(self, data: bytes, key: bytes) -> bytes:
                """✅ Encrypt with Fernet (AES)"""
                f = Fernet(key)
                encrypted = f.encrypt(data)
                return encrypted
            
            def decrypt_data(self, encrypted: bytes, key: bytes) -> bytes:
                """✅ Decrypt data"""
                f = Fernet(key)
                decrypted = f.decrypt(encrypted)
                return decrypted
        
        # Esempio pratico
        print("\nSecure Implementation:")
        crypto = SecureCrypto()
        
        # Password hashing
        password = "MySecurePass123!"
        hashed = crypto.hash_password(password)
        print(f"Password: {password}")
        print(f"Hashed: {hashed[:20]}...")
        print(f"Verified: {crypto.verify_password(password, hashed)}")
        
        # Data encryption
        key = Fernet.generate_key()
        data = b"Sensitive trading data"
        encrypted = crypto.encrypt_data(data, key)
        decrypted = crypto.decrypt_data(encrypted, key)
        print(f"\nOriginal: {data}")
        print(f"Encrypted: {encrypted[:30]}...")
        print(f"Decrypted: {decrypted}")
        
        return crypto
    
    def a03_injection(self):
        """A03: Injection Attacks"""
        
        print("\n💉 A03: INJECTION ATTACKS")
        print("=" * 60)
        
        # SQL INJECTION
        print("SQL Injection:")
        
        # VULNERABILE
        def vulnerable_query(user_input):
            """❌ Vulnerable to SQL injection"""
            query = f"SELECT * FROM users WHERE name = '{user_input}'"
            # user_input = "admin' OR '1'='1" → Returns all users!
            return query
        
        # SICURO
        def secure_query(user_input):
            """✅ Parameterized query"""
            query = "SELECT * FROM users WHERE name = ?"
            # Execute with parameter binding
            # cursor.execute(query, (user_input,))
            return query
        
        print(f"❌ Vulnerable: {vulnerable_query(\"admin' OR '1'='1\")}")
        print(f"✅ Secure: {secure_query('admin')}")
        
        # Command Injection
        print("\nCommand Injection:")
        
        import subprocess
        
        # VULNERABILE
        def vulnerable_command(filename):
            """❌ Vulnerable to command injection"""
            # filename = "file.txt; rm -rf /"  # DANGER!
            cmd = f"cat {filename}"
            # subprocess.run(cmd, shell=True)  # NEVER with user input!
            return cmd
        
        # SICURO  
        def secure_command(filename):
            """✅ Safe command execution"""
            # Validate input
            if not re.match(r'^[\w\-\.]+$', filename):
                raise ValueError("Invalid filename")
            
            # Use list, not string
            cmd = ["cat", filename]
            # subprocess.run(cmd, shell=False)  # shell=False!
            return cmd
        
        # NoSQL Injection
        print("\nNoSQL Injection Prevention:")
        
        def secure_mongodb_query(user_input):
            """✅ Sanitize for MongoDB"""
            # Remove dangerous characters
            sanitized = re.sub(r'[\$\{\}]', '', user_input)
            
            # Use parameterized query
            query = {"username": sanitized}
            # collection.find(query)
            return query
        
        # Best Practices
        print("\n✅ Injection Prevention:")
        prevention = [
            "1. ALWAYS use parameterized queries",
            "2. Input validation with whitelisting",
            "3. Escape special characters",
            "4. Least privilege database users",
            "5. Stored procedures where appropriate",
            "6. WAF (Web Application Firewall)"
        ]
        for p in prevention:
            print(f"  {p}")

# ============================================================================
# PART 2: AUTHENTICATION & AUTHORIZATION
# ============================================================================

class SecureAuthentication:
    """Implementazione autenticazione sicura"""
    
    def __init__(self):
        self.secret_key = secrets.token_urlsafe(32)
        self.users_db = {}  # In production: use real database
        
    def secure_registration(self):
        """Registrazione sicura"""
        
        print("\n👤 SECURE REGISTRATION")
        print("=" * 60)
        
        class UserRegistration:
            def __init__(self):
                self.password_requirements = {
                    'min_length': 12,
                    'require_upper': True,
                    'require_lower': True,
                    'require_digit': True,
                    'require_special': True,
                    'check_common': True
                }
            
            def validate_password(self, password: str) -> Tuple[bool, List[str]]:
                """Validate password strength"""
                errors = []
                
                if len(password) < self.password_requirements['min_length']:
                    errors.append(f"Password must be at least {self.password_requirements['min_length']} characters")
                
                if self.password_requirements['require_upper'] and not re.search(r'[A-Z]', password):
                    errors.append("Password must contain uppercase letter")
                
                if self.password_requirements['require_lower'] and not re.search(r'[a-z]', password):
                    errors.append("Password must contain lowercase letter")
                
                if self.password_requirements['require_digit'] and not re.search(r'\d', password):
                    errors.append("Password must contain digit")
                
                if self.password_requirements['require_special'] and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
                    errors.append("Password must contain special character")
                
                # Check common passwords
                common_passwords = ['password123', 'admin123', '12345678']
                if password.lower() in common_passwords:
                    errors.append("Password is too common")
                
                return len(errors) == 0, errors
            
            def register_user(self, username: str, password: str, email: str):
                """Register new user securely"""
                # Validate input
                if not re.match(r'^[a-zA-Z0-9_]{3,20}$', username):
                    raise ValueError("Invalid username format")
                
                if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
                    raise ValueError("Invalid email format")
                
                # Validate password
                valid, errors = self.validate_password(password)
                if not valid:
                    raise ValueError(f"Password validation failed: {', '.join(errors)}")
                
                # Hash password
                salt = bcrypt.gensalt()
                hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
                
                # Store user
                user = {
                    'username': username,
                    'email': email,
                    'password_hash': hashed.decode('utf-8'),
                    'created_at': datetime.now().isoformat(),
                    'failed_attempts': 0,
                    'locked_until': None,
                    'mfa_secret': None
                }
                
                return user
        
        return UserRegistration()
    
    def jwt_authentication(self):
        """JWT token implementation"""
        
        print("\n🎫 JWT AUTHENTICATION")
        print("=" * 60)
        
        class JWTManager:
            def __init__(self, secret_key: str):
                self.secret_key = secret_key
                self.algorithm = 'HS256'
                self.access_token_expire = timedelta(minutes=15)
                self.refresh_token_expire = timedelta(days=30)
            
            def generate_tokens(self, user_id: str) -> Dict[str, str]:
                """Generate access and refresh tokens"""
                now = datetime.utcnow()
                
                # Access token
                access_payload = {
                    'user_id': user_id,
                    'type': 'access',
                    'exp': now + self.access_token_expire,
                    'iat': now,
                    'jti': secrets.token_urlsafe(16)  # JWT ID for revocation
                }
                access_token = jwt.encode(access_payload, self.secret_key, algorithm=self.algorithm)
                
                # Refresh token
                refresh_payload = {
                    'user_id': user_id,
                    'type': 'refresh',
                    'exp': now + self.refresh_token_expire,
                    'iat': now,
                    'jti': secrets.token_urlsafe(16)
                }
                refresh_token = jwt.encode(refresh_payload, self.secret_key, algorithm=self.algorithm)
                
                return {
                    'access_token': access_token,
                    'refresh_token': refresh_token,
                    'token_type': 'Bearer',
                    'expires_in': self.access_token_expire.total_seconds()
                }
            
            def verify_token(self, token: str, token_type: str = 'access') -> Optional[Dict]:
                """Verify and decode token"""
                try:
                    payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
                    
                    # Verify token type
                    if payload.get('type') != token_type:
                        raise jwt.InvalidTokenError(f"Expected {token_type} token")
                    
                    return payload
                    
                except jwt.ExpiredSignatureError:
                    return None  # Token expired
                except jwt.InvalidTokenError:
                    return None  # Invalid token
            
            def refresh_access_token(self, refresh_token: str) -> Optional[str]:
                """Use refresh token to get new access token"""
                payload = self.verify_token(refresh_token, 'refresh')
                if payload:
                    return self.generate_tokens(payload['user_id'])
                return None
        
        # Example usage
        jwt_manager = JWTManager(self.secret_key)
        tokens = jwt_manager.generate_tokens("user_123")
        
        print("Generated tokens:")
        print(f"Access Token: {tokens['access_token'][:50]}...")
        print(f"Refresh Token: {tokens['refresh_token'][:50]}...")
        print(f"Expires in: {tokens['expires_in']} seconds")
        
        return jwt_manager
    
    def multi_factor_authentication(self):
        """MFA Implementation"""
        
        print("\n🔑 MULTI-FACTOR AUTHENTICATION")
        print("=" * 60)
        
        class MFAManager:
            def setup_totp(self, user_id: str) -> Dict[str, str]:
                """Setup TOTP (Time-based One-Time Password)"""
                # Generate secret
                secret = pyotp.random_base32()
                
                # Create provisioning URI for QR code
                totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
                    name=user_id,
                    issuer_name='TradingBot'
                )
                
                return {
                    'secret': secret,
                    'uri': totp_uri,
                    'backup_codes': self.generate_backup_codes()
                }
            
            def generate_backup_codes(self, count: int = 10) -> List[str]:
                """Generate backup codes"""
                codes = []
                for _ in range(count):
                    code = ''.join(secrets.choice('0123456789') for _ in range(8))
                    codes.append(f"{code[:4]}-{code[4:]}")
                return codes
            
            def verify_totp(self, secret: str, code: str) -> bool:
                """Verify TOTP code"""
                totp = pyotp.TOTP(secret)
                # Allow 1 time window drift (30 seconds)
                return totp.verify(code, valid_window=1)
            
            def verify_backup_code(self, stored_codes: List[str], provided_code: str) -> bool:
                """Verify and consume backup code"""
                if provided_code in stored_codes:
                    stored_codes.remove(provided_code)  # One-time use
                    return True
                return False
        
        # Demo
        mfa = MFAManager()
        mfa_setup = mfa.setup_totp("user@example.com")
        
        print("MFA Setup:")
        print(f"Secret: {mfa_setup['secret']}")
        print(f"URI for QR: {mfa_setup['uri'][:50]}...")
        print(f"Backup codes: {mfa_setup['backup_codes'][:3]}...")
        
        return mfa

# ============================================================================
# PART 3: INPUT VALIDATION & SANITIZATION
# ============================================================================

class InputSecurity:
    """Input validation e sanitization"""
    
    def input_validation(self):
        """Validazione input sicura"""
        
        print("\n✅ INPUT VALIDATION")
        print("=" * 60)
        
        class InputValidator:
            def __init__(self):
                self.validators = {
                    'email': self.validate_email,
                    'username': self.validate_username,
                    'phone': self.validate_phone,
                    'url': self.validate_url,
                    'credit_card': self.validate_credit_card
                }
            
            def validate_email(self, email: str) -> bool:
                """Validate email format"""
                pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                if not re.match(pattern, email):
                    return False
                
                # Additional checks
                if len(email) > 254:  # RFC 5321
                    return False
                
                # Check for dangerous patterns
                dangerous = ['<script', 'javascript:', 'onclick']
                for danger in dangerous:
                    if danger in email.lower():
                        return False
                
                return True
            
            def validate_username(self, username: str) -> bool:
                """Validate username"""
                # Alphanumeric + underscore, 3-20 chars
                pattern = r'^[a-zA-Z0-9_]{3,20}$'
                return bool(re.match(pattern, username))
            
            def validate_phone(self, phone: str) -> bool:
                """Validate phone number"""
                # Remove common separators
                cleaned = re.sub(r'[\s\-\(\)]', '', phone)
                # Check if valid format
                pattern = r'^\+?[1-9]\d{1,14}$'  # E.164 format
                return bool(re.match(pattern, cleaned))
            
            def validate_url(self, url: str) -> bool:
                """Validate URL"""
                pattern = r'^https?://[a-zA-Z0-9\-\.]+(:[0-9]+)?(/.*)?$'
                if not re.match(pattern, url):
                    return False
                
                # Check for SSRF attempts
                dangerous_hosts = ['localhost', '127.0.0.1', '0.0.0.0', '169.254']
                for host in dangerous_hosts:
                    if host in url:
                        return False
                
                return True
            
            def validate_credit_card(self, card_number: str) -> bool:
                """Validate credit card with Luhn algorithm"""
                # Remove spaces and dashes
                card_number = re.sub(r'[\s\-]', '', card_number)
                
                if not card_number.isdigit():
                    return False
                
                # Luhn algorithm
                def luhn_check(card):
                    def digits_of(n):
                        return [int(d) for d in str(n)]
                    
                    digits = digits_of(card)
                    odd_digits = digits[-1::-2]
                    even_digits = digits[-2::-2]
                    
                    checksum = sum(odd_digits)
                    for d in even_digits:
                        checksum += sum(digits_of(d*2))
                    
                    return checksum % 10 == 0
                
                return luhn_check(card_number)
            
            def sanitize_html(self, html: str) -> str:
                """Remove dangerous HTML"""
                # Remove script tags
                html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
                # Remove event handlers
                html = re.sub(r'\bon\w+\s*=\s*["\'][^"\']*["\']', '', html)
                # Remove javascript: URLs
                html = re.sub(r'javascript:', '', html, flags=re.IGNORECASE)
                
                return html
            
            def sanitize_sql(self, input_str: str) -> str:
                """Escape SQL special characters"""
                # Better: use parameterized queries!
                replacements = {
                    "'": "''",
                    '"': '""',
                    ';': '',
                    '--': '',
                    '/*': '',
                    '*/': '',
                    'xp_': '',
                    'sp_': ''
                }
                
                for old, new in replacements.items():
                    input_str = input_str.replace(old, new)
                
                return input_str
        
        return InputValidator()

# ============================================================================
# PART 4: SECURE CODING PRACTICES
# ============================================================================

class SecureCodingPractices:
    """Best practices per codice sicuro"""
    
    def secure_file_operations(self):
        """File operations sicure"""
        
        print("\n📁 SECURE FILE OPERATIONS")
        print("=" * 60)
        
        import tempfile
        import pathlib
        
        class SecureFileHandler:
            def __init__(self, base_dir: str):
                self.base_dir = pathlib.Path(base_dir).resolve()
            
            def validate_path(self, user_path: str) -> pathlib.Path:
                """Prevent path traversal attacks"""
                # Resolve to absolute path
                requested = (self.base_dir / user_path).resolve()
                
                # Check if still within base_dir
                if not str(requested).startswith(str(self.base_dir)):
                    raise ValueError("Path traversal attempt detected!")
                
                return requested
            
            def secure_upload(self, file_data: bytes, filename: str) -> str:
                """Secure file upload"""
                # Sanitize filename
                safe_name = re.sub(r'[^a-zA-Z0-9\-\_\.]', '', filename)
                
                # Add random suffix to prevent overwrites
                suffix = secrets.token_hex(8)
                safe_name = f"{suffix}_{safe_name}"
                
                # Validate extension
                allowed_extensions = ['.jpg', '.png', '.pdf', '.txt']
                if not any(safe_name.endswith(ext) for ext in allowed_extensions):
                    raise ValueError("File type not allowed")
                
                # Check file size
                max_size = 10 * 1024 * 1024  # 10MB
                if len(file_data) > max_size:
                    raise ValueError("File too large")
                
                # Check file content (magic numbers)
                file_signatures = {
                    b'\xFF\xD8\xFF': 'jpg',
                    b'\x89PNG': 'png',
                    b'%PDF': 'pdf'
                }
                
                # Save to secure location
                file_path = self.validate_path(safe_name)
                file_path.write_bytes(file_data)
                
                return str(file_path)
            
            def secure_temp_file(self) -> str:
                """Create secure temporary file"""
                # Use system temp with secure permissions
                with tempfile.NamedTemporaryFile(
                    mode='w',
                    delete=False,
                    prefix='trading_',
                    suffix='.tmp'
                ) as tmp:
                    tmp.write("Temporary data")
                    return tmp.name
        
        return SecureFileHandler('/safe/uploads')
    
    def secure_api_design(self):
        """API Security best practices"""
        
        print("\n🔌 SECURE API DESIGN")
        print("=" * 60)
        
        class SecureAPI:
            def __init__(self):
                self.rate_limits = {}
                self.api_keys = {}
            
            def rate_limiting(self, user_id: str, limit: int = 100, window: int = 60):
                """Implement rate limiting"""
                now = time.time()
                
                if user_id not in self.rate_limits:
                    self.rate_limits[user_id] = []
                
                # Remove old requests outside window
                self.rate_limits[user_id] = [
                    timestamp for timestamp in self.rate_limits[user_id]
                    if now - timestamp < window
                ]
                
                # Check limit
                if len(self.rate_limits[user_id]) >= limit:
                    raise Exception("Rate limit exceeded")
                
                # Add current request
                self.rate_limits[user_id].append(now)
                return True
            
            def api_key_validation(self, api_key: str) -> bool:
                """Validate API key"""
                # Check format
                if not re.match(r'^[A-Za-z0-9]{32}$', api_key):
                    return False
                
                # Check if exists and not expired
                if api_key in self.api_keys:
                    key_data = self.api_keys[api_key]
                    if key_data['expires'] > datetime.now():
                        return True
                
                return False
            
            def cors_headers(self) -> Dict[str, str]:
                """Secure CORS headers"""
                return {
                    'Access-Control-Allow-Origin': 'https://trusted-domain.com',
                    'Access-Control-Allow-Methods': 'GET, POST',
                    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
                    'Access-Control-Max-Age': '3600',
                    'X-Content-Type-Options': 'nosniff',
                    'X-Frame-Options': 'DENY',
                    'X-XSS-Protection': '1; mode=block',
                    'Strict-Transport-Security': 'max-age=31536000; includeSubDomains'
                }
            
            def secure_response(self, data: Any) -> Dict[str, Any]:
                """Secure API response"""
                response = {
                    'status': 'success',
                    'data': data,
                    'timestamp': datetime.now().isoformat(),
                    'version': 'v1'
                }
                
                # Remove sensitive fields
                sensitive_fields = ['password', 'token', 'secret', 'api_key']
                
                def remove_sensitive(obj):
                    if isinstance(obj, dict):
                        return {
                            k: remove_sensitive(v) 
                            for k, v in obj.items() 
                            if k not in sensitive_fields
                        }
                    elif isinstance(obj, list):
                        return [remove_sensitive(item) for item in obj]
                    return obj
                
                response['data'] = remove_sensitive(data)
                return response
        
        return SecureAPI()

# ============================================================================
# PART 5: SECURITY PROJECTS
# ============================================================================

class SecurityProjects:
    """Progetti pratici di sicurezza"""
    
    def project_vulnerability_scanner(self):
        """Build a vulnerability scanner"""
        
        print("\n🔨 PROJECT: Vulnerability Scanner")
        print("=" * 60)
        
        class VulnerabilityScanner:
            def __init__(self):
                self.vulnerabilities = []
            
            def scan_code(self, code: str) -> List[Dict]:
                """Scan Python code for vulnerabilities"""
                issues = []
                
                # Check for hardcoded secrets
                secret_patterns = [
                    r'api_key\s*=\s*["\'][^"\']+["\']',
                    r'password\s*=\s*["\'][^"\']+["\']',
                    r'secret\s*=\s*["\'][^"\']+["\']'
                ]
                
                for pattern in secret_patterns:
                    if re.search(pattern, code, re.IGNORECASE):
                        issues.append({
                            'severity': 'HIGH',
                            'type': 'Hardcoded Secret',
                            'description': 'Found hardcoded sensitive data'
                        })
                
                # Check for SQL injection
                if 'f"SELECT' in code or "f'SELECT" in code:
                    issues.append({
                        'severity': 'CRITICAL',
                        'type': 'SQL Injection',
                        'description': 'Potential SQL injection with f-strings'
                    })
                
                # Check for unsafe eval
                if 'eval(' in code:
                    issues.append({
                        'severity': 'CRITICAL',
                        'type': 'Code Injection',
                        'description': 'Use of eval() is dangerous'
                    })
                
                # Check for weak crypto
                weak_crypto = ['md5', 'sha1', 'DES']
                for crypto in weak_crypto:
                    if crypto in code:
                        issues.append({
                            'severity': 'MEDIUM',
                            'type': 'Weak Cryptography',
                            'description': f'Use of weak algorithm: {crypto}'
                        })
                
                return issues
            
            def scan_dependencies(self) -> List[Dict]:
                """Check for vulnerable dependencies"""
                # In real implementation: check against CVE database
                pass
            
            def generate_report(self, issues: List[Dict]) -> str:
                """Generate security report"""
                report = "SECURITY SCAN REPORT\n" + "="*50 + "\n"
                
                by_severity = {'CRITICAL': [], 'HIGH': [], 'MEDIUM': [], 'LOW': []}
                for issue in issues:
                    by_severity[issue['severity']].append(issue)
                
                for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
                    if by_severity[severity]:
                        report += f"\n{severity} ({len(by_severity[severity])} issues)\n"
                        for issue in by_severity[severity]:
                            report += f"  - {issue['type']}: {issue['description']}\n"
                
                return report
        
        print("Scanner features to implement:")
        print("✅ Code analysis for vulnerabilities")
        print("✅ Dependency checking")
        print("✅ Configuration review")
        print("✅ Report generation")
        
        return VulnerabilityScanner()

# ============================================================================
# EXERCISES
# ============================================================================

def security_exercises():
    """30 security exercises"""
    
    print("\n🔒 SECURITY EXERCISES")
    print("=" * 60)
    
    exercises = {
        "Authentication (1-10)": [
            "Implement secure password hashing with bcrypt",
            "Create JWT authentication system",
            "Add MFA with TOTP",
            "Build account lockout mechanism",
            "Implement OAuth2 flow",
            "Create password reset with tokens",
            "Add biometric authentication",
            "Implement session management",
            "Build role-based access control",
            "Create API key system"
        ],
        
        "Input Security (11-20)": [
            "Fix SQL injection vulnerability",
            "Sanitize HTML input",
            "Validate email addresses",
            "Prevent path traversal",
            "Implement CSRF protection",
            "Add XSS prevention",
            "Validate file uploads",
            "Sanitize JSON input",
            "Prevent XXE attacks",
            "Implement rate limiting"
        ],
        
        "Cryptography (21-30)": [
            "Encrypt sensitive data at rest",
            "Implement TLS communication",
            "Create secure random tokens",
            "Build encryption key management",
            "Implement digital signatures",
            "Create secure password generator",
            "Build data masking system",
            "Implement secret sharing",
            "Create secure backup system",
            "Build audit logging system"
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
    """Run security module"""
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║                 🔒 SECURITY FUNDAMENTALS MODULE             ║
    ║                  From Vulnerable to Secure Code             ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    modules = {
        "1": ("OWASP Top 10", OWASPTop10),
        "2": ("Authentication", SecureAuthentication),
        "3": ("Input Security", InputSecurity),
        "4": ("Secure Coding", SecureCodingPractices),
        "5": ("Projects", SecurityProjects),
        "6": ("Exercises", security_exercises)
    }
    
    while True:
        print("\n📚 SELECT MODULE:")
        for key, (name, _) in modules.items():
            print(f"  {key}. {name}")
        print("  Q. Quit")
        
        choice = input("\nChoice: ").strip().upper()
        
        if choice == 'Q':
            break
        elif choice == '6':
            security_exercises()
        else:
            # Run module demonstrations
            pass

if __name__ == "__main__":
    main()
