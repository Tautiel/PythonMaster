"""
🏗️ SESSION 4 - PART 5: SYSTEM DESIGN & ARCHITECTURE
===================================================
Design Patterns for Production ML Systems
Scalable Architecture & Best Practices

Author: Python Master Course
Level: SYSTEM ARCHITECT
"""

from typing import Dict, List, Optional, Any, Protocol, TypeVar
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio
from enum import Enum
import yaml
import json
import logging
from datetime import datetime
import hashlib
import uuid

print("=" * 80)
print("🏗️ SYSTEM DESIGN & ARCHITECTURE PATTERNS")
print("=" * 80)

# =============================================================================
# DESIGN PATTERN 1: MICROSERVICES ARCHITECTURE
# =============================================================================

print("\n" + "=" * 80)
print("📦 PATTERN 1: MICROSERVICES FOR ML")
print("=" * 80)

class MicroserviceArchitecture:
    """
    Microservices Design Pattern
    - Service decomposition
    - API Gateway
    - Service mesh
    - Event-driven communication
    """
    
    @dataclass
    class ServiceDefinition:
        """Service specification"""
        name: str
        version: str
        endpoints: List[str]
        dependencies: List[str]
        resources: Dict[str, Any]
        health_check: str
        
    def design_trading_platform(self) -> Dict:
        """Complete microservices architecture"""
        
        architecture = {
            "api_gateway": {
                "type": "Kong/AWS API Gateway",
                "features": [
                    "Rate limiting",
                    "Authentication",
                    "Load balancing",
                    "Circuit breaking"
                ]
            },
            
            "services": {
                # Data ingestion service
                "market_data_service": self.ServiceDefinition(
                    name="market-data",
                    version="v1",
                    endpoints=[
                        "/prices",
                        "/orderbook",
                        "/trades"
                    ],
                    dependencies=["redis", "kafka"],
                    resources={"cpu": "2", "memory": "4Gi"},
                    health_check="/health"
                ),
                
                # ML prediction service
                "prediction_service": self.ServiceDefinition(
                    name="ml-predictions",
                    version="v1",
                    endpoints=[
                        "/predict",
                        "/batch-predict",
                        "/model-info"
                    ],
                    dependencies=["market_data_service", "mlflow"],
                    resources={"cpu": "4", "memory": "8Gi", "gpu": "1"},
                    health_check="/health"
                ),
                
                # Trading execution service
                "execution_service": self.ServiceDefinition(
                    name="order-execution",
                    version="v1",
                    endpoints=[
                        "/place-order",
                        "/cancel-order",
                        "/order-status"
                    ],
                    dependencies=["prediction_service", "risk_service"],
                    resources={"cpu": "2", "memory": "2Gi"},
                    health_check="/health"
                ),
                
                # Risk management service
                "risk_service": self.ServiceDefinition(
                    name="risk-management",
                    version="v1",
                    endpoints=[
                        "/validate-trade",
                        "/position-limits",
                        "/exposure"
                    ],
                    dependencies=["portfolio_service"],
                    resources={"cpu": "1", "memory": "2Gi"},
                    health_check="/health"
                ),
                
                # Portfolio service
                "portfolio_service": self.ServiceDefinition(
                    name="portfolio",
                    version="v1",
                    endpoints=[
                        "/positions",
                        "/balance",
                        "/performance"
                    ],
                    dependencies=["database"],
                    resources={"cpu": "1", "memory": "1Gi"},
                    health_check="/health"
                )
            },
            
            "communication": {
                "sync": "gRPC/REST",
                "async": "Kafka/RabbitMQ",
                "service_mesh": "Istio/Linkerd"
            },
            
            "data_layer": {
                "databases": {
                    "postgres": "Transactional data",
                    "mongodb": "Document store",
                    "redis": "Cache layer",
                    "timescaledb": "Time-series data",
                    "neo4j": "Graph relationships"
                },
                "data_lake": "S3/MinIO",
                "data_warehouse": "Snowflake/BigQuery"
            },
            
            "observability": {
                "metrics": "Prometheus + Grafana",
                "logging": "ELK Stack",
                "tracing": "Jaeger/Zipkin",
                "alerting": "PagerDuty"
            }
        }
        
        return architecture

# =============================================================================
# DESIGN PATTERN 2: EVENT-DRIVEN ARCHITECTURE
# =============================================================================

print("\n" + "=" * 80)
print("📡 PATTERN 2: EVENT-DRIVEN SYSTEMS")
print("=" * 80)

class EventDrivenArchitecture:
    """
    Event-Driven Design Pattern
    - Event sourcing
    - CQRS
    - Saga pattern
    - Event streaming
    """
    
    class EventType(Enum):
        """Event types in the system"""
        MARKET_DATA_RECEIVED = "market.data.received"
        SIGNAL_GENERATED = "signal.generated"
        ORDER_PLACED = "order.placed"
        ORDER_FILLED = "order.filled"
        RISK_ALERT = "risk.alert"
        MODEL_UPDATED = "model.updated"
    
    @dataclass
    class Event:
        """Event structure"""
        event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
        event_type: str = ""
        timestamp: datetime = field(default_factory=datetime.now)
        payload: Dict = field(default_factory=dict)
        metadata: Dict = field(default_factory=dict)
        
    class EventBus:
        """Central event bus"""
        
        def __init__(self):
            self.handlers: Dict[str, List] = {}
            self.event_store = []
            
        def subscribe(self, event_type: str, handler):
            """Subscribe to events"""
            if event_type not in self.handlers:
                self.handlers[event_type] = []
            self.handlers[event_type].append(handler)
            
        async def publish(self, event: 'Event'):
            """Publish event"""
            # Store event (Event Sourcing)
            self.event_store.append(event)
            
            # Notify handlers
            if event.event_type in self.handlers:
                for handler in self.handlers[event.event_type]:
                    await handler(event)
    
    def implement_saga_pattern(self):
        """Distributed transaction management"""
        
        class TradingSaga:
            """Saga for trade execution"""
            
            def __init__(self):
                self.steps = []
                self.compensations = []
                
            async def execute_trade(self, trade_request):
                """Execute trade with saga"""
                
                try:
                    # Step 1: Validate funds
                    funds = await self.validate_funds(trade_request)
                    self.steps.append(("validate_funds", funds))
                    self.compensations.append(
                        lambda: self.release_funds(funds)
                    )
                    
                    # Step 2: Check risk
                    risk_approval = await self.check_risk(trade_request)
                    self.steps.append(("check_risk", risk_approval))
                    self.compensations.append(
                        lambda: self.cancel_risk_approval(risk_approval)
                    )
                    
                    # Step 3: Place order
                    order = await self.place_order(trade_request)
                    self.steps.append(("place_order", order))
                    self.compensations.append(
                        lambda: self.cancel_order(order)
                    )
                    
                    # Step 4: Update portfolio
                    portfolio = await self.update_portfolio(order)
                    self.steps.append(("update_portfolio", portfolio))
                    
                    return {"status": "success", "order": order}
                    
                except Exception as e:
                    # Compensate in reverse order
                    for compensation in reversed(self.compensations):
                        await compensation()
                    
                    return {"status": "failed", "error": str(e)}
        
        return TradingSaga()

# =============================================================================
# DESIGN PATTERN 3: LAMBDA ARCHITECTURE
# =============================================================================

print("\n" + "=" * 80)
print("🌊 PATTERN 3: LAMBDA ARCHITECTURE")
print("=" * 80)

class LambdaArchitecture:
    """
    Lambda Architecture for Big Data
    - Batch layer
    - Speed layer
    - Serving layer
    """
    
    def design_data_pipeline(self) -> Dict:
        """Complete Lambda architecture"""
        
        architecture = {
            "batch_layer": {
                "storage": "HDFS/S3",
                "processing": "Spark/Hadoop",
                "workflow": "Airflow/Dagster",
                "frequency": "Daily/Hourly",
                "use_cases": [
                    "Historical analysis",
                    "Model training",
                    "Backtesting",
                    "Report generation"
                ]
            },
            
            "speed_layer": {
                "ingestion": "Kafka/Kinesis",
                "processing": "Flink/Storm",
                "storage": "Redis/Cassandra",
                "latency": "< 100ms",
                "use_cases": [
                    "Real-time prices",
                    "Live predictions",
                    "Alert generation",
                    "Order execution"
                ]
            },
            
            "serving_layer": {
                "database": "HBase/DynamoDB",
                "cache": "Redis/Memcached",
                "api": "GraphQL/REST",
                "features": [
                    "Query optimization",
                    "Result caching",
                    "Load balancing",
                    "Data federation"
                ]
            },
            
            "implementation": self.implement_lambda_architecture()
        }
        
        return architecture
    
    def implement_lambda_architecture(self):
        """Implementation example"""
        
        class DataPipeline:
            """Lambda architecture implementation"""
            
            async def process_batch(self, data_path: str):
                """Batch processing"""
                # Spark job for historical data
                spark_job = """
                from pyspark.sql import SparkSession
                
                spark = SparkSession.builder \
                    .appName("BatchProcessing") \
                    .getOrCreate()
                
                # Read historical data
                df = spark.read.parquet(data_path)
                
                # Calculate indicators
                df = df.withColumn(
                    "sma_20",
                    avg("price").over(Window.partitionBy("symbol")
                        .orderBy("timestamp")
                        .rowsBetween(-19, 0))
                )
                
                # Save results
                df.write.mode("overwrite").parquet("output/")
                """
                return spark_job
            
            async def process_stream(self, topic: str):
                """Stream processing"""
                # Flink job for real-time
                flink_job = """
                from pyflink.datastream import StreamExecutionEnvironment
                
                env = StreamExecutionEnvironment.get_execution_environment()
                
                # Kafka source
                kafka_source = KafkaSource.builder() \
                    .setBootstrapServers("localhost:9092") \
                    .setTopics(topic) \
                    .build()
                
                # Process stream
                stream = env.from_source(kafka_source)
                
                # Window aggregation
                stream.key_by(lambda x: x.symbol) \
                    .window(TumblingWindow.of(Time.seconds(10))) \
                    .aggregate(calculate_metrics) \
                    .sink_to(redis_sink)
                
                env.execute("StreamProcessing")
                """
                return flink_job
        
        return DataPipeline()

# =============================================================================
# DESIGN PATTERN 4: HEXAGONAL ARCHITECTURE
# =============================================================================

print("\n" + "=" * 80)
print("🔯 PATTERN 4: HEXAGONAL ARCHITECTURE")
print("=" * 80)

class HexagonalArchitecture:
    """
    Hexagonal (Ports & Adapters) Architecture
    - Domain-driven design
    - Dependency inversion
    - Clean architecture
    """
    
    # Domain layer (core business logic)
    class TradingDomain:
        """Pure domain logic"""
        
        @dataclass
        class Trade:
            """Domain entity"""
            id: str
            symbol: str
            quantity: float
            price: float
            timestamp: datetime
            
            def calculate_value(self) -> float:
                return self.quantity * self.price
            
            def is_profitable(self, current_price: float) -> bool:
                return current_price > self.price
        
        class TradingStrategy(Protocol):
            """Domain interface (port)"""
            def generate_signal(
                self,
                market_data: Dict
            ) -> str:
                ...
        
        class RiskManager(Protocol):
            """Domain interface (port)"""
            def validate_trade(
                self,
                trade: 'Trade'
            ) -> bool:
                ...
    
    # Application layer (use cases)
    class TradingApplication:
        """Application services"""
        
        def __init__(
            self,
            strategy: 'TradingStrategy',
            risk_manager: 'RiskManager',
            repository: 'TradeRepository'
        ):
            self.strategy = strategy
            self.risk_manager = risk_manager
            self.repository = repository
        
        async def execute_trade(
            self,
            market_data: Dict
        ) -> Optional['Trade']:
            """Use case: Execute trade"""
            
            # Generate signal
            signal = self.strategy.generate_signal(market_data)
            
            if signal == "BUY":
                # Create trade
                trade = self.create_trade(market_data)
                
                # Validate risk
                if self.risk_manager.validate_trade(trade):
                    # Save trade
                    await self.repository.save(trade)
                    return trade
            
            return None
    
    # Infrastructure layer (adapters)
    class InfrastructureAdapters:
        """External adapters"""
        
        class MLStrategy:
            """ML-based strategy adapter"""
            def generate_signal(self, market_data: Dict) -> str:
                # ML model prediction
                features = self.extract_features(market_data)
                prediction = self.model.predict(features)
                return "BUY" if prediction > 0.7 else "HOLD"
        
        class DatabaseRepository:
            """Database adapter"""
            async def save(self, trade):
                async with self.db.transaction():
                    await self.db.trades.insert(trade)

# =============================================================================
# DESIGN PATTERN 5: DISTRIBUTED SYSTEM PATTERNS
# =============================================================================

print("\n" + "=" * 80)
print("🌐 PATTERN 5: DISTRIBUTED SYSTEMS")
print("=" * 80)

class DistributedPatterns:
    """
    Distributed System Design Patterns
    - Circuit breaker
    - Bulkhead
    - Retry with backoff
    - Rate limiting
    """
    
    class CircuitBreaker:
        """Circuit breaker pattern"""
        
        def __init__(
            self,
            failure_threshold: int = 5,
            recovery_timeout: int = 60
        ):
            self.failure_threshold = failure_threshold
            self.recovery_timeout = recovery_timeout
            self.failure_count = 0
            self.last_failure_time = None
            self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
        async def call(self, func, *args, **kwargs):
            """Execute with circuit breaker"""
            
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = await func(*args, **kwargs)
                self._on_success()
                return result
                
            except Exception as e:
                self._on_failure()
                raise e
        
        def _on_success(self):
            """Handle successful call"""
            self.failure_count = 0
            self.state = "CLOSED"
        
        def _on_failure(self):
            """Handle failed call"""
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
    
    class BulkheadPattern:
        """Bulkhead isolation"""
        
        def __init__(self, max_concurrent: int = 10):
            self.semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute(self, func, *args, **kwargs):
            """Execute with bulkhead"""
            async with self.semaphore:
                return await func(*args, **kwargs)
    
    class RetryWithBackoff:
        """Exponential backoff retry"""
        
        def __init__(
            self,
            max_retries: int = 3,
            base_delay: float = 1.0
        ):
            self.max_retries = max_retries
            self.base_delay = base_delay
        
        async def execute(self, func, *args, **kwargs):
            """Execute with retry"""
            for attempt in range(self.max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    
                    delay = self.base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)

# =============================================================================
# BEST PRACTICES & PRINCIPLES
# =============================================================================

print("\n" + "=" * 80)
print("📚 ARCHITECTURAL BEST PRACTICES")
print("=" * 80)

class ArchitecturalPrinciples:
    """
    SOLID, DRY, KISS, YAGNI, and more
    """
    
    # SOLID Principles
    class SOLIDPrinciples:
        """SOLID in practice"""
        
        # Single Responsibility
        class DataFetcher:
            """Only fetches data"""
            def fetch(self, source: str) -> Dict:
                pass
        
        class DataProcessor:
            """Only processes data"""
            def process(self, data: Dict) -> Dict:
                pass
        
        # Open/Closed
        class Strategy(ABC):
            """Open for extension, closed for modification"""
            @abstractmethod
            def execute(self, data: Dict) -> str:
                pass
        
        class ScalpingStrategy(Strategy):
            def execute(self, data: Dict) -> str:
                return "BUY" if data['rsi'] < 30 else "SELL"
        
        # Liskov Substitution
        class Bird(ABC):
            @abstractmethod
            def move(self):
                pass
        
        class FlyingBird(Bird):
            def move(self):
                return "fly"
        
        class SwimmingBird(Bird):
            def move(self):
                return "swim"
        
        # Interface Segregation
        class Readable(Protocol):
            def read(self) -> str:
                ...
        
        class Writable(Protocol):
            def write(self, data: str):
                ...
        
        # Dependency Inversion
        class HighLevel:
            def __init__(self, low_level: Protocol):
                self.low_level = low_level
    
    # Design Principles
    @staticmethod
    def architectural_guidelines():
        """Key architectural guidelines"""
        
        guidelines = {
            "scalability": [
                "Horizontal scaling over vertical",
                "Stateless services",
                "Database sharding",
                "Caching strategies",
                "CDN for static assets"
            ],
            
            "reliability": [
                "Redundancy at every layer",
                "Graceful degradation",
                "Health checks",
                "Circuit breakers",
                "Chaos engineering"
            ],
            
            "security": [
                "Zero trust architecture",
                "Encryption at rest and transit",
                "API rate limiting",
                "OAuth 2.0 / JWT",
                "Regular security audits"
            ],
            
            "performance": [
                "Async/await patterns",
                "Connection pooling",
                "Query optimization",
                "Lazy loading",
                "Edge computing"
            ],
            
            "maintainability": [
                "Clear documentation",
                "Consistent coding standards",
                "Automated testing",
                "CI/CD pipelines",
                "Monitoring and alerting"
            ]
        }
        
        return guidelines

# =============================================================================
# TECHNOLOGY STACK RECOMMENDATIONS
# =============================================================================

print("\n" + "=" * 80)
print("🛠️ RECOMMENDED TECH STACK")
print("=" * 80)

class TechStackRecommendations:
    """Production-ready technology choices"""
    
    @staticmethod
    def get_stack_for_use_case(use_case: str) -> Dict:
        """Recommend stack based on use case"""
        
        stacks = {
            "trading_platform": {
                "languages": ["Python", "Go", "Rust"],
                "frameworks": ["FastAPI", "Django", "Gin"],
                "databases": {
                    "timeseries": "TimescaleDB",
                    "transactional": "PostgreSQL",
                    "cache": "Redis",
                    "document": "MongoDB"
                },
                "message_queue": "Kafka",
                "ml_platform": "MLflow + Kubeflow",
                "monitoring": "Prometheus + Grafana",
                "orchestration": "Kubernetes",
                "ci_cd": "GitLab CI / GitHub Actions"
            },
            
            "ml_platform": {
                "languages": ["Python", "Scala"],
                "frameworks": ["PyTorch", "JAX", "Ray"],
                "feature_store": "Feast",
                "model_registry": "MLflow",
                "data_processing": "Spark + Dask",
                "workflow": "Airflow + Prefect",
                "serving": "TorchServe / TF Serving",
                "monitoring": "Weights & Biases",
                "infrastructure": "Kubeflow + Argo"
            },
            
            "data_pipeline": {
                "ingestion": "Kafka + Debezium",
                "processing": "Flink + Spark",
                "storage": "Delta Lake + S3",
                "warehouse": "Snowflake / BigQuery",
                "orchestration": "Dagster",
                "quality": "Great Expectations",
                "catalog": "DataHub",
                "visualization": "Superset / Tableau"
            }
        }
        
        return stacks.get(use_case, {})

# =============================================================================
# SYSTEM DESIGN INTERVIEW PATTERNS
# =============================================================================

print("\n" + "=" * 80)
print("🎯 SYSTEM DESIGN PATTERNS")
print("=" * 80)

class SystemDesignPatterns:
    """Common system design scenarios"""
    
    def design_high_frequency_trading(self) -> Dict:
        """Ultra-low latency trading system"""
        
        return {
            "requirements": {
                "latency": "< 1 microsecond",
                "throughput": "1M orders/sec",
                "availability": "99.999%"
            },
            
            "architecture": {
                "colocation": "Exchange data center",
                "networking": "Kernel bypass (DPDK)",
                "programming": "C++ / Rust",
                "memory": "Lock-free data structures",
                "storage": "In-memory only",
                "redundancy": "Active-active",
                "monitoring": "Custom hardware counters"
            },
            
            "optimizations": [
                "CPU affinity",
                "NUMA awareness",
                "Busy polling",
                "Prefetching",
                "Branch prediction",
                "Cache line optimization"
            ]
        }
    
    def design_ml_serving_platform(self) -> Dict:
        """Scalable ML serving system"""
        
        return {
            "components": {
                "gateway": "Kong / Envoy",
                "load_balancer": "HAProxy / NGINX",
                "model_server": "TorchServe clusters",
                "feature_store": "Redis + Feast",
                "monitoring": "Prometheus + Grafana",
                "logging": "ELK stack",
                "tracing": "Jaeger"
            },
            
            "scaling_strategy": {
                "horizontal": "Pod autoscaling",
                "vertical": "Resource limits",
                "geographic": "Multi-region",
                "caching": "Result caching",
                "batching": "Request batching"
            },
            
            "deployment": {
                "blue_green": "Zero downtime",
                "canary": "Gradual rollout",
                "shadow": "Traffic mirroring",
                "rollback": "Instant revert"
            }
        }

# =============================================================================
# EXECUTION & SUMMARY
# =============================================================================

def main():
    """Demonstrate all patterns"""
    print("\n" + "=" * 80)
    print("🏗️ SYSTEM ARCHITECTURE MASTERY")
    print("=" * 80)
    
    # Initialize patterns
    microservices = MicroserviceArchitecture()
    event_driven = EventDrivenArchitecture()
    lambda_arch = LambdaArchitecture()
    hexagonal = HexagonalArchitecture()
    distributed = DistributedPatterns()
    
    # Show architectures
    print("\n📋 Architecture Patterns Covered:")
    print("1️⃣ Microservices Architecture")
    print("2️⃣ Event-Driven Architecture")
    print("3️⃣ Lambda Architecture")
    print("4️⃣ Hexagonal Architecture")
    print("5️⃣ Distributed System Patterns")
    
    print("\n🎓 Key Principles Mastered:")
    print("• SOLID principles")
    print("• DRY, KISS, YAGNI")
    print("• Scalability patterns")
    print("• Reliability patterns")
    print("• Security best practices")
    
    print("\n🚀 You're Now Ready To:")
    print("• Design production systems")
    print("• Lead architecture decisions")
    print("• Pass system design interviews")
    print("• Build enterprise-grade platforms")

if __name__ == "__main__":
    main()
    
    print("\n" + "=" * 80)
    print("🎉 CONGRATULATIONS!")
    print("You've completed the FULL Python Master Course!")
    print("From basics to ML Engineer/Architect level!")
    print("=" * 80)
    print("\n🏆 YOUR ACHIEVEMENTS:")
    print("✅ Session 1: Python Mastery")
    print("✅ Session 2: Advanced Concepts")
    print("✅ Session 3: Data Science & ML")
    print("✅ Session 4: Production Systems")
    print("\n🚀 You're now a FULL-STACK ML ENGINEER!")
    print("=" * 80)
