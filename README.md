# AI Inference Gateway

A scalable, production-ready FastAPI-based service for distributed AI model inference with request batching, async processing, load balancing, monitoring, and multi-tenant support.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-00a393.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Features

- **Multi-Model Support**: Load and serve multiple AI models (text generation, embeddings, classification)
- **Request Batching**: Intelligent request aggregation for improved throughput (configurable 10-1000ms windows)
- **Async Processing**: Celery-based background task processing with retry logic
- **Authentication & Authorization**: JWT-based authentication with tiered access (Free/Pro/Enterprise)
- **Rate Limiting**: Configurable per-tier request limits with Redis counters
- **Caching**: Redis-based output caching with 24h TTL
- **Monitoring**: Prometheus metrics + Grafana dashboards
- **Multi-Tenant**: User isolation with API key management
- **Health Checks**: Comprehensive health endpoints for load balancers
- **Auto-Fallback**: Graceful degradation to backup models on failure

## Tech Stack

| Component | Technology |
|-----------|------------|
| Web Framework | FastAPI |
| Task Queue | Celery + Redis |
| Database | PostgreSQL + SQLAlchemy 2.0 |
| Cache | Redis |
| Monitoring | Prometheus + Grafana |
| Auth | JWT (python-jose) |
| ML Models | Transformers, PyTorch |

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+ (for local development)

### Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/artasyaskar/ai-inference-gateway.git
cd ai-inference-gateway

# Copy environment file
cp .env.example .env

# Start all services
docker-compose up -d

# Run migrations
docker-compose exec api alembic upgrade head

# Check health
curl http://localhost:8000/api/v1/health
```

Services will be available at:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Flower**: http://localhost:5555

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup database (requires PostgreSQL running)
# Edit .env with your DATABASE_URL
alembic upgrade head

# Start Redis (required for caching and Celery)
redis-server

# Start Celery worker
celery -A celery_worker worker --loglevel=info

# Start API server (with hot reload)
uvicorn app.main:app --reload
```

## API Documentation

### Authentication

```bash
# Login with API key to get JWT token
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"api_key": "your_api_key"}'
```

### Inference Request

```bash
# Single inference (async)
curl -X POST http://localhost:8000/api/v1/inference \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "input": "Once upon a time",
    "task_type": "text-generation",
    "parameters": {
      "max_length": 100,
      "temperature": 0.8
    }
  }'

# Check result
curl http://localhost:8000/api/v1/inference/{request_id} \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### Batch Inference

```bash
curl -X POST http://localhost:8000/api/v1/inference/batch \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "all-MiniLM-L6-v2",
    "inputs": [
      "First text to embed",
      "Second text to embed",
      "Third text to embed"
    ],
    "task_type": "embeddings",
    "batch_size": 32
  }'
```

### List Available Models

```bash
curl http://localhost:8000/api/v1/models
```

### Health Check

```bash
# Full health check
curl http://localhost:8000/api/v1/health

# Kubernetes liveness probe
curl http://localhost:8000/api/v1/health/live

# Kubernetes readiness probe
curl http://localhost:8000/api/v1/health/ready
```

## Configuration

Environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://...` | PostgreSQL connection |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `SECRET_KEY` | - | JWT signing key |
| `BATCH_SIZE` | `32` | Max requests per batch |
| `BATCH_WAIT_TIME_MS` | `100` | Max wait time for batch |
| `DEVICE` | `cpu` | `cpu`, `cuda`, or `auto` |
| `MODEL_CACHE_DIR` | `./models_cache` | Local model storage |
| `RATE_LIMIT_FREE_TIER` | `100` | Daily requests for free tier |

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│   FastAPI   │────▶│    Redis    │
│             │     │   Gateway   │     │   (Cache)   │
└─────────────┘     └──────┬──────┘     └─────────────┘
                           │
           ┌──────────────┼──────────────┐
           │              │              │
           ▼              ▼              ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │PostgreSQL│   │  Celery  │   │Prometheus│
    │  (DB)    │   │  Worker  │   │(Metrics) │
    └──────────┘   └────┬─────┘   └──────────┘
                        │
                        ▼
                 ┌─────────────┐
                 │  ML Models  │
                 │ (GPU/CPU)   │
                 └─────────────┘
```

### Key Components

1. **API Layer** (`app/api/`): REST endpoints with validation
2. **Auth Layer** (`app/auth/`): JWT authentication & middleware
3. **Inference Engine** (`app/inference/`): Model loading & execution
4. **Batch Processor** (`app/inference/batch_processor.py`): Request aggregation
5. **Celery Tasks** (`app/celery_tasks/`): Async background processing
6. **Monitoring** (`app/monitoring/`): Prometheus metrics & logging

## Adding New Models

Edit the `SUPPORTED_MODELS` environment variable:

```bash
# Format: name:path_or_hf_id:task_type
SUPPORTED_MODELS="gpt2:gpt2:text-generation,my-model:./models/custom:text-generation"
```

Or register programmatically:

```python
from app.inference.model_manager import model_manager

model_manager.register_model(
    model_id="my-model",
    model_path="organization/model-name",
    task_types=["text-generation"],
    description="My custom model"
)
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_auth.py

# Run with async support
pytest -v --asyncio-mode=auto
```

### Test Structure

- `tests/test_auth.py`: JWT and authentication tests
- `tests/test_inference.py`: API endpoint tests
- `tests/test_batch_processing.py`: Batching logic tests

## Performance Benchmarks

Expected latencies on CPU (Intel i7/AMD Ryzen 7):

| Model | Task | Input Length | Latency (ms) |
|-------|------|--------------|--------------|
| GPT-2 | Generation | 50 tokens | 150-300 |
| GPT-2 | Generation | 100 tokens | 300-600 |
| all-MiniLM | Embeddings | 128 tokens | 50-100 |
| all-MiniLM | Batch 32 | 128 tokens | 800-1200 |

Batch processing improves throughput by 3-5x for embeddings.

## Production Deployment

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-gateway
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: api
        image: ai-inference-gateway:latest
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /api/v1/health/live
            port: 8000
        readinessProbe:
          httpGet:
            path: /api/v1/health/ready
            port: 8000
```

### Scaling Considerations

1. **Horizontal Scaling**: Run multiple API instances behind a load balancer
2. **GPU Workers**: Deploy dedicated GPU nodes for model inference
3. **Database**: Use connection pooling (PgBouncer)
4. **Redis**: Cluster mode for high availability

### Security Checklist

- [ ] Change `SECRET_KEY` in production
- [ ] Use HTTPS only
- [ ] Enable rate limiting
- [ ] Configure CORS appropriately
- [ ] Use strong API keys
- [ ] Enable request validation
- [ ] Set up audit logging

## Troubleshooting

### Common Issues

**Database connection errors**
```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Run migrations
docker-compose exec api alembic upgrade head
```

**Model loading fails**
```bash
# Check model cache directory exists
mkdir -p models_cache

# Verify SUPPORTED_MODELS format
# Format: name:path:task_type
```

**Celery tasks not processing**
```bash
# Check Celery worker is running
docker-compose logs -f celery_worker

# Check Redis connection
docker-compose exec redis redis-cli ping
```

**Out of memory errors**
```bash
# Reduce batch size
BATCH_SIZE=16

# Use CPU instead of GPU
DEVICE=cpu

# Enable model unloading
# Models auto-unload after 30min idle
```

## Monitoring & Alerting

### Key Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `inference_latency_seconds` | Request latency | p95 > 5s |
| `active_requests` | Concurrent requests | > 80% of limit |
| `inference_requests_total` | Request rate | Error rate > 1% |
| `health_check_status` | Service health | < 1 (degraded) |
| `rate_limit_hits_total` | Rate limit events | Sudden spike |

### Grafana Dashboards

Access at http://localhost:3000 (admin/admin):

1. **Overview Dashboard**: System health and request rates
2. **Model Performance**: Per-model latency and throughput
3. **Resource Usage**: CPU, memory, GPU utilization

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add tests for new features
- Update documentation
- Use type hints
- Add docstrings to functions

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) for the excellent web framework
- [Hugging Face](https://huggingface.co/) for transformer models
- [Celery](https://docs.celeryproject.org/) for distributed task processing

---

**Built with ❤️ for scalable AI inference**
