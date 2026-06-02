# Testing Guide - SOUL-GATE-AI-MODELS

**Last Updated**: 2025-12-12

---

## Testing Framework

**Framework**: pytest + pytest-asyncio  
**Coverage Tool**: pytest-cov  
**Target Coverage**: 80%+ para código nuevo

---

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Run specific file
pytest tests/test_main.py

# Run with coverage
pytest --cov=app --cov-report=html

# Verbose
pytest -v

# Stop on first failure
pytest -x
```

---

## Test Structure

```
service/
├── app/
│   └── main.py
└── tests/
    ├── __init__.py
    ├── conftest.py        # Fixtures
    ├── test_main.py       # Endpoint tests
    ├── test_pipeline.py   # ML pipeline tests
    └── fixtures/
        └── test_image.jpg # Test data
```

---

## Required Tests

### 1. Health Endpoint (CRITICAL)

```python
from fastapi.testclient import TestClient

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

### 2. Model Loading

```python
def test_pipeline_initialization():
    pipeline = Pipeline()
    assert pipeline.model is not None
    assert pipeline.device.type in ['cuda', 'cpu']
```

### 3. Basic Inference

```python
@pytest.mark.asyncio
async def test_basic_inference():
    result = await pipeline.analyze(dummy_image)
    assert result is not None
```

---

## Fixtures

```python
# conftest.py
import pytest

@pytest.fixture
def pipeline():
    return Pipeline(model_path="models/test_model.pth")

@pytest.fixture
def dummy_image():
    return np.zeros((224, 224, 3), dtype=np.uint8)
```

---

## Integration Tests

```python
@pytest.mark.integration
def test_full_workflow():
    with open("tests/fixtures/test_face.jpg", "rb") as f:
        response = client.post("/analyze", files={"file": f})
    
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
```

---

## Parametrize

```python
@pytest.mark.parametrize("confidence,expected", [
    (0.3, 5),
    (0.5, 3),
    (0.7, 1),
])
def test_thresholds(confidence, expected):
    result = pipeline.analyze(image, confidence_threshold=confidence)
    assert len(result["detections"]) == expected
```

---

## Test Markers

```python
# pytest.ini
[tool:pytest]
markers =
    slow: slow tests
    integration: integration tests
    gpu: requires GPU

# Usage
pytest -m "not slow"  # Skip slow tests
pytest -m gpu         # Only GPU tests
```

---

## Coverage

```bash
# Generate HTML report
pytest --cov=app --cov-report=html

# Open in browser
open htmlcov/index.html

# Fail if coverage < 80%
pytest --cov=app --cov-fail-under=80
```

---

## Mocking

```python
from unittest.mock import Mock, patch

def test_with_mock():
    with patch('app.models.pipeline.Model') as MockModel:
        mock_instance = MockModel.return_value
        mock_instance.predict.return_value = {"result": "mocked"}
        # ... test code ...
```

---

## Best Practices

1. **Test isolation**: Cada test independiente
2. **Fast tests**: Usar mocks para tests rápidos
3. **Clear names**: `test_health_returns_200_ok()`
4. **One assertion focus**: Cada test verifica una cosa
5. **Fixtures**: Reusa setup común

---

**Testing es esencial. Escribe tests para código nuevo.**

