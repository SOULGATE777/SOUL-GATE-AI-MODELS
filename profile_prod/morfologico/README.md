# Profile Morfologico Analysis

Advanced profile morphological analysis using a 3-model ensemble architecture.

## Features
- Bounding box detection with Faster R-CNN
- Landmark classification with custom CNN
- Anthropometric point detection
- Profile side detection (left/right)
- Advanced visualization

## Quick Start

### Local Development
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8003
