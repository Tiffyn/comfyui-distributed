# ComfyUI Distributed

Distributed image generation using ComfyUI's native queue system.

## Architecture

```
Frontend (queue manager) <--- API requests
    │
    ├── Worker 1 (GPU box)
    ├── Worker 2 (GPU box)
    └── Worker N (GPU box)
```

Workers dynamically load models based on workflow requirements.

## Setup

### Frontend (Main Machine)

```bash
./start-frontend.sh
```

Or manually:
```bash
cd /path/to/ComfyUI
python main.py --listen 0.0.0.0 --distributed-queue-frontend
```

### Worker (GPU Machines)

```bash
FRONTEND_HOST=192.168.1.100 ./start-worker.sh
```

Or manually:
```bash
cd /path/to/ComfyUI
python main.py --distributed-queue-worker --distributed-queue-name <frontend-ip>:8188
```

## API Usage

Submit workflows to the frontend:

```bash
curl -X POST http://frontend:8188/prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt": <workflow-json>}'
```

## Requirements

- ComfyUI installed on all machines
- Models synced across workers (or shared storage)
- Network connectivity between frontend and workers
