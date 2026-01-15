# ComfyUI Distributed

Scale ComfyUI across multiple GPU machines using the native distributed queue.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend                               │
│              (receives API requests, manages queue)         │
│                    port 8188                                │
└─────────────────────────┬───────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
   ┌────▼────┐       ┌────▼────┐       ┌────▼────┐
   │ Worker 1│       │ Worker 2│       │ Worker N│
   │  (GPU)  │       │  (GPU)  │       │  (GPU)  │
   └─────────┘       └─────────┘       └─────────┘
```

Workers pull jobs from the queue and dynamically load models as needed.

## Quick Start

### Frontend (Main Machine)

```bash
git clone https://github.com/Tiffyn/comfyui-distributed.git
cd comfyui-distributed
./start-frontend.sh
```

### Worker (GPU Machines)

```bash
git clone https://github.com/Tiffyn/comfyui-distributed.git
cd comfyui-distributed
FRONTEND_HOST=<frontend-ip> ./start-worker.sh
```

## Configuration

Set environment variables or create `.env` file (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `COMFYUI_PATH` | `/home/wile/projects/ComfyUI` | Path to ComfyUI installation |
| `PORT` | `8188` | Frontend port |
| `FRONTEND_HOST` | `127.0.0.1` | Frontend IP (for workers) |
| `FRONTEND_PORT` | `8188` | Frontend port (for workers) |

## API

Submit workflows to the frontend:

```bash
curl -X POST http://<frontend>:8188/prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt": <workflow-api-json>}'
```

Check queue status:
```bash
curl http://<frontend>:8188/queue
```

## Requirements

- Python 3.10+
- ComfyUI installed on all machines
- Models available on workers (sync or shared storage)
- Network access from workers to frontend on port 8188

## Model Sync

Workers need access to the same models. Options:

1. **Shared NFS/SMB mount** - Mount models directory from central storage
2. **Rsync** - Sync models directory to each worker
3. **Symlink** - Point to existing model directories

## Troubleshooting

**Worker can't connect to frontend:**
- Check firewall allows port 8188
- Verify `FRONTEND_HOST` is correct IP (not localhost)
- Ensure frontend started with `--listen 0.0.0.0`

**Model not found on worker:**
- Sync models to worker machine
- Check `models/` directory structure matches frontend
