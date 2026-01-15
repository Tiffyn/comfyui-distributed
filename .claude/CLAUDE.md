# ComfyUI Distributed

Distributed image generation using ComfyUI's native queue system.

## Project Structure

```
comfyui-distributed/
├── start-frontend.sh   # Launch queue frontend
├── start-worker.sh     # Launch worker node
├── .env.example        # Config template
└── README.md
```

## Key Concepts

- **Frontend**: Single instance that receives API requests and manages job queue
- **Workers**: GPU machines that pull jobs and execute workflows
- **Dynamic loading**: Workers load models on-demand based on workflow requirements

## ComfyUI Distributed Flags

```bash
--distributed-queue-frontend    # Run as queue manager
--distributed-queue-worker      # Run as worker
--distributed-queue-name HOST   # Frontend address for workers
```

## Related Projects

- ComfyUI: `/home/wile/projects/ComfyUI`
- comfyui-mcp-server: `/home/wile/projects/comfyui-mcp-server`
