#!/bin/bash
# Start ComfyUI as distributed queue frontend

COMFYUI_PATH="${COMFYUI_PATH:-/home/wile/projects/ComfyUI}"
PORT="${PORT:-8188}"

cd "$COMFYUI_PATH"

echo "Starting ComfyUI frontend on port $PORT..."
python main.py \
    --listen 0.0.0.0 \
    --port "$PORT" \
    --distributed-queue-frontend
