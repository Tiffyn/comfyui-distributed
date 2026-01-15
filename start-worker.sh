#!/bin/bash
# Start ComfyUI as distributed queue worker

COMFYUI_PATH="${COMFYUI_PATH:-/home/wile/projects/ComfyUI}"
FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${FRONTEND_PORT:-8188}"

cd "$COMFYUI_PATH"

echo "Starting ComfyUI worker, connecting to $FRONTEND_HOST:$FRONTEND_PORT..."
python main.py \
    --distributed-queue-worker \
    --distributed-queue-name "$FRONTEND_HOST:$FRONTEND_PORT"
