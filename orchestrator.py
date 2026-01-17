#!/usr/bin/env python
"""
Distributed Batch Orchestrator
Central server that coordinates multiple batch processing nodes.
Provides web dashboard for monitoring and queue management.
"""

import os
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from dotenv import load_dotenv

# Template/relationship features removed - using external microservice

load_dotenv()

app = Flask(__name__)
CORS(app)

# ============== Configuration ==============
RECIPES_PATH = os.getenv("RECIPES_PATH", r"C:\Users\dhruv\Downloads\ingredients_flux2_v12_packaging_bowl_final.json")
NODE_TIMEOUT = 60  # Seconds before node considered dead
HEARTBEAT_INTERVAL = 10  # Expected heartbeat interval
MANIFEST_BATCH_SIZE = 10  # Sync manifest to R2 every N completions

# R2 Configuration
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET = os.getenv("R2_BUCKET", "tiffyn-test")
R2_PUBLIC_URL = os.getenv("R2_PUBLIC_URL", "https://pub-1c778b9e0f364547b6d2c459a25da054.r2.dev")

# ============== R2/S3 Client ==============
_s3_client = None

def get_s3_client():
    """Get or create S3 client for R2"""
    global _s3_client
    if _s3_client is None:
        if not all([R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY]):
            print("Warning: R2 credentials not configured - manifest sync disabled")
            return None
        try:
            import boto3
            _s3_client = boto3.client(
                's3',
                endpoint_url=f'https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com',
                aws_access_key_id=R2_ACCESS_KEY_ID,
                aws_secret_access_key=R2_SECRET_ACCESS_KEY,
                region_name='auto'
            )
        except ImportError:
            print("Warning: boto3 not installed - manifest sync disabled")
            return None
    return _s3_client

# ============== Manifest Sync (Split by Cuisine) ==============

def normalize_cuisine(cuisine):
    """Normalize cuisine name for file paths"""
    if not cuisine:
        return "uncategorized"
    return cuisine.lower().strip().replace(" ", "_").replace("-", "_")


def sync_manifest_to_r2(force=False):
    """Sync pending manifest entries to R2 (split by cuisine)

    Structure:
      data/manifest.json - Master index
      data/manifests/{cuisine}.json - Per-cuisine manifests
    """
    global _pending_manifest_entries

    with _manifest_lock:
        if not force and len(_pending_manifest_entries) < MANIFEST_BATCH_SIZE:
            return False  # Not enough entries yet

        if not _pending_manifest_entries:
            return False  # Nothing to sync

        entries_to_sync = _pending_manifest_entries.copy()
        _pending_manifest_entries = []

    s3 = get_s3_client()
    if s3 is None:
        return False

    try:
        # Group entries by cuisine
        from collections import defaultdict
        entries_by_cuisine = defaultdict(list)
        for entry in entries_to_sync:
            cuisine = normalize_cuisine(entry.get("cuisine"))
            entries_by_cuisine[cuisine].append(entry)

        # Download master manifest
        try:
            response = s3.get_object(Bucket=R2_BUCKET, Key="data/manifest.json")
            master = json.loads(response['Body'].read().decode('utf-8'))
            # Handle old flat format
            if isinstance(master, list):
                master = {"version": 2, "cuisines": {}, "total_images": 0, "updated_at": ""}
        except Exception:
            master = {"version": 2, "cuisines": {}, "total_images": 0, "updated_at": ""}

        total_new = 0

        # Update each cuisine's manifest
        for cuisine, new_entries in entries_by_cuisine.items():
            cuisine_key = f"data/manifests/{cuisine}.json"

            # Download existing cuisine manifest
            try:
                response = s3.get_object(Bucket=R2_BUCKET, Key=cuisine_key)
                cuisine_manifest = json.loads(response['Body'].read().decode('utf-8'))
                # Index by unique key
                cuisine_dict = {f"{item['id']}_{item['model']}": item for item in cuisine_manifest}
            except Exception:
                cuisine_dict = {}

            # Merge new entries
            for entry in new_entries:
                key = f"{entry['id']}_{entry['model']}"
                cuisine_dict[key] = entry
                total_new += 1

            # Sort by generated_at (newest first) and upload
            cuisine_manifest = sorted(
                cuisine_dict.values(),
                key=lambda x: x.get("generated_at", ""),
                reverse=True
            )

            s3.put_object(
                Bucket=R2_BUCKET,
                Key=cuisine_key,
                Body=json.dumps(cuisine_manifest, indent=2, ensure_ascii=False).encode('utf-8'),
                ContentType='application/json'
            )

            # Update master manifest entry for this cuisine
            latest = max((item.get("generated_at", "") for item in cuisine_manifest), default="")
            master["cuisines"][cuisine] = {
                "count": len(cuisine_manifest),
                "path": f"manifests/{cuisine}.json",
                "updated_at": latest
            }

        # Update master manifest totals
        master["total_images"] = sum(c["count"] for c in master["cuisines"].values())
        master["updated_at"] = datetime.now().isoformat()

        # Upload master manifest
        s3.put_object(
            Bucket=R2_BUCKET,
            Key="data/manifest.json",
            Body=json.dumps(master, indent=2, ensure_ascii=False).encode('utf-8'),
            ContentType='application/json'
        )

        cuisines_updated = list(entries_by_cuisine.keys())
        print(f"  [Manifest synced: {total_new} entries to {len(cuisines_updated)} cuisines: {', '.join(cuisines_updated)}]")
        return True

    except Exception as e:
        print(f"  Warning: Failed to sync manifest: {e}")
        # Put entries back for retry
        with _manifest_lock:
            _pending_manifest_entries = entries_to_sync + _pending_manifest_entries
        return False


def queue_manifest_entry(task, result):
    """Queue a completed task for manifest sync"""
    # Handle both numeric and string IDs
    try:
        entry_id = int(task["recipe_id"])
    except (ValueError, TypeError):
        entry_id = task["recipe_id"]

    entry = {
        "id": entry_id,
        "data_type": "meals",
        "meal_name": task["meal_name"],
        "cuisine": task["cuisine"],
        "image_url": result.get("r2_url", ""),
        "prompt": task.get("prompt", ""),
        "model": result.get("model", task.get("model", "")),
        "generated_at": datetime.now().isoformat(),
        "generation_count": 1,
    }

    # Include slug if provided (used for filename)
    if task.get("slug"):
        entry["slug"] = task["slug"]

    with _manifest_lock:
        _pending_manifest_entries.append(entry)
        pending_count = len(_pending_manifest_entries)

    # Check if we should sync
    if pending_count >= MANIFEST_BATCH_SIZE:
        sync_manifest_to_r2()


# ============== State Management ==============
_lock = threading.Lock()
_manifest_lock = threading.Lock()
_pending_manifest_entries = []

# Registered nodes: {node_id: {name, gpu, last_heartbeat, status, current_task, stats}}
nodes = {}

# Task queue: [{id, recipe_id, prompt, cuisine, status, assigned_to, ...}]
task_queue = []

# Completed tasks for tracking
completed_tasks = {}
failed_tasks = {}

# Settings
settings = {
    "models": ["flux2_dev", "flux2_turbo", "flux2_klein", "zimage_turbo", "comfyui_flux", "qwen_turbo"],  # Active models for queue
    "auto_assign": True,
    "paused": False,
}


def load_recipes():
    """Load recipes from JSON file"""
    if not os.path.exists(RECIPES_PATH):
        return []
    with open(RECIPES_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def structured_prompt_to_text(structured):
    """Convert structured prompt object to text prompt

    Handles flux2_structured_prompt format with:
    - scene, subjects (description, position, action, constraints)
    - style, mood, background, lighting, composition
    - camera (angle, lens, depth_of_field)
    """
    if not structured or not isinstance(structured, dict):
        return ""

    parts = []

    # Scene description
    if structured.get("scene"):
        parts.append(structured["scene"])

    # Subject descriptions
    subjects = structured.get("subjects", [])
    for subj in subjects:
        if subj.get("description"):
            parts.append(subj["description"])
        if subj.get("position"):
            parts.append(subj["position"])
        if subj.get("action"):
            parts.append(subj["action"])
        if subj.get("constraints"):
            parts.append(subj["constraints"])

    # Style and mood
    if structured.get("style"):
        parts.append(structured["style"])
    if structured.get("mood"):
        parts.append(f"Mood: {structured['mood']}")

    # Background and lighting
    if structured.get("background"):
        parts.append(structured["background"])
    if structured.get("lighting"):
        parts.append(structured["lighting"])

    # Composition and camera
    if structured.get("composition"):
        parts.append(structured["composition"])
    camera = structured.get("camera", {})
    if camera.get("angle"):
        parts.append(f"Camera angle: {camera['angle']}")
    if camera.get("lens"):
        parts.append(f"Lens: {camera['lens']}")
    if camera.get("depth_of_field"):
        parts.append(camera["depth_of_field"])

    return " ".join(parts)


def init_queue(recipes=None, models=None):
    """Initialize task queue from recipes for multiple models"""
    global task_queue, completed_tasks, failed_tasks

    if recipes is None:
        recipes = load_recipes()

    if models is None:
        models = settings["models"]

    # Ensure models is a list
    if isinstance(models, str):
        models = [models]

    with _lock:
        task_queue = []
        for model in models:
            for recipe in recipes:
                recipe_id = str(recipe["id"])
                task_key = f"{model}:{recipe_id}"

                # Skip if already completed for this model
                if task_key in completed_tasks:
                    continue

                # Extract prompt - check for structured prompt first
                prompt = ""
                if recipe.get("flux2_structured_prompt"):
                    prompt = structured_prompt_to_text(recipe["flux2_structured_prompt"])
                elif recipe.get("flux_prompt"):
                    prompt = recipe["flux_prompt"]
                elif recipe.get("image_prompt"):
                    prompt = recipe["image_prompt"]

                task = {
                    "id": task_key,
                    "recipe_id": recipe_id,
                    "meal_name": recipe.get("meal_name") or recipe.get("ingredient_name", f"Item {recipe_id}"),
                    "cuisine": recipe.get("cuisine_name") or recipe.get("category", "general"),
                    "category": recipe.get("category", ""),
                    "packaging_type": recipe.get("packaging_type", ""),
                    "prompt": prompt,
                    "seed": recipe.get("recommended_seed", -1),
                    "model": model,
                    "status": "pending",  # pending, assigned, generating, completed, failed
                    "assigned_to": None,
                    "assigned_at": None,
                    "created_at": datetime.now().isoformat(),
                }
                task_queue.append(task)

    return len(task_queue)


def get_next_task(node_id, node_models=None):
    """Get next available task for a node (filtered by node's supported models)

    Priority: Fresh tasks first, then retried tasks (sorted by retry_count ascending)
    """
    with _lock:
        if settings["paused"]:
            return None

        # Get node's supported models if not provided
        if node_models is None and node_id in nodes:
            node_models = nodes[node_id].get("models", [])

        # Convert single model to list for backwards compatibility
        if isinstance(node_models, str):
            node_models = [node_models]

        # First pass: look for fresh tasks (no retries)
        for task in task_queue:
            if task["status"] == "pending" and task.get("retry_count", 0) == 0:
                # If node has models specified, only assign matching tasks
                if node_models and task["model"] not in node_models:
                    continue

                task["status"] = "assigned"
                task["assigned_to"] = node_id
                task["assigned_at"] = datetime.now().isoformat()
                return task

        # Second pass: get retried tasks (lowest retry_count first)
        retried_tasks = [t for t in task_queue
                        if t["status"] == "pending" and t.get("retry_count", 0) > 0]
        retried_tasks.sort(key=lambda t: t.get("retry_count", 0))

        for task in retried_tasks:
            if node_models and task["model"] not in node_models:
                continue

            task["status"] = "assigned"
            task["assigned_to"] = node_id
            task["assigned_at"] = datetime.now().isoformat()
            return task

    return None


def complete_task(task_id, result):
    """Mark task as completed and queue for manifest sync"""
    task_copy = None
    queue_is_empty = False
    with _lock:
        for task in task_queue:
            if task["id"] == task_id:
                task["status"] = "completed"
                task["completed_at"] = datetime.now().isoformat()
                task["result"] = result
                completed_tasks[task["recipe_id"]] = task
                task_copy = task.copy()  # Copy before removing
                task_queue.remove(task)
                # Check if queue is now empty (no pending/assigned tasks)
                queue_is_empty = len(task_queue) == 0
                break

    if task_copy:
        # Queue for manifest sync (outside lock to avoid deadlock)
        queue_manifest_entry(task_copy, result)

        # Force sync if queue is empty to ensure final entries are saved
        # The batch sync inside queue_manifest_entry may have already synced
        # if we hit MANIFEST_BATCH_SIZE, but force=True handles that gracefully
        if queue_is_empty:
            sync_manifest_to_r2(force=True)
            print("  [Queue empty - final manifest sync complete]")

        return True
    return False


def fail_task(task_id, error, node_id=None):
    """Mark task as failed - requeue with lower priority for retry"""
    with _lock:
        for task in task_queue:
            if task["id"] == task_id:
                # Track retry count and last error
                task["retry_count"] = task.get("retry_count", 0) + 1
                task["last_error"] = error
                task["last_failed_at"] = datetime.now().isoformat()
                task["last_failed_by"] = node_id

                # Reset to pending for retry, but with lower priority
                task["status"] = "pending"
                task["assigned_to"] = None
                task["assigned_at"] = None

                # Track in failed_tasks for reporting
                failed_tasks[task_id] = {
                    "task_id": task_id,
                    "error": error,
                    "retry_count": task["retry_count"],
                    "failed_at": task["last_failed_at"],
                }
                return True
    return False


def cleanup_stale_tasks():
    """Reassign tasks from dead nodes"""
    now = time.time()
    with _lock:
        dead_nodes = [nid for nid, n in nodes.items()
                      if now - n.get("last_heartbeat", 0) > NODE_TIMEOUT]

        for task in task_queue:
            if task["status"] == "assigned" and task["assigned_to"] in dead_nodes:
                task["status"] = "pending"
                task["assigned_to"] = None
                task["assigned_at"] = None


# Background cleanup thread
def cleanup_thread():
    sync_counter = 0
    while True:
        time.sleep(30)
        cleanup_stale_tasks()

        # Force sync manifest every 60 seconds (2 iterations) to flush any remaining entries
        sync_counter += 1
        if sync_counter >= 2:
            sync_manifest_to_r2(force=True)
            sync_counter = 0

threading.Thread(target=cleanup_thread, daemon=True).start()


# ============== API Endpoints ==============

@app.route("/api/register", methods=["POST"])
def register_node():
    """Register a new node"""
    data = request.json
    node_id = data.get("node_id")

    # Support both single model and list of models
    models = data.get("models") or data.get("model")
    if isinstance(models, str):
        models = [models]

    with _lock:
        nodes[node_id] = {
            "name": data.get("name", node_id),
            "gpu": data.get("gpu", "Unknown"),
            "vram_gb": data.get("vram_gb", 0),
            "models": models or [],  # List of models this node can process
            "last_heartbeat": time.time(),
            "status": "idle",
            "current_task": None,
            "current_model": None,  # Track which model is currently loaded
            "stats": {
                "completed": 0,
                "failed": 0,
                "total_time": 0,
            },
            "registered_at": datetime.now().isoformat(),
        }

    return jsonify({"status": "registered", "node_id": node_id, "models": models})


@app.route("/api/heartbeat", methods=["POST"])
def heartbeat():
    """Receive heartbeat from node"""
    data = request.json
    node_id = data.get("node_id")

    with _lock:
        if node_id not in nodes:
            return jsonify({"error": "Node not registered"}), 404

        nodes[node_id]["last_heartbeat"] = time.time()
        nodes[node_id]["status"] = data.get("status", "idle")
        nodes[node_id]["current_task"] = data.get("current_task")
        nodes[node_id]["current_model"] = data.get("current_model")

        if "gpu_util" in data:
            nodes[node_id]["gpu_util"] = data["gpu_util"]
        if "vram_used" in data:
            nodes[node_id]["vram_used"] = data["vram_used"]

    return jsonify({"status": "ok", "paused": settings["paused"]})


@app.route("/api/task/request", methods=["POST"])
def request_task():
    """Node requests next task"""
    data = request.json
    node_id = data.get("node_id")

    if node_id not in nodes:
        return jsonify({"error": "Node not registered"}), 404

    task = get_next_task(node_id)
    if task:
        with _lock:
            nodes[node_id]["status"] = "generating"
            nodes[node_id]["current_task"] = task["id"]
        return jsonify({"task": task})

    return jsonify({"task": None, "message": "No tasks available"})


@app.route("/api/task/complete", methods=["POST"])
def task_complete():
    """Node reports task completion"""
    data = request.json
    node_id = data.get("node_id")
    task_id = data.get("task_id")
    result = data.get("result", {})

    if complete_task(task_id, result):
        with _lock:
            if node_id in nodes:
                nodes[node_id]["status"] = "idle"
                nodes[node_id]["current_task"] = None
                nodes[node_id]["stats"]["completed"] += 1
                if "generation_time" in result:
                    nodes[node_id]["stats"]["total_time"] += result["generation_time"]
        return jsonify({"status": "ok"})

    return jsonify({"error": "Task not found"}), 404


@app.route("/api/task/fail", methods=["POST"])
def task_fail():
    """Node reports task failure - task is requeued with lower priority"""
    data = request.json
    node_id = data.get("node_id")
    task_id = data.get("task_id")
    error = data.get("error", "Unknown error")

    if fail_task(task_id, error, node_id):
        with _lock:
            if node_id in nodes:
                nodes[node_id]["status"] = "idle"
                nodes[node_id]["current_task"] = None
                nodes[node_id]["stats"]["failed"] += 1
        return jsonify({"status": "ok", "requeued": True})

    return jsonify({"error": "Task not found"}), 404


@app.route("/api/task/progress", methods=["POST"])
def task_progress():
    """Node reports task progress (0-100)"""
    data = request.json
    node_id = data.get("node_id")
    task_id = data.get("task_id")
    progress = data.get("progress", 0)
    step = data.get("step")
    total_steps = data.get("total_steps")

    with _lock:
        for task in task_queue:
            if task["id"] == task_id:
                task["progress"] = progress
                if step is not None:
                    task["step"] = step
                if total_steps is not None:
                    task["total_steps"] = total_steps
                return jsonify({"status": "ok"})

    return jsonify({"error": "Task not found"}), 404


@app.route("/api/status")
def status():
    """Get overall status"""
    now = time.time()

    with _lock:
        active_nodes = sum(1 for n in nodes.values()
                          if now - n.get("last_heartbeat", 0) < NODE_TIMEOUT)

        pending = sum(1 for t in task_queue if t["status"] == "pending")
        assigned = sum(1 for t in task_queue if t["status"] == "assigned")
        generating = sum(1 for t in task_queue if t["status"] == "generating")

        return jsonify({
            "nodes": {
                "total": len(nodes),
                "active": active_nodes,
            },
            "tasks": {
                "pending": pending,
                "assigned": assigned,
                "generating": generating,
                "completed": len(completed_tasks),
                "failed": len(failed_tasks),
                "total": pending + assigned + generating + len(completed_tasks),
            },
            "settings": settings,
            "uptime": time.time(),
        })


@app.route("/api/nodes")
def get_nodes():
    """Get all node details"""
    now = time.time()
    with _lock:
        result = []
        for node_id, node in nodes.items():
            node_info = node.copy()
            node_info["node_id"] = node_id
            node_info["online"] = (now - node.get("last_heartbeat", 0)) < NODE_TIMEOUT
            node_info["last_seen"] = int(now - node.get("last_heartbeat", now))
            result.append(node_info)
        return jsonify(result)


@app.route("/api/queue")
def get_queue():
    """Get current task queue (first 100)"""
    with _lock:
        return jsonify(task_queue[:100])


@app.route("/api/settings", methods=["GET", "POST"])
def manage_settings():
    """Get or update settings"""
    if request.method == "POST":
        data = request.json
        with _lock:
            if "model" in data:
                settings["model"] = data["model"]
            if "paused" in data:
                settings["paused"] = data["paused"]
            if "auto_assign" in data:
                settings["auto_assign"] = data["auto_assign"]
        return jsonify(settings)

    return jsonify(settings)


@app.route("/api/queue/init", methods=["POST"])
def init_queue_endpoint():
    """Initialize queue from recipes for one or more models"""
    data = request.json or {}
    models = data.get("models", settings["models"])
    count = init_queue(models=models)
    return jsonify({"status": "ok", "tasks_added": count, "models": models})


@app.route("/api/queue/upload", methods=["POST"])
def upload_queue():
    """Upload recipes JSON and initialize queue

    POST body should be:
    {
        "recipes": [...],  // Array of recipe objects with id, prompt, category fields
        "models": ["flux2_dev", "zimage_turbo"]  // Optional, defaults to settings
    }
    """
    data = request.json
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    recipes = data.get("recipes")
    if not recipes:
        return jsonify({"error": "No recipes array provided"}), 400

    if not isinstance(recipes, list):
        return jsonify({"error": "recipes must be an array"}), 400

    models = data.get("models", settings["models"])
    if isinstance(models, str):
        models = [models]

    # Initialize queue with uploaded recipes
    count = init_queue(recipes=recipes, models=models)
    return jsonify({
        "status": "ok",
        "tasks_added": count,
        "models": models,
        "recipes_count": len(recipes)
    })


@app.route("/api/queue/clear", methods=["POST"])
def clear_queue():
    """Clear all pending tasks"""
    with _lock:
        task_queue.clear()
    return jsonify({"status": "ok"})


@app.route("/api/queue/remove", methods=["POST"])
def remove_task():
    """Remove a specific task from the queue by ID or name

    POST body:
    {
        "task_id": "zimage_turbo:123"  // Remove by task ID
    }
    OR
    {
        "name": "Test Item"  // Remove by name (meal_name)
    }
    """
    data = request.json or {}
    task_id = data.get("task_id")
    name = data.get("name")

    if not task_id and not name:
        return jsonify({"error": "task_id or name required"}), 400

    removed = []
    with _lock:
        to_remove = []
        for task in task_queue:
            if task_id and task["id"] == task_id:
                to_remove.append(task)
            elif name and task.get("meal_name") == name:
                to_remove.append(task)

        for task in to_remove:
            task_queue.remove(task)
            removed.append(task["id"])

    if removed:
        return jsonify({"status": "ok", "removed": removed})
    return jsonify({"error": "Task not found"}), 404


@app.route("/api/queue/add", methods=["POST"])
def add_single_task():
    """Add a single task to the queue

    POST body:
    {
        "id": "unique_id",           // Required: unique identifier
        "prompt": "Your prompt...",  // Required: the generation prompt
        "model": "zimage_turbo",     // Required: model to use
        "name": "Task name",         // Optional: display name
        "category": "Category"       // Optional: category/cuisine
    }
    """
    data = request.json
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    # Validate required fields
    recipe_id = data.get("id")
    prompt = data.get("prompt")
    model = data.get("model")

    if not recipe_id:
        return jsonify({"error": "id is required"}), 400
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400
    if not model:
        return jsonify({"error": "model is required"}), 400

    # Optional fields
    name = data.get("name", f"Task {recipe_id}")
    category = data.get("category", "custom")
    slug = data.get("slug")  # Optional slug for filename (defaults to recipe_id)

    # Generate unique task ID
    task_id = f"{model}:{recipe_id}"

    # Check for duplicates
    with _lock:
        for existing in task_queue:
            if existing["id"] == task_id:
                return jsonify({"error": f"Task {task_id} already exists in queue"}), 409

        # Create task object
        task = {
            "id": task_id,
            "recipe_id": str(recipe_id),
            "meal_name": name,
            "prompt": prompt,
            "cuisine": category,
            "model": model,
            "slug": slug,  # Use for filename if provided
            "status": "pending",
            "assigned_to": None,
            "assigned_at": None,
            "created_at": datetime.now().isoformat(),
        }
        task_queue.append(task)

    return jsonify({
        "status": "ok",
        "task_id": task_id,
        "task": task
    })


@app.route("/api/queue/bulk", methods=["POST"])
def add_bulk_tasks():
    """Add multiple tasks to the queue, respecting model field in each item

    POST body:
    {
        "tasks": [
            {
                "id": "unique_id",
                "prompt": "Your prompt...",
                "model": "zimage_turbo",
                "name": "Task name",
                "category": "Category",
                "slug": "filename-slug"
            },
            ...
        ]
    }
    """
    data = request.json
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    tasks = data.get("tasks")
    if not tasks:
        return jsonify({"error": "No tasks array provided"}), 400

    if not isinstance(tasks, list):
        return jsonify({"error": "tasks must be an array"}), 400

    added = 0
    skipped = 0
    errors = []

    with _lock:
        for item in tasks:
            recipe_id = item.get("id")
            prompt = item.get("prompt")
            model = item.get("model")

            if not recipe_id or not prompt or not model:
                errors.append(f"Missing required field in item: {recipe_id or 'unknown'}")
                skipped += 1
                continue

            name = item.get("name", f"Task {recipe_id}")
            category = item.get("category", "custom")
            slug = item.get("slug")

            task_id = f"{model}:{recipe_id}"

            # Check for duplicates
            duplicate = False
            for existing in task_queue:
                if existing["id"] == task_id:
                    skipped += 1
                    duplicate = True
                    break

            if duplicate:
                continue

            task = {
                "id": task_id,
                "recipe_id": str(recipe_id),
                "meal_name": name,
                "prompt": prompt,
                "cuisine": category,
                "model": model,
                "slug": slug,
                "status": "pending",
                "assigned_to": None,
                "assigned_at": None,
                "created_at": datetime.now().isoformat(),
            }
            task_queue.append(task)
            added += 1

    return jsonify({
        "status": "ok",
        "added": added,
        "skipped": skipped,
        "errors": errors[:10] if errors else []
    })


@app.route("/api/prompt/preview", methods=["POST"])
def preview_prompt():
    """Preview the rendered prompt for an ingredient"""
    from db.template_renderer import build_full_prompt, load_defaults
    import json as json_mod

    data = request.json or {}
    ingredient_name = data.get("ingredient_name", "")
    packaging_type = data.get("packaging_type", "")
    category = data.get("category", "")

    if not ingredient_name or not packaging_type:
        return jsonify({"error": "ingredient_name and packaging_type are required"}), 400

    # Load category template if available
    category_file = os.path.join(os.path.dirname(__file__), "db", "category_templates.json")
    category_hints = ""
    if category and os.path.exists(category_file):
        with open(category_file, 'r') as f:
            cat_data = json_mod.load(f)
            cat_template = cat_data.get("categories", {}).get(category, {})
            if cat_template:
                category_hints = f"\nCategory context: {cat_template.get('description', '')} {cat_template.get('visual_hints', '')}"

    try:
        prompt = build_full_prompt(packaging_type, ingredient_name, category)

        # Add category hints to the scene if available
        if category_hints:
            prompt["_category_hints"] = category_hints.strip()

        return jsonify({
            "status": "ok",
            "prompt": prompt,
            "prompt_text": format_prompt_as_text(prompt)
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


def format_prompt_as_text(prompt: dict) -> str:
    """Convert structured prompt to readable text format"""
    lines = []
    lines.append(f"Scene: {prompt.get('scene', '')}")

    if prompt.get('subjects'):
        for i, subj in enumerate(prompt['subjects']):
            lines.append(f"\nSubject {i+1}:")
            lines.append(f"  Description: {subj.get('description', '')}")
            lines.append(f"  Position: {subj.get('position', '')}")
            lines.append(f"  Constraints: {subj.get('constraints', '')}")

    lines.append(f"\nStyle: {prompt.get('style', '')}")
    lines.append(f"Lighting: {prompt.get('lighting', '')}")
    lines.append(f"Background: {prompt.get('background', '')}")
    lines.append(f"Mood: {prompt.get('mood', '')}")
    lines.append(f"Composition: {prompt.get('composition', '')}")

    camera = prompt.get('camera', {})
    if camera:
        lines.append(f"\nCamera:")
        lines.append(f"  Angle: {camera.get('angle', '')}")
        lines.append(f"  Lens: {camera.get('lens', '')}")
        lines.append(f"  DoF: {camera.get('depth_of_field', '')}")

    if prompt.get('_category_hints'):
        lines.append(f"\n{prompt['_category_hints']}")

    return "\n".join(lines)


# ============== Web Dashboard ==============

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <title>Batch Orchestrator Dashboard</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect x='10' y='10' width='35' height='35' rx='5' fill='%2322c55e'/><rect x='55' y='10' width='35' height='35' rx='5' fill='%2322c55e' opacity='0.7'/><rect x='10' y='55' width='35' height='35' rx='5' fill='%2322c55e' opacity='0.5'/><rect x='55' y='55' width='35' height='35' rx='5' fill='%2322c55e' opacity='0.3'/></svg>">
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    colors: {
                        border: "hsl(240 3.7% 15.9%)",
                        input: "hsl(240 3.7% 15.9%)",
                        ring: "hsl(142.1 76.2% 36.3%)",
                        background: "hsl(240 10% 3.9%)",
                        foreground: "hsl(0 0% 98%)",
                        primary: {
                            DEFAULT: "hsl(142.1 76.2% 36.3%)",
                            foreground: "hsl(144.9 80.4% 10%)",
                        },
                        secondary: {
                            DEFAULT: "hsl(240 3.7% 15.9%)",
                            foreground: "hsl(0 0% 98%)",
                        },
                        destructive: {
                            DEFAULT: "hsl(0 62.8% 30.6%)",
                            foreground: "hsl(0 0% 98%)",
                        },
                        muted: {
                            DEFAULT: "hsl(240 3.7% 15.9%)",
                            foreground: "hsl(240 5% 64.9%)",
                        },
                        accent: {
                            DEFAULT: "hsl(240 3.7% 15.9%)",
                            foreground: "hsl(0 0% 98%)",
                        },
                        card: {
                            DEFAULT: "hsl(240 10% 3.9%)",
                            foreground: "hsl(0 0% 98%)",
                        },
                    },
                    borderRadius: {
                        lg: "0.5rem",
                        md: "calc(0.5rem - 2px)",
                        sm: "calc(0.5rem - 4px)",
                    },
                }
            }
        }
    </script>
    <style>
        @keyframes pulse-glow {
            0%, 100% { opacity: 1; box-shadow: 0 0 8px currentColor; }
            50% { opacity: 0.5; box-shadow: 0 0 4px currentColor; }
        }
        .animate-pulse-glow { animation: pulse-glow 2s ease-in-out infinite; }
        .hidden { display: none !important; }

        /* Custom scrollbar */
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: hsl(240 10% 3.9%); }
        ::-webkit-scrollbar-thumb { background: hsl(240 3.7% 15.9%); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: hsl(240 5% 25%); }
    </style>
</head>
<body class="bg-background text-foreground min-h-screen antialiased">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <!-- Header -->
        <div class="flex items-center justify-between mb-8">
            <div>
                <h1 class="text-2xl font-bold tracking-tight">Batch Orchestrator</h1>
                <p class="text-muted-foreground text-sm mt-1">Manage image generation tasks and templates</p>
            </div>
            <div class="flex items-center gap-2 text-xs text-muted-foreground">
                <div class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
                <span>Last updated: <span id="lastUpdate">-</span></span>
            </div>
        </div>

        <!-- Navigation Tabs -->
        <div class="border-b border-border mb-6">
            <nav class="flex gap-4" aria-label="Tabs">
                <button onclick="showTab('queue')" id="tab-queue"
                    class="tab-btn relative px-1 pb-3 text-sm font-medium text-foreground border-b-2 border-primary">
                    Queue
                </button>
            </nav>
        </div>

        <!-- Queue Tab -->
        <div id="queue-tab" class="tab-content">
            <!-- Stats Cards -->
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                <div class="rounded-xl border border-border bg-card p-6">
                    <div class="flex items-center justify-between">
                        <p class="text-sm font-medium text-muted-foreground">Completed</p>
                        <svg class="w-4 h-4 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                        </svg>
                    </div>
                    <p class="text-3xl font-bold text-emerald-500 mt-2" id="completed">0</p>
                </div>
                <div class="rounded-xl border border-border bg-card p-6">
                    <div class="flex items-center justify-between">
                        <p class="text-sm font-medium text-muted-foreground">Pending</p>
                        <svg class="w-4 h-4 text-amber-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                        </svg>
                    </div>
                    <p class="text-3xl font-bold text-amber-500 mt-2" id="pending">0</p>
                </div>
                <div class="rounded-xl border border-border bg-card p-6">
                    <div class="flex items-center justify-between">
                        <p class="text-sm font-medium text-muted-foreground">Failed</p>
                        <svg class="w-4 h-4 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                        </svg>
                    </div>
                    <p class="text-3xl font-bold text-red-500 mt-2" id="failed">0</p>
                </div>
                <div class="rounded-xl border border-border bg-card p-6">
                    <div class="flex items-center justify-between">
                        <p class="text-sm font-medium text-muted-foreground">Active Nodes</p>
                        <svg class="w-4 h-4 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2"></path>
                        </svg>
                    </div>
                    <p class="text-3xl font-bold text-blue-500 mt-2"><span id="activeNodes">0</span><span class="text-lg text-muted-foreground">/<span id="totalNodes">0</span></span></p>
                </div>
            </div>

            <!-- Progress Bar -->
            <div class="rounded-xl border border-border bg-card p-6 mb-6">
                <div class="flex items-center justify-between mb-3">
                    <h3 class="text-sm font-medium">Overall Progress</h3>
                    <span class="text-sm text-muted-foreground" id="progressPercent">0%</span>
                </div>
                <div class="h-2 bg-secondary rounded-full overflow-hidden">
                    <div id="progress" class="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 transition-all duration-500" style="width: 0%"></div>
                </div>
                <div class="flex gap-3 mt-4">
                    <button onclick="initQueue()" class="inline-flex items-center justify-center rounded-md text-sm font-medium h-9 px-4 bg-primary text-primary-foreground hover:bg-primary/90 transition-colors">
                        <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                        </svg>
                        Load Queue
                    </button>
                    <button id="pauseBtn" onclick="togglePause()" class="inline-flex items-center justify-center rounded-md text-sm font-medium h-9 px-4 bg-secondary text-secondary-foreground hover:bg-secondary/80 transition-colors">
                        <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                        </svg>
                        Pause
                    </button>
                    <button onclick="clearQueue()" class="inline-flex items-center justify-center rounded-md text-sm font-medium h-9 px-4 bg-destructive text-destructive-foreground hover:bg-destructive/90 transition-colors">
                        <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path>
                        </svg>
                        Clear
                    </button>
                </div>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                <!-- Upload JSON Section (collapsed) -->
                <div class="rounded-xl border border-border bg-card">
                    <button onclick="toggleUploadSection()" class="w-full p-4 flex items-center justify-between text-left hover:bg-secondary/30 transition-colors rounded-xl">
                        <div class="flex items-center gap-3">
                            <svg class="w-5 h-5 text-muted-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                            </svg>
                            <span class="text-sm font-medium">Upload JSON File</span>
                        </div>
                        <svg id="uploadChevron" class="w-4 h-4 text-muted-foreground transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
                        </svg>
                    </button>
                    <div id="uploadContent" class="hidden p-4 pt-0">
                        <div id="uploadSection" class="border-2 border-dashed border-border rounded-lg p-6 text-center hover:border-muted-foreground/50 transition-colors cursor-pointer">
                            <input type="file" id="fileInput" class="hidden" accept=".json" onchange="handleFileSelect(event)">
                            <label for="fileInput" class="cursor-pointer">
                                <p class="text-sm text-muted-foreground">Drop JSON file here or click to browse</p>
                                <p class="text-xs text-muted-foreground/70 mt-1">Format: [{id, prompt, model, name?, category?, slug?}]</p>
                            </label>
                            <p id="fileName" class="text-sm text-foreground mt-2 font-medium"></p>
                        </div>
                        <button onclick="uploadFile()" class="w-full mt-4 inline-flex items-center justify-center rounded-md text-sm font-medium h-9 px-4 bg-primary text-primary-foreground hover:bg-primary/90 transition-colors">
                            Upload & Add to Queue
                        </button>
                        <p id="uploadStatus" class="text-sm mt-2 text-center"></p>
                    </div>
                </div>

                <!-- Nodes Section -->
                <div class="rounded-xl border border-border bg-card p-6">
                    <h3 class="text-sm font-medium mb-4">Connected Nodes</h3>
                    <div id="nodesList" class="space-y-3">
                        <p class="text-sm text-muted-foreground text-center py-8">No nodes connected</p>
                    </div>
                </div>
            </div>

            <!-- Queue List -->
            <div class="rounded-xl border border-border bg-card">
                <div class="p-4 border-b border-border">
                    <h3 class="text-sm font-medium">Task Queue</h3>
                    <p class="text-xs text-muted-foreground mt-1">Showing next 20 tasks</p>
                </div>
                <div id="queueList" class="divide-y divide-border">
                    <p class="text-sm text-muted-foreground text-center py-8">Queue empty - click "Load Queue" to start</p>
                </div>
            </div>
        </div><!-- End Queue Tab -->
    </div>

    <script>
        let paused = false;

        async function fetchStatus() {
            try {
                const [status, nodes, queue] = await Promise.all([
                    fetch('/api/status').then(r => r.json()),
                    fetch('/api/nodes').then(r => r.json()),
                    fetch('/api/queue').then(r => r.json())
                ]);

                // Update stats
                document.getElementById('completed').textContent = status.tasks.completed;
                document.getElementById('pending').textContent = status.tasks.pending;
                document.getElementById('failed').textContent = status.tasks.failed;
                document.getElementById('activeNodes').textContent = status.nodes.active;
                document.getElementById('totalNodes').textContent = status.nodes.total;

                // Update progress bar
                const total = status.tasks.total || 1;
                const pct = Math.round((status.tasks.completed / total) * 100);
                document.getElementById('progress').style.width = pct + '%';
                document.getElementById('progressPercent').textContent = pct + '%';

                // Update pause button
                paused = status.settings.paused;
                const pauseBtn = document.getElementById('pauseBtn');
                pauseBtn.innerHTML = paused
                    ? `<svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>Resume`
                    : `<svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>Pause`;

                // Update nodes list
                const nodesList = document.getElementById('nodesList');
                if (nodes.length === 0) {
                    nodesList.innerHTML = '<p class="text-sm text-muted-foreground text-center py-8">No nodes connected</p>';
                } else {
                    nodesList.innerHTML = nodes.map(n => {
                        const models = n.models || [];
                        const modelStr = models.length > 0 ? models.join(', ') : 'any';
                        const statusColor = n.online
                            ? (n.status === 'generating' ? 'bg-amber-500 animate-pulse' : 'bg-emerald-500')
                            : 'bg-red-500';
                        const statusGlow = n.online && n.status !== 'generating' ? 'shadow-[0_0_8px_rgba(16,185,129,0.6)]' : '';
                        return `
                        <div class="flex items-center gap-4 p-3 rounded-lg bg-secondary/50">
                            <div class="w-3 h-3 rounded-full ${statusColor} ${statusGlow} flex-shrink-0"></div>
                            <div class="flex-1 min-w-0">
                                <div class="flex items-center gap-2">
                                    <span class="font-medium text-sm truncate">${n.name}</span>
                                    <span class="text-xs text-blue-400">[${modelStr}]</span>
                                    ${n.current_model ? `<span class="text-xs text-emerald-400">(${n.current_model})</span>` : ''}
                                </div>
                                <div class="text-xs text-muted-foreground">${n.gpu} (${n.vram_gb}GB)</div>
                                ${n.current_task ? `<div class="text-xs text-blue-400 mt-1 truncate">Working on: ${n.current_task}</div>` : ''}
                            </div>
                            <div class="text-right text-xs text-muted-foreground flex-shrink-0">
                                <div>${n.stats.completed} done</div>
                                <div>${n.online ? 'Online' : `Offline ${n.last_seen}s`}</div>
                            </div>
                        </div>
                    `}).join('');
                }

                // Update queue list
                const queueList = document.getElementById('queueList');
                const first20 = queue.slice(0, 20);
                if (first20.length === 0) {
                    queueList.innerHTML = '<p class="text-sm text-muted-foreground text-center py-8">Queue empty - click "Load Queue" to start</p>';
                } else {
                    queueList.innerHTML = first20.map(t => {
                        const retryCount = t.retry_count || 0;
                        const isGenerating = t.status === 'assigned' || t.status === 'generating';
                        const progress = t.progress || 0;
                        const step = t.step || 0;
                        const totalSteps = t.total_steps || 0;

                        let statusColor = 'bg-muted-foreground';
                        if (t.status === 'assigned' || t.status === 'generating') statusColor = 'bg-amber-500 animate-pulse';
                        else if (retryCount > 0) statusColor = 'bg-orange-500';

                        let progressHtml = '';
                        if (isGenerating) {
                            const stepText = totalSteps > 0 ? `Step ${step}/${totalSteps}` : `${progress}%`;
                            progressHtml = `
                                <div class="mt-2 ml-5">
                                    <div class="h-1.5 bg-secondary rounded-full overflow-hidden">
                                        <div class="h-full bg-gradient-to-r from-blue-500 to-blue-400 transition-all duration-300" style="width: ${progress}%"></div>
                                    </div>
                                    <div class="text-xs text-muted-foreground mt-1">${stepText} - ${t.assigned_to || 'Processing...'}</div>
                                </div>
                            `;
                        }

                        const taskIndex = first20.indexOf(t);
                        const packagingType = t.packaging_type || '';
                        const category = t.category || t.cuisine || '';

                        return `
                            <div class="px-4 py-3">
                                <div class="flex items-center gap-3">
                                    <div class="w-2 h-2 rounded-full ${statusColor} flex-shrink-0"></div>
                                    <div class="flex-1 min-w-0">
                                        <span class="font-medium text-sm">${t.meal_name}</span>
                                        <span class="text-xs text-muted-foreground ml-2">(${category})</span>
                                        <span class="text-xs text-blue-400 ml-2">${t.model}</span>
                                        ${packagingType ? `<span class="text-xs text-purple-400 ml-2">[${packagingType.replace(/_/g, ' ')}]</span>` : ''}
                                        ${retryCount > 0 ? `<span class="ml-2 px-1.5 py-0.5 text-[10px] font-semibold bg-orange-500 text-white rounded">Retry #${retryCount}</span>` : ''}
                                    </div>
                                    <button onclick="togglePromptPreview(${taskIndex})"
                                        class="text-xs px-2 py-1 rounded bg-secondary hover:bg-secondary/80 text-muted-foreground hover:text-foreground transition-colors flex-shrink-0"
                                        data-prompt="${encodeURIComponent(t.prompt || '')}">
                                        Prompt
                                    </button>
                                    <span class="text-xs text-muted-foreground capitalize flex-shrink-0">${t.status}</span>
                                </div>
                                ${progressHtml}
                                <div id="prompt-preview-${taskIndex}" class="hidden mt-3 ml-5 p-3 bg-secondary/50 rounded-lg text-xs font-mono whitespace-pre-wrap text-muted-foreground max-h-60 overflow-y-auto"></div>
                            </div>
                        `;
                    }).join('');
                }

                document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
            } catch (e) {
                console.error('Error fetching status:', e);
            }
        }

        async function initQueue() {
            await fetch('/api/queue/init', { method: 'POST' });
            fetchStatus();
        }

        async function togglePause() {
            await fetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ paused: !paused })
            });
            fetchStatus();
        }

        async function clearQueue() {
            if (confirm('Clear all pending tasks?')) {
                await fetch('/api/queue/clear', { method: 'POST' });
                fetchStatus();
            }
        }

        function togglePromptPreview(index) {
            const previewDiv = document.getElementById('prompt-preview-' + index);
            if (!previewDiv) return;

            // Toggle visibility
            if (!previewDiv.classList.contains('hidden')) {
                previewDiv.classList.add('hidden');
                return;
            }

            // Get the prompt from the button's data attribute
            const button = document.querySelector(`button[onclick="togglePromptPreview(${index})"]`);
            const encodedPrompt = button ? button.getAttribute('data-prompt') : '';
            const prompt = decodeURIComponent(encodedPrompt);

            previewDiv.classList.remove('hidden');
            previewDiv.textContent = prompt || 'No prompt available for this item.';
        }

        function toggleUploadSection() {
            const content = document.getElementById('uploadContent');
            const chevron = document.getElementById('uploadChevron');
            content.classList.toggle('hidden');
            chevron.classList.toggle('rotate-180');
        }

        // ============== File Upload Functions ==============
        let selectedFile = null;

        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (file) {
                selectedFile = file;
                document.getElementById('fileName').textContent = file.name;
                document.getElementById('uploadStatus').textContent = '';
            }
        }

        // Drag and drop
        const uploadSection = document.getElementById('uploadSection');
        uploadSection.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadSection.classList.add('border-emerald-500', 'bg-emerald-500/10');
        });
        uploadSection.addEventListener('dragleave', () => {
            uploadSection.classList.remove('border-emerald-500', 'bg-emerald-500/10');
        });
        uploadSection.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadSection.classList.remove('border-emerald-500', 'bg-emerald-500/10');
            const file = e.dataTransfer.files[0];
            if (file && file.name.endsWith('.json')) {
                selectedFile = file;
                document.getElementById('fileName').textContent = file.name;
                document.getElementById('uploadStatus').textContent = '';
            } else {
                const statusEl = document.getElementById('uploadStatus');
                statusEl.textContent = 'Please drop a .json file';
                statusEl.className = 'text-sm mt-2 text-center text-red-400';
            }
        });

        async function uploadFile() {
            const statusEl = document.getElementById('uploadStatus');
            if (!selectedFile) {
                statusEl.textContent = 'Please select a file first';
                statusEl.className = 'text-sm mt-2 text-center text-red-400';
                return;
            }

            statusEl.textContent = 'Uploading...';
            statusEl.className = 'text-sm mt-2 text-center text-muted-foreground';

            try {
                const text = await selectedFile.text();
                let tasks = JSON.parse(text);

                // Handle both array and {tasks: [...]} formats
                if (!Array.isArray(tasks)) {
                    tasks = tasks.tasks || tasks.recipes || [];
                }

                // Validate each task has required fields
                const invalid = tasks.filter(t => !t.id || !t.prompt || !t.model);
                if (invalid.length > 0) {
                    statusEl.textContent = `${invalid.length} items missing id, prompt, or model`;
                    statusEl.className = 'text-sm mt-2 text-center text-red-400';
                    return;
                }

                const response = await fetch('/api/queue/bulk', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ tasks })
                });

                const result = await response.json();
                if (response.ok) {
                    statusEl.textContent = `Added ${result.added} tasks (${result.skipped} skipped)`;
                    statusEl.className = 'text-sm mt-2 text-center text-emerald-400';
                    fetchStatus();
                } else {
                    statusEl.textContent = result.error || 'Upload failed';
                    statusEl.className = 'text-sm mt-2 text-center text-red-400';
                }
            } catch (e) {
                statusEl.textContent = 'Error: ' + e.message;
                statusEl.className = 'text-sm mt-2 text-center text-red-400';
            }
        }

        // Initial fetch and auto-refresh
        fetchStatus();
        setInterval(fetchStatus, 2000);
    </script>
</body>
</html>
"""

@app.route("/")
def dashboard():
    """Serve the dashboard"""
    return render_template_string(DASHBOARD_HTML)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Batch Orchestrator Server")
    print("=" * 60)
    print(f"Dashboard: http://localhost:5002")
    print(f"API: http://localhost:5002/api/")
    print("=" * 60 + "\n")

    debug_mode = os.getenv("FLASK_DEBUG", "0").lower() in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=5002, debug=debug_mode)
