#!/usr/bin/env python
"""
Batch Recipe Image Generator
Generates images for recipes using FLUX.2-dev and uploads to Cloudflare R2.
Tracks progress and supports resume from interruption.
"""

import os
import io
import gc
import sys
import json
import time
import signal
import atexit
import hashlib
import argparse
import threading
from datetime import datetime
from pathlib import Path

import boto3
from botocore.config import Config
from PIL import Image
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import LocalEntryNotFoundError

load_dotenv()


# ============== Model Loading ==============
_pipes = {}
_compel_instances = {}  # For SDXL long prompt handling
_torch = None
_current_model = None
_use_compile = False  # Set via --compile flag for long batch runs
_model_last_used = 0  # Timestamp of last model usage
_idle_check_thread = None
_stop_idle_check = False

# Auto-unload models after 1 minute of inactivity
MODEL_IDLE_TIMEOUT = 60  # seconds


# ============== Cleanup & Signal Handling ==============
def cleanup_gpu():
    """Release all GPU memory and kill child processes"""
    global _pipes, _torch

    print("\n[Cleanup] Releasing GPU memory...")

    # Clear all loaded models
    if _pipes:
        for name in list(_pipes.keys()):
            try:
                del _pipes[name]
            except:
                pass
        _pipes.clear()

    # Force garbage collection
    gc.collect()

    # Clear CUDA cache if torch is loaded (skip sync to avoid hangs)
    if _torch is not None:
        try:
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception as e:
            print(f"[Cleanup] Error clearing CUDA: {e}")

    print("[Cleanup] Done!")


def check_idle_and_unload():
    """Background thread to unload models after idle timeout"""
    global _stop_idle_check, _model_last_used, _pipes, _current_model

    while not _stop_idle_check:
        time.sleep(10)  # Check every 10 seconds

        if _pipes and _model_last_used > 0:
            idle_time = time.time() - _model_last_used
            if idle_time > MODEL_IDLE_TIMEOUT:
                print(f"\n[Idle] Model idle for {idle_time:.0f}s, unloading to free GPU memory...")
                cleanup_gpu()
                _current_model = None
                _model_last_used = 0
                print("[Idle] Model unloaded. Will reload on next task.")


def start_idle_check_thread():
    """Start background thread for idle model unloading"""
    global _idle_check_thread, _stop_idle_check

    if _idle_check_thread is None or not _idle_check_thread.is_alive():
        _stop_idle_check = False
        _idle_check_thread = threading.Thread(target=check_idle_and_unload, daemon=True)
        _idle_check_thread.start()
        print(f"[Idle] Auto-unload enabled (timeout: {MODEL_IDLE_TIMEOUT}s)")


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    sig_name = signal.Signals(signum).name
    print(f"\n\nReceived {sig_name}, cleaning up...")
    cleanup_gpu()
    sys.exit(0)


# Register cleanup handlers
atexit.register(cleanup_gpu)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ============== Configuration ==============
# Auto-detect Windows vs WSL paths
import sys
if sys.platform == "win32":
    RECIPES_PATH = r"C:\Users\dhruv\Downloads\ingredients_flux_prompts_v2_quantity_crosssection_seeded.json"
else:
    RECIPES_PATH = "/mnt/c/Users/dhruv/Downloads/ingredients_flux_prompts_v2_quantity_crosssection_seeded.json"
STATE_FILE = "generation_state.json"
OUTPUT_DIR = "generated_images"
R2_BUCKET = os.getenv("R2_BUCKET", "tiffyn-test")

# R2 Configuration (set these in .env)
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_PUBLIC_URL = os.getenv("R2_PUBLIC_URL", "")  # e.g., https://pub-xxx.r2.dev

# Generation settings
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024
NUM_INFERENCE_STEPS = 28
GUIDANCE_SCALE = 4.0


def get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


# Model repository IDs for update checking
MODEL_REPOS = {
    "flux2_dev": "diffusers/FLUX.2-dev-bnb-4bit",
    "flux2_fp8": "black-forest-labs/FLUX.2-dev",
    "flux2_turbo": "black-forest-labs/FLUX.2-dev",
    "flux2_klein": "black-forest-labs/FLUX.2-klein-9B",
    "zimage_turbo": "Tongyi-MAI/Z-Image-Turbo",
    "sdxl_food": "stabilityai/stable-diffusion-xl-base-1.0",
    "chroma1_hd": "lodestones/Chroma1-HD",
    "qwen_turbo": "Qwen/Qwen-Image-2512",
}

LORA_REPOS = {
    "sdxl_food": ("jcjo/pyc-food-sdxl-lora", "pytorch_lora_weights.safetensors"),
    "flux2_turbo": ("fal/FLUX.2-dev-Turbo", "flux.2-turbo-lora.safetensors"),
}


def check_and_sync_model(model: str, auto_update: bool = True) -> bool:
    """
    Check if a specific model is up to date with HuggingFace and optionally sync it.
    Returns True if model is up to date (or was updated), False if update failed.
    """
    from huggingface_hub import scan_cache_dir, HfApi, snapshot_download

    api = HfApi()
    cache_info = scan_cache_dir()

    # Build a map of cached repos
    cached_repos = {}
    for repo in cache_info.repos:
        cached_repos[repo.repo_id] = [rev.commit_hash for rev in repo.revisions]

    repo_id = MODEL_REPOS.get(model)
    if not repo_id:
        print(f"  Unknown model: {model}")
        return False

    print(f"Checking {model} for updates...")

    try:
        repo_info = api.repo_info(repo_id=repo_id, repo_type="model")
        latest_sha = repo_info.sha

        if repo_id in cached_repos and latest_sha in cached_repos[repo_id]:
            print(f"  {model}: Up to date ({latest_sha[:8]})")
        else:
            if repo_id in cached_repos:
                print(f"  {model}: UPDATE AVAILABLE (local: {cached_repos[repo_id][0][:8]}, remote: {latest_sha[:8]})")
            else:
                print(f"  {model}: Not cached, will download")

            if auto_update:
                print(f"  Downloading latest version...")
                snapshot_download(repo_id=repo_id)
                print(f"  {model}: Updated to {latest_sha[:8]}")

        # Also check LoRA if applicable
        if model in LORA_REPOS:
            lora_repo_id, weight_file = LORA_REPOS[model]
            lora_info = api.repo_info(repo_id=lora_repo_id, repo_type="model")
            lora_sha = lora_info.sha

            if lora_repo_id in cached_repos and lora_sha in cached_repos[lora_repo_id]:
                print(f"  {model} LoRA: Up to date ({lora_sha[:8]})")
            else:
                if lora_repo_id in cached_repos:
                    print(f"  {model} LoRA: UPDATE AVAILABLE")
                else:
                    print(f"  {model} LoRA: Not cached, will download")

                if auto_update:
                    print(f"  Downloading latest LoRA...")
                    hf_hub_download(repo_id=lora_repo_id, filename=weight_file)
                    print(f"  {model} LoRA: Updated to {lora_sha[:8]}")

        return True

    except Exception as e:
        print(f"  Error checking/updating {model}: {e}")
        return False


def check_model_updates():
    """Check for updates to HuggingFace models and LoRAs"""
    from huggingface_hub import scan_cache_dir, HfApi

    api = HfApi()
    cache_info = scan_cache_dir()

    # Build a map of cached repos to their commit hashes
    cached_repos = {}
    for repo in cache_info.repos:
        cached_repos[repo.repo_id] = {
            "revisions": [rev.commit_hash for rev in repo.revisions],
            "last_modified": max((rev.last_modified for rev in repo.revisions), default=None)
        }

    updates_available = []

    print("\n" + "=" * 60)
    print("Checking for Model Updates from HuggingFace")
    print("=" * 60)

    # Check main models
    for model_key, repo_id in MODEL_REPOS.items():
        try:
            # Get latest commit from HF
            repo_info = api.repo_info(repo_id=repo_id, repo_type="model")
            latest_sha = repo_info.sha

            if repo_id in cached_repos:
                cached_shas = cached_repos[repo_id]["revisions"]
                if latest_sha in cached_shas:
                    print(f"  {model_key}: Up to date ({latest_sha[:8]})")
                else:
                    print(f"  {model_key}: UPDATE AVAILABLE")
                    print(f"    Local: {cached_shas[0][:8] if cached_shas else 'N/A'}")
                    print(f"    Remote: {latest_sha[:8]}")
                    updates_available.append((model_key, repo_id, "model"))
            else:
                print(f"  {model_key}: Not cached (will download on first use)")
        except Exception as e:
            print(f"  {model_key}: Error checking - {e}")

    # Check LoRAs
    print("\nLoRAs:")
    for model_key, (repo_id, weight_file) in LORA_REPOS.items():
        try:
            repo_info = api.repo_info(repo_id=repo_id, repo_type="model")
            latest_sha = repo_info.sha

            if repo_id in cached_repos:
                cached_shas = cached_repos[repo_id]["revisions"]
                if latest_sha in cached_shas:
                    print(f"  {model_key} LoRA: Up to date ({latest_sha[:8]})")
                else:
                    print(f"  {model_key} LoRA: UPDATE AVAILABLE")
                    print(f"    Local: {cached_shas[0][:8] if cached_shas else 'N/A'}")
                    print(f"    Remote: {latest_sha[:8]}")
                    updates_available.append((model_key, repo_id, "lora"))
            else:
                print(f"  {model_key} LoRA: Not cached (will download on first use)")
        except Exception as e:
            print(f"  {model_key} LoRA: Error checking - {e}")

    print("=" * 60)

    if updates_available:
        print(f"\n{len(updates_available)} update(s) available.")
        print("Run with --update-models to download updates.")
    else:
        print("\nAll models are up to date.")

    return updates_available


def update_models():
    """Force re-download of all models and LoRAs"""
    from huggingface_hub import snapshot_download

    print("\n" + "=" * 60)
    print("Updating Models from HuggingFace")
    print("=" * 60)

    for model_key, repo_id in MODEL_REPOS.items():
        print(f"\nUpdating {model_key} ({repo_id})...")
        try:
            snapshot_download(repo_id=repo_id, force_download=True)
            print(f"  Done!")
        except Exception as e:
            print(f"  Error: {e}")

    for model_key, (repo_id, weight_file) in LORA_REPOS.items():
        print(f"\nUpdating {model_key} LoRA ({repo_id})...")
        try:
            hf_hub_download(repo_id=repo_id, filename=weight_file, force_download=True)
            print(f"  Done!")
        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 60)
    print("Model updates complete!")
    print("=" * 60)


def is_high_vram_gpu():
    """Check GPU type and VRAM for optimal loading strategy

    Returns:
        (has_very_high_vram, is_blackwell, can_compile)
        - has_very_high_vram: 28GB+ (5090, A100) - can load directly to CUDA
        - is_blackwell: RTX 5090/5080 architecture
        - can_compile: torch.compile available (Linux only)
    """
    import sys
    torch = get_torch()
    if not torch.cuda.is_available():
        return False, False, False

    gpu_name = torch.cuda.get_device_name(0).lower()
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    # RTX 5090 (32GB), A100 (40/80GB) - can load directly to CUDA
    # RTX 4090 (24GB) - needs CPU offload, not enough headroom
    is_blackwell = "5090" in gpu_name or "5080" in gpu_name  # Blackwell architecture
    has_very_high_vram = vram_gb >= 28  # Only 5090/A100 class GPUs

    # Check if torch.compile is available (requires triton, Linux-only)
    can_compile = sys.platform != "win32"  # triton not available on Windows

    print(f"  GPU detected: {torch.cuda.get_device_name(0)} ({vram_gb:.1f}GB VRAM)")
    print(f"  Blackwell architecture: {is_blackwell}, Very high VRAM (28GB+): {has_very_high_vram}")
    if is_blackwell and not can_compile:
        print(f"  torch.compile unavailable (triton requires Linux)")

    return has_very_high_vram, is_blackwell, can_compile


def get_pipeline(model: str = "flux2_dev"):
    """Load specified model (lazy loading)"""
    global _pipes, _compel_instances, _current_model

    if model in _pipes:
        _current_model = model
        return _pipes[model]

    torch = get_torch()

    if model == "flux2_dev":
        from diffusers import Flux2Pipeline, AutoModel
        from transformers import Mistral3ForConditionalGeneration, BitsAndBytesConfig

        has_very_high_vram, is_blackwell, _ = is_high_vram_gpu()

        if is_blackwell:
            # ============== RTX 5090 (32GB) - On-the-fly quantization ==============
            # Load to CPU first, use CPU offload during inference
            repo_id = "black-forest-labs/FLUX.2-dev"

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type='nf4',
            )

            print(f"Loading FLUX.2-dev with on-the-fly 4-bit quantization (Blackwell/5090)...")

            print("  Loading text encoder to CPU...")
            text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
                repo_id, subfolder="text_encoder", torch_dtype=torch.bfloat16,
                quantization_config=quant_config, device_map="cpu"
            )

            print("  Loading transformer to CPU...")
            dit = AutoModel.from_pretrained(
                repo_id, subfolder="transformer", torch_dtype=torch.bfloat16,
                quantization_config=quant_config, device_map="cpu"
            )

            print("  Building pipeline...")
            pipe = Flux2Pipeline.from_pretrained(
                repo_id,
                text_encoder=text_encoder,
                transformer=dit,
                torch_dtype=torch.bfloat16,
            )

            pipe.enable_model_cpu_offload()
            print(f"FLUX.2-dev (4-bit on-the-fly) loaded with CPU offload!")

        else:
            # ============== RTX 4090 (24GB) and others - Pre-quantized model ==============
            # Use pre-quantized model which works reliably with CPU offload
            repo_id = "diffusers/FLUX.2-dev-bnb-4bit"

            print(f"Loading FLUX.2-dev from {repo_id} (4-bit pre-quantized for RTX 4090)...")

            print("  Loading text encoder to CPU...")
            text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
                repo_id, subfolder="text_encoder", torch_dtype=torch.bfloat16, device_map="cpu"
            )

            print("  Loading transformer to CPU...")
            dit = AutoModel.from_pretrained(
                repo_id, subfolder="transformer", torch_dtype=torch.bfloat16, device_map="cpu"
            )

            print("  Building pipeline...")
            pipe = Flux2Pipeline.from_pretrained(
                repo_id,
                text_encoder=text_encoder,
                transformer=dit,
                torch_dtype=torch.bfloat16,
            )

            # CPU offload moves components to GPU only when needed during inference
            pipe.enable_model_cpu_offload()
            print("FLUX.2-dev (4-bit pre-quantized) loaded for RTX 4090!")

    elif model == "flux2_fp8":
        from diffusers import Flux2Pipeline
        from torchao.quantization import quantize_, float8_weight_only

        print("Loading FLUX.2-dev (FP8 quantized via torchao)...")
        has_high_vram, is_blackwell, can_compile = is_high_vram_gpu()

        # Load in bfloat16 first, then quantize to FP8
        pipe = Flux2Pipeline.from_pretrained(
            "black-forest-labs/FLUX.2-dev",
            torch_dtype=torch.bfloat16,
        )

        # Quantize transformer to FP8 weights
        quantize_(pipe.transformer, float8_weight_only())

        # Also quantize text encoder for memory savings
        quantize_(pipe.text_encoder, float8_weight_only())

        if has_high_vram:
            # High VRAM GPU: Keep everything on GPU
            print("  Using direct GPU placement (high VRAM detected)")
            pipe.to("cuda")

            # torch.compile: only enabled with --compile flag (causes 5+ min warmup)
            # Only beneficial for very long batch runs (50+ images in single session)
            if _use_compile and is_blackwell and can_compile:
                print("  Compiling transformer with torch.compile...")
                try:
                    pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")
                    print("  torch.compile complete!")
                except Exception as e:
                    print(f"  torch.compile failed ({e}), falling back to uncompiled")
        else:
            # Low VRAM GPU: Use CPU offload to save memory
            print("  Using CPU offload (low VRAM detected)")
            pipe.enable_model_cpu_offload()

        print("FLUX.2-dev (FP8) loaded!")

    elif model == "flux2_turbo":
        from diffusers import Flux2Pipeline

        print("Loading FLUX.2-dev with Turbo LoRA (8-step)...")
        pipe = Flux2Pipeline.from_pretrained(
            "black-forest-labs/FLUX.2-dev",
            torch_dtype=torch.bfloat16,
        )
        pipe.load_lora_weights(
            "fal/FLUX.2-dev-Turbo",
            weight_name="flux.2-turbo-lora.safetensors"
        )
        pipe.enable_sequential_cpu_offload()  # Model too large for VRAM with desktop running
        print("FLUX.2-dev Turbo loaded!")

    elif model == "zimage_turbo":
        from diffusers import ZImagePipeline

        print("Loading Z-Image-Turbo...")
        pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )
        pipe.to("cuda")
        print("Z-Image-Turbo loaded!")

    elif model == "chroma1_hd":
        from diffusers import ChromaPipeline
        print("Loading Chroma1-HD...")
        pipe = ChromaPipeline.from_pretrained(
            "lodestones/Chroma1-HD",
            torch_dtype=torch.bfloat16,
        )
        pipe.to("cuda")
        if _use_compile:
            print("  Compiling Chroma1-HD transformer...")
            pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")
        print("Chroma1-HD loaded!")

    elif model == "sdxl_food":
        from diffusers import StableDiffusionXLPipeline
        from compel import Compel, ReturnedEmbeddingsType

        print("Loading SDXL with Food LoRA...")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )

        # Load Food LoRA - jcjo/pyc-food-sdxl-lora (SDXL compatible)
        pipe.load_lora_weights(
            "jcjo/pyc-food-sdxl-lora",
            weight_name="pytorch_lora_weights.safetensors",
            adapter_name="food"
        )
        pipe.set_adapters(["food"], adapter_weights=[1.0])
        pipe.to("cuda")

        # Initialize Compel for long prompt handling
        compel = Compel(
            tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
            text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True]
        )
        _compel_instances[model] = compel

        print("SDXL with Food LoRA loaded!")

    elif model == "qwen_turbo":
        from diffusers import DiffusionPipeline

        print("Loading Qwen-Image-2512 (via diffusers)...")
        pipe = DiffusionPipeline.from_pretrained(
            "Qwen/Qwen-Image-2512",
            torch_dtype=torch.bfloat16,
        )
        pipe.enable_sequential_cpu_offload()
        print("Qwen-Image-2512 loaded!")

    elif model == "flux2_klein":
        from diffusers import Flux2Pipeline

        print("Loading FLUX.2-klein-9B...")
        pipe = Flux2Pipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-9B",
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        print("FLUX.2-klein-9B loaded!")

    else:
        raise ValueError(f"Unknown model: {model}")

    _pipes[model] = pipe
    _current_model = model
    return pipe


def get_compel(model: str):
    """Get compel instance for a model"""
    return _compel_instances.get(model)


# Model-specific generation settings
MODEL_SETTINGS = {
    "flux2_dev": {
        "num_inference_steps": 28,
        "guidance_scale": 12.0,  # Higher CFG for strict prompt adherence
    },
    "flux2_fp8": {
        "num_inference_steps": 28,
        "guidance_scale": 12.0,  # Same as 4-bit for fair comparison
    },
    "flux2_turbo": {
        "num_inference_steps": 8,
        "guidance_scale": 2.5,
        "sigmas": [1.0, 0.6509, 0.4374, 0.2932, 0.1893, 0.1108, 0.0495, 0.00031],
    },
    "zimage_turbo": {
        "num_inference_steps": 8,
        "guidance_scale": 1.0,  # Light guidance for better prompt adherence
        "max_sequence_length": 1024,  # Default is 512, increase for longer prompts
    },
    "chroma1_hd": {
        "num_inference_steps": 40,
        "guidance_scale": 3.0,
    },
    "sdxl_food": {
        "num_inference_steps": 30,
        "guidance_scale": 7.5,  # Standard SDXL guidance
    },
    "comfyui_flux": {
        "num_inference_steps": 20,
        "guidance_scale": 4.0,  # FluxGuidance value
        "width": 1024,
        "height": 1024,
    },
    "qwen_turbo": {
        "num_inference_steps": 50,
        "true_cfg_scale": 4.0,  # Qwen uses true_cfg_scale, not guidance_scale
        "width": 1328,
        "height": 1328,  # Native 2K resolution
    },
    "flux2_klein": {
        "num_inference_steps": 4,  # Step-distilled to 4 steps
        "guidance_scale": 1.0,
    },
}


# ============== R2 Storage ==============
_s3_client = None


def get_s3_client():
    """Get S3 client configured for Cloudflare R2"""
    global _s3_client
    if _s3_client is not None:
        return _s3_client

    if not all([R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY]):
        print("Warning: R2 credentials not configured. Images will only be saved locally.")
        return None

    _s3_client = boto3.client(
        's3',
        endpoint_url=f'https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com',
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version='s3v4'),
        region_name='auto'
    )
    return _s3_client


def upload_to_r2(image: Image.Image, recipe_id: int, cuisine: str, model: str, data_type: str = "meals", slug: str = None) -> str:
    """Upload image to R2 and return public URL

    Args:
        image: PIL Image to upload
        recipe_id: ID of the recipe/ingredient
        cuisine: Cuisine name (for meals) or category (for ingredients)
        model: Model name (flux2_dev, zimage_turbo, etc.)
        data_type: Either "meals" or "ingredients" for folder separation
        slug: Optional slug for filename (defaults to recipe_id)
    """
    from urllib.parse import quote

    s3 = get_s3_client()
    if s3 is None:
        return ""

    # Convert image to bytes
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    buffer.seek(0)

    # Create key with data_type (meals/ingredients) for folder separation
    # Use slug for filename if provided, otherwise use recipe_id
    safe_cuisine = cuisine.lower().replace(" ", "_")
    filename = slug if slug else recipe_id
    key = f"images/{model}/{data_type}/{safe_cuisine}/{filename}.png"

    try:
        s3.put_object(
            Bucket=R2_BUCKET,
            Key=key,
            Body=buffer.getvalue(),
            ContentType='image/png'
        )

        if R2_PUBLIC_URL:
            # URL-encode the path components for proper URL handling
            encoded_key = quote(key, safe='/')
            return f"{R2_PUBLIC_URL}/{encoded_key}"
        return f"r2://{R2_BUCKET}/{key}"
    except Exception as e:
        print(f"  Warning: Failed to upload to R2: {e}")
        return ""


# ============== State Management ==============
_last_manifest_upload = 0
_last_state_upload = 0


def get_state_file(model: str) -> str:
    """Get model-specific state file path"""
    return f"generation_state_{model}.json"


def load_state(model: str) -> dict:
    """Load generation state from model-specific file"""
    state_file = get_state_file(model)
    if os.path.exists(state_file):
        with open(state_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "completed": {},  # recipe_id -> generation info
        "failed": {},     # recipe_id -> error info
        "last_index": 0,
        "started_at": None,
        "model": model
    }


def save_state(state: dict, model: str):
    """Save generation state to model-specific file"""
    state_file = get_state_file(model)
    with open(state_file, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def get_state_r2_key(model: str) -> str:
    """Get R2 key for state file"""
    return f"data/state_{model}.json"


def download_state_from_r2(model: str) -> dict | None:
    """Download state from R2, returns None if not found"""
    s3 = get_s3_client()
    if s3 is None:
        return None

    try:
        response = s3.get_object(Bucket=R2_BUCKET, Key=get_state_r2_key(model))
        return json.loads(response['Body'].read().decode('utf-8'))
    except s3.exceptions.NoSuchKey:
        return None
    except Exception as e:
        print(f"  Warning: Failed to download state from R2: {e}")
        return None


def upload_state_to_r2(state: dict, model: str):
    """Upload state to R2"""
    s3 = get_s3_client()
    if s3 is None:
        return

    try:
        s3.put_object(
            Bucket=R2_BUCKET,
            Key=get_state_r2_key(model),
            Body=json.dumps(state, indent=2, ensure_ascii=False).encode('utf-8'),
            ContentType='application/json'
        )
    except Exception as e:
        print(f"  Warning: Failed to upload state to R2: {e}")


def merge_states(local: dict, remote: dict) -> dict:
    """Merge local and remote states, taking union of completed/failed"""
    merged = {
        "completed": {},
        "failed": {},
        "last_index": max(local.get("last_index", 0), remote.get("last_index", 0)),
        "started_at": local.get("started_at") or remote.get("started_at"),
        "model": local.get("model") or remote.get("model")
    }

    # Merge completed: take all completed from both, prefer newer timestamp
    for state in [local, remote]:
        for key, info in state.get("completed", {}).items():
            if key not in merged["completed"]:
                merged["completed"][key] = info
            else:
                # Keep the one with newer timestamp
                existing_time = merged["completed"][key].get("generated_at", "")
                new_time = info.get("generated_at", "")
                if new_time > existing_time:
                    merged["completed"][key] = info

    # Merge failed: only keep failures that aren't completed
    for state in [local, remote]:
        for key, info in state.get("failed", {}).items():
            if key not in merged["completed"] and key not in merged["failed"]:
                merged["failed"][key] = info

    return merged


def sync_state_from_r2(model: str) -> dict:
    """Sync state: download from R2, merge with local, save and upload merged"""
    print(f"Syncing state for {model} from R2...")

    local_state = load_state(model)
    remote_state = download_state_from_r2(model)

    if remote_state is None:
        print(f"  No remote state found, using local only")
        return local_state

    merged = merge_states(local_state, remote_state)

    local_completed = len(local_state.get("completed", {}))
    remote_completed = len(remote_state.get("completed", {}))
    merged_completed = len(merged["completed"])

    print(f"  Local: {local_completed}, Remote: {remote_completed}, Merged: {merged_completed}")

    # Save merged state locally and to R2
    save_state(merged, model)
    upload_state_to_r2(merged, model)

    return merged


def sync_state_if_needed(state: dict, model: str, force: bool = False):
    """Merge local state with R2 and upload if 60+ seconds since last sync"""
    global _last_state_upload
    now = time.time()

    if not force and (now - _last_state_upload) < 60:
        return  # Skip if less than 60 seconds since last upload

    # Download current R2 state and merge to avoid overwriting other machine's progress
    remote_state = download_state_from_r2(model)
    if remote_state:
        merged = merge_states(state, remote_state)
        # Update local state with merged data
        state["completed"] = merged["completed"]
        state["failed"] = merged["failed"]

    upload_state_to_r2(state, model)
    _last_state_upload = now


def load_recipes() -> list:
    """Load recipes from JSON file"""
    with open(RECIPES_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def sync_manifest_if_needed(recipes: list, force: bool = False):
    """Upload manifest to R2 if 60+ seconds since last upload

    Merges local state with existing R2 manifest, keeping newer timestamps.
    This prevents one machine from overwriting fresh timestamps from another.
    """
    global _last_manifest_upload
    now = time.time()

    if not force and (now - _last_manifest_upload) < 60:
        return  # Skip if less than 60 seconds since last upload

    s3 = get_s3_client()
    if s3 is None:
        return

    recipe_lookup = {str(r["id"]): r for r in recipes}

    # First, download existing manifest from R2 to preserve timestamps from other machines
    existing_manifest = {}
    try:
        response = s3.get_object(Bucket=R2_BUCKET, Key="data/manifest.json")
        existing_items = json.loads(response['Body'].read().decode('utf-8'))
        # Index by unique key (id_model)
        for item in existing_items:
            key = f"{item['id']}_{item['model']}"
            existing_manifest[key] = item
    except Exception:
        pass  # No existing manifest, start fresh

    # Build manifest from local state, but preserve newer timestamps from R2
    for model in ["zimage_turbo", "flux2_dev", "sdxl_food"]:
        state = load_state(model)
        for state_key, info in state["completed"].items():
            # Handle composite keys (data_type:id) or legacy numeric keys
            if ":" in state_key:
                data_type, recipe_id = state_key.split(":", 1)
            else:
                data_type = "meals"
                recipe_id = state_key

            recipe = recipe_lookup.get(recipe_id, {})
            entry_model = info.get("model", model)
            manifest_key = f"{recipe_id}_{entry_model}"

            # Handle both numeric and string IDs
            try:
                entry_id = int(recipe_id)
            except (ValueError, TypeError):
                entry_id = recipe_id

            new_entry = {
                "id": entry_id,
                "data_type": data_type,
                "meal_name": info["meal_name"],
                "cuisine": info["cuisine"],
                "image_url": info.get("r2_url", ""),
                "prompt": recipe.get("image_prompt") or recipe.get("flux_prompt", ""),
                "model": entry_model,
                "generated_at": info.get("generated_at", ""),
                "generation_count": info.get("generation_count", 1),
            }

            # Only update if local timestamp is newer or entry doesn't exist
            if manifest_key in existing_manifest:
                existing_time = existing_manifest[manifest_key].get("generated_at", "")
                local_time = new_entry.get("generated_at", "")
                if local_time > existing_time:
                    existing_manifest[manifest_key] = new_entry
                # else: keep existing (it has newer timestamp from another machine)
            else:
                existing_manifest[manifest_key] = new_entry

    # Convert back to list
    manifest = list(existing_manifest.values())

    try:
        manifest_json = json.dumps(manifest, indent=2, ensure_ascii=False)
        s3.put_object(
            Bucket=R2_BUCKET,
            Key="data/manifest.json",
            Body=manifest_json.encode('utf-8'),
            ContentType='application/json'
        )
        _last_manifest_upload = now
        print(f"  [Manifest synced: {len(manifest)} images]")
    except Exception as e:
        print(f"  Warning: Failed to sync manifest: {e}")


# Manifest lock for thread-safe updates
_manifest_lock = threading.Lock()


def add_to_manifest(entry: dict):
    """Add or update a single entry in the R2 manifest (thread-safe)

    Used in orchestrator mode where we need to update manifest incrementally.
    """
    s3 = get_s3_client()
    if s3 is None:
        return False

    with _manifest_lock:
        try:
            # Download current manifest
            try:
                response = s3.get_object(Bucket=R2_BUCKET, Key="data/manifest.json")
                manifest = json.loads(response['Body'].read().decode('utf-8'))
            except Exception:
                manifest = []

            # Create unique key for this entry
            entry_key = f"{entry['id']}_{entry['model']}"

            # Update or add entry
            updated = False
            for i, item in enumerate(manifest):
                if f"{item['id']}_{item['model']}" == entry_key:
                    manifest[i] = entry
                    updated = True
                    break

            if not updated:
                manifest.append(entry)

            # Upload updated manifest
            s3.put_object(
                Bucket=R2_BUCKET,
                Key="data/manifest.json",
                Body=json.dumps(manifest, indent=2, ensure_ascii=False).encode('utf-8'),
                ContentType='application/json'
            )
            return True
        except Exception as e:
            print(f"  Warning: Failed to update manifest: {e}")
            return False


# ============== Image Generation ==============
def structured_prompt_to_text(prompt_dict: dict) -> str:
    """Convert structured prompt dict to text prompt for diffusers pipeline.

    Expected structure:
        scene_description: Main scene description
        subject: {dish, garnish, texture, presentation, ...}
        camera: {angle, lens}
        lighting: {style}
        style: {genre, presentation}
        setting: {surface}
        quality: {detail, colors, bokeh}
        composition: {depth_of_field, focus}
    """
    parts = []

    # Main scene description
    if scene := prompt_dict.get("scene_description"):
        parts.append(scene)

    # Subject details
    if subject := prompt_dict.get("subject"):
        if isinstance(subject, dict):
            subject_parts = []
            for key in ["dish", "presentation", "texture", "garnish"]:
                if val := subject.get(key):
                    # Handle list values (e.g., garnish: ["mint", "ice"])
                    if isinstance(val, list):
                        subject_parts.append(", ".join(str(v) for v in val))
                    else:
                        subject_parts.append(str(val))
            if subject_parts:
                parts.append(" ".join(subject_parts))
        elif isinstance(subject, str):
            parts.append(subject)

    # Camera and lighting
    if camera := prompt_dict.get("camera"):
        if isinstance(camera, dict):
            if angle := camera.get("angle"):
                parts.append(angle)
            if lens := camera.get("lens"):
                parts.append(f"{lens} lens")
        elif isinstance(camera, str):
            parts.append(camera)

    if lighting := prompt_dict.get("lighting"):
        if isinstance(lighting, dict):
            if style := lighting.get("style"):
                parts.append(style)
        elif isinstance(lighting, str):
            parts.append(lighting)

    # Style
    if style := prompt_dict.get("style"):
        if isinstance(style, dict):
            if genre := style.get("genre"):
                parts.append(genre)
        elif isinstance(style, str):
            parts.append(style)

    # Quality hints
    if quality := prompt_dict.get("quality"):
        if isinstance(quality, dict):
            quality_parts = []
            for key in ["detail", "colors"]:
                if val := quality.get(key):
                    quality_parts.append(f"{val} {key}")
            if quality_parts:
                parts.append(", ".join(quality_parts))

    return ". ".join(parts) if parts else str(prompt_dict)


def generate_image(prompt, model: str = "flux2_dev", seed: int = -1, progress_callback=None) -> Image.Image:
    """Generate a single image using specified model

    Args:
        prompt: Text prompt string or structured prompt dict (converted to text)
        progress_callback: Optional callback(step, total_steps) for progress reporting
    """
    global _model_last_used
    _model_last_used = time.time()  # Reset idle timer

    # Convert structured dict prompts to text
    if isinstance(prompt, dict):
        prompt = structured_prompt_to_text(prompt)

    torch = get_torch()
    pipe = get_pipeline(model)
    settings = MODEL_SETTINGS[model]

    generator = None
    if seed >= 0:
        generator = torch.Generator("cuda").manual_seed(seed)

    # Create step callback for progress reporting
    def step_callback(pipe, step, timestep, callback_kwargs):
        if progress_callback:
            total = settings["num_inference_steps"]
            progress_callback(step + 1, total)
        return callback_kwargs

    callback_on_step_end = step_callback if progress_callback else None

    # Handle model-specific generation APIs
    if model == "qwen_turbo":
        # Qwen-Image-2512 via diffusers - uses true_cfg_scale instead of guidance_scale
        width = settings.get("width", IMAGE_WIDTH)
        height = settings.get("height", IMAGE_HEIGHT)
        result = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=settings["num_inference_steps"],
            true_cfg_scale=settings["true_cfg_scale"],
            generator=generator,
            callback_on_step_end=callback_on_step_end,
        )

    elif model == "sdxl_food":
        compel = get_compel(model)
        if compel:
            # Generate embeddings from long prompt
            conditioning, pooled = compel(prompt)
            result = pipe(
                prompt_embeds=conditioning,
                pooled_prompt_embeds=pooled,
                height=IMAGE_HEIGHT,
                width=IMAGE_WIDTH,
                num_inference_steps=settings["num_inference_steps"],
                guidance_scale=settings["guidance_scale"],
                generator=generator,
                callback_on_step_end=callback_on_step_end,
            )
        else:
            # Fallback to standard prompt (will truncate)
            result = pipe(
                prompt=prompt,
                height=IMAGE_HEIGHT,
                width=IMAGE_WIDTH,
                num_inference_steps=settings["num_inference_steps"],
                guidance_scale=settings["guidance_scale"],
                generator=generator,
                callback_on_step_end=callback_on_step_end,
            )
    else:
        # Build kwargs - add max_sequence_length for FLUX-based models (zimage_turbo)
        pipe_kwargs = {
            "prompt": prompt,
            "height": IMAGE_HEIGHT,
            "width": IMAGE_WIDTH,
            "num_inference_steps": settings["num_inference_steps"],
            "guidance_scale": settings["guidance_scale"],
            "generator": generator,
            "callback_on_step_end": callback_on_step_end,
        }
        if "max_sequence_length" in settings:
            pipe_kwargs["max_sequence_length"] = settings["max_sequence_length"]
        if "sigmas" in settings:
            pipe_kwargs["sigmas"] = settings["sigmas"]
        result = pipe(**pipe_kwargs)

    return result.images[0]


# ============== ComfyUI Generation ==============
_comfyui_client = None


def get_comfyui_client(url: str = None):
    """Get or create ComfyUI client

    Args:
        url: ComfyUI server URL. If not provided, uses COMFYUI_URL env var
             or defaults to http://127.0.0.1:8188
    """
    global _comfyui_client

    # Allow override with different URL
    comfyui_url = url or os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")

    # Create new client if URL changed or client doesn't exist
    if _comfyui_client is None or getattr(_comfyui_client, 'base_url', None) != comfyui_url:
        from comfyui_client import ComfyUIClient
        _comfyui_client = ComfyUIClient(base_url=comfyui_url)

    return _comfyui_client


def generate_image_comfyui(
    prompt: str | dict,
    model: str = "comfyui_flux",
    seed: int = -1,
    comfyui_url: str = None,
    progress_callback=None,
    max_retries: int = 3,
) -> Image.Image:
    """Generate image using ComfyUI backend

    Args:
        prompt: Text prompt for image generation
        model: Model identifier (used to get settings from MODEL_SETTINGS)
        seed: Random seed (-1 for random)
        comfyui_url: ComfyUI server URL (optional, uses env var if not provided)
        progress_callback: Optional callback (not fully supported for ComfyUI)
        max_retries: Number of retries on connection failure

    Returns:
        PIL Image of the generated result
    """
    # Convert structured dict prompts to text
    if isinstance(prompt, dict):
        prompt = structured_prompt_to_text(prompt)

    import time
    from comfyui_workflows import build_flux2_txt2img_workflow

    settings = MODEL_SETTINGS.get(model, MODEL_SETTINGS["comfyui_flux"])
    client = get_comfyui_client(comfyui_url)

    # Build the workflow with parameters
    workflow = build_flux2_txt2img_workflow(
        prompt=prompt,
        seed=seed,
        steps=settings.get("num_inference_steps", 20),
        guidance=settings.get("guidance_scale", 4.0),
        width=settings.get("width", 1024),
        height=settings.get("height", 1024),
    )

    # Execute with retry logic
    last_error = None
    for attempt in range(max_retries):
        try:
            result = client.execute_workflow(workflow, timeout=600)

            if not result.get("images"):
                raise Exception("No images returned from ComfyUI")

            return result["images"][0]

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                print(f"  ComfyUI error: {e}, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise Exception(f"ComfyUI generation failed after {max_retries} attempts: {last_error}")


def save_local(image: Image.Image, recipe_id: int, cuisine: str, model: str, data_type: str = "meals") -> str:
    """Save image locally with model and data type distinction"""
    safe_cuisine = cuisine.lower().replace(" ", "_")
    # Organize by model/data_type/cuisine for clear distinction between outputs
    output_path = Path(OUTPUT_DIR) / model / data_type / safe_cuisine
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / f"{recipe_id}.png"
    image.save(filepath, format="PNG", optimize=True)
    return str(filepath)


# ============== Main Batch Processing ==============
def process_batch(
    start_index: int = 0,
    limit: int = None,
    skip_uploaded: bool = True,
    regenerate_failed: bool = False,
    dry_run: bool = False,
    model: str = "flux2_dev",
    skip_sync: bool = False,
    no_compile: bool = False,
    sync_state: bool = False
):
    """Process recipes in batch"""

    # Check and sync model with HuggingFace before starting
    if not skip_sync and not dry_run:
        print("\n" + "=" * 60)
        print("Pre-flight Check: Syncing model with HuggingFace")
        print("=" * 60)
        if not check_and_sync_model(model, auto_update=True):
            print("Warning: Could not verify model is up to date. Continuing anyway...")
        print("=" * 60 + "\n")

    recipes = load_recipes()

    # Sync state from R2 if requested (for multi-system coordination)
    if sync_state and not dry_run:
        state = sync_state_from_r2(model)
    else:
        state = load_state(model)

    if state.get("started_at") is None:
        state["started_at"] = datetime.now().isoformat()

    # Ensure required keys exist
    if "completed" not in state:
        state["completed"] = {}
    if "failed" not in state:
        state["failed"] = {}

    # Detect data type based on first recipe's schema (needed for state key)
    sample_recipe = recipes[0] if recipes else {}
    data_type = "ingredients" if "ingredient_name" in sample_recipe else "meals"

    # Determine which recipes to process
    to_process = []
    for i, recipe in enumerate(recipes):
        recipe_id = str(recipe["id"])
        # Use composite key: data_type:id to allow overlapping IDs between meals and ingredients
        state_key = f"{data_type}:{recipe_id}"

        # Skip already completed unless regenerating
        if state_key in state["completed"] and skip_uploaded:
            continue

        # Handle failed recipes
        if state_key in state["failed"]:
            if not regenerate_failed:
                continue

        to_process.append((i, recipe))

    # Apply start index and limit
    if start_index > 0:
        to_process = to_process[start_index:]
    if limit:
        to_process = to_process[:limit]

    # Auto-set compile flag based on batch size (unless explicitly disabled)
    global _use_compile
    if no_compile:
        _use_compile = False
    elif len(to_process) < 50:
        _use_compile = False
    else:
        _use_compile = True

    model_names = {"flux2_dev": "FLUX.2-dev (4-bit)", "flux2_fp8": "FLUX.2-dev (FP8)", "flux2_klein": "FLUX.2-klein-9B", "zimage_turbo": "Z-Image-Turbo (8-step)", "sdxl_food": "SDXL + Food LoRA (compel)"}
    print(f"\n{'='*60}")
    print(f"Batch Recipe Image Generator")
    print(f"{'='*60}")
    print(f"Total recipes: {len(recipes)}")
    print(f"Already completed: {len(state['completed'])}")
    print(f"Previously failed: {len(state['failed'])}")
    print(f"To process this run: {len(to_process)}")
    print(f"Model: {model_names.get(model, model)}")
    print(f"Data type: {data_type}")
    print(f"torch.compile: {'enabled' if _use_compile else 'disabled'} ({'<50 images' if len(to_process) < 50 else '>=50 images'})")
    print(f"R2 path: images/{model}/{data_type}/<category>/<id>.png")
    print(f"{'='*60}\n")

    if dry_run:
        print("DRY RUN - No images will be generated")
        for i, recipe in to_process[:10]:
            item_name = recipe.get("meal_name") or recipe.get("ingredient_name", f"Item {recipe['id']}")
            print(f"  Would process: {recipe['id']} - {item_name}")
        if len(to_process) > 10:
            print(f"  ... and {len(to_process) - 10} more")
        return

    # Process recipes
    for idx, (orig_idx, recipe) in enumerate(to_process):
        recipe_id = str(recipe["id"])
        # Use composite key: data_type:id to allow overlapping IDs between meals and ingredients
        state_key = f"{data_type}:{recipe_id}"
        # Support both meal and ingredient schemas
        meal_name = recipe.get("meal_name") or recipe.get("ingredient_name", f"Item {recipe_id}")
        cuisine = recipe.get("cuisine_name") or recipe.get("category", "general")
        prompt = recipe.get("image_prompt") or recipe.get("flux_prompt")
        seed = recipe.get("recommended_seed", -1)  # Use recommended seed if provided

        seed_msg = f", seed={seed}" if seed != -1 else ""
        print(f"[{idx+1}/{len(to_process)}] Processing: {meal_name} (ID: {recipe_id}{seed_msg})")

        try:
            start_time = time.time()

            # Generate image with seed from JSON (if provided)
            image = generate_image(prompt, model=model, seed=seed)
            gen_time = time.time() - start_time

            # Skip local save - only upload to R2
            local_path = ""

            # Upload to R2 with data_type for folder separation
            r2_url = upload_to_r2(image, recipe["id"], cuisine, model, data_type)
            if r2_url:
                print(f"  Uploaded to R2: {r2_url}")

            # Update state with composite key (data_type:id)
            state["completed"][state_key] = {
                "meal_name": meal_name,
                "cuisine": cuisine,
                "data_type": data_type,
                "local_path": local_path,
                "r2_url": r2_url,
                "model": model,
                "generated_at": datetime.now().isoformat(),
                "generation_time": round(gen_time, 2),
                "generation_count": state["completed"].get(state_key, {}).get("generation_count", 0) + 1
            }

            # Remove from failed if it was there
            if state_key in state["failed"]:
                del state["failed"][state_key]

            state["last_index"] = orig_idx
            save_state(state, model)

            print(f"  Done in {gen_time:.1f}s")

            # Sync manifest to R2 (rate limited to once per minute)
            sync_manifest_if_needed(recipes)

            # Sync state to R2 for multi-system coordination (rate limited)
            if sync_state:
                sync_state_if_needed(state, model)

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Progress saved.")
            save_state(state, model)
            if sync_state:
                upload_state_to_r2(state, model)
            break

        except Exception as e:
            print(f"  ERROR: {e}")
            state["failed"][state_key] = {
                "meal_name": meal_name,
                "cuisine": cuisine,
                "data_type": data_type,
                "error": str(e),
                "failed_at": datetime.now().isoformat()
            }
            save_state(state, model)
            continue

    # Final summary
    print(f"\n{'='*60}")
    print(f"Batch Complete")
    print(f"{'='*60}")
    print(f"Completed: {len(state['completed'])}")
    print(f"Failed: {len(state['failed'])}")
    print(f"Remaining: {len(recipes) - len(state['completed'])}")
    print(f"{'='*60}")


# ============== Orchestrator Node Mode ==============
import requests
import socket
import uuid

_node_id = None
_orchestrator_url = None
_heartbeat_thread = None
_stop_heartbeat = False


def get_node_id():
    """Generate or retrieve persistent node ID"""
    global _node_id
    if _node_id:
        return _node_id

    node_file = Path(".node_id")
    if node_file.exists():
        _node_id = node_file.read_text().strip()
    else:
        _node_id = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
        node_file.write_text(_node_id)

    return _node_id


def get_gpu_stats():
    """Get current GPU utilization and memory"""
    torch = get_torch()
    if not torch.cuda.is_available():
        return {"gpu_util": 0, "vram_used": 0, "vram_total": 0}

    try:
        # Use nvidia-smi for accurate utilization (torch doesn't provide this)
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            return {
                "gpu_util": int(parts[0].strip()),
                "vram_used": float(parts[1].strip()) / 1024,  # Convert MB to GB
                "vram_total": float(parts[2].strip()) / 1024,
            }
    except Exception:
        pass

    # Fallback to torch memory stats
    props = torch.cuda.get_device_properties(0)
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    total = props.total_memory / (1024**3)
    return {"gpu_util": 0, "vram_used": allocated, "vram_total": total}


def heartbeat_worker(node_id, orchestrator_url, current_task_ref):
    """Background thread that sends heartbeats to orchestrator"""
    global _stop_heartbeat

    while not _stop_heartbeat:
        try:
            stats = get_gpu_stats()
            requests.post(
                f"{orchestrator_url}/api/heartbeat",
                json={
                    "node_id": node_id,
                    "status": "generating" if current_task_ref[0] else "idle",
                    "current_task": current_task_ref[0],
                    "gpu_util": stats["gpu_util"],
                    "vram_used": round(stats["vram_used"], 1),
                },
                timeout=5
            )
        except Exception as e:
            print(f"  [Heartbeat] Error: {e}")

        time.sleep(10)


def register_with_orchestrator(orchestrator_url, models, node_name=None):
    """Register this node with the orchestrator"""
    torch = get_torch()
    node_id = get_node_id()

    # Ensure models is a list
    if isinstance(models, str):
        models = [models]

    gpu_name = "CPU"
    vram_gb = 0
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)

    response = requests.post(
        f"{orchestrator_url}/api/register",
        json={
            "node_id": node_id,
            "name": node_name or socket.gethostname(),
            "gpu": gpu_name,
            "vram_gb": vram_gb,
            "models": models,
        },
        timeout=10
    )

    if response.status_code == 200:
        models_str = ", ".join(models)
        print(f"Registered with orchestrator as: {node_id} (models: {models_str})")
        return node_id
    else:
        raise Exception(f"Failed to register: {response.text}")


def run_as_node(orchestrator_url, models="flux2_dev", node_name=None, comfyui_url=None):
    """Run as orchestrated node - request tasks from orchestrator

    Args:
        models: Single model string or comma-separated list (e.g., "flux2_dev,zimage_turbo")
        comfyui_url: ComfyUI server URL for comfyui_* models (optional, uses env var if not provided)
    """
    global _stop_heartbeat, _heartbeat_thread

    # Parse comma-separated models
    if isinstance(models, str):
        models = [m.strip() for m in models.split(",")]

    print("\n" + "=" * 60)
    print("Running as Orchestrated Node")
    print("=" * 60)
    print(f"Orchestrator: {orchestrator_url}")
    print(f"Models: {', '.join(models)}")

    # Start idle model unload thread
    start_idle_check_thread()

    # Current task and model reference for heartbeat thread
    current_task_ref = [None]
    current_model_ref = [None]
    is_registered = [False]
    node_id_ref = [None]

    def try_register():
        """Attempt to register with orchestrator, returns True on success"""
        try:
            node_id_ref[0] = register_with_orchestrator(orchestrator_url, models, node_name)
            is_registered[0] = True
            return True
        except Exception as e:
            print(f"  [Register] Failed: {e}")
            is_registered[0] = False
            return False

    # Initial registration with retries
    max_retries = 5
    retry_delay = 5
    for attempt in range(max_retries):
        if try_register():
            break
        print(f"  Retrying registration in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
        time.sleep(retry_delay)
        retry_delay = min(retry_delay * 2, 60)  # Exponential backoff, max 60s
    else:
        print("Failed to register after multiple attempts. Exiting.")
        return

    # Modified heartbeat worker with reconnection detection
    def heartbeat_worker_with_model():
        global _stop_heartbeat
        consecutive_failures = 0
        while not _stop_heartbeat:
            try:
                stats = get_gpu_stats()
                response = requests.post(
                    f"{orchestrator_url}/api/heartbeat",
                    json={
                        "node_id": node_id_ref[0],
                        "status": "generating" if current_task_ref[0] else "idle",
                        "current_task": current_task_ref[0],
                        "current_model": current_model_ref[0],
                        "gpu_util": stats["gpu_util"],
                        "vram_used": round(stats["vram_used"], 1),
                    },
                    timeout=5
                )
                # Check if orchestrator says we need to re-register
                if response.status_code == 404:
                    print("\n  [Heartbeat] Orchestrator doesn't recognize us - will re-register")
                    is_registered[0] = False
                consecutive_failures = 0
            except Exception as e:
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    print(f"\n  [Heartbeat] Connection lost ({consecutive_failures} failures)")
                    is_registered[0] = False
            time.sleep(10)

    # Start heartbeat thread
    _stop_heartbeat = False
    _heartbeat_thread = threading.Thread(target=heartbeat_worker_with_model, daemon=True)
    _heartbeat_thread.start()
    print("Heartbeat thread started")

    # Track currently loaded model (lazy loading - don't preload)
    loaded_model = None

    tasks_completed = 0
    tasks_failed = 0
    consecutive_request_failures = 0

    try:
        while True:
            # Check if we need to re-register (orchestrator may have restarted)
            if not is_registered[0]:
                print("\n  Attempting to reconnect to orchestrator...")
                reconnect_delay = 5
                while not is_registered[0] and not _stop_heartbeat:
                    if try_register():
                        print("  Reconnected successfully!")
                        consecutive_request_failures = 0
                        break
                    print(f"  Retrying in {reconnect_delay}s...")
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, 60)
                continue

            # Request next task
            try:
                response = requests.post(
                    f"{orchestrator_url}/api/task/request",
                    json={"node_id": node_id_ref[0]},
                    timeout=10
                )

                # Check for 404 (node not registered)
                if response.status_code == 404:
                    print("  Orchestrator doesn't recognize us - re-registering...")
                    is_registered[0] = False
                    continue

                data = response.json()
                consecutive_request_failures = 0

            except requests.exceptions.ConnectionError:
                consecutive_request_failures += 1
                if consecutive_request_failures >= 3:
                    print(f"  Connection lost after {consecutive_request_failures} failures - will reconnect")
                    is_registered[0] = False
                else:
                    print(f"  Connection error, retrying... ({consecutive_request_failures}/3)")
                time.sleep(5)
                continue

            except Exception as e:
                print(f"Error requesting task: {e}")
                time.sleep(5)
                continue

            task = data.get("task")
            if not task:
                # No tasks available, wait and retry
                print("No tasks available, waiting...")
                time.sleep(5)
                continue

            # Process task
            task_id = task["id"]
            task_model = task["model"]
            current_task_ref[0] = task_id

            # Check if we need to switch models
            # Skip pipeline loading for ComfyUI models (they use external server)
            is_comfyui_model = task_model.startswith("comfyui_")

            if loaded_model != task_model:
                if loaded_model:
                    print(f"\n  Switching model: {loaded_model} -> {task_model}")
                    # Unload previous model to free VRAM (only for diffusers models)
                    if loaded_model in _pipes:
                        del _pipes[loaded_model]
                        gc.collect()
                        torch = get_torch()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                else:
                    print(f"\n  Loading model: {task_model}")

                # Only load pipeline for non-ComfyUI models
                if not is_comfyui_model:
                    get_pipeline(task_model)
                loaded_model = task_model
                current_model_ref[0] = task_model

            print(f"\n[Task {tasks_completed + 1}] {task['meal_name']} (ID: {task['recipe_id']}) [{task_model}]")

            try:
                start_time = time.time()

                # Progress callback to report steps to orchestrator
                def report_progress(step, total_steps):
                    try:
                        progress_pct = int((step / total_steps) * 100)
                        requests.post(
                            f"{orchestrator_url}/api/task/progress",
                            json={
                                "node_id": node_id_ref[0],
                                "task_id": task_id,
                                "progress": progress_pct,
                                "step": step,
                                "total_steps": total_steps,
                            },
                            timeout=2
                        )
                    except:
                        pass  # Don't let progress reporting failures affect generation

                # Generate image using task's model with progress reporting
                # Route to ComfyUI for comfyui_* models
                if task_model.startswith("comfyui_"):
                    image = generate_image_comfyui(
                        task["prompt"],
                        model=task_model,
                        seed=task.get("seed", -1),
                        comfyui_url=comfyui_url,
                    )
                else:
                    image = generate_image(task["prompt"], model=task_model, seed=task.get("seed", -1), progress_callback=report_progress)
                gen_time = time.time() - start_time

                # Upload to R2 - handle both numeric and string IDs
                try:
                    upload_id = int(task["recipe_id"])
                except (ValueError, TypeError):
                    upload_id = task["recipe_id"]

                # Use slug for filename if provided by orchestrator
                slug = task.get("slug")

                r2_url = upload_to_r2(
                    image,
                    upload_id,
                    task["cuisine"],
                    task_model,
                    "meals",  # Default data type
                    slug=slug
                )

                # Report completion (orchestrator handles manifest updates)
                # This eliminates race conditions from multiple nodes writing to manifest
                requests.post(
                    f"{orchestrator_url}/api/task/complete",
                    json={
                        "node_id": node_id_ref[0],
                        "task_id": task_id,
                        "result": {
                            "r2_url": r2_url,
                            "generation_time": round(gen_time, 2),
                            "model": task_model,
                        }
                    },
                    timeout=10
                )

                tasks_completed += 1
                print(f"  Done in {gen_time:.1f}s -> {r2_url}")

            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"  ERROR: {e}")

                # Report failure
                try:
                    requests.post(
                        f"{orchestrator_url}/api/task/fail",
                        json={
                            "node_id": node_id_ref[0],
                            "task_id": task_id,
                            "error": str(e),
                        },
                        timeout=10
                    )
                except Exception:
                    pass

                tasks_failed += 1

            finally:
                current_task_ref[0] = None

    except KeyboardInterrupt:
        print("\n\nShutting down node...")

    finally:
        _stop_heartbeat = True
        print(f"\nSession complete: {tasks_completed} completed, {tasks_failed} failed")


def export_manifest():
    """Export a manifest JSON for the review app and upload to R2

    Merges local state with existing R2 manifest, keeping newer timestamps.
    """
    recipes = load_recipes()

    # Build recipe lookup
    recipe_lookup = {str(r["id"]): r for r in recipes}

    s3 = get_s3_client()

    # First, download existing manifest from R2 to preserve timestamps from other machines
    existing_manifest = {}
    if s3:
        try:
            response = s3.get_object(Bucket=R2_BUCKET, Key="data/manifest.json")
            existing_items = json.loads(response['Body'].read().decode('utf-8'))
            # Index by unique key (id_model)
            for item in existing_items:
                key = f"{item['id']}_{item['model']}"
                existing_manifest[key] = item
        except Exception:
            pass  # No existing manifest, start fresh

    # Merge local state, keeping newer timestamps
    for model in ["zimage_turbo", "flux2_dev", "sdxl_food"]:
        state = load_state(model)
        for state_key, info in state["completed"].items():
            # Handle composite keys (data_type:id) or legacy numeric keys
            if ":" in state_key:
                data_type, recipe_id = state_key.split(":", 1)
            else:
                data_type = "meals"
                recipe_id = state_key

            recipe = recipe_lookup.get(recipe_id, {})
            entry_model = info.get("model", model)
            manifest_key = f"{recipe_id}_{entry_model}"

            # Handle both numeric and string IDs
            try:
                entry_id = int(recipe_id)
            except (ValueError, TypeError):
                entry_id = recipe_id

            new_entry = {
                "id": entry_id,
                "data_type": data_type,
                "meal_name": info["meal_name"],
                "cuisine": info["cuisine"],
                "image_url": info.get("r2_url", ""),
                "prompt": recipe.get("image_prompt") or recipe.get("flux_prompt", ""),
                "model": entry_model,
                "generated_at": info.get("generated_at", ""),
                "generation_count": info.get("generation_count", 1),
            }

            # Only update if local timestamp is newer or entry doesn't exist
            if manifest_key in existing_manifest:
                existing_time = existing_manifest[manifest_key].get("generated_at", "")
                local_time = new_entry.get("generated_at", "")
                if local_time > existing_time:
                    existing_manifest[manifest_key] = new_entry
            else:
                existing_manifest[manifest_key] = new_entry

    # Convert back to list
    manifest = list(existing_manifest.values())

    # Upload manifest to R2
    if s3:
        manifest_json = json.dumps(manifest, indent=2, ensure_ascii=False)
        try:
            s3.put_object(
                Bucket=R2_BUCKET,
                Key="data/manifest.json",
                Body=manifest_json.encode('utf-8'),
                ContentType='application/json'
            )
            print(f"Uploaded manifest to R2: {R2_PUBLIC_URL}/data/manifest.json")
        except Exception as e:
            print(f"Failed to upload manifest to R2: {e}")

    # Also save locally for reference
    manifest_path = "image_manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Exported manifest with {len(manifest)} images to {manifest_path}")
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Recipe Image Generator")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--limit", type=int, default=None, help="Max images to generate")
    parser.add_argument("--model", type=str, default="zimage_turbo",
                        help="Model(s) to use, comma-separated for multi-model node (e.g., flux2_dev,zimage_turbo)")
    parser.add_argument("--regenerate-failed", action="store_true", help="Retry failed images")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed")
    parser.add_argument("--export-manifest", action="store_true", help="Export manifest for review app")
    parser.add_argument("--status", action="store_true", help="Show current progress")
    parser.add_argument("--check-updates", action="store_true", help="Check for model updates from HuggingFace")
    parser.add_argument("--update-models", action="store_true", help="Force re-download all models from HuggingFace")
    parser.add_argument("--no-sync", action="store_true", help="Skip model sync check with HuggingFace before generation")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile even for large batches")
    parser.add_argument("--sync-state", action="store_true", help="Sync state with R2 before starting (for multi-system runs)")
    parser.add_argument("--push-state", action="store_true", help="Push current local state to R2")

    # Orchestrator mode
    parser.add_argument("--orchestrator", type=str, default=None,
                        help="Run as orchestrated node (e.g., --orchestrator http://192.168.1.100:5002)")
    parser.add_argument("--node-name", type=str, default=None,
                        help="Custom node name for orchestrator display")
    parser.add_argument("--comfyui-url", type=str, default=None,
                        help="ComfyUI server URL for comfyui_* models (default: COMFYUI_URL env or http://127.0.0.1:8188)")

    args = parser.parse_args()

    # Validate model names (support comma-separated for orchestrator mode)
    valid_models = {"flux2_dev", "flux2_fp8", "flux2_turbo", "flux2_klein", "zimage_turbo", "sdxl_food", "comfyui_flux", "chroma1_hd", "qwen_turbo"}
    models_list = [m.strip() for m in args.model.split(",")]
    for m in models_list:
        if m not in valid_models:
            parser.error(f"Invalid model '{m}'. Valid choices: {', '.join(valid_models)}")

    if args.check_updates:
        check_model_updates()
    elif args.update_models:
        update_models()
    elif args.export_manifest:
        export_manifest()
    elif args.push_state:
        # Push current local state to R2
        state = load_state(args.model)
        print(f"Pushing state for {args.model} to R2...")
        print(f"  Completed: {len(state.get('completed', {}))}")
        print(f"  Failed: {len(state.get('failed', {}))}")
        upload_state_to_r2(state, args.model)
        print("Done!")
    elif args.orchestrator:
        # Run as orchestrated node (supports comma-separated models)
        run_as_node(
            orchestrator_url=args.orchestrator,
            models=args.model,  # Can be single or comma-separated list
            node_name=args.node_name,
            comfyui_url=args.comfyui_url,
        )
    elif args.status:
        # Optionally sync state from R2 first
        if args.sync_state:
            state = sync_state_from_r2(args.model)
        else:
            state = load_state(args.model)
        recipes = load_recipes()
        print(f"Model: {args.model}")
        print(f"Total recipes: {len(recipes)}")
        print(f"Completed: {len(state['completed'])}")
        print(f"Failed: {len(state['failed'])}")
        print(f"Remaining: {len(recipes) - len(state['completed'])}")
        if state["started_at"]:
            print(f"Started: {state['started_at']}")
    else:
        process_batch(
            start_index=args.start,
            limit=args.limit,
            regenerate_failed=args.regenerate_failed,
            dry_run=args.dry_run,
            model=args.model,
            skip_sync=args.no_sync,
            no_compile=args.no_compile,
            sync_state=args.sync_state
        )
