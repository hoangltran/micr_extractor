# Ollama Guide for MICR Extraction

## What is Ollama?

Ollama is a tool that runs large language models (LLMs) locally on your machine. It handles downloading, managing, and serving models through a simple HTTP API. For this project, we use Ollama to run Vision Language Models (VLMs) that can "see" check images and extract fields like routing numbers, amounts, and more.

**Key benefits:**
- Runs 100% locally — no data leaves your machine
- Supports Apple Silicon GPU acceleration (Metal)
- Simple CLI and HTTP API

## Installation

### macOS
```bash
# Download from https://ollama.com/download
# Or install via Homebrew:
brew install ollama
```

### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Verify installation
```bash
ollama --version
```

## Starting Ollama

Ollama runs as a background server that listens on `http://localhost:11434`.

```bash
# Start the Ollama server (runs in background)
ollama serve
```

On macOS, Ollama typically auto-starts when you open the app. You can check if it's running:

```bash
# Check if Ollama is responding
curl http://localhost:11434/api/tags
```

## Managing Models

### Pull (download) a model
```bash
# Pull the default model for MICR extraction
ollama pull qwen2.5vl:7b

# Pull a smaller/faster model
ollama pull qwen2.5vl:3b
```

First pull downloads the model (~3-6 GB). Subsequent pulls only check for updates.

### List downloaded models
```bash
ollama list
```

Example output:
```
NAME             ID           SIZE     MODIFIED
qwen2.5vl:7b    5ced39dfa4    6.0 GB   2 hours ago
qwen2.5vl:3b    e9758e589d    3.2 GB   1 hour ago
```

### Check running models
```bash
ollama ps
```

Example output:
```
NAME            SIZE     PROCESSOR          UNTIL
qwen2.5vl:7b   14 GB    62%/38% CPU/GPU    4 minutes from now
```

The `PROCESSOR` column shows how the model is split between CPU and GPU. **100% GPU** is fastest. If the model is too large for your GPU memory, Ollama splits it across CPU/GPU (slower).

### Remove a model
```bash
ollama rm qwen2.5vl:3b
```

## Memory & Performance

### How much RAM do I need?

| Model | Disk Size | RAM When Running | Apple Silicon GPU |
|-------|-----------|-----------------|-------------------|
| qwen2.5vl:3b | 3.2 GB | ~6 GB | Fits in 8GB+ Mac |
| qwen2.5vl:7b | 6.0 GB | ~14 GB | Partial fit on 16GB Mac |
| qwen2.5vl:32b | 21 GB | ~40 GB | Needs 64GB+ Mac |
| qwen2.5vl:72b | 49 GB | ~80 GB | Needs 128GB+ Mac |

### Understanding CPU/GPU split

When you run `ollama ps`, the `PROCESSOR` column shows:
- **100% GPU** — Entire model fits in GPU memory. Fastest.
- **62%/38% CPU/GPU** — Model is split. GPU processes 38%, CPU handles 62%. Slower because CPU inference is much slower than GPU.

**Rule of thumb:** Your Mac's unified memory is shared between system and GPU. A 16GB Mac has roughly 10-12GB available for models after the OS takes its share.

### Cold start vs warm inference

- **Cold start** (~10-140s): First request after pulling a model or after the model is unloaded from memory. Ollama loads the model weights into RAM/GPU.
- **Warm inference** (~3-5s): Model is already loaded. Just processes the image.

Ollama unloads models after 5 minutes of inactivity by default. You can change this:

```bash
# Keep model loaded for 30 minutes
OLLAMA_KEEP_ALIVE=30m ollama serve

# Keep model loaded indefinitely
OLLAMA_KEEP_ALIVE=-1 ollama serve
```

## Using with MICR Extraction

### List available model presets
```bash
micr-extract --list-vlm-models
```

### Extract from a check image (default 7b model)
```bash
micr-extract Check.jpg --check
```

### Use a specific model
```bash
# Use preset shorthand
micr-extract Check.jpg --check --vlm-model 3b
micr-extract Check.jpg --check --vlm-model 7b

# Or use full Ollama model name
micr-extract Check.jpg --check --vlm-model qwen2.5vl:7b
```

### Increase timeout for slow machines
```bash
# Default timeout is 120 seconds
# Increase to 5 minutes for cold starts
micr-extract Check.jpg --check --vlm-timeout 300
```

### JSON output
```bash
micr-extract Check.jpg --check --format json
```

### Connect to remote Ollama server
```bash
micr-extract Check.jpg --check --ollama-url http://192.168.1.100:11434
```

## Troubleshooting

### "Ollama is not running"
```bash
# Start the server
ollama serve

# Or on macOS, open the Ollama app
```

### "HTTP Error 404: Not Found"
The model isn't downloaded yet:
```bash
ollama pull qwen2.5vl:7b
```

### Slow inference
Check the CPU/GPU split:
```bash
ollama ps
```

If you see high CPU %, the model is too large for your GPU memory. Options:
1. Use a smaller model (`--vlm-model 3b`)
2. Close other apps to free memory
3. Use a machine with more RAM

### Model keeps unloading
Ollama unloads models after 5 minutes. To keep it loaded:
```bash
OLLAMA_KEEP_ALIVE=30m ollama serve
```

### Check Ollama logs (macOS)
```bash
cat ~/.ollama/logs/server.log
```

## Ollama HTTP API (for developers)

Ollama exposes a REST API at `http://localhost:11434`. This is what our VLM engine uses internally.

### Check server health
```bash
curl http://localhost:11434/api/tags
```

### Generate with an image (what our engine does)
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5vl:7b",
  "prompt": "What text is in this image?",
  "images": ["<base64-encoded-image>"],
  "stream": false,
  "options": {
    "temperature": 0.1,
    "num_predict": 512
  }
}'
```

### Key API endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/tags` | GET | List downloaded models |
| `/api/generate` | POST | Generate text (with optional images) |
| `/api/pull` | POST | Pull a model |
| `/api/delete` | DELETE | Remove a model |

Full API docs: https://github.com/ollama/ollama/blob/main/docs/api.md
