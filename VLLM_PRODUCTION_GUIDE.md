# vLLM Production Guide for MICR Extraction

## What is vLLM?

vLLM is a high-throughput inference engine for large language models. Unlike Ollama (which processes one request at a time), vLLM uses **continuous batching** — it groups multiple incoming requests and processes them simultaneously on the GPU. This is what makes 100K+ checks/day possible on a single GPU.

**Ollama vs vLLM:**

| | Ollama | vLLM |
|---|--------|------|
| Use case | Development, single-user | Production, high-throughput |
| Concurrency | 1 request at a time | Hundreds of concurrent requests |
| API | Custom `/api/generate` | OpenAI-compatible `/v1/chat/completions` |
| Setup | `brew install ollama` | Docker + NVIDIA GPU |
| Platform | macOS, Linux, Windows | Linux + NVIDIA GPU (CUDA) |
| Batching | None | Continuous batching |

**Key takeaway:** For development on your Mac, keep using Ollama. For production on NVIDIA GPUs, switch to vLLM. Our code supports both — just change the URL.

## Prerequisites

- Linux server with NVIDIA GPU(s)
- NVIDIA driver 535+ installed
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- At least 24GB GPU VRAM for the 7B model (A10, L4, RTX 4090, A100, H100)

### Verify your GPU setup

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker can see GPUs
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## Quick Start

### 1. Pull and run vLLM with Qwen2.5-VL-7B

```bash
docker run -d \
  --name vllm-micr \
  --runtime nvidia \
  --gpus all \
  --ipc=host \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt image=1,video=0
```

First run downloads the model weights (~14 GB). Subsequent starts use the cached weights.

### 2. Wait for the model to load

```bash
# Watch startup logs (model loading takes 1-3 minutes)
docker logs -f vllm-micr

# Wait until you see:
# INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 3. Verify it's running

```bash
# Health check
curl http://localhost:8000/health

# List loaded models
curl http://localhost:8000/v1/models
```

### 4. Extract from a check image

```bash
micr-extract Check.jpg --check --ollama-url http://localhost:8000
```

That's it. The `--ollama-url` flag works for both Ollama and vLLM — our engine detects the server type automatically and uses the correct API format.

## Docker Flags Explained

| Flag | What it does |
|------|-------------|
| `--runtime nvidia` | Enables GPU access inside the container |
| `--gpus all` | Passes all GPUs to the container (or `--gpus '"device=0"'` for a specific GPU) |
| `--ipc=host` | Shares host memory with container (required for PyTorch tensor operations) |
| `-v ~/.cache/huggingface:...` | Caches model weights on the host so they survive container restarts |
| `-p 8000:8000` | Exposes the API on port 8000 |

## vLLM Serve Arguments Explained

| Argument | Value | Why |
|----------|-------|-----|
| `--model Qwen/Qwen2.5-VL-7B-Instruct` | HuggingFace model ID | The vision-language model we use |
| `--dtype bfloat16` | BF16 precision | Training precision for Qwen2.5-VL, best accuracy |
| `--max-model-len 4096` | 4K context | Our prompts + responses are small (~1K tokens). Reducing from the default 128K saves massive GPU memory for more concurrent requests |
| `--gpu-memory-utilization 0.90` | 90% of VRAM | How much GPU memory vLLM can use. Push to 0.95 for maximum throughput |
| `--limit-mm-per-prompt image=1,video=0` | 1 image per request | We send exactly 1 check image per request. Disabling video saves memory |

## Performance Tuning

### For maximum throughput (100K+ checks/day)

```bash
docker run -d \
  --name vllm-micr \
  --runtime nvidia \
  --gpus all \
  --ipc=host \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 256 \
  --enable-chunked-prefill \
  --limit-mm-per-prompt image=1,video=0
```

**Key tuning parameters:**

| Parameter | Effect | Recommendation |
|-----------|--------|----------------|
| `--max-model-len` | Lower = more concurrent requests | 4096 is plenty for check extraction |
| `--gpu-memory-utilization` | Higher = more KV cache = more concurrency | 0.95 for dedicated servers |
| `--max-num-seqs` | Max concurrent sequences | 256 is a good default, increase if GPU memory allows |
| `--enable-chunked-prefill` | Prevents long prefills from blocking other requests | Recommended for mixed workloads |

### Expected throughput

| GPU | VRAM | Est. Throughput | Checks/Day |
|-----|------|----------------|------------|
| RTX 4090 | 24 GB | ~1.5 req/s | ~130,000 |
| A10 | 24 GB | ~1.2 req/s | ~100,000 |
| L4 | 24 GB | ~1.0 req/s | ~86,000 |
| A100 | 80 GB | ~2.5 req/s | ~216,000 |
| H100 | 80 GB | ~4+ req/s | ~345,000 |

## Scaling with Multiple GPUs

### Data Parallelism (recommended for 7B)

Since the 7B model fits on a single GPU (~14 GB), run **independent copies** on each GPU. Each copy processes different requests. Throughput scales linearly.

```bash
docker run -d \
  --name vllm-micr \
  --runtime nvidia \
  --gpus all \
  --ipc=host \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --data-parallel-size 4 \
  --limit-mm-per-prompt image=1,video=0
```

With 4x A100 GPUs using data parallelism: **~10 req/s = 860K+ checks/day**.

### Tensor Parallelism (for larger models only)

Use tensor parallelism when the model doesn't fit on a single GPU (e.g., 72B model):

```bash
docker run -d \
  --name vllm-micr \
  --runtime nvidia \
  --gpus all \
  --ipc=host \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-VL-72B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --tensor-parallel-size 4 \
  --limit-mm-per-prompt image=1,video=0
```

This splits the model across 4 GPUs. Requires NVLink for good performance — check with `nvidia-smi topo -m`.

## Docker Compose (Production)

Create a `docker-compose.yml`:

```yaml
services:
  vllm:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    ipc: host
    ports:
      - "8000:8000"
    volumes:
      - huggingface-cache:/root/.cache/huggingface
    command: >
      --model Qwen/Qwen2.5-VL-7B-Instruct
      --host 0.0.0.0
      --port 8000
      --dtype bfloat16
      --max-model-len 4096
      --gpu-memory-utilization 0.90
      --limit-mm-per-prompt image=1,video=0
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 180s
    restart: unless-stopped

volumes:
  huggingface-cache:
```

```bash
# Start
docker compose up -d

# View logs
docker compose logs -f vllm

# Stop
docker compose down
```

## Using with MICR Extraction

### CLI

```bash
# Point to your vLLM server
micr-extract Check.jpg --check --ollama-url http://your-gpu-server:8000

# Same works with remote server
micr-extract Check.jpg --check --ollama-url http://192.168.1.100:8000
```

### Python API

```python
from micr import MICRExtractor

extractor = MICRExtractor(
    use_vlm=True,
    ollama_url="http://your-gpu-server:8000",
)
result = extractor.extract_check("Check.jpg")
print(result.routing_number)
print(result.legal_amount)
```

### Direct API Call (for custom integrations)

vLLM exposes an OpenAI-compatible API. You can call it directly:

```python
import base64
import json
import urllib.request

# Encode check image
with open("Check.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract routing number, account number, and check number from this check image. Return JSON."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    }
                }
            ]
        }
    ],
    "max_tokens": 512,
    "temperature": 0.1
}

data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(
    "http://localhost:8000/v1/chat/completions",
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urllib.request.urlopen(req) as resp:
    result = json.loads(resp.read())
    print(result["choices"][0]["message"]["content"])
```

### Using the OpenAI Python SDK

```bash
pip install openai
```

```python
import base64
from openai import OpenAI

client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8000/v1",
)

with open("Check.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract the check fields. Return JSON."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    }
                }
            ]
        }
    ],
    max_tokens=512,
    temperature=0.1,
)

print(response.choices[0].message.content)
```

## Monitoring

### Health endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Returns 200 if vLLM is alive |
| `GET /v1/models` | Returns loaded models (confirms model is ready) |
| `GET /metrics` | Prometheus metrics (throughput, latency, queue depth) |

### Check server status

```bash
# Is it alive?
curl http://localhost:8000/health

# What model is loaded?
curl http://localhost:8000/v1/models | python3 -m json.tool

# Prometheus metrics
curl http://localhost:8000/metrics
```

### Key metrics to watch

From the `/metrics` endpoint:

| Metric | What it tells you |
|--------|------------------|
| `vllm:num_requests_running` | Currently processing requests |
| `vllm:num_requests_waiting` | Queued requests (if consistently > 0, you need more capacity) |
| `vllm:gpu_cache_usage_perc` | KV cache utilization (if near 100%, reduce `--max-model-len` or add GPUs) |
| `vllm:request_success_total` | Total successful requests |
| `vllm:e2e_request_latency_seconds` | End-to-end latency distribution |

### Grafana dashboard

For a production setup, pipe `/metrics` into Prometheus and visualize with Grafana:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['your-gpu-server:8000']
```

## Troubleshooting

### Model fails to load (CUDA OOM)

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Fix:** Reduce context length:
```bash
# Reduce from default 128K to 4K
--max-model-len 4096
```

For check extraction, 4K tokens is more than enough. The default 128K reserves huge amounts of GPU memory for KV cache that we don't need.

### Container starts but hangs during inference

Known issue with some vLLM versions. Try:

```bash
# Use a specific known-good version instead of latest
docker run ... vllm/vllm-openai:v0.8.4 ...
```

### Slow inference

Check GPU utilization:
```bash
# Inside the host
nvidia-smi -l 1
```

If GPU utilization is low, you may need more concurrent requests to keep the GPU busy. vLLM batches requests — it's fastest when processing many requests simultaneously.

### "Connection refused" from micr-extract

```bash
# Check if vLLM is running
docker ps | grep vllm

# Check if the port is accessible
curl http://localhost:8000/health

# Check logs for errors
docker logs vllm-micr
```

### Model download is slow

The first run downloads ~14 GB from HuggingFace. To speed this up:

```bash
# Pre-download the model on the host
pip install huggingface-hub
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct

# Then mount the cache into the container
-v ~/.cache/huggingface:/root/.cache/huggingface
```

### Using a different model

```bash
# Use the 3B model (faster, less accurate)
--model Qwen/Qwen2.5-VL-3B-Instruct

# Use the 72B model (most accurate, needs 4+ GPUs)
--model Qwen/Qwen2.5-VL-72B-Instruct --tensor-parallel-size 4
```

## Ollama to vLLM Migration Checklist

1. **Set up GPU server** — Linux + NVIDIA GPU + Docker + NVIDIA Container Toolkit
2. **Start vLLM** — `docker run` command from Quick Start above
3. **Wait for model to load** — Watch logs until "Uvicorn running"
4. **Test** — `curl http://your-server:8000/health`
5. **Update MICR config** — Change `--ollama-url` to point to your vLLM server
6. **Verify** — `micr-extract Check.jpg --check --ollama-url http://your-server:8000`
7. **Monitor** — Set up Prometheus + Grafana on the `/metrics` endpoint

No code changes needed. Same `--ollama-url` flag, same results, 10-100x more throughput.
