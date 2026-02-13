# MICR Extraction

Enterprise MICR E-13B text extraction from check images. Extracts routing numbers, account numbers, check numbers, legal amounts, and courtesy amounts using a combination of template matching and Vision Language Models (VLMs).

Runs 100% locally — no data leaves your machine.

## Features

- **Template matching engine** — Hu moments + structural feature matching for MICR line extraction. Works on scanner and camera images, including low-resolution (25px+).
- **VLM extraction** — Uses Qwen2.5-VL via Ollama to extract all check fields (MICR, legal amount, courtesy amount) from full check images.
- **Amount cross-validation** — VLM compares legal amount (words) against courtesy amount (numbers) and flags mismatches.
- **Multiple model support** — Choose from 7 model presets (3B to 72B) or any Ollama model.
- **No post-processing** — VLM output is returned as-is. If the model can't extract a field, it reports low confidence rather than hiding errors behind cleanup heuristics.

## Quick Start

### Install

```bash
pip install -e .
```

### MICR-only extraction (no VLM needed)

```bash
micr-extract MICR.png
```

Output:
```
MICR Line:       ⑈267084131⑈790319013⑆1024
Routing Number:  267084131
Account Number:  790319013
Check Number:    1024
Routing Valid:   True
Confidence:      97.54%
```

### Full check extraction (requires Ollama or vLLM)

```bash
# Install and start Ollama (see OLLAMA_GUIDE.md)
ollama pull qwen2.5vl:7b

# Extract all fields from a check image
micr-extract Check.jpg --check

# Or use a vLLM server (see VLLM_PRODUCTION_GUIDE.md)
micr-extract Check.jpg --check --ollama-url http://gpu-server:8000
```

Output:
```
Routing Number:   267084131
Account Number:   790319013
Check Number:     1024
Legal Amount:     One thousand dollars and 00/100 cent
Courtesy Amount:  1000.00
Amounts Match:    True
Confidence:       100.00%
Method:           vlm
VLM Model:        qwen2.5vl:7b
VLM Latency:      5187ms
```

### JSON output

```bash
micr-extract Check.jpg --check --format json
```

### Python API

```python
from micr import MICRExtractor

# MICR-only (fast, no VLM)
extractor = MICRExtractor()
result = extractor.extract("MICR.png")
print(result.routing_number)  # "267084131"

# Full check extraction (requires Ollama)
extractor = MICRExtractor(use_vlm=True)
result = extractor.extract_check("Check.jpg")
print(result.routing_number)   # "267084131"
print(result.legal_amount)     # "One thousand dollars and 00/100 cent"
print(result.courtesy_amount)  # "1000.00"
print(result.amounts_match)    # True
print(result.confidence)       # 1.0
```

## VLM Model Options

List available models:

```bash
micr-extract --list-vlm-models
```

| Preset | Model | Disk | RAM | Description |
|--------|-------|------|-----|-------------|
| `3b` | qwen2.5vl:3b | 3.2 GB | ~6 GB | Fast, fits fully in GPU on 16GB Mac |
| **`7b`** | **qwen2.5vl:7b** | **6.0 GB** | **~14 GB** | **Best accuracy/speed balance (default)** |
| `32b` | qwen2.5vl:32b | 21 GB | ~40 GB | Highest accuracy, needs 64GB+ RAM |
| `72b` | qwen2.5vl:72b | 49 GB | ~80 GB | Maximum accuracy, needs 128GB+ RAM |
| `minicpm` | minicpm-v | 4.9 GB | ~10 GB | Alternative VLM, good OCR |
| `moondream` | moondream | 1.7 GB | ~4 GB | Ultra-fast, lower accuracy |
| `granite` | granite3.2-vision:2b | 1.8 GB | ~5 GB | Document-focused, compact |

Use a specific model:

```bash
micr-extract Check.jpg --check --vlm-model 3b
micr-extract Check.jpg --check --vlm-model 32b
```

### Model accuracy comparison (Check.jpg test image)

| Field | 7b | 3b |
|-------|----|----|
| Routing | 267084131 | 267084131**1** (reads Transit symbol as digit) |
| Account | 790319013 | 790319013**1** (reads On-Us symbol as digit) |
| Check | 1024 | 1024 |
| Legal Amount | One thousand dollars and 00/100 cent | One thousand dollars and 2/100 cent |
| Courtesy Amount | 1000.00 | 1000.00 |
| Warm latency | ~5s | ~3s |

The 3b model confuses the E-13B special symbols (Transit ⑈, On-Us ⑆) with the digit 1. The 7b model handles them correctly.

## Performance & Scaling

### Development (Ollama, single request)

| Hardware | Latency/check | Checks/day |
|----------|---------------|------------|
| Apple M5 16GB | ~5s | ~17,000 |
| RTX 4090 24GB | ~1.5s | ~58,000 |
| A100 80GB | ~0.8s | ~108,000 |

### Production (vLLM, batched inference)

For high-throughput production (100K+ checks/day), switch from Ollama to [vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen2.5-VL.html) which supports concurrent batch processing:

| Hardware | Throughput | Checks/day |
|----------|-----------|------------|
| RTX 4090 24GB | ~1.5 req/s | ~130,000 |
| A100 80GB | ~2.5 req/s | ~216,000 |
| H100 80GB | ~4+ req/s | ~345,000 |

A single A100 with vLLM handles **200K+ checks/day** — well above 100K targets.

The VLM engine's HTTP API interface is compatible with both Ollama and OpenAI-compatible endpoints (vLLM, TGI).

## Architecture

```
Full check image
    |
    +---> VLM Engine (Ollama/vLLM) ---> VLMExtractionResult
    |       All fields: routing, account, check, legal amt, courtesy amt
    |       Self-reported confidence, amount cross-validation
    |
    +---> Template Engine (offline) ---> MICRResult
    |       MICR only: routing, account, check (with checksum validation)
    |
    +---> Hybrid Merger ---> CheckResult
            VLM drives all output
            Template result kept for reference
            Informational warnings (checksum, mismatches)
```

### Design decisions

- **VLM-first** — VLM output is used directly without cleanup. If the VLM makes errors, they're reflected in low confidence rather than hidden by post-processing.
- **No new dependencies for VLM** — Uses `urllib.request` (stdlib) to call Ollama's HTTP API. Only `opencv-python-headless` and `numpy` are required.
- **Dual server support** — Auto-detects Ollama (`/api/generate`) or OpenAI-compatible servers (`/v1/chat/completions`). Just change the URL to switch between Ollama and vLLM.
- **Backward compatible** — `extract()` / `extract_from_array()` for MICR-only extraction remain unchanged. New `extract_check()` / `extract_check_from_array()` methods add VLM support.
- **Graceful fallback** — If Ollama isn't running, falls back to template-only extraction for MICR fields.

## MICR E-13B Standard

The MICR line at the bottom of checks uses the E-13B font with 14 characters:

- **Digits**: 0-9
- **Transit** ⑈ — encloses routing number
- **On-Us** ⑆ — separates account and check number
- **Amount** ⑇ — encloses amount field
- **Dash** ⑉ — separator within a field

Standard layout (ANSI X9.13):
```
⑈RRRRRRRRR⑈ AAAAAAAAAA⑆CCCC

R = routing number (exactly 9 digits)
A = account number
C = check number
```

Routing number validation uses the ABA modulo-10 checksum with weights `[3, 7, 1, 3, 7, 1, 3, 7, 1]`.

## CLI Reference

```
micr-extract <image> [options]

Positional:
  image                    Path to the check image file

Extraction mode:
  --check                  Extract full check fields using VLM
  --vlm                    Enable VLM engine

VLM options:
  --vlm-model MODEL        Model preset (3b, 7b, 32b, 72b, minicpm, moondream, granite)
                           or full Ollama model name (default: 7b)
  --list-vlm-models        List available model presets and exit
  --ollama-url URL         Ollama API URL (default: http://localhost:11434)
  --vlm-timeout SECONDS    VLM inference timeout (default: 120)

Output:
  --format {text,json}     Output format (default: text)
  -v, --verbose            Show per-character details

Template engine:
  --templates-dir DIR      Path to custom E-13B templates
  --tesseract              Enable Tesseract as secondary engine
```

## Testing

```bash
# Unit tests (no Ollama needed)
python -m pytest tests/ -v

# Integration tests (requires Ollama + models pulled)
python -m pytest tests/test_vlm_integration.py -v -m integration
```

## Project Structure

```
micr/
  __init__.py              # Public exports
  api.py                   # MICRExtractor — main entry point
  cli.py                   # Command-line interface
  models.py                # Data models (MICRResult, CheckResult, VLMExtractionResult)
  engines/
    template_engine.py     # Hu moments + structural feature matching
    vlm_engine.py          # Ollama VLM integration + model presets
    tesseract_engine.py    # Optional Tesseract OCR engine
  preprocessing/
    scanner_pipeline.py    # Scanner image preprocessing
    camera_pipeline.py     # Camera image preprocessing
    micr_locator.py        # MICR line region detection
    common.py              # Shared preprocessing utilities
  parsing/
    parser.py              # MICR line field parsing
    validator.py           # Routing number checksum validation
    hybrid.py              # VLM + template result merging
    consensus.py           # Multi-engine consensus
  resources/
    templates/             # E-13B character templates

tests/
  test_extractor.py        # Template engine + validation tests (19 tests)
  test_vlm_engine.py       # VLM parsing, hybrid merge, server detection tests (25 tests)
  test_vlm_integration.py  # Live Ollama integration tests (3 tests)

OLLAMA_GUIDE.md            # Ollama setup, usage, and troubleshooting guide
VLLM_PRODUCTION_GUIDE.md   # vLLM production deployment guide
```
