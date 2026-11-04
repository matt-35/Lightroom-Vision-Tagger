# ⚡ Performance & Optimization Guide

This guide details best practices and configuration options to maximize tagging throughput when running local Vision Language Models (VLMs) with **LM Studio**.

---

## ⏱️ Performance Overview

| Setup | Speed per Image | Time for 300 Photos | Best Use Case |
|---|---|---|---|
| **Nano Vision Models** (`nvidia/nemotron-3-nano-omni`) | **~2–5 seconds** | **~10–25 minutes** | Fast bulk tagging, standard landscape & travel |
| **Mid-Size Vision Models** (`google/gemma-3-12b`) | **~10–15 seconds** | **~50–75 minutes** | High-accuracy landmarks, detailed hierarchical tags |
| **Large Reasoning Models** (`google/gemma-4-31b-qat`) | **~120–180 seconds** | **~10–15 hours** | Deep exploratory reasoning, complex composition analysis |

---

## 🚀 Optimization Strategies (Ranked by Impact)

### 1. Select a Fast, Non-Reasoning Vision Model (20x – 50x Speedup)
Large reasoning models (like `gemma-4-31b`) generate hundreds of hidden internal "thinking" tokens before emitting the final JSON metadata. For image tagging, structured output models without chain-of-thought overhead are vastly faster:

- **Recommended Fast Models**:
  - `google/gemma-3-12b`: Excellent balance of landmark recognition, detailed tagging, and speed.
  - `nvidia/nemotron-3-nano-omni`: Lightweight vision model designed for rapid multimodal inference.

```bash
# Tag using Gemma 3 12B:
./tagger -r '/Volumes/Lexar/Arches NP/Raw' --model google/gemma-3-12b

# Tag using Nemotron Nano:
./tagger -r '/Volumes/Lexar/Arches NP/Raw' --model nvidia/nemotron-3-nano-omni
```

---

### 2. Reduce Image Dimension (`--max-size`) (2x – 3x Prefill Speedup)
By default, the tagger resizes images to `1536px` on the long edge (`--max-size 1536`). 

Vision encoders (SigLIP / ViT) slice high-resolution images into multiple grid tiles:
- **1536px**: 4–6 tiles (~1,000 to 2,500 vision prompt tokens per image).
- **1024px** or **768px**: 1–2 tiles (~300 to 700 vision prompt tokens per image).

Downscaling to `1024` or `768` cuts the vision prefill computation dramatically with virtually no impact on landmark or scene identification:

```bash
# Run with 1024px maximum dimension
./tagger -r '/Volumes/Lexar/Arches NP/Raw' --model google/gemma-3-12b --max-size 1024

# Ultra-fast mode with 768px
./tagger -r '/Volumes/Lexar/Arches NP/Raw' --model google/gemma-3-12b --max-size 768
```

---

### 3. Leverage Incremental Processing (Omit `--force`)
The tagger automatically inspects existing `.xmp` sidecar files before sending anything to the model:
- If a photo has already been tagged in a previous run, the tagger checks its `.xmp` file and skips it in **< 5ms**.
- Only supply `--force` if you intentionally want to re-process and re-tag images from scratch.

```bash
# Safe, resumable run: skips previously processed photos automatically
./tagger -r '/Volumes/Lexar/Arches NP/Raw' --model google/gemma-3-12b
```

---

### 4. Optimize LM Studio Server Settings (Apple Silicon / Metal)

1. **GPU Offload to Maximum**:
   - In LM Studio, ensure **GPU Offload** / **Acceleration (Metal)** slider is set to **MAX** (100% of layers offloaded to unified memory).
2. **Limit Context Length**:
   - In LM Studio model settings, set context length between `4096` and `8192` tokens.
   - Avoid `32k+` context sizes for image tagging; oversized context buffers consume memory bandwidth for KV caching.
3. **Disable Thinking / Chain of Thought**:
   - If using a reasoning model, disable "Thinking" or reduce "Reasoning Effort" in the LM Studio model configuration panel.

---

## 💡 Recommended Presets

### 🏎️ Max Speed Preset (~2–4 sec/image)
```bash
./tagger -r '/path/to/photos' \
  --model nvidia/nemotron-3-nano-omni \
  --max-size 768
```

### ⚖️ Balanced Quality & Speed Preset (~10–12 sec/image)
```bash
./tagger -r '/path/to/photos' \
  --model google/gemma-3-12b \
  --max-size 1024
```

