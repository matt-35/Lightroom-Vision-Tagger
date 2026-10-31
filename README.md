# 📸 Lightroom Local Vision Auto-Tagger

A fast, pure-Python tool that reads camera RAW files and image folders, analyzes them using a local Vision Language Model (VLM) running in **LM Studio**, and writes Lightroom-compatible keywords and hierarchical tags directly into **`.xmp` sidecar files**.

---

## ✨ Features

- **🚀 Ultra-Fast RAW Preview Extraction**:
  - Leverages embedded full/high-resolution JPEG previews found inside camera RAW files (Canon `.CR2`/`.CR3`, Sony `.ARW`, Nikon `.NEF`, Fuji `.RAF`, `.DNG`, etc.) using `rawpy`.
  - Avoids slow demosaicing while retaining optimal quality for AI vision inference (< 50ms extraction).
- **🏛️ Named Iconic Places & Landmarks**:
  - Detects recognized world landmarks, national parks, monuments, skylines, and mountain peaks (e.g., *Yosemite National Park*, *Half Dome*, *Eiffel Tower*, *Golden Gate Bridge*).
- **🌲 Lightroom Hierarchical Tags**:
  - Writes standard flat keywords (`dc:subject`) and Adobe hierarchical categories (`lr:hierarchicalSubject`), e.g.:
    - `Places|USA|California|Yosemite`
    - `Nature|Landscape|Mountains`
    - `Lighting|Golden Hour`
- **🔒 Safe & Non-Destructive**:
  - Preserves existing Lightroom develop adjustments (`crs:*`), star ratings (`xmp:Rating`), color labels, and crop settings.
  - Safely merges new tags into existing `.xmp` sidecars without duplicates (or replaces them if `--overwrite` is specified).
- **📴 100% Offline & Pure Python**:
  - No external binary dependencies (no `exiftool` installation required).
  - Uses your local LM Studio instance (`http://localhost:1234/v1`).

---

## 📋 Requirements

- **Python Environment**: Conda `torch` environment (Python 3.14 with `rawpy`, `Pillow`, `httpx`, `rich`, `pydantic`).
- **LM Studio**: Running locally with a Vision-capable model loaded (e.g., `google/gemma-3-12b`, `qwen/qwen3.6-35b-a3b`, `nvidia/nemotron-3-nano-omni`, `flint`, `pixtral`, or `llava`).

---

## 🚀 Quick Start

### 1. Start LM Studio Server
1. Open **LM Studio**.
2. Load a Vision model (e.g. `google/gemma-3-12b`, `nvidia/nemotron-3-nano-omni`, or any multimodal model).
3. Start the Local Server on port `1234` (OpenAI-compatible server).

### 2. Run the Tagger

You can use the included `./tagger` launcher directly:

```bash
# Preview what tags will be generated without modifying any files:
./tagger --dry-run /path/to/photos

# Process a folder and write .xmp sidecars:
./tagger /path/to/photos

# Process subfolders recursively:
./tagger -r /path/to/photos

# Re-tag images that already have tags in their XMP:
./tagger --force /path/to/photos

# Overwrite existing tags instead of merging:
./tagger --overwrite /path/to/photos
```

Alternatively, run with your conda environment:
```bash
/opt/miniconda3/envs/torch/bin/python -m src /path/to/photos
```

---

## 🔄 Lightroom Classic Workflow

Once the script has finished creating or updating `.xmp` files:

1. Open **Adobe Lightroom Classic**.
2. Select the photos or folder you just tagged in the Library Grid (`G`).
3. In the top menu, click **Metadata** ➔ **Read Metadata from File**  
   *(Shortcut: `Cmd + Option + S` on macOS or `Ctrl + Alt + S` on Windows)*.
4. Lightroom will read the `.xmp` sidecars and immediately display the new keywords in the **Keywording** and **Keyword List** panels!

---

## ⚙️ CLI Options

| Option | Description | Default |
|---|---|---|
| `folder` | Path to folder containing photos/RAWs | *(Required)* |
| `-r`, `--recursive` | Scan subdirectories recursively | `False` |
| `--dry-run` | Analyze photos and print tags to terminal without writing XMP files | `False` |
| `--force` | Process photos even if they already have tags in their XMP sidecar | `False` |
| `--overwrite` | Overwrite existing tags instead of merging | `False` |
| `--model` | Specific model ID in LM Studio | Auto-detected |
| `--url` | LM Studio API endpoint | `http://localhost:1234/v1` |
| `--max-size` | Long edge pixel dimension for vision input | `1536` |
| `--timeout` | Timeout in seconds per image | `60.0` |

---

## 🧪 Running Tests

Run the test suite:
```bash
/opt/miniconda3/envs/torch/bin/python -m unittest discover -s tests -p "test_*.py"
```
