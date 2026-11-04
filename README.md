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

- **Python Environment**: Python 3.14 with `rawpy`, `Pillow`, `httpx`, `rich`, `pydantic`.
- **LM Studio**: Running locally with a Vision-capable model loaded (e.g., `google/gemma-3-12b`, `qwen/qwen3.6-35b-a3b`, `nvidia/nemotron-3-nano-omni`, `flint`, `pixtral`, or `llava`).

---

## 🚀 Quick Start

### 1. Start LM Studio Server
1. Open **LM Studio**.
2. Load a Vision model (e.g. `google/gemma-3-12b`, `nvidia/nemotron-3-nano-omni`, or any multimodal model).
3. Start the Local Server on port `1234` (OpenAI-compatible server).

### 2. Launch the Desktop GUI

You can launch the modern desktop GUI using `./tagger-gui`:

```bash
./tagger-gui
```

*(or via python: `./tagger --gui` or `python -m src --gui`)*

**GUI Features**:
- **🌐 LM Studio Models Dropdown**: Automatically detects available models loaded in LM Studio, badging Vision-capable models with `👁️ [Vision]` and auto-selecting the best model.
- **📁 Visual File Browser**: Select any folder to scan and see untagged vs. tagged counts, file formats, and status icons.
- **🖼️ High-Res RAW Image Preview**: Extracts embedded RAW previews on the fly without slow demosaicing.
- **🧠 System Prompt Editor Window**: Click **"⚙️ Edit System Prompt"** to inspect or modify the AI prompt and customize instructions for your photography style.
- **🏷️ Tag Inspector & Editor**: View recognized landmarks, flat keywords, and Lightroom hierarchy side-by-side, or tag individual images on demand.
- **🚀 Batch Execution**: Real-time progress bar, pause/stop controls, and dry-run testing.

---

### 3. Run via CLI (Terminal)

You can also use the included `./tagger` CLI launcher directly:

```bash
# List available models in LM Studio and see which support Vision:
./tagger -l

# Use a specific vision model (exact ID or substring like 'gemma-3' or 'nemotron'):
./tagger -m gemma-3 /path/to/photos

# Preview what tags will be generated without modifying any files:
./tagger --dry-run -m gemma-3 /path/to/photos

# Process a folder and write .xmp sidecars:
./tagger -m gemma-3 /path/to/photos

# Process subfolders recursively:
./tagger -r -m gemma-3 /path/to/photos

# Re-tag images that already have tags in their XMP:
./tagger --force -m gemma-3 /path/to/photos

# Overwrite existing tags instead of merging:
./tagger --overwrite -m gemma-3 /path/to/photos
```

Alternatively, run with your python environment:
```bash
python -m src /path/to/photos
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
| `--timeout` | Timeout in seconds per image | `300.0` |
| `--gui` | Launch the modern desktop GUI | `False` |

---

## ⚡ Performance & Optimization

It is recomended to use smaller or quantised models (Ex: `google/gemma-4-e4b`) and turn off reasoning/thinking in LM Studio. Note that smaller models may make more mistakes. 

---

## 🧪 Running Tests

Run the test suite:
```bash
python -m unittest discover -s tests -p "test_*.py"
```
