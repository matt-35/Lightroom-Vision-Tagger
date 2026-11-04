"""
Threaded worker and event queue system for asynchronous GUI operations.
Ensures the UI remains completely responsive during scanning, extraction, and VLM queries.
"""

from dataclasses import dataclass
from pathlib import Path
import queue
import threading
from typing import List, Optional, Any, Callable
from PIL import Image

from src.config import TaggerConfig
from src.scanner import DirectoryScanner, ImageFileItem
from src.raw_extractor import RawExtractor
from src.vision_tagger import VisionTagger, TaggingResult
from src.xmp_manager import XMPManager


# Event Types
EVENT_MODELS_LOADED = "models_loaded"
EVENT_MODELS_FAILED = "models_failed"

EVENT_SCAN_COMPLETED = "scan_completed"
EVENT_SCAN_FAILED = "scan_failed"

EVENT_ITEM_STARTED = "item_started"
EVENT_ITEM_SUCCESS = "item_success"
EVENT_ITEM_FAILED = "item_failed"
EVENT_BATCH_FINISHED = "batch_finished"

EVENT_PREVIEW_READY = "preview_ready"


@dataclass
class GuiEvent:
    event_type: str
    data: Any


class ModelsWorker(threading.Thread):
    """Background worker to query LM Studio for available models."""

    def __init__(self, api_url: str, event_queue: queue.Queue):
        super().__init__(daemon=True)
        self.api_url = api_url
        self.event_queue = event_queue

    def run(self):
        try:
            tagger = VisionTagger(api_url=self.api_url, timeout=10.0)
            models = tagger.get_available_models()
            vision_models = [m for m in models if tagger.is_likely_vision_model(m)]
            self.event_queue.put(
                GuiEvent(
                    EVENT_MODELS_LOADED,
                    {"models": models, "vision_models": vision_models, "api_url": self.api_url},
                )
            )
        except Exception as e:
            self.event_queue.put(
                GuiEvent(EVENT_MODELS_FAILED, {"error": str(e), "api_url": self.api_url})
            )


class ScanWorker(threading.Thread):
    """Background worker to discover and inspect image files in a folder."""

    def __init__(self, folder: Path, recursive: bool, skip_tagged: bool, event_queue: queue.Queue):
        super().__init__(daemon=True)
        self.folder = folder
        self.recursive = recursive
        self.skip_tagged = skip_tagged
        self.event_queue = event_queue

    def run(self):
        try:
            scanner = DirectoryScanner(
                folder=self.folder,
                recursive=self.recursive,
                skip_tagged=False,  # Discover all so GUI can display both tagged & untagged
            )
            all_items = scanner.discover_files()
            total = len(all_items)
            tagged = sum(1 for item in all_items if item.is_already_tagged)
            untagged = total - tagged

            self.event_queue.put(
                GuiEvent(
                    EVENT_SCAN_COMPLETED,
                    {
                        "items": all_items,
                        "total": total,
                        "tagged": tagged,
                        "untagged": untagged,
                        "folder": self.folder,
                    },
                )
            )
        except Exception as e:
            self.event_queue.put(GuiEvent(EVENT_SCAN_FAILED, {"error": str(e)}))


class PreviewManager:
    """Manages background preview extraction with request deduplication and cancellation."""

    def __init__(self, event_queue: queue.Queue, max_dimension: int = 400):
        self.event_queue = event_queue
        self.max_dimension = max_dimension
        self._lock = threading.Lock()
        self._pending_path: Optional[Path] = None
        self._extractor = RawExtractor(max_dimension=max_dimension, jpeg_quality=80)
        self._has_task = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def request_preview(self, image_path: Path):
        """Replaces any pending preview request with this image path."""
        with self._lock:
            self._pending_path = image_path
            self._has_task.set()

    def _worker_loop(self):
        while True:
            self._has_task.wait()
            with self._lock:
                target_path = self._pending_path
                self._has_task.clear()

            if target_path is None:
                continue

            try:
                img = self._extractor.extract_preview(target_path, max_dimension=self.max_dimension)
                # If a newer request was queued while processing, skip posting stale image
                with self._lock:
                    if self._pending_path != target_path and self._has_task.is_set():
                        continue

                self.event_queue.put(
                    GuiEvent(
                        EVENT_PREVIEW_READY,
                        {"image_path": target_path, "pil_image": img, "error": None},
                    )
                )
            except Exception as e:
                self.event_queue.put(
                    GuiEvent(
                        EVENT_PREVIEW_READY,
                        {"image_path": target_path, "pil_image": None, "error": str(e)},
                    )
                )


class PreviewWorker(threading.Thread):
    """Background worker to load/extract image thumbnail for preview."""

    def __init__(self, image_path: Path, max_dimension: int, event_queue: queue.Queue):
        super().__init__(daemon=True)
        self.image_path = image_path
        self.max_dimension = max_dimension
        self.event_queue = event_queue

    def run(self):
        try:
            extractor = RawExtractor(max_dimension=self.max_dimension, jpeg_quality=80)
            img = extractor.extract_preview(self.image_path, max_dimension=self.max_dimension)
            self.event_queue.put(
                GuiEvent(
                    EVENT_PREVIEW_READY,
                    {"image_path": self.image_path, "pil_image": img, "error": None},
                )
            )
        except Exception as e:
            self.event_queue.put(
                GuiEvent(
                    EVENT_PREVIEW_READY,
                    {"image_path": self.image_path, "pil_image": None, "error": str(e)},
                )
            )


class BatchTaggingWorker(threading.Thread):
    """Background worker to process a list of images through the VLM."""

    def __init__(
        self,
        items: List[ImageFileItem],
        config: TaggerConfig,
        event_queue: queue.Queue,
        stop_event: threading.Event,
    ):
        super().__init__(daemon=True)
        self.items = items
        self.config = config
        self.event_queue = event_queue
        self.stop_event = stop_event

    def run(self):
        tagger = VisionTagger(
            api_url=self.config.api_url,
            model=self.config.model,
            timeout=self.config.timeout,
            system_prompt=self.config.system_prompt,
        )
        extractor = RawExtractor(
            max_dimension=self.config.max_dimension,
            jpeg_quality=self.config.jpeg_quality,
        )

        success_count = 0
        fail_count = 0
        consecutive_errors = 0

        for idx, item in enumerate(self.items):
            if self.stop_event.is_set():
                break

            self.event_queue.put(
                GuiEvent(
                    EVENT_ITEM_STARTED,
                    {"index": idx, "total": len(self.items), "item": item},
                )
            )

            try:
                # 1. Extract base64
                b64_img, dims = extractor.prepare_base64_jpeg(item.image_path)

                # 2. Check cancel before slow network request
                if self.stop_event.is_set():
                    break

                # 3. Analyze
                result = tagger.analyze_image(b64_img)
                consecutive_errors = 0

                # 4. Save to XMP unless dry run
                final_flat = result.tags
                final_hier = result.hierarchical_tags
                if not self.config.dry_run:
                    final_flat, final_hier = XMPManager.write_tags(
                        xmp_path=item.xmp_path,
                        new_flat_tags=result.tags,
                        new_hierarchical_tags=result.hierarchical_tags,
                        overwrite=self.config.overwrite_tags,
                    )

                # Update item state
                item.has_existing_xmp = item.xmp_path.exists()
                item.existing_tags = final_flat
                item.existing_hierarchical = final_hier

                success_count += 1
                self.event_queue.put(
                    GuiEvent(
                        EVENT_ITEM_SUCCESS,
                        {
                            "index": idx,
                            "item": item,
                            "result": result,
                            "final_flat": final_flat,
                            "final_hier": final_hier,
                            "dry_run": self.config.dry_run,
                        },
                    )
                )

            except Exception as e:
                fail_count += 1
                consecutive_errors += 1
                self.event_queue.put(
                    GuiEvent(
                        EVENT_ITEM_FAILED,
                        {"index": idx, "item": item, "error": str(e)},
                    )
                )

                # Abort if server is consistently rejecting requests
                if consecutive_errors >= 3:
                    break

        self.event_queue.put(
            GuiEvent(
                EVENT_BATCH_FINISHED,
                {
                    "success_count": success_count,
                    "fail_count": fail_count,
                    "cancelled": self.stop_event.is_set(),
                    "total": len(self.items),
                },
            )
        )

