import os
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.config import TaggerConfig
from src.scanner import ImageFileItem
from src.vision_tagger import SYSTEM_PROMPT, VisionTagger, TaggingResult
from src.gui.worker import (
    GuiEvent,
    ScanWorker,
    ModelsWorker,
    BatchTaggingWorker,
    EVENT_SCAN_COMPLETED,
    EVENT_MODELS_LOADED,
    EVENT_ITEM_STARTED,
    EVENT_ITEM_SUCCESS,
    EVENT_BATCH_FINISHED,
)


class TestGuiStateAndWorkers(unittest.TestCase):

    def test_custom_system_prompt_configuration(self):
        """Test that VisionTagger and TaggerConfig properly accept a custom system prompt."""
        custom_prompt = "You are a specialized astro-photography tagger."
        tagger = VisionTagger(system_prompt=custom_prompt)
        self.assertEqual(tagger.system_prompt, custom_prompt)

        # Default prompt fallback
        default_tagger = VisionTagger()
        self.assertEqual(default_tagger.system_prompt, SYSTEM_PROMPT)

        config = TaggerConfig(system_prompt=custom_prompt)
        self.assertEqual(config.system_prompt, custom_prompt)

    def test_scan_worker(self):
        """Test that ScanWorker correctly inspects files and posts EVENT_SCAN_COMPLETED."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            img1 = tmppath / "test1.jpg"
            img2 = tmppath / "test2.jpg"
            img1.write_bytes(b"fake image 1")
            img2.write_bytes(b"fake image 2")

            # Create an XMP sidecar with tags for img1
            xmp1 = tmppath / "test1.xmp"
            xmp1.write_text(
                """<x:xmpmeta xmlns:x="adobe:ns:meta/">
                <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
                  <rdf:Description xmlns:dc="http://purl.org/dc/elements/1.1/">
                    <dc:subject><rdf:Bag><rdf:li>mountain</rdf:li></rdf:Bag></dc:subject>
                  </rdf:Description>
                </rdf:RDF></x:xmpmeta>"""
            )

            evt_queue = queue.Queue()
            worker = ScanWorker(
                folder=tmppath,
                recursive=False,
                skip_tagged=False,
                event_queue=evt_queue,
            )
            worker.start()
            worker.join(timeout=3.0)

            self.assertFalse(evt_queue.empty())
            event: GuiEvent = evt_queue.get_nowait()
            self.assertEqual(event.event_type, EVENT_SCAN_COMPLETED)
            self.assertEqual(event.data["total"], 2)
            self.assertEqual(event.data["tagged"], 1)
            self.assertEqual(event.data["untagged"], 1)

    @patch("src.gui.worker.VisionTagger")
    def test_models_worker(self, mock_tagger_cls):
        """Test that ModelsWorker queries LM Studio and segregates vision models."""
        mock_instance = MagicMock()
        mock_instance.get_available_models.return_value = [
            "google/gemma-3-12b",
            "meta-llama/llama-3-8b-instruct",
            "nvidia/nemotron-3-nano-omni",
        ]
        mock_instance.is_likely_vision_model.side_effect = lambda m: "gemma-3" in m or "omni" in m
        mock_tagger_cls.return_value = mock_instance

        evt_queue = queue.Queue()
        worker = ModelsWorker(api_url="http://localhost:1234/v1", event_queue=evt_queue)
        worker.start()
        worker.join(timeout=3.0)

        self.assertFalse(evt_queue.empty())
        event: GuiEvent = evt_queue.get_nowait()
        self.assertEqual(event.event_type, EVENT_MODELS_LOADED)
        self.assertEqual(len(event.data["models"]), 3)
        self.assertEqual(
            event.data["vision_models"],
            ["google/gemma-3-12b", "nvidia/nemotron-3-nano-omni"],
        )

    @patch("src.gui.worker.VisionTagger")
    @patch("src.gui.worker.RawExtractor")
    def test_batch_tagging_worker_dry_run(self, mock_extractor_cls, mock_tagger_cls):
        """Test BatchTaggingWorker executing a dry run."""
        mock_extractor = MagicMock()
        mock_extractor.prepare_base64_jpeg.return_value = ("base64data", (1920, 1080))
        mock_extractor_cls.return_value = mock_extractor

        mock_tagger = MagicMock()
        mock_tagger.analyze_image.return_value = TaggingResult(
            tags=["sunset", "half dome"],
            places=["Yosemite National Park"],
            hierarchical_tags=["Places|Yosemite"],
            description="Half Dome at sunset.",
        )
        mock_tagger_cls.return_value = mock_tagger

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            img_file = tmppath / "photo.jpg"
            img_file.write_bytes(b"dummy")
            xmp_file = tmppath / "photo.xmp"

            item = ImageFileItem(
                image_path=img_file,
                xmp_path=xmp_file,
                has_existing_xmp=False,
                existing_tags=[],
                existing_hierarchical=[],
            )

            config = TaggerConfig(
                folder=tmppath,
                dry_run=True,
            )

            evt_queue = queue.Queue()
            stop_evt = threading.Event()
            worker = BatchTaggingWorker(
                items=[item],
                config=config,
                event_queue=evt_queue,
                stop_event=stop_evt,
            )
            worker.start()
            worker.join(timeout=3.0)

            events = []
            while not evt_queue.empty():
                events.append(evt_queue.get_nowait())

            event_types = [e.event_type for e in events]
            self.assertIn(EVENT_ITEM_STARTED, event_types)
            self.assertIn(EVENT_ITEM_SUCCESS, event_types)
            self.assertIn(EVENT_BATCH_FINISHED, event_types)

            # In dry-run, XMP file should NOT have been created
            self.assertFalse(xmp_file.exists())

    def test_preview_widget_image_clearing_no_tcl_error(self):
        """Test that setting and clearing images on PreviewWidget does not raise pyimageX TclErrors."""
        import gc
        from PIL import Image as PILImage
        import customtkinter as ctk
        from src.gui.preview_widget import PreviewWidget

        root = ctk.CTk()
        root.withdraw()
        try:
            widget = PreviewWidget(root)
            widget.pack()

            # Set an image
            im1 = PILImage.new("RGB", (100, 100), color="blue")
            widget.set_preview_image(im1)
            root.update_idletasks()

            # Clear image
            widget._clear_image("Loading...")
            gc.collect()

            # Set another text or image
            im2 = PILImage.new("RGB", (100, 100), color="green")
            widget.set_preview_image(im2)
            root.update_idletasks()

            widget._clear_image("Finished")
            root.update_idletasks()
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()

