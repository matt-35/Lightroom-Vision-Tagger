import unittest
import tempfile
import shutil
from pathlib import Path
from PIL import Image
from src.scanner import DirectoryScanner
from src.raw_extractor import RawExtractor
from src.xmp_manager import XMPManager


class TestScannerAndExtractor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_directory_scanner(self):
        # Create sample files
        img1 = self.temp_dir / "photo1.jpg"
        img2 = self.temp_dir / "photo2.cr3"
        txt = self.temp_dir / "notes.txt"

        Image.new("RGB", (100, 100), color="blue").save(img1)
        img2.write_bytes(b"dummy raw content")
        txt.write_text("not an image")

        # Initially neither is tagged
        scanner = DirectoryScanner(folder=self.temp_dir, recursive=False, skip_tagged=True)
        items = scanner.discover_files()
        self.assertEqual(len(items), 2)
        paths = {item.image_path.name for item in items}
        self.assertIn("photo1.jpg", paths)
        self.assertIn("photo2.cr3", paths)

        # Tag photo1
        xmp1 = XMPManager.get_sidecar_path(img1)
        XMPManager.write_tags(xmp1, new_flat_tags=["blue", "test"])

        # Scanner with skip_tagged=True should now only return photo2
        items_after = scanner.discover_files()
        self.assertEqual(len(items_after), 1)
        self.assertEqual(items_after[0].image_path.name, "photo2.cr3")

    def test_raw_extractor_raster(self):
        img_path = self.temp_dir / "sample.jpg"
        test_img = Image.new("RGB", (3000, 2000), color="red")
        test_img.save(img_path)

        extractor = RawExtractor(max_dimension=1000)
        b64_str, dims = extractor.prepare_base64_jpeg(img_path)

        self.assertGreater(len(b64_str), 0)
        self.assertEqual(dims[0], 1000)
        self.assertEqual(dims[1], round(2000 * (1000 / 3000)))

    def test_raw_extractor_preview(self):
        img_path = self.temp_dir / "preview_sample.jpg"
        test_img = Image.new("RGB", (2000, 1500), color="green")
        test_img.save(img_path)

        extractor = RawExtractor()
        preview = extractor.extract_preview(img_path, max_dimension=400)
        self.assertLessEqual(max(preview.width, preview.height), 400)

        # Second call should hit the LRU preview cache
        cached_preview = extractor.extract_preview(img_path, max_dimension=400)
        self.assertEqual(preview.size, cached_preview.size)


if __name__ == "__main__":
    unittest.main()
