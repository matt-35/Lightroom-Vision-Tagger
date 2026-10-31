import unittest
from pathlib import Path
import tempfile
import shutil
from src.xmp_manager import XMPManager, XPACKET_HEADER, XPACKET_FOOTER

SAMPLE_EXISTING_XMP = f"""{XPACKET_HEADER}<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Adobe XMP Core 7.0-c000">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:lr="http://ns.adobe.com/lightroom/1.0/"
    xmp:Rating="5"
    crs:Exposure2012="+0.50">
   <dc:subject>
    <rdf:Bag>
     <rdf:li>Original Tag</rdf:li>
    </rdf:Bag>
   </dc:subject>
   <lr:hierarchicalSubject>
    <rdf:Bag>
     <rdf:li>Category|Original</rdf:li>
    </rdf:Bag>
   </lr:hierarchicalSubject>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>{XPACKET_FOOTER}"""


class TestXMPManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_new_xmp(self):
        xmp_file = self.temp_dir / "test_photo.xmp"
        flat_tags = ["Landscape", "Mountain", "Sunset"]
        hier_tags = ["Nature|Landscape", "Nature|Mountain"]

        final_flat, final_hier = XMPManager.write_tags(
            xmp_path=xmp_file,
            new_flat_tags=flat_tags,
            new_hierarchical_tags=hier_tags,
        )

        self.assertTrue(xmp_file.exists())
        self.assertEqual(set(final_flat), set(flat_tags))
        self.assertEqual(set(final_hier), set(hier_tags))

        # Read back
        read_flat, read_hier = XMPManager.read_tags(xmp_file)
        self.assertEqual(set(read_flat), set(flat_tags))
        self.assertEqual(set(read_hier), set(hier_tags))

        content = xmp_file.read_text(encoding="utf-8")
        self.assertIn("<?xpacket begin=", content)
        self.assertIn("<?xpacket end=", content)

    def test_update_existing_xmp_preserve_edits(self):
        xmp_file = self.temp_dir / "existing.xmp"
        xmp_file.write_text(SAMPLE_EXISTING_XMP, encoding="utf-8")

        new_tags = ["New Tag", "Sunset", "Original Tag"]
        new_hier = ["Nature|Sunset"]

        final_flat, final_hier = XMPManager.write_tags(
            xmp_path=xmp_file,
            new_flat_tags=new_tags,
            new_hierarchical_tags=new_hier,
            overwrite=False,
        )

        self.assertIn("Original Tag", final_flat)
        self.assertIn("New Tag", final_flat)
        self.assertIn("Sunset", final_flat)
        self.assertEqual(len(final_flat), 3)

        self.assertIn("Category|Original", final_hier)
        self.assertIn("Nature|Sunset", final_hier)

        updated_content = xmp_file.read_text(encoding="utf-8")
        self.assertTrue('Rating="5"' in updated_content or "Rating" in updated_content)
        self.assertIn("+0.50", updated_content)

    def test_update_existing_xmp_overwrite(self):
        xmp_file = self.temp_dir / "overwrite.xmp"
        xmp_file.write_text(SAMPLE_EXISTING_XMP, encoding="utf-8")

        new_tags = ["Brand New Tag"]
        new_hier = ["Places|Yosemite"]

        final_flat, final_hier = XMPManager.write_tags(
            xmp_path=xmp_file,
            new_flat_tags=new_tags,
            new_hierarchical_tags=new_hier,
            overwrite=True,
        )

        self.assertEqual(final_flat, ["Brand New Tag"])
        self.assertEqual(final_hier, ["Places|Yosemite"])

        read_flat, read_hier = XMPManager.read_tags(xmp_file)
        self.assertEqual(read_flat, ["Brand New Tag"])
        self.assertEqual(read_hier, ["Places|Yosemite"])


if __name__ == "__main__":
    unittest.main()
