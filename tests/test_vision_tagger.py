import unittest
from src.vision_tagger import VisionTagger, TaggingResult


class TestVisionTagger(unittest.TestCase):
    def test_parse_json_response_clean(self):
        raw_json = """{
            "tags": ["mountain", "sunset", "lake", "golden hour"],
            "hierarchical_tags": ["Nature|Landscape|Mountains"],
            "description": "Sunset over a serene alpine lake."
        }"""

        result = VisionTagger._parse_json_response(raw_json)
        self.assertIn("mountain", result.tags)
        self.assertIn("sunset", result.tags)
        self.assertIn("Nature|Landscape|Mountains", result.hierarchical_tags)
        self.assertEqual(result.description, "Sunset over a serene alpine lake.")

    def test_parse_json_response_with_markdown_fences(self):
        raw_markdown = """Here is the tagging analysis:
```json
{
  "tags": ["bridge", "ocean", "fog"],
  "hierarchical_tags": ["Architecture|Bridge"],
  "description": "A suspension bridge shrouded in morning fog."
}
```
Hope this helps!"""

        result = VisionTagger._parse_json_response(raw_markdown)
        self.assertIn("bridge", result.tags)
        self.assertIn("Architecture|Bridge", result.hierarchical_tags)
        self.assertEqual(result.description, "A suspension bridge shrouded in morning fog.")


if __name__ == "__main__":
    unittest.main()
