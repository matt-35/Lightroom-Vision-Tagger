import unittest
from src.vision_tagger import VisionTagger, TaggingResult


class TestVisionTagger(unittest.TestCase):
    def test_parse_json_response_clean(self):
        raw_json = """{
            "tags": ["mountain", "sunset", "lake", "golden hour"],
            "places": ["Yosemite National Park"],
            "hierarchical_tags": ["Nature|Landscape|Mountains"],
            "description": "Sunset over a serene alpine lake in Yosemite."
        }"""

        result = VisionTagger._parse_json_response(raw_json)
        self.assertIn("mountain", result.tags)
        self.assertIn("Yosemite National Park", result.tags)
        self.assertIn("Yosemite National Park", result.places)
        self.assertIn("Nature|Landscape|Mountains", result.hierarchical_tags)
        self.assertIn("Places|Yosemite National Park", result.hierarchical_tags)
        self.assertEqual(result.description, "Sunset over a serene alpine lake in Yosemite.")

    def test_parse_json_response_with_markdown_fences(self):
        raw_markdown = """Here is the tagging analysis:
```json
{
  "tags": ["bridge", "ocean", "fog"],
  "places": ["Golden Gate Bridge"],
  "hierarchical_tags": ["Architecture|Bridge"],
  "description": "The Golden Gate Bridge shrouded in morning fog."
}
```
Hope this helps!"""

        result = VisionTagger._parse_json_response(raw_markdown)
        self.assertIn("bridge", result.tags)
        self.assertIn("Golden Gate Bridge", result.tags)
        self.assertIn("Golden Gate Bridge", result.places)
        self.assertIn("Architecture|Bridge", result.hierarchical_tags)
        self.assertIn("Places|Golden Gate Bridge", result.hierarchical_tags)


if __name__ == "__main__":
    unittest.main()
