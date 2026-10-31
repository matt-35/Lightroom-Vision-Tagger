import json
import re
from typing import List, Dict, Any, Optional, Tuple
import httpx
from pydantic import BaseModel, Field


class TaggingResult(BaseModel):
    """Structured response from the vision model."""
    tags: List[str] = Field(default_factory=list, description="Descriptive flat keywords")
    places: List[str] = Field(default_factory=list, description="Recognized iconic places or landmarks")
    hierarchical_tags: List[str] = Field(
        default_factory=list,
        description="Lightroom hierarchical tags using pipe delimiter like Category|Subcategory"
    )
    description: Optional[str] = Field(default="", description="1-2 sentence descriptive caption")


SYSTEM_PROMPT = """You are an expert photography metadata tagger for Adobe Lightroom.
Analyze the provided image and generate comprehensive, high-quality tags for cataloging and searching.

Focus on:
1. General descriptive tags: subjects, objects, animals, flora, landscape elements, architecture.
2. Photographic conditions: lighting (e.g., golden hour, blue hour, overcast, harsh sunlight, backlit, studio lighting), weather, season, mood, composition (e.g., wide angle, macro, close-up, aerial, silhouette, long exposure).
3. Named & iconic places: Identify any specific famous landmark, monument, national park, iconic skyline, or recognizable location (e.g., "Grand Canyon", "Yosemite National Park", "Half Dome", "Eiffel Tower", "Golden Gate Bridge", "Mount Rainier", "Times Square", "Kyoto Fushimi Inari"). If the specific location cannot be determined with certainty, provide geographic scene types (e.g., "Pacific Northwest", "Alpine Mountain Range", "Coastal Cliff").
4. Hierarchical tags: Structured categories separated by pipe '|' for Lightroom hierarchies, such as:
   - "Places|Landmark Name" or "Places|Region|Landmark"
   - "Nature|Landscape|Mountains"
   - "Nature|Wildlife|Mammal"
   - "Lighting|Golden Hour"
   - "Architecture|Modern|Bridge"

Return ONLY a valid JSON object matching this schema:
{
  "tags": ["keyword1", "keyword2", ...],
  "places": ["Landmark/Place Name 1", ...],
  "hierarchical_tags": ["Category|Subcategory|Item", ...],
  "description": "Short 1-2 sentence summary of the image."
}
Keep keywords concise, singular where appropriate, lower-case for common tags, and Title Case for proper nouns/places.
"""


class VisionTagger:
    """Client for querying LM Studio's local vision models."""

    def __init__(self, api_url: str = "http://localhost:1234/v1", model: Optional[str] = None, timeout: float = 60.0):
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def get_available_models(self) -> List[str]:
        """Fetch list of available model IDs from LM Studio."""
        try:
            with httpx.Client(timeout=10.0) as client:
                res = client.get(f"{self.api_url}/models")
                res.raise_for_status()
                data = res.json()
                return [m["id"] for m in data.get("data", [])]
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to LM Studio at {self.api_url}. "
                f"Ensure LM Studio is running and local server is started. Error: {e}"
            )

    def resolve_model(self) -> str:
        """Determines which model to use, auto-detecting if none specified."""
        if self.model:
            return self.model

        available = self.get_available_models()
        if not available:
            raise ValueError(f"No models found loaded in LM Studio at {self.api_url}")

        # Prefer vision/omni/multimodal models if identifiable by name
        vision_candidates = [
            m for m in available
            if any(term in m.lower() for term in ["vl", "vision", "omni", "gemma-3", "flint", "pixtral", "llava", "internvl"])
        ]
        if vision_candidates:
            self.model = vision_candidates[0]
            return self.model

        # Fallback to the first available model
        self.model = available[0]
        return self.model

    def analyze_image(self, base64_jpeg: str) -> TaggingResult:
        """
        Sends base64 image to LM Studio and parses structured tags.
        """
        model_name = self.resolve_model()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this photo and provide keywords, recognized places, hierarchical tags, and description as JSON."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_jpeg}"
                        }
                    }
                ]
            }
        ]

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.2,
            "response_format": {"type": "json_object"}
        }

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.api_url}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            resp.raise_for_status()
            data = resp.json()

        content = data["choices"][0]["message"]["content"]
        return self._parse_json_response(content)

    @staticmethod
    def _parse_json_response(content: str) -> TaggingResult:
        """Robustly parses JSON response from the model, stripping markdown fences if present."""
        cleaned = content.strip()

        # Handle ```json ... ``` codeblocks
        if "```" in cleaned:
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1).strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback regex extraction if full JSON parsing fails
            tags_match = re.findall(r'"tags"\s*:\s*\[(.*?)\]', cleaned, re.DOTALL)
            places_match = re.findall(r'"places"\s*:\s*\[(.*?)\]', cleaned, re.DOTALL)
            hier_match = re.findall(r'"hierarchical_tags"\s*:\s*\[(.*?)\]', cleaned, re.DOTALL)

            tags = [t.strip(' "\'\n\r') for t in tags_match[0].split(",")] if tags_match else []
            places = [p.strip(' "\'\n\r') for p in places_match[0].split(",")] if places_match else []
            hier = [h.strip(' "\'\n\r') for h in hier_match[0].split(",")] if hier_match else []

            return TaggingResult(
                tags=[t for t in tags if t],
                places=[p for p in places if p],
                hierarchical_tags=[h for h in hier if h],
                description=""
            )

        # Merge places into tags if not already present
        flat_tags = [t.strip() for t in parsed.get("tags", []) if isinstance(t, str) and t.strip()]
        places = [p.strip() for p in parsed.get("places", []) if isinstance(p, str) and p.strip()]

        for place in places:
            if place.lower() not in {t.lower() for t in flat_tags}:
                flat_tags.append(place)

        hierarchical = [
            h.strip() for h in parsed.get("hierarchical_tags", [])
            if isinstance(h, str) and h.strip()
        ]

        # Add places into hierarchical tags if missing
        for place in places:
            place_hier = f"Places|{place}"
            if place_hier.lower() not in {h.lower() for h in hierarchical}:
                hierarchical.append(place_hier)

        return TaggingResult(
            tags=flat_tags,
            places=places,
            hierarchical_tags=hierarchical,
            description=parsed.get("description", "")
        )
