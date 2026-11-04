import json
import re
from typing import List, Dict, Any, Optional, Tuple
import httpx
from pydantic import BaseModel, Field


class TaggingResult(BaseModel):
    """Structured response from the vision model."""
    tags: List[str] = Field(default_factory=list, description="Descriptive flat keywords")
    places: List[str] = Field(default_factory=list, description="Deprecated / empty")
    hierarchical_tags: List[str] = Field(
        default_factory=list,
        description="Lightroom hierarchical tags using pipe delimiter like Category|Subcategory"
    )
    description: Optional[str] = Field(default="", description="1-2 sentence descriptive caption")


SYSTEM_PROMPT = """You are an expert photography metadata tagger for Adobe Lightroom.
Analyze the provided image and generate comprehensive, high-quality tags for cataloging and searching.

Focus on:
1. General descriptive tags: subjects, objects, animals, flora, landscape elements, architecture, materials, colors.
2. Photographic conditions: lighting (e.g., golden hour, blue hour, overcast, harsh sunlight, backlit, studio lighting), weather, season, mood, composition (e.g., wide angle, macro, close-up, aerial, silhouette, long exposure).
3. Hierarchical tags: Structured categories separated by pipe '|' for Lightroom hierarchies, such as:
   - "Nature|Landscape|Mountains"
   - "Nature|Wildlife|Mammal"
   - "Lighting|Golden Hour"
   - "Architecture|Modern|Bridge"
   - "Subject|People|Portrait"
   Do NOT attempt to guess or identify specific named landmarks, monuments, or geographic locations unless explicitly visible in signage or text. Instead, use general scene descriptors (e.g., "mountain", "coastal cliff", "city skyline").

Return ONLY a valid JSON object matching this schema:
{
  "tags": ["keyword1", "keyword2", ...],
  "hierarchical_tags": ["Category|Subcategory|Item", ...],
  "description": "Short 1-2 sentence summary of the image."
}
Keep keywords concise, singular where appropriate, and lower-case.
"""


TAGGING_RESPONSE_FORMAT: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "tagging_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Descriptive flat keywords"
                },
                "hierarchical_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Lightroom hierarchical tags using pipe delimiter like Category|Subcategory"
                },
                "description": {
                    "type": "string",
                    "description": "1-2 sentence descriptive caption"
                }
            },
            "required": ["tags", "hierarchical_tags", "description"],
            "additionalProperties": False
        }
    }
}


class VisionTagger:
    """Client for querying LM Studio's local vision models."""

    def __init__(
        self,
        api_url: str = "http://localhost:1234/v1",
        model: Optional[str] = None,
        timeout: float = 60.0,
        system_prompt: Optional[str] = None,
    ):
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.system_prompt = system_prompt or SYSTEM_PROMPT

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

    @staticmethod
    def is_likely_vision_model(model_name: str) -> bool:
        """Heuristic to check if a model ID corresponds to a vision/multimodal model."""
        name = model_name.lower()
        if any(term in name for term in ["embed", "bert", "flint"]):
            return False
        vision_indicators = ["vision", "-vl", "vl-", "omni", "gemma-3", "gemma-4", "pixtral", "llava", "internvl", "minicpm-v"]
        return any(term in name for term in vision_indicators)

    def resolve_model(self) -> str:
        """Determines which model to use, supporting exact or partial substring matching,
        or auto-detecting a vision model if none specified."""
        available = self.get_available_models()
        if not available:
            raise ValueError(f"No models found loaded in LM Studio at {self.api_url}")

        if self.model:
            # 1. Exact match
            if self.model in available:
                return self.model

            # 2. Case-insensitive exact match
            for m in available:
                if m.lower() == self.model.lower():
                    self.model = m
                    return self.model

            # 3. Partial substring match (e.g. 'gemma-3' -> 'google/gemma-3-12b')
            matches = [m for m in available if self.model.lower() in m.lower()]
            if len(matches) == 1:
                self.model = matches[0]
                return self.model
            elif len(matches) > 1:
                vision_subset = [m for m in matches if self.is_likely_vision_model(m)]
                self.model = vision_subset[0] if vision_subset else matches[0]
                return self.model

            models_list = "\n  - " + "\n  - ".join(available)
            raise ValueError(
                f"Model '{self.model}' not found in LM Studio.\n"
                f"Available models in LM Studio:{models_list}"
            )

        # Auto-detect if no model specified: pick first vision-capable model
        vision_candidates = [m for m in available if self.is_likely_vision_model(m)]
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
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this photo and provide keywords, hierarchical tags, and description as JSON."
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
            "response_format": TAGGING_RESPONSE_FORMAT
        }

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.api_url}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            # If server does not support json_schema or rejects response_format, retry without response_format
            if resp.status_code == 400 and ("response_format" in resp.text.lower() or "schema" in resp.text.lower()):
                payload.pop("response_format", None)
                resp = client.post(
                    f"{self.api_url}/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )

            if resp.status_code != 200:
                try:
                    err_json = resp.json()
                    err_msg = (
                        err_json.get("error", {}).get("message")
                        or err_json.get("message")
                        or str(err_json)
                    )
                except Exception:
                    err_msg = resp.text

                if "image" in err_msg.lower() or "vision" in err_msg.lower():
                    raise RuntimeError(
                        f"LM Studio Error ({resp.status_code}): {err_msg}\n"
                        f"   💡 Note: Model '{model_name}' does not appear to support Vision/Image inputs.\n"
                        f"   Please use a vision model such as google/gemma-3-12b or nvidia/nemotron-3-nano-omni."
                    )
                raise RuntimeError(f"LM Studio API Error ({resp.status_code}): {err_msg}")

            data = resp.json()

        choice = data["choices"][0]["message"]
        content = choice.get("content") or ""
        if not content.strip():
            content = choice.get("reasoning_content") or ""
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
            hier_match = re.findall(r'"hierarchical_tags"\s*:\s*\[(.*?)\]', cleaned, re.DOTALL)

            tags = [t.strip(' "\'\n\r') for t in tags_match[0].split(",")] if tags_match else []
            hier = [h.strip(' "\'\n\r') for h in hier_match[0].split(",")] if hier_match else []

            return TaggingResult(
                tags=[t for t in tags if t],
                places=[],
                hierarchical_tags=[h for h in hier if h],
                description=""
            )

        flat_tags = [t.strip() for t in parsed.get("tags", []) if isinstance(t, str) and t.strip()]
        hierarchical = [
            h.strip() for h in parsed.get("hierarchical_tags", [])
            if isinstance(h, str) and h.strip()
        ]

        return TaggingResult(
            tags=flat_tags,
            places=[],
            hierarchical_tags=hierarchical,
            description=parsed.get("description", "")
        )
