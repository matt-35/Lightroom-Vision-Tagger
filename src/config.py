from dataclasses import dataclass, field
from pathlib import Path
from typing import Set, List, Optional


# Standard camera RAW formats + common photography formats
SUPPORTED_RAW_EXTENSIONS: Set[str] = {
    ".cr2",  # Canon
    ".cr3",  # Canon
    ".crw",  # Canon
    ".nef",  # Nikon
    ".nrw",  # Nikon
    ".arw",  # Sony
    ".srf",  # Sony
    ".sr2",  # Sony
    ".raf",  # Fujifilm
    ".dng",  # Adobe DNG / Leica / Pentax / DJI
    ".rw2",  # Panasonic Lumix
    ".raw",  # Panasonic / Leica
    ".orf",  # Olympus / OM System
    ".ori",  # Olympus
    ".pef",  # Pentax
    ".ptx",  # Pentax
    ".3fr",  # Hasselblad
    ".fff",  # Hasselblad
    ".iiq",  # Phase One
    ".gpr",  # GoPro RAW
}

SUPPORTED_RASTER_EXTENSIONS: Set[str] = {
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".png",
    ".webp",
}

ALL_SUPPORTED_EXTENSIONS: Set[str] = SUPPORTED_RAW_EXTENSIONS | SUPPORTED_RASTER_EXTENSIONS


@dataclass
class TaggerConfig:
    """Configuration settings for the image tagger."""
    folder: Path
    recursive: bool = False
    api_url: str = "http://localhost:1234/v1"
    model: Optional[str] = None
    dry_run: bool = False
    overwrite_tags: bool = False  # If False, merges with existing tags
    skip_tagged: bool = True      # Skip photos that already contain tags in their XMP
    max_dimension: int = 1536     # Long edge resize for VLM inference
    jpeg_quality: int = 85
    timeout: float = 120.0
    hierarchical: bool = True     # Generate lr:hierarchicalSubject tags
    temperature: float = 0.2      # Low temperature for deterministic, accurate tags
    confidence_threshold: str = "high"
