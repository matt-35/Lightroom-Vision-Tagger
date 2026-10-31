from dataclasses import dataclass
from pathlib import Path
from typing import List, Generator
from src.config import ALL_SUPPORTED_EXTENSIONS
from src.xmp_manager import XMPManager


@dataclass
class ImageFileItem:
    """Represents a discovered image file and its sidecar status."""
    image_path: Path
    xmp_path: Path
    has_existing_xmp: bool
    existing_tags: List[str]
    existing_hierarchical: List[str]

    @property
    def is_already_tagged(self) -> bool:
        return len(self.existing_tags) > 0


class DirectoryScanner:
    """Scans folders for supported camera RAW and image files."""

    def __init__(self, folder: Path, recursive: bool = False, skip_tagged: bool = True):
        self.folder = folder
        self.recursive = recursive
        self.skip_tagged = skip_tagged

    def discover_files(self) -> List[ImageFileItem]:
        """Discovers all candidate files and checks their XMP status."""
        if not self.folder.exists() or not self.folder.is_dir():
            raise FileNotFoundError(f"Folder not found or is not a directory: {self.folder}")

        candidates = self._find_files()
        items: List[ImageFileItem] = []

        for img_path in candidates:
            xmp_path = XMPManager.get_sidecar_path(img_path)
            has_xmp = xmp_path.exists()
            existing_tags, existing_hier = (
                XMPManager.read_tags(xmp_path) if has_xmp else ([], [])
            )

            item = ImageFileItem(
                image_path=img_path,
                xmp_path=xmp_path,
                has_existing_xmp=has_xmp,
                existing_tags=existing_tags,
                existing_hierarchical=existing_hier,
            )

            if self.skip_tagged and item.is_already_tagged:
                continue

            items.append(item)

        return items

    def _find_files(self) -> List[Path]:
        """Finds all matching file paths in the directory, skipping hidden and macOS AppleDouble files."""
        pattern = "**/*" if self.recursive else "*"
        matched = []
        for p in self.folder.glob(pattern):
            # Skip hidden files and macOS AppleDouble resource fork sidecars (e.g. ._* or .DS_Store)
            if p.name.startswith("."):
                continue
            if p.is_file() and p.suffix.lower() in ALL_SUPPORTED_EXTENSIONS:
                matched.append(p)
        return sorted(matched)
