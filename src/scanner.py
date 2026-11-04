import os
from concurrent.futures import ThreadPoolExecutor
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
        if not candidates:
            return []

        def _check_xmp(img_path: Path) -> ImageFileItem:
            xmp_path = XMPManager.get_sidecar_path(img_path)
            has_xmp = xmp_path.exists()
            existing_tags, existing_hier = (
                XMPManager.read_tags(xmp_path) if has_xmp else ([], [])
            )
            return ImageFileItem(
                image_path=img_path,
                xmp_path=xmp_path,
                has_existing_xmp=has_xmp,
                existing_tags=existing_tags,
                existing_hierarchical=existing_hier,
            )

        # Parallelize XMP inspection across threads for speed
        max_workers = min(32, (os.cpu_count() or 4) * 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            items = list(executor.map(_check_xmp, candidates))

        if self.skip_tagged:
            items = [item for item in items if not item.is_already_tagged]

        return items

    def _find_files(self) -> List[Path]:
        """Finds all matching file paths using fast os.scandir."""
        matched = []
        exts = ALL_SUPPORTED_EXTENSIONS

        def _scan(dir_path: Path):
            try:
                with os.scandir(dir_path) as it:
                    for entry in it:
                        name = entry.name
                        if name.startswith("."):
                            continue
                        try:
                            if entry.is_file(follow_symlinks=False):
                                dot_idx = name.rfind(".")
                                if dot_idx != -1 and name[dot_idx:].lower() in exts:
                                    matched.append(Path(entry.path))
                            elif self.recursive and entry.is_dir(follow_symlinks=False):
                                _scan(Path(entry.path))
                        except (PermissionError, OSError):
                            pass
            except (PermissionError, FileNotFoundError, OSError):
                pass

        _scan(self.folder)
        return sorted(matched)
