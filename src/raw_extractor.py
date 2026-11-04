from collections import OrderedDict
import io
import base64
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image, ImageOps
import rawpy

from src.config import SUPPORTED_RAW_EXTENSIONS, SUPPORTED_RASTER_EXTENSIONS

# In-memory LRU thumbnail cache for fast previewing
_PREVIEW_CACHE: "OrderedDict[str, Image.Image]" = OrderedDict()
_PREVIEW_CACHE_MAX_SIZE = 120


class RawExtractor:
    """Extracts and prepares images from RAW files or standard raster formats for VLM inference."""

    def __init__(self, max_dimension: int = 1536, jpeg_quality: int = 85):
        self.max_dimension = max_dimension
        self.jpeg_quality = jpeg_quality

    def extract_preview(self, file_path: Path, max_dimension: int = 400) -> Image.Image:
        """Fast preview extraction optimized specifically for GUI responsiveness with caching."""
        cache_key = f"{file_path.resolve()}_{max_dimension}"
        if cache_key in _PREVIEW_CACHE:
            _PREVIEW_CACHE.move_to_end(cache_key)
            return _PREVIEW_CACHE[cache_key]

        ext = file_path.suffix.lower()
        if ext in SUPPORTED_RAW_EXTENSIONS:
            img = self._extract_raw_preview(file_path, max_dimension)
        elif ext in SUPPORTED_RASTER_EXTENSIONS:
            img = self._extract_raster_preview(file_path, max_dimension)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        if img.mode != "RGB":
            img = img.convert("RGB")

        # Cache result
        _PREVIEW_CACHE[cache_key] = img
        if len(_PREVIEW_CACHE) > _PREVIEW_CACHE_MAX_SIZE:
            _PREVIEW_CACHE.popitem(last=False)

        return img

    def _extract_raw_preview(self, file_path: Path, max_dimension: int) -> Image.Image:
        """Extracts embedded thumbnail/preview from RAW without demosaicing when possible."""
        with rawpy.imread(str(file_path)) as raw:
            # Try fast embedded preview/thumbnail first
            try:
                thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    img = Image.open(io.BytesIO(thumb.data))
                    img = ImageOps.exif_transpose(img)
                    if max(img.width, img.height) > max_dimension:
                        img.thumbnail((max_dimension, max_dimension), Image.Resampling.BILINEAR)
                    return img
                elif thumb.format == rawpy.ThumbFormat.BITMAP:
                    img = Image.fromarray(thumb.data)
                    if max(img.width, img.height) > max_dimension:
                        img.thumbnail((max_dimension, max_dimension), Image.Resampling.BILINEAR)
                    return img
            except Exception:
                pass

            # Fast linear half-size demosaic fallback if no embedded thumbnail
            rgb_array = raw.postprocess(
                use_camera_wb=False,
                half_size=True,
                four_color_rgb=False,
                demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,
                output_bps=8,
            )
            img = Image.fromarray(rgb_array)
            if max(img.width, img.height) > max_dimension:
                img.thumbnail((max_dimension, max_dimension), Image.Resampling.BILINEAR)
            return img

    def _extract_raster_preview(self, file_path: Path, max_dimension: int) -> Image.Image:
        """Fast raster preview extraction with JPEG DCT draft decoding."""
        img = Image.open(file_path)
        try:
            # Draft mode lets libjpeg decode at reduced DCT resolution for instant loading
            img.draft("RGB", (max_dimension, max_dimension))
        except Exception:
            pass
        img = ImageOps.exif_transpose(img)
        if max(img.width, img.height) > max_dimension:
            img.thumbnail((max_dimension, max_dimension), Image.Resampling.BILINEAR)
        return img

    def extract_image(self, file_path: Path) -> Image.Image:
        """Extract PIL Image from RAW or standard image file."""
        ext = file_path.suffix.lower()

        if ext in SUPPORTED_RAW_EXTENSIONS:
            return self._extract_from_raw(file_path)
        elif ext in SUPPORTED_RASTER_EXTENSIONS:
            return self._extract_from_raster(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _extract_from_raw(self, file_path: Path) -> Image.Image:
        """Extract image from RAW, attempting fast embedded preview first."""
        with rawpy.imread(str(file_path)) as raw:
            # Step 1: Try fast embedded preview/thumbnail
            try:
                thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    img = Image.open(io.BytesIO(thumb.data))
                    # Check if extracted preview is substantial enough for vision model
                    if max(img.width, img.height) >= 640:
                        return ImageOps.exif_transpose(img)
                elif thumb.format == rawpy.ThumbFormat.BITMAP:
                    img = Image.fromarray(thumb.data)
                    if max(img.width, img.height) >= 640:
                        return img
            except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError, Exception):
                pass

            # Step 2: Fallback to fast half-size demosaicing
            # half_size=True skips full interpolation and is ~4x faster than standard demosaic
            rgb_array = raw.postprocess(
                use_camera_wb=True,
                half_size=True,
                no_auto_bright=False,
                output_bps=8,
            )
            return Image.fromarray(rgb_array)

    def _extract_from_raster(self, file_path: Path) -> Image.Image:
        """Load and normalize standard raster image."""
        img = Image.open(file_path)
        img = ImageOps.exif_transpose(img)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        return img

    def prepare_base64_jpeg(self, file_path: Path) -> Tuple[str, Tuple[int, int]]:
        """
        Extracts, resizes, and base64-encodes the image.
        Returns:
            (base64_jpeg_string, (resized_width, resized_height))
        """
        img = self.extract_image(file_path)

        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Downscale if long edge exceeds max_dimension
        if max(img.width, img.height) > self.max_dimension:
            img.thumbnail((self.max_dimension, self.max_dimension), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=self.jpeg_quality, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return encoded, (img.width, img.height)
