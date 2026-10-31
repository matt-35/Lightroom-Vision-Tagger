import io
import base64
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image, ImageOps
import rawpy

from src.config import SUPPORTED_RAW_EXTENSIONS, SUPPORTED_RASTER_EXTENSIONS


class RawExtractor:
    """Extracts and prepares images from RAW files or standard raster formats for VLM inference."""

    def __init__(self, max_dimension: int = 1536, jpeg_quality: int = 85):
        self.max_dimension = max_dimension
        self.jpeg_quality = jpeg_quality

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
