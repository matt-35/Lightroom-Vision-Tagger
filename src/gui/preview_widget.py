"""
Preview and Tag Inspector widget for Lightroom Vision Auto-Tagger GUI.
Displays image preview, existing XMP metadata, and AI-generated tags.
"""

from pathlib import Path
from typing import Optional, Callable, List
from PIL import Image
import customtkinter as ctk

from src.scanner import ImageFileItem
from src.vision_tagger import TaggingResult
from src.xmp_manager import XMPManager
from src.gui.theme import (
    CARD_BG, CARD_BORDER, ACCENT_PRIMARY, ACCENT_HOVER,
    ACCENT_SUCCESS, ACCENT_PURPLE, ACCENT_WARNING,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED
)


class PreviewWidget(ctk.CTkFrame):
    """Card displaying the image preview, metadata tags, and single-image actions."""

    def __init__(
        self,
        parent,
        on_tag_single_callback: Optional[Callable[[ImageFileItem], None]] = None,
        on_save_manual_callback: Optional[Callable[[ImageFileItem, List[str], List[str]], None]] = None,
        **kwargs
    ):
        super().__init__(parent, fg_color=CARD_BG, border_width=1, border_color=CARD_BORDER, corner_radius=12, **kwargs)
        self.on_tag_single_callback = on_tag_single_callback
        self.on_save_manual_callback = on_save_manual_callback
        self.current_item: Optional[ImageFileItem] = None
        self.current_pil_image: Optional[Image.Image] = None
        self._current_ctk_image: Optional[ctk.CTkImage] = None

        self._build_ui()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # 1. Header
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 8))
        header_frame.grid_columnconfigure(0, weight=1)

        self.filename_lbl = ctk.CTkLabel(
            header_frame,
            text="No Image Selected",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=TEXT_PRIMARY,
            anchor="w",
        )
        self.filename_lbl.grid(row=0, column=0, sticky="w")

        self.format_badge = ctk.CTkLabel(
            header_frame,
            text="",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#FFFFFF",
            fg_color="#2D3043",
            corner_radius=6,
            padx=8,
            pady=2,
        )
        self.format_badge.grid(row=0, column=1, sticky="e")

        # 2. Preview Canvas / Image Box
        self.preview_container = ctk.CTkFrame(self, fg_color="#0F1017", corner_radius=8, height=220)
        self.preview_container.grid(row=1, column=0, sticky="ew", padx=16, pady=4)
        self.preview_container.pack_propagate(False)

        self.img_lbl = ctk.CTkLabel(
            self.preview_container,
            text="Select an image from the list to preview",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
        )
        self.img_lbl.pack(expand=True, fill="both")

        # 3. Scrollable Inspector Details
        self.scroll_details = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scroll_details.grid(row=2, column=0, sticky="nsew", padx=14, pady=8)
        self.scroll_details.grid_columnconfigure(0, weight=1)

        # AI Caption / Description
        desc_title = ctk.CTkLabel(
            self.scroll_details,
            text="📝 AI Description",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT_SECONDARY,
            anchor="w",
        )
        desc_title.grid(row=0, column=0, sticky="w", pady=(4, 2))

        self.desc_val = ctk.CTkLabel(
            self.scroll_details,
            text="—",
            font=ctk.CTkFont(size=12, slant="italic"),
            text_color=TEXT_PRIMARY,
            wraplength=340,
            justify="left",
            anchor="w",
        )
        self.desc_val.grid(row=1, column=0, sticky="w", pady=(0, 8))

        # Recognized Places
        places_title = ctk.CTkLabel(
            self.scroll_details,
            text="🏛️ Recognized Places / Landmarks",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=ACCENT_PURPLE,
            anchor="w",
        )
        places_title.grid(row=2, column=0, sticky="w", pady=(4, 2))

        self.places_val = ctk.CTkLabel(
            self.scroll_details,
            text="—",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT_PRIMARY,
            wraplength=340,
            justify="left",
            anchor="w",
        )
        self.places_val.grid(row=3, column=0, sticky="w", pady=(0, 8))

        # Flat Keywords
        tags_title = ctk.CTkLabel(
            self.scroll_details,
            text="🏷️ Keywords (XMP dc:subject)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=ACCENT_SUCCESS,
            anchor="w",
        )
        tags_title.grid(row=4, column=0, sticky="w", pady=(4, 2))

        self.tags_textbox = ctk.CTkTextbox(
            self.scroll_details,
            height=65,
            font=ctk.CTkFont(size=12),
            border_width=1,
            border_color=CARD_BORDER,
            fg_color="#0F1017",
            wrap="word",
        )
        self.tags_textbox.grid(row=5, column=0, sticky="ew", pady=(0, 8))

        # Hierarchical Tags
        hier_title = ctk.CTkLabel(
            self.scroll_details,
            text="🌲 Hierarchy (lr:hierarchicalSubject)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=ACCENT_PRIMARY,
            anchor="w",
        )
        hier_title.grid(row=6, column=0, sticky="w", pady=(4, 2))

        self.hier_textbox = ctk.CTkTextbox(
            self.scroll_details,
            height=65,
            font=ctk.CTkFont(size=12),
            border_width=1,
            border_color=CARD_BORDER,
            fg_color="#0F1017",
            wrap="word",
        )
        self.hier_textbox.grid(row=7, column=0, sticky="ew", pady=(0, 8))

        # 4. Action Buttons Footer
        action_bar = ctk.CTkFrame(self, fg_color="transparent")
        action_bar.grid(row=3, column=0, sticky="ew", padx=16, pady=(4, 14))
        action_bar.grid_columnconfigure((0, 1), weight=1)

        self.tag_single_btn = ctk.CTkButton(
            action_bar,
            text="⚡ Tag This Image",
            fg_color=ACCENT_PRIMARY,
            hover_color=ACCENT_HOVER,
            command=self._on_tag_single_clicked,
            state="disabled",
        )
        self.tag_single_btn.grid(row=0, column=0, padx=(0, 6), sticky="ew")

        self.save_btn = ctk.CTkButton(
            action_bar,
            text="💾 Save Tags to XMP",
            fg_color="#2D3043",
            hover_color="#3B3F58",
            command=self._on_save_clicked,
            state="disabled",
        )
        self.save_btn.grid(row=0, column=1, padx=(6, 0), sticky="ew")

    def _clear_image(self, placeholder_text: str = "Loading preview..."):
        """Safely clears image from CTkLabel and underlying Tk label to prevent pyimageX TclErrors."""
        self._current_ctk_image = None
        try:
            self.img_lbl._label.configure(image="")
        except Exception:
            pass
        self.img_lbl.configure(image=None, text=placeholder_text)

    def display_item(self, item: ImageFileItem):
        """Populates item metadata."""
        self.current_item = item
        self.filename_lbl.configure(text=item.image_path.name)

        ext = item.image_path.suffix.upper().lstrip(".")
        is_raw = ext in {"CR2", "CR3", "ARW", "NEF", "RAF", "DNG", "RW2", "ORF"}
        badge_text = f"RAW ({ext})" if is_raw else ext
        self.format_badge.configure(text=badge_text)

        # Clear preview safely until worker delivers it
        self._clear_image("Loading preview...")

        # Populate current tags from XMP
        self._populate_tags(item.existing_tags, item.existing_hierarchical)
        self.desc_val.configure(text="Select 'Tag This Image' or run Batch to generate description")
        self.places_val.configure(text="—")

        self.tag_single_btn.configure(state="normal")
        self.save_btn.configure(state="normal")

    def set_preview_image(self, pil_img: Optional[Image.Image], error_msg: Optional[str] = None):
        """Sets the loaded PIL preview image."""
        if error_msg or pil_img is None:
            msg = f"Preview unavailable: {error_msg}" if error_msg else "No preview available"
            self._clear_image(msg)
            return

        self.current_pil_image = pil_img
        # Calculate aspect ratio scaled down to container size (e.g. 340w x 210h)
        target_w, target_h = 340, 210
        img_copy = pil_img.copy()
        img_copy.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)

        ctk_img = ctk.CTkImage(
            light_image=img_copy,
            dark_image=img_copy,
            size=(img_copy.width, img_copy.height),
        )
        self._current_ctk_image = ctk_img
        self.img_lbl.configure(image=self._current_ctk_image, text="")

    def set_tagging_result(self, result: TaggingResult):
        """Displays AI tagging results."""
        if result.description:
            self.desc_val.configure(text=result.description)
        if result.places:
            self.places_val.configure(text=", ".join(result.places))
        else:
            self.places_val.configure(text="None detected")

        self._populate_tags(result.tags, result.hierarchical_tags)

    def _populate_tags(self, flat_tags: List[str], hier_tags: List[str]):
        self.tags_textbox.delete("1.0", "end")
        if flat_tags:
            self.tags_textbox.insert("1.0", ", ".join(flat_tags))

        self.hier_textbox.delete("1.0", "end")
        if hier_tags:
            self.hier_textbox.insert("1.0", "\n".join(hier_tags))

    def _on_tag_single_clicked(self):
        if self.current_item and self.on_tag_single_callback:
            self.on_tag_single_callback(self.current_item)

    def _on_save_clicked(self):
        if not self.current_item:
            return

        # Parse tags from textboxes
        flat_str = self.tags_textbox.get("1.0", "end-1c").strip()
        hier_str = self.hier_textbox.get("1.0", "end-1c").strip()

        flat_tags = [t.strip() for t in flat_str.split(",") if t.strip()]
        hier_tags = [h.strip() for h in hier_str.splitlines() if h.strip()]

        if self.on_save_manual_callback:
            self.on_save_manual_callback(self.current_item, flat_tags, hier_tags)
        else:
            XMPManager.write_tags(
                xmp_path=self.current_item.xmp_path,
                new_flat_tags=flat_tags,
                new_hierarchical_tags=hier_tags,
                overwrite=True,
            )
            self.current_item.existing_tags = flat_tags
            self.current_item.existing_hierarchical = hier_tags
            self.current_item.has_existing_xmp = True

