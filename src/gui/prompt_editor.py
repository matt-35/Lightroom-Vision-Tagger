"""
System Prompt Editor Window for Lightroom Vision Auto-Tagger.
Allows viewing, customizing, and resetting the VLM system prompt.
"""

from typing import Callable, Optional
import customtkinter as ctk

from src.vision_tagger import SYSTEM_PROMPT
from src.gui.theme import (
    CARD_BG, CARD_BORDER, ACCENT_PRIMARY, ACCENT_HOVER,
    ACCENT_WARNING, ACCENT_SUCCESS, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED
)


class PromptEditorWindow(ctk.CTkToplevel):
    """Modal window for inspecting and modifying the Vision model's system prompt."""

    def __init__(
        self,
        parent,
        current_prompt: str,
        on_save_callback: Callable[[str], None],
    ):
        super().__init__(parent)
        self.title("⚙️ System Prompt Editor")
        self.geometry("820x620")
        self.minsize(680, 480)

        self.current_prompt = current_prompt
        self.on_save_callback = on_save_callback

        # Make it stay on top and focus
        self.attributes("-topmost", True)
        self.lift()
        self.focus_force()

        self._build_ui()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # 1. Header Frame
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))

        title_lbl = ctk.CTkLabel(
            header_frame,
            text="🧠 Vision Model System Prompt",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=TEXT_PRIMARY,
        )
        title_lbl.pack(anchor="w")

        subtitle_lbl = ctk.CTkLabel(
            header_frame,
            text="Customize instructions sent to the vision model. Ensure your changes preserve the required JSON schema fields below.",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_SECONDARY,
        )
        subtitle_lbl.pack(anchor="w", pady=(2, 0))

        # 2. Schema Notice Card
        schema_frame = ctk.CTkFrame(self, fg_color=CARD_BG, border_width=1, border_color=CARD_BORDER, corner_radius=8)
        schema_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=5)

        schema_text = (
            "⚠️ Important: The model must return valid JSON with these keys:\n"
            "• tags (list of strings)  • hierarchical_tags (list of Category|Subcategory)  • description (string)"
        )
        schema_lbl = ctk.CTkLabel(
            schema_frame,
            text=schema_text,
            font=ctk.CTkFont(size=11),
            text_color=ACCENT_WARNING,
            justify="left",
        )
        schema_lbl.pack(padx=14, pady=8, anchor="w")

        # 3. Editor Textbox
        self.textbox = ctk.CTkTextbox(
            self,
            font=ctk.CTkFont(family="Courier", size=12),
            wrap="word",
            border_width=1,
            border_color=CARD_BORDER,
            fg_color="#0F1017",
            text_color="#E0E2EC",
        )
        self.textbox.grid(row=2, column=0, sticky="nsew", padx=20, pady=10)
        self.textbox.insert("1.0", self.current_prompt)

        # 4. Action Buttons Bar
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.grid(row=3, column=0, sticky="ew", padx=20, pady=(10, 20))
        btn_frame.grid_columnconfigure(1, weight=1)

        reset_btn = ctk.CTkButton(
            btn_frame,
            text="↺ Reset to Default",
            fg_color="#2D3043",
            hover_color="#3B3F58",
            text_color=TEXT_PRIMARY,
            command=self._reset_to_default,
            width=140,
        )
        reset_btn.grid(row=0, column=0, sticky="w")

        status_lbl = ctk.CTkLabel(
            btn_frame,
            text="",
            font=ctk.CTkFont(size=12),
            text_color=ACCENT_SUCCESS,
        )
        self.status_lbl = status_lbl
        status_lbl.grid(row=0, column=1, sticky="w", padx=15)

        cancel_btn = ctk.CTkButton(
            btn_frame,
            text="Cancel",
            fg_color="#2D3043",
            hover_color="#3B3F58",
            text_color=TEXT_SECONDARY,
            command=self.destroy,
            width=90,
        )
        cancel_btn.grid(row=0, column=2, padx=10)

        save_btn = ctk.CTkButton(
            btn_frame,
            text="💾 Apply & Save",
            fg_color=ACCENT_PRIMARY,
            hover_color=ACCENT_HOVER,
            text_color="#FFFFFF",
            font=ctk.CTkFont(weight="bold"),
            command=self._apply_and_save,
            width=130,
        )
        save_btn.grid(row=0, column=3)

    def _reset_to_default(self):
        self.textbox.delete("1.0", "end")
        self.textbox.insert("1.0", SYSTEM_PROMPT)
        self.status_lbl.configure(text="Restored default prompt (click Apply to save)")

    def _apply_and_save(self):
        new_prompt = self.textbox.get("1.0", "end-1c").strip()
        if not new_prompt:
            self.status_lbl.configure(text="Prompt cannot be empty!")
            return
        self.on_save_callback(new_prompt)
        self.destroy()

