"""
Main Application Window for Lightroom Vision Auto-Tagger.
Built with CustomTkinter and high-performance Treeview for instant folder loading and smooth browsing.
"""

from pathlib import Path
import queue
import sys
import threading
from typing import List, Optional, Dict
from tkinter import filedialog, ttk
import customtkinter as ctk

from src.config import TaggerConfig
from src.scanner import ImageFileItem
from src.vision_tagger import SYSTEM_PROMPT, VisionTagger, TaggingResult
from src.xmp_manager import XMPManager
from src.gui.theme import (
    apply_theme, BG_DARK, CARD_BG, CARD_BORDER, SIDEBAR_BG,
    ACCENT_PRIMARY, ACCENT_HOVER, ACCENT_SUCCESS, ACCENT_SUCCESS_HOVER,
    ACCENT_WARNING, ACCENT_DANGER, ACCENT_DANGER_HOVER, ACCENT_PURPLE,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, SELECTION_BG
)
from src.gui.worker import (
    GuiEvent, ModelsWorker, ScanWorker, PreviewManager, BatchTaggingWorker,
    EVENT_MODELS_LOADED, EVENT_MODELS_FAILED,
    EVENT_SCAN_COMPLETED, EVENT_SCAN_FAILED,
    EVENT_ITEM_STARTED, EVENT_ITEM_SUCCESS, EVENT_ITEM_FAILED,
    EVENT_BATCH_FINISHED, EVENT_PREVIEW_READY,
)
from src.gui.preview_widget import PreviewWidget
from src.gui.prompt_editor import PromptEditorWindow


class ImageTaggerApp(ctk.CTk):
    """Main window of the Lightroom Vision Auto-Tagger GUI."""

    def __init__(self, initial_folder: Optional[Path] = None):
        super().__init__()
        apply_theme()

        self.title("📸 Lightroom Vision Auto-Tagger")
        self.geometry("1280x820")
        self.minsize(1050, 680)

        # State
        self.event_queue: queue.Queue = queue.Queue()
        self.stop_event: threading.Event = threading.Event()
        self.current_folder: Optional[Path] = initial_folder
        self.all_items: List[ImageFileItem] = []
        self.filtered_items: List[ImageFileItem] = []
        self.selected_item: Optional[ImageFileItem] = None
        self.item_by_path: Dict[str, ImageFileItem] = {}

        self.available_models: List[str] = []
        self.vision_models: List[str] = []
        self.system_prompt: str = SYSTEM_PROMPT
        self.is_custom_prompt: bool = False

        self.batch_worker: Optional[BatchTaggingWorker] = None
        self.is_processing: bool = False

        # Dedicated background preview manager with cancellation & cache
        self.preview_manager = PreviewManager(event_queue=self.event_queue, max_dimension=400)

        # Set window icon
        self._set_app_icon()

        self._build_ui()

        # Start background event loop polling
        self.after(40, self._process_events)

        # Initial model discovery
        self._refresh_models()

        # Initial scan if folder provided
        if self.current_folder and self.current_folder.is_dir():
            self._trigger_scan()

    def _set_app_icon(self):
        """Loads and sets the desktop application and dock icon."""
        icon_path = Path(__file__).resolve().parent.parent.parent / "assets" / "icon.png"
        if icon_path.exists():
            try:
                from PIL import Image as PILImage, ImageTk
                img = PILImage.open(icon_path)
                photo = ImageTk.PhotoImage(img.resize((64, 64), PILImage.Resampling.LANCZOS))
                self.wm_iconphoto(True, photo)
                self._app_icon_ref = photo
            except Exception:
                pass

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # =====================================================================
        # 1. LEFT SIDEBAR
        # =====================================================================
        sidebar = ctk.CTkFrame(self, fg_color=SIDEBAR_BG, width=320, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.grid_columnconfigure(0, weight=1)
        sidebar.grid_rowconfigure(4, weight=1)

        # App Brand Header
        brand_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        brand_frame.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 10))

        brand_row = ctk.CTkFrame(brand_frame, fg_color="transparent")
        brand_row.pack(fill="x")

        icon_path = Path(__file__).resolve().parent.parent.parent / "assets" / "icon.png"
        if icon_path.exists():
            try:
                from PIL import Image as PILImage
                icon_pil = PILImage.open(icon_path).resize((38, 38), PILImage.Resampling.LANCZOS)
                self.brand_icon = ctk.CTkImage(light_image=icon_pil, dark_image=icon_pil, size=(38, 38))
                icon_lbl = ctk.CTkLabel(brand_row, image=self.brand_icon, text="")
                icon_lbl.pack(side="left", padx=(0, 10))
            except Exception:
                pass

        title_container = ctk.CTkFrame(brand_row, fg_color="transparent")
        title_container.pack(side="left", fill="both", expand=True)

        app_title = ctk.CTkLabel(
            title_container,
            text="Image Tagger",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=TEXT_PRIMARY,
        )
        app_title.pack(anchor="w")

        app_subtitle = ctk.CTkLabel(
            title_container,
            text="Local Vision AI for Lightroom",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_SECONDARY,
        )
        app_subtitle.pack(anchor="w")

        # --- Card A: LM Studio Connection & Model Selector ---
        conn_card = ctk.CTkFrame(sidebar, fg_color=CARD_BG, border_width=1, border_color=CARD_BORDER, corner_radius=10)
        conn_card.grid(row=1, column=0, sticky="ew", padx=14, pady=6)
        conn_card.grid_columnconfigure(0, weight=1)

        conn_title = ctk.CTkLabel(
            conn_card,
            text="🌐 LM Studio Model",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_PRIMARY,
        )
        conn_title.grid(row=0, column=0, sticky="w", padx=12, pady=(10, 4))

        # URL entry & refresh button in one row
        url_frame = ctk.CTkFrame(conn_card, fg_color="transparent")
        url_frame.grid(row=1, column=0, sticky="ew", padx=12, pady=2)
        url_frame.grid_columnconfigure(0, weight=1)

        self.url_entry = ctk.CTkEntry(
            url_frame,
            placeholder_text="http://localhost:1234/v1",
            font=ctk.CTkFont(size=11),
            height=28,
        )
        self.url_entry.insert(0, "http://localhost:1234/v1")
        self.url_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.refresh_btn = ctk.CTkButton(
            url_frame,
            text="🔄",
            width=32,
            height=28,
            fg_color="#2D3043",
            hover_color="#3B3F58",
            command=self._refresh_models,
        )
        self.refresh_btn.grid(row=0, column=1)

        # Model Dropdown Label
        model_lbl = ctk.CTkLabel(
            conn_card,
            text="Select Model:",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_SECONDARY,
        )
        model_lbl.grid(row=2, column=0, sticky="w", padx=12, pady=(6, 2))

        # Model Dropdown (ComboBox)
        self.model_combo = ctk.CTkComboBox(
            conn_card,
            values=["Searching models..."],
            font=ctk.CTkFont(size=11),
            dropdown_font=ctk.CTkFont(size=11),
            height=30,
        )
        self.model_combo.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 4))

        # Connection status label
        self.conn_status_lbl = ctk.CTkLabel(
            conn_card,
            text="Connecting to LM Studio...",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_MUTED,
            anchor="w",
        )
        self.conn_status_lbl.grid(row=4, column=0, sticky="w", padx=12, pady=(0, 10))

        # --- Card B: System Prompt Customization ---
        prompt_card = ctk.CTkFrame(sidebar, fg_color=CARD_BG, border_width=1, border_color=CARD_BORDER, corner_radius=10)
        prompt_card.grid(row=2, column=0, sticky="ew", padx=14, pady=6)
        prompt_card.grid_columnconfigure(0, weight=1)

        prompt_btn = ctk.CTkButton(
            prompt_card,
            text="⚙️ Edit System Prompt",
            fg_color="#2D3043",
            hover_color="#3B3F58",
            text_color=TEXT_PRIMARY,
            command=self._open_prompt_editor,
            height=30,
        )
        prompt_btn.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 4))

        self.prompt_status_lbl = ctk.CTkLabel(
            prompt_card,
            text="Using default photo prompt",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_MUTED,
        )
        self.prompt_status_lbl.grid(row=1, column=0, sticky="w", padx=12, pady=(0, 10))

        # --- Card C: Tagging & Scanning Options ---
        options_card = ctk.CTkFrame(sidebar, fg_color=CARD_BG, border_width=1, border_color=CARD_BORDER, corner_radius=10)
        options_card.grid(row=3, column=0, sticky="ew", padx=14, pady=6)
        options_card.grid_columnconfigure(0, weight=1)

        opt_title = ctk.CTkLabel(
            options_card,
            text="⚙️ Options",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_PRIMARY,
        )
        opt_title.grid(row=0, column=0, sticky="w", padx=12, pady=(8, 4))

        self.opt_recursive = ctk.CTkSwitch(
            options_card,
            text="Include Subfolders (-r)",
            font=ctk.CTkFont(size=11),
            command=self._on_option_changed,
        )
        self.opt_recursive.grid(row=1, column=0, sticky="w", padx=12, pady=3)

        self.opt_skip_tagged = ctk.CTkSwitch(
            options_card,
            text="Skip Already Tagged",
            font=ctk.CTkFont(size=11),
        )
        self.opt_skip_tagged.select()
        self.opt_skip_tagged.grid(row=2, column=0, sticky="w", padx=12, pady=3)

        self.opt_overwrite = ctk.CTkSwitch(
            options_card,
            text="Overwrite Tags (No Merge)",
            font=ctk.CTkFont(size=11),
        )
        self.opt_overwrite.grid(row=3, column=0, sticky="w", padx=12, pady=3)

        self.opt_dry_run = ctk.CTkSwitch(
            options_card,
            text="Dry Run (Test Only)",
            font=ctk.CTkFont(size=11),
        )
        self.opt_dry_run.grid(row=4, column=0, sticky="w", padx=12, pady=(3, 10))

        # --- Card D: Execution Controls (Bottom of Sidebar) ---
        exec_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        exec_frame.grid(row=5, column=0, sticky="ew", padx=14, pady=(10, 16))
        exec_frame.grid_columnconfigure(0, weight=1)

        self.start_btn = ctk.CTkButton(
            exec_frame,
            text="▶ Start Auto-Tagging",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=ACCENT_PRIMARY,
            hover_color=ACCENT_HOVER,
            height=38,
            command=self._start_batch_tagging,
        )
        self.start_btn.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        self.stop_btn = ctk.CTkButton(
            exec_frame,
            text="⏹ Stop / Cancel",
            font=ctk.CTkFont(size=12),
            fg_color="#2D3043",
            hover_color=ACCENT_DANGER_HOVER,
            text_color=TEXT_MUTED,
            height=32,
            state="disabled",
            command=self._stop_batch_tagging,
        )
        self.stop_btn.grid(row=1, column=0, sticky="ew")

        # =====================================================================
        # 2. MAIN CONTENT AREA (RIGHT)
        # =====================================================================
        main_area = ctk.CTkFrame(self, fg_color="transparent")
        main_area.grid(row=0, column=1, sticky="nsew", padx=16, pady=16)
        main_area.grid_columnconfigure(0, weight=1)
        main_area.grid_rowconfigure(1, weight=1)

        # --- Top Header: Folder Picker & Statistics ---
        folder_card = ctk.CTkFrame(main_area, fg_color=CARD_BG, border_width=1, border_color=CARD_BORDER, corner_radius=10)
        folder_card.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        folder_card.grid_columnconfigure(1, weight=1)

        folder_lbl = ctk.CTkLabel(
            folder_card,
            text="📁 Folder:",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_PRIMARY,
        )
        folder_lbl.grid(row=0, column=0, padx=(14, 6), pady=12)

        self.folder_path_entry = ctk.CTkEntry(
            folder_card,
            placeholder_text="Select a photo directory to scan...",
            font=ctk.CTkFont(size=12),
            height=32,
        )
        if self.current_folder:
            self.folder_path_entry.insert(0, str(self.current_folder))
        self.folder_path_entry.grid(row=0, column=1, sticky="ew", padx=6, pady=12)

        browse_btn = ctk.CTkButton(
            folder_card,
            text="Browse...",
            width=90,
            height=32,
            fg_color="#2D3043",
            hover_color="#3B3F58",
            command=self._browse_folder,
        )
        browse_btn.grid(row=0, column=2, padx=6, pady=12)

        self.rescan_btn = ctk.CTkButton(
            folder_card,
            text="Scan",
            width=70,
            height=32,
            fg_color=ACCENT_PRIMARY,
            hover_color=ACCENT_HOVER,
            command=self._trigger_scan,
        )
        self.rescan_btn.grid(row=0, column=3, padx=(0, 14), pady=12)

        # Stats badges row
        stats_frame = ctk.CTkFrame(folder_card, fg_color="transparent")
        stats_frame.grid(row=1, column=0, columnspan=4, sticky="ew", padx=14, pady=(0, 10))

        self.stat_total_lbl = ctk.CTkLabel(
            stats_frame,
            text="Total: 0",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=TEXT_SECONDARY,
        )
        self.stat_total_lbl.pack(side="left", padx=(0, 14))

        self.stat_untagged_lbl = ctk.CTkLabel(
            stats_frame,
            text="Untagged: 0",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=ACCENT_WARNING,
        )
        self.stat_untagged_lbl.pack(side="left", padx=(0, 14))

        self.stat_tagged_lbl = ctk.CTkLabel(
            stats_frame,
            text="Tagged: 0",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=ACCENT_SUCCESS,
        )
        self.stat_tagged_lbl.pack(side="left")

        # --- Center: Split View (File List + Preview/Inspector) ---
        split_frame = ctk.CTkFrame(main_area, fg_color="transparent")
        split_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 12))
        split_frame.grid_columnconfigure(0, weight=5)
        split_frame.grid_columnconfigure(1, weight=5)
        split_frame.grid_rowconfigure(0, weight=1)

        # Left side: File List Card (Fast Treeview)
        file_list_card = ctk.CTkFrame(split_frame, fg_color=CARD_BG, border_width=1, border_color=CARD_BORDER, corner_radius=10)
        file_list_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        file_list_card.grid_columnconfigure(0, weight=1)
        file_list_card.grid_rowconfigure(1, weight=1)

        # File List Header & Filter
        fl_header = ctk.CTkFrame(file_list_card, fg_color="transparent")
        fl_header.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 6))
        fl_header.grid_columnconfigure(0, weight=1)

        fl_title = ctk.CTkLabel(
            fl_header,
            text="Discovered Images",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_PRIMARY,
        )
        fl_title.grid(row=0, column=0, sticky="w")

        self.filter_menu = ctk.CTkOptionMenu(
            fl_header,
            values=["All Files", "Untagged Only", "Tagged Only"],
            command=self._on_filter_changed,
            width=130,
            height=26,
            font=ctk.CTkFont(size=11),
        )
        self.filter_menu.grid(row=0, column=1, sticky="e")

        # High-performance ttk.Treeview with CTkScrollbar
        self.tree_container = ctk.CTkFrame(file_list_card, fg_color="#12131A", corner_radius=6)
        self.tree_container.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self.tree_container.grid_columnconfigure(0, weight=1)
        self.tree_container.grid_rowconfigure(0, weight=1)

        self._setup_treeview_style()

        columns = ("status", "name", "type", "tags")
        self.file_tree = ttk.Treeview(
            self.tree_container,
            columns=columns,
            show="headings",
            selectmode="browse",
            style="Custom.Treeview",
        )
        self.file_tree.heading("status", text="Stat", anchor="center")
        self.file_tree.heading("name", text="Filename", anchor="w")
        self.file_tree.heading("type", text="Type", anchor="center")
        self.file_tree.heading("tags", text="Tags", anchor="e")

        self.file_tree.column("status", width=42, minwidth=36, anchor="center")
        self.file_tree.column("name", width=180, minwidth=110, anchor="w")
        self.file_tree.column("type", width=55, minwidth=45, anchor="center")
        self.file_tree.column("tags", width=70, minwidth=50, anchor="e")

        self.file_scrollbar = ctk.CTkScrollbar(self.tree_container, command=self.file_tree.yview)
        self.file_tree.configure(yscrollcommand=self.file_scrollbar.set)

        self.file_tree.grid(row=0, column=0, sticky="nsew")
        self.file_scrollbar.grid(row=0, column=1, sticky="ns")

        self.file_tree.bind("<<TreeviewSelect>>", self._on_tree_selected)

        # Right side: Preview Widget
        self.preview_widget = PreviewWidget(
            split_frame,
            on_tag_single_callback=self._tag_single_image,
            on_save_manual_callback=self._save_manual_tags,
        )
        self.preview_widget.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        # --- Bottom: Progress & Live Activity Console ---
        bottom_frame = ctk.CTkFrame(main_area, fg_color=CARD_BG, border_width=1, border_color=CARD_BORDER, corner_radius=10)
        bottom_frame.grid(row=2, column=0, sticky="ew")
        bottom_frame.grid_columnconfigure(0, weight=1)

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(bottom_frame, height=10, fg_color="#12131A", progress_color=ACCENT_PRIMARY)
        self.progress_bar.set(0)
        self.progress_bar.grid(row=0, column=0, sticky="ew", padx=14, pady=(10, 4))

        # Status text
        status_row = ctk.CTkFrame(bottom_frame, fg_color="transparent")
        status_row.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 6))
        status_row.grid_columnconfigure(0, weight=1)

        self.status_text_lbl = ctk.CTkLabel(
            status_row,
            text="Ready",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_SECONDARY,
            anchor="w",
        )
        self.status_text_lbl.grid(row=0, column=0, sticky="w")

        self.progress_percent_lbl = ctk.CTkLabel(
            status_row,
            text="0%",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=TEXT_PRIMARY,
            anchor="e",
        )
        self.progress_percent_lbl.grid(row=0, column=1, sticky="e")

        # Collapsible log text box
        self.log_textbox = ctk.CTkTextbox(
            bottom_frame,
            height=65,
            font=ctk.CTkFont(family="Courier", size=11),
            fg_color="#0F1017",
            text_color="#CBD5E1",
        )
        self.log_textbox.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 10))
        self._log_message("Lightroom Vision Auto-Tagger initialized.")

    def _setup_treeview_style(self):
        """Configures modern dark theme styling for ttk.Treeview."""
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(
            "Custom.Treeview",
            background="#161722",
            foreground="#E2E8F0",
            fieldbackground="#161722",
            rowheight=30,
            font=("Helvetica", 11),
            borderwidth=0,
            relief="flat",
        )
        style.configure(
            "Custom.Treeview.Heading",
            background="#25273C",
            foreground="#94A3B8",
            font=("Helvetica", 11, "bold"),
            borderwidth=0,
            relief="flat",
        )
        style.map(
            "Custom.Treeview.Heading",
            background=[("active", "#2D304B")],
        )
        style.map(
            "Custom.Treeview",
            background=[("selected", "#2E344A")],
            foreground=[("selected", "#38BDF8")],
        )

    # =========================================================================
    # EVENT LOOP & BACKGROUND MESSAGE PROCESSING
    # =========================================================================
    def _process_events(self):
        """Drains the worker thread-safe event queue and updates the UI."""
        while not self.event_queue.empty():
            try:
                event: GuiEvent = self.event_queue.get_nowait()
            except queue.Empty:
                break

            self._handle_single_event(event)

        # Re-schedule event polling
        self.after(40, self._process_events)

    def _handle_single_event(self, event: GuiEvent):
        etype = event.event_type
        data = event.data

        if etype == EVENT_MODELS_LOADED:
            models = data["models"]
            vision_models = data["vision_models"]
            self.available_models = models
            self.vision_models = vision_models

            # Populate combo box
            display_values = []
            for m in models:
                prefix = "👁️ " if m in vision_models else "📄 "
                display_values.append(f"{prefix}{m}")

            if not display_values:
                display_values = ["No models loaded in LM Studio"]

            self.model_combo.configure(values=display_values)

            # Auto-select the first vision model, or first model
            if vision_models:
                self.model_combo.set(f"👁️ {vision_models[0]}")
                status_text = f"🟢 Connected ({len(models)} models, {len(vision_models)} vision)"
                self.conn_status_lbl.configure(text=status_text, text_color=ACCENT_SUCCESS)
            elif models:
                self.model_combo.set(f"📄 {models[0]}")
                status_text = f"🟡 Connected ({len(models)} models, ⚠️ no vision model detected)"
                self.conn_status_lbl.configure(text=status_text, text_color=ACCENT_WARNING)
            else:
                self.model_combo.set("No models loaded")
                self.conn_status_lbl.configure(text="🟡 Connected to LM Studio, but no models are loaded", text_color=ACCENT_WARNING)

            self._log_message(f"Connected to LM Studio: found {len(models)} models ({len(vision_models)} vision-capable).")

        elif etype == EVENT_MODELS_FAILED:
            self.conn_status_lbl.configure(
                text="🔴 Cannot connect to LM Studio (Ensure server is started on port 1234)",
                text_color=ACCENT_DANGER,
            )
            self.model_combo.configure(values=["Server offline"])
            self.model_combo.set("Server offline")
            self._log_message(f"LM Studio connection error: {data['error']}")

        elif etype == EVENT_SCAN_COMPLETED:
            self.all_items = data["items"]
            self._update_stats(data["total"], data["untagged"], data["tagged"])
            self._render_file_list()
            self._log_message(f"Discovered {data['total']} photo(s): {data['untagged']} untagged, {data['tagged']} already tagged.")
            self.rescan_btn.configure(state="normal", text="Scan")

        elif etype == EVENT_SCAN_FAILED:
            self.rescan_btn.configure(state="normal", text="Scan")
            self._log_message(f"Folder scan error: {data['error']}")

        elif etype == EVENT_PREVIEW_READY:
            if self.selected_item and self.selected_item.image_path == data["image_path"]:
                self.preview_widget.set_preview_image(data["pil_image"], data["error"])

        elif etype == EVENT_ITEM_STARTED:
            idx = data["index"]
            total = data["total"]
            item: ImageFileItem = data["item"]

            pct = (idx / max(total, 1))
            self.progress_bar.set(pct)
            self.progress_percent_lbl.configure(text=f"{int(pct * 100)}%")
            self.status_text_lbl.configure(text=f"Tagging ({idx+1}/{total}): {item.image_path.name}...")
            self._update_item_row_status(item.image_path, "processing")

        elif etype == EVENT_ITEM_SUCCESS:
            item: ImageFileItem = data["item"]
            result: TaggingResult = data["result"]
            final_flat = data["final_flat"]

            self._update_item_row_status(item.image_path, "tagged", f"{len(final_flat)} tags")
            self._log_message(f"Tagged {item.image_path.name}: {len(final_flat)} tags -> {item.xmp_path.name}")

            # If this is currently displayed in the preview panel, update its results!
            if self.selected_item and self.selected_item.image_path == item.image_path:
                self.preview_widget.set_tagging_result(result)

        elif etype == EVENT_ITEM_FAILED:
            item: ImageFileItem = data["item"]
            err = data["error"]
            self._update_item_row_status(item.image_path, "error")
            self._log_message(f"❌ Failed {item.image_path.name}: {err}")

        elif etype == EVENT_BATCH_FINISHED:
            self.is_processing = False
            self.start_btn.configure(state="normal", text="▶ Start Auto-Tagging", fg_color=ACCENT_PRIMARY)
            self.stop_btn.configure(state="disabled", text_color=TEXT_MUTED)
            self.progress_bar.set(1.0)
            self.progress_percent_lbl.configure(text="100%")

            cancelled = data["cancelled"]
            status = f"Batch finished: {data['success_count']} tagged, {data['fail_count']} failed."
            if cancelled:
                status += " (Stopped by user)"
            self.status_text_lbl.configure(text=status)
            self._log_message(f"=== {status} ===")
            if not self.opt_dry_run.get() and data["success_count"] > 0:
                self._log_message("💡 Tip: In Lightroom Classic, select photos and press Cmd+Option+S ('Read Metadata from File') to load tags.")

            # Update stats badges
            tagged = sum(1 for item in self.all_items if item.is_already_tagged)
            self._update_stats(len(self.all_items), len(self.all_items) - tagged, tagged)

    # =========================================================================
    # ACTIONS & WORKER TRIGGERS
    # =========================================================================
    def _refresh_models(self):
        """Queries LM Studio for available models in background."""
        api_url = self.url_entry.get().strip() or "http://localhost:1234/v1"
        self.conn_status_lbl.configure(text="Connecting to LM Studio...", text_color=TEXT_MUTED)
        self.model_combo.configure(values=["Connecting..."])
        worker = ModelsWorker(api_url=api_url, event_queue=self.event_queue)
        worker.start()

    def _browse_folder(self):
        chosen = filedialog.askdirectory(
            title="Select Directory with RAW / Photos",
            initialdir=str(self.current_folder) if self.current_folder else str(Path.home()),
        )
        if chosen:
            self.current_folder = Path(chosen)
            self.folder_path_entry.delete(0, "end")
            self.folder_path_entry.insert(0, str(self.current_folder))
            self._trigger_scan()

    def _trigger_scan(self):
        folder_str = self.folder_path_entry.get().strip()
        if not folder_str:
            return

        folder = Path(folder_str)
        if not folder.exists() or not folder.is_dir():
            self._log_message(f"Directory not found: {folder}")
            return

        self.current_folder = folder
        self.rescan_btn.configure(state="disabled", text="Scanning...")

        # Clear existing rows instantly
        self.file_tree.delete(*self.file_tree.get_children())
        self.item_by_path.clear()

        worker = ScanWorker(
            folder=self.current_folder,
            recursive=bool(self.opt_recursive.get()),
            skip_tagged=False,
            event_queue=self.event_queue,
        )
        worker.start()

    def _on_option_changed(self):
        if self.current_folder:
            self._trigger_scan()

    def _update_stats(self, total: int, untagged: int, tagged: int):
        self.stat_total_lbl.configure(text=f"Total: {total}")
        self.stat_untagged_lbl.configure(text=f"Untagged: {untagged}")
        self.stat_tagged_lbl.configure(text=f"Tagged: {tagged}")

    def _render_file_list(self):
        """Populates file rows into the fast Treeview."""
        self.file_tree.delete(*self.file_tree.get_children())
        self.item_by_path.clear()

        filter_mode = self.filter_menu.get()
        if filter_mode == "Untagged Only":
            items = [it for it in self.all_items if not it.is_already_tagged]
        elif filter_mode == "Tagged Only":
            items = [it for it in self.all_items if it.is_already_tagged]
        else:
            items = self.all_items

        self.filtered_items = items

        for item in items:
            key = str(item.image_path)
            self.item_by_path[key] = item

            status_icon = "✅" if item.is_already_tagged else "⚪"
            ext = item.image_path.suffix.upper().lstrip(".")
            is_raw = ext in {"CR2", "CR3", "ARW", "NEF", "RAF", "DNG", "RW2", "ORF"}
            type_str = f"RAW" if is_raw else ext
            tag_str = f"{len(item.existing_tags)} tags" if item.is_already_tagged else "—"

            self.file_tree.insert(
                "",
                "end",
                iid=key,
                values=(status_icon, item.image_path.name, type_str, tag_str),
            )

        if items:
            # Auto-select the first item
            first_key = str(items[0].image_path)
            self.file_tree.selection_set(first_key)
            self.file_tree.see(first_key)
            self._select_item(items[0])

    def _on_tree_selected(self, event=None):
        selected_iids = self.file_tree.selection()
        if not selected_iids:
            return
        key = selected_iids[0]
        item = self.item_by_path.get(key)
        if item:
            self._select_item(item)

    def _select_item(self, item: ImageFileItem):
        self.selected_item = item
        self.preview_widget.display_item(item)
        # Fast preview extraction via dedicated manager with cancellation & LRU cache
        self.preview_manager.request_preview(item.image_path)

    def _update_item_row_status(self, path: Path, status: str, tag_text: Optional[str] = None):
        key = str(path)
        if not self.file_tree.exists(key):
            return

        current_vals = list(self.file_tree.item(key, "values"))
        if status == "processing":
            current_vals[0] = "⏳"
        elif status == "tagged":
            current_vals[0] = "✅"
        elif status == "error":
            current_vals[0] = "❌"

        if tag_text:
            current_vals[3] = tag_text

        self.file_tree.item(key, values=current_vals)

    def _on_filter_changed(self, choice: str):
        self._render_file_list()

    def _open_prompt_editor(self):
        def on_prompt_saved(new_prompt: str):
            self.system_prompt = new_prompt
            self.is_custom_prompt = (new_prompt != SYSTEM_PROMPT)
            if self.is_custom_prompt:
                self.prompt_status_lbl.configure(text="✨ Custom prompt active", text_color=ACCENT_PURPLE)
                self._log_message("Applied custom Vision system prompt.")
            else:
                self.prompt_status_lbl.configure(text="Using default photo prompt", text_color=TEXT_MUTED)
                self._log_message("Restored default system prompt.")

        PromptEditorWindow(
            parent=self,
            current_prompt=self.system_prompt,
            on_save_callback=on_prompt_saved,
        )

    def _get_selected_model_id(self) -> Optional[str]:
        raw_val = self.model_combo.get().strip()
        if not raw_val or "Connecting" in raw_val or "No models" in raw_val or "Server offline" in raw_val:
            return None
        # Remove "👁️ " or "📄 " prefix
        clean = raw_val.replace("👁️ ", "").replace("📄 ", "").strip()
        return clean or None

    def _start_batch_tagging(self):
        if not self.all_items:
            self._log_message("No items to tag. Please scan a folder first.")
            return

        model_id = self._get_selected_model_id()
        api_url = self.url_entry.get().strip() or "http://localhost:1234/v1"

        skip_tagged = bool(self.opt_skip_tagged.get())
        targets = [it for it in self.all_items if (not it.is_already_tagged if skip_tagged else True)]

        if not targets:
            self._log_message("All images are already tagged! Uncheck 'Skip Already Tagged' to re-tag them.")
            return

        config = TaggerConfig(
            folder=self.current_folder,
            recursive=bool(self.opt_recursive.get()),
            api_url=api_url,
            model=model_id,
            dry_run=bool(self.opt_dry_run.get()),
            overwrite_tags=bool(self.opt_overwrite.get()),
            skip_tagged=skip_tagged,
            max_dimension=1536,
            timeout=300.0,
            system_prompt=self.system_prompt,
        )

        self.stop_event.clear()
        self.is_processing = True
        self.start_btn.configure(state="disabled", text="Tagging in Progress...", fg_color="#2D3043")
        self.stop_btn.configure(state="normal", text_color=ACCENT_DANGER)
        self.progress_bar.set(0)
        self.progress_percent_lbl.configure(text="0%")

        self._log_message(f"Starting batch tagging for {len(targets)} image(s) using model '{model_id or 'auto-detected'}'...")

        self.batch_worker = BatchTaggingWorker(
            items=targets,
            config=config,
            event_queue=self.event_queue,
            stop_event=self.stop_event,
        )
        self.batch_worker.start()

    def _stop_batch_tagging(self):
        if self.is_processing:
            self.stop_event.set()
            self.stop_btn.configure(text="Stopping...", state="disabled")
            self._log_message("Stopping batch after current image...")

    def _tag_single_image(self, item: ImageFileItem):
        """Tags only the selected image on demand."""
        model_id = self._get_selected_model_id()
        api_url = self.url_entry.get().strip() or "http://localhost:1234/v1"

        config = TaggerConfig(
            folder=self.current_folder,
            api_url=api_url,
            model=model_id,
            dry_run=bool(self.opt_dry_run.get()),
            overwrite_tags=bool(self.opt_overwrite.get()),
            system_prompt=self.system_prompt,
        )

        worker = BatchTaggingWorker(
            items=[item],
            config=config,
            event_queue=self.event_queue,
            stop_event=threading.Event(),
        )
        worker.start()

    def _save_manual_tags(self, item: ImageFileItem, flat_tags: List[str], hier_tags: List[str]):
        """Saves manually edited tags directly to XMP sidecar."""
        final_flat, final_hier = XMPManager.write_tags(
            xmp_path=item.xmp_path,
            new_flat_tags=flat_tags,
            new_hierarchical_tags=hier_tags,
            overwrite=True,
        )
        item.has_existing_xmp = True
        item.existing_tags = final_flat
        item.existing_hierarchical = final_hier
        self._update_item_row_status(item.image_path, "tagged", f"{len(final_flat)} tags")
        self._log_message(f"Saved manual edits to {item.xmp_path.name} ({len(final_flat)} keywords).")

    def _log_message(self, message: str):
        self.log_textbox.insert("end", f"> {message}\n")
        self.log_textbox.see("end")


def launch_gui(initial_folder: Optional[Path] = None):
    """Entry point to launch the CustomTkinter GUI."""
    app = ImageTaggerApp(initial_folder=initial_folder)
    app.mainloop()
