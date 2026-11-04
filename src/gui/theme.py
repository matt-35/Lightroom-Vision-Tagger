"""
Theme and styling configuration for Lightroom Vision Auto-Tagger GUI.
"""

import customtkinter as ctk

# Color Palette (Modern Dark / Slate)
BG_DARK = "#12131A"
CARD_BG = "#1A1B26"
CARD_BORDER = "#2A2B3D"
SIDEBAR_BG = "#161722"

# Accents
ACCENT_PRIMARY = "#3B82F6"       # Blue
ACCENT_HOVER = "#2563EB"
ACCENT_SUCCESS = "#10B981"       # Emerald
ACCENT_SUCCESS_HOVER = "#059669"
ACCENT_WARNING = "#F59E0B"       # Amber
ACCENT_DANGER = "#EF4444"        # Red
ACCENT_DANGER_HOVER = "#DC2626"
ACCENT_PURPLE = "#8B5CF6"        # Purple / Landmarks

# Text colors
TEXT_PRIMARY = "#F3F4F6"
TEXT_SECONDARY = "#9CA3AF"
TEXT_MUTED = "#6B7280"

# Table / List selection
SELECTION_BG = "#2E344A"
STATUS_TAGGED = "#10B981"
STATUS_UNTAGGED = "#6B7280"
STATUS_PROCESSING = "#F59E0B"
STATUS_ERROR = "#EF4444"


def apply_theme():
    """Sets standard appearance mode and default color theme."""
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

