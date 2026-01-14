"""
UI Module
Provides the graphical user interface for the emotion music player.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional
from Modules.music import EmotionMusicPlayer, emotion_dict, music_library
from Modules.logger import get_logger
import os

class EmotionMusicUI:
    """Graphical user interface for the emotion music player."""
    
    def __init__(
        self,
        player: EmotionMusicPlayer,
        window_width: int = 512,
        window_height: int = 900,
        show_confidence: bool = True
    ):
        """
        Initialize the UI.
        
        Args:
            player: EmotionMusicPlayer instance
            window_width: Window width in pixels
            window_height: Window height in pixels
            show_confidence: Whether to show confidence indicator
        """
        self.player = player
        self.logger = get_logger()
        self.show_confidence = show_confidence
        self.current_emotion: Optional[int] = 6  # Default to Neutral
        self.current_confidence: float = 0.0
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Vibify - Emotion Music Player")
        self.root.geometry(f"{window_width}x{window_height}+650+100")
        self.root.resizable(False, False)
        self.root.configure(bg="#121212")
        
        self._create_widgets()
        
        self.logger.info("UI initialized successfully")

    def _create_widgets(self) -> None:
        """Create all UI widgets."""
        # Header
        header = tk.Label(
            self.root,
            text="Vibify",
            font=("Arial", 50, "underline", "bold"),
            bg="#121212",
            fg="ghostwhite"
        )
        header.pack(pady=20)

        # Emotion Display
        self.emotion_display = tk.Label(
            self.root,
            text=f"Current Emotion: {emotion_dict[self.current_emotion]}",
            font=("Arial", 25),
            bg="#121212",
            fg="white"
        )
        self.emotion_display.pack(pady=20)

        # Confidence Display
        if self.show_confidence:
            confidence_frame = tk.Frame(self.root, bg="#121212")
            confidence_frame.pack(pady=10)
            
            tk.Label(
                confidence_frame,
                text="Confidence:",
                font=("Arial", 14),
                bg="#121212",
                fg="white"
            ).pack(side="left", padx=5)
            
            self.confidence_label = tk.Label(
                confidence_frame,
                text="0%",
                font=("Arial", 14, "bold"),
                bg="#121212",
                fg="#00ff00"
            )
            self.confidence_label.pack(side="left", padx=5)
            
            # Progress bar for confidence
            self.confidence_bar = ttk.Progressbar(
                self.root,
                length=400,
                mode='determinate',
                maximum=100
            )
            self.confidence_bar.pack(pady=5)

        # Current Track Display
        self.track_display = tk.Label(
            self.root,
            text="No track playing",
            font=("Arial", 12),
            bg="#121212",
            fg="#888888",
            wraplength=450
        )
        self.track_display.pack(pady=10)

        # Style for the Combobox
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(
            "TCombobox",
            fieldbackground="#333333",
            background="#333333",
            foreground="black",
            arrowcolor="white",
            selectbackground="gray",
            selectforeground="white",
            padding=(15, 12, 15, 12),
            font=("Arial", 20)
        )

        # Dropdown for manual emotion selection
        self.emotion_var = tk.StringVar()
        self.emotion_dropdown = ttk.Combobox(
            self.root,
            textvariable=self.emotion_var,
            state="readonly",
            style="TCombobox"
        )
        self.emotion_dropdown['values'] = list(emotion_dict.values())
        self.emotion_dropdown.set(emotion_dict[self.current_emotion])
        self.emotion_dropdown.pack(pady=10)
        self.emotion_dropdown.bind("<<ComboboxSelected>>", self.on_emotion_select)

        # Control buttons frame
        controls_frame = tk.Frame(self.root, bg="#121212")
        controls_frame.pack(pady=15)

        self.play_button = tk.Button(
            controls_frame,
            text="Play",
            command=self.on_play_pause,
            font=("Arial", 12, "bold"),
            bg="lime",
            fg="black",
            padx=15,
            pady=8,
            width=10
        )
        self.play_button.pack(side="left", padx=10)

        next_button = tk.Button(
            controls_frame,
            text="Next",
            command=self.on_next,
            font=("Arial", 12, "bold"),
            bg="lime",
            fg="black",
            padx=15,
            pady=8,
            width=10
        )
        next_button.pack(side="left", padx=10)

        # Volume control
        volume_frame = tk.Frame(self.root, bg="#121212")
        volume_frame.pack(pady=10)
        
        tk.Label(
            volume_frame,
            text="Volume:",
            font=("Arial", 12),
            bg="#121212",
            fg="white"
        ).pack(side="left", padx=5)
        
        self.volume_slider = tk.Scale(
            volume_frame,
            from_=0,
            to=100,
            orient="horizontal",
            command=self.on_volume_change,
            bg="#333333",
            fg="white",
            highlightthickness=0,
            length=300
        )
        self.volume_slider.set(int(self.player.get_volume() * 100))
        self.volume_slider.pack(side="left", padx=5)

        # Quit button
        quit_button = tk.Button(
            self.root,
            text="Quit",
            command=self.on_quit,
            font=("Arial", 12, "bold"),
            bg="red",
            fg="white",
            padx=15,
            pady=8,
            width=15
        )
        quit_button.pack(pady=20)

    def update_emotion_display(self, emotion_index: int, confidence: float = 0.0) -> None:
        """
        Update the emotion display.
        
        Args:
            emotion_index: Index of the emotion
            confidence: Confidence score (0.0 to 1.0)
        """
        self.current_emotion = emotion_index
        self.current_confidence = confidence
        
        emotion_name = emotion_dict[emotion_index]
        self.emotion_display.config(text=f"Current Emotion: {emotion_name}")
        self.emotion_var.set(emotion_name)
        
        if self.show_confidence:
            confidence_pct = int(confidence * 100)
            self.confidence_label.config(text=f"{confidence_pct}%")
            self.confidence_bar['value'] = confidence_pct
            
            # Color code confidence
            if confidence >= 0.8:
                color = "#00ff00"  # Green
            elif confidence >= 0.6:
                color = "#ffff00"  # Yellow
            else:
                color = "#ff6600"  # Orange
            self.confidence_label.config(fg=color)
        
        self.logger.debug(f"UI updated: {emotion_name} ({confidence:.2%})")

    def update_track_display(self, track_path: Optional[str] = None) -> None:
        """
        Update the current track display.
        
        Args:
            track_path: Path to the current track
        """
        if track_path:
            track_name = os.path.basename(track_path)
            # Remove extension
            track_name = os.path.splitext(track_name)[0]
            # Replace underscores with spaces for better readability
            track_name = track_name.replace('_', ' ')
            self.track_display.config(text=f"♪ {track_name}", fg="white")
        else:
            self.track_display.config(text="No track playing", fg="#888888")

    def update_not_found(self) -> None:
        """Update display when no emotion is recognised."""
        self.emotion_display.config(text="Emotion not recognised!")
        self.emotion_var.set("")
        if self.show_confidence:
            self.confidence_label.config(text="0%")
            self.confidence_bar['value'] = 0

    def on_play_pause(self) -> None:
        """Handle play/pause button click."""
        if self.player.is_playing and not self.player.is_paused:
            self.player.pause_music()
            self.play_button.config(text="Resume")
        else:
            emotion_index = self.current_emotion if self.current_emotion is not None else 6
            self.player.play_music(emotion_index)
            self.update_emotion_display(emotion_index, self.current_confidence)
            self.update_track_display(self.player.current_track)
            self.play_button.config(text="Pause")

    def on_next(self) -> None:
        """Handle next button click."""
        self.player.next_track()
        self.update_track_display(self.player.current_track)
        self.play_button.config(text="Pause")

    def on_volume_change(self, value: str) -> None:
        """
        Handle volume slider change.
        
        Args:
            value: New volume value (0-100)
        """
        volume = int(value) / 100.0
        self.player.set_volume(volume)

    def on_quit(self) -> None:
        """Handle quit button click."""
        self.logger.info("User quit application")
        self.player.quit_player()
        self.root.destroy()

    def on_emotion_select(self, event) -> None:
        """Handle manual emotion selection from dropdown."""
        selected_emotion = self.emotion_var.get()
        emotion_index = list(emotion_dict.values()).index(selected_emotion)
        self.update_emotion_display(emotion_index, 1.0)  # Manual selection = 100% confidence
        self.player.play_music(emotion_index)
        self.update_track_display(self.player.current_track)
        self.play_button.config(text="Pause")

    def run(self) -> None:
        """Start the UI main loop."""
        self.logger.info("Starting UI main loop")
        self.root.mainloop()
