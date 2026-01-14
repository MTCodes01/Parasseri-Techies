"""
Music Player Module
Handles music playback based on detected emotions with cross-platform support.
"""

import pygame
import os
import random
import time
from typing import Dict, List, Optional
from Modules.logger import get_logger

class EmotionMusicPlayer:
    """Music player that plays songs based on detected emotions."""
    
    def __init__(
        self,
        emotion_dict: Dict[int, str],
        music_library: Dict[str, List[str]],
        fade_duration: int = 1000,
        default_volume: float = 0.7
    ):
        """
        Initialize the emotion music player.
        
        Args:
            emotion_dict: Mapping of emotion indices to names
            music_library: Mapping of emotion names to track lists
            fade_duration: Fade duration in milliseconds
            default_volume: Default volume (0.0 to 1.0)
        """
        self.emotion_dict = emotion_dict
        self.music_library = music_library
        self.fade_duration = fade_duration
        self.current_emotion: Optional[str] = None
        self.current_track: Optional[str] = None
        self.last_track: Optional[str] = None
        self.is_playing = False
        self.is_paused = False
        self.play_history: List[str] = []  # Track play history
        self.max_history = 10  # Remember last 10 tracks
        
        self.logger = get_logger()
        
        # Initialize pygame mixer
        try:
            pygame.mixer.init()
            pygame.mixer.music.set_volume(default_volume)
            self.logger.info("Pygame mixer initialized successfully")
        except pygame.error as e:
            self.logger.error(f"Failed to initialize pygame mixer: {e}")
            raise

    def play_music(self, emotion_index: int) -> None:
        """
        Play music for the given emotion.
        
        Args:
            emotion_index: Index of the emotion
        """
        self.logger.debug(f"play_music called with emotion_index: {emotion_index}")
        emotion = self.emotion_dict.get(emotion_index, "Neutral")
        self.logger.debug(f"Resolved emotion: {emotion}")

        if emotion != self.current_emotion or not self.is_playing:
            self.current_emotion = emotion
            self.switch_track(emotion)
        elif self.is_paused:
            self.resume_music()

    def switch_track(self, emotion: str) -> None:
        """
        Switch to a new track for the given emotion.
        
        Args:
            emotion: Name of the emotion
        """
        # Fade out current track if playing
        if self.current_track and self.is_playing:
            try:
                pygame.mixer.music.fadeout(self.fade_duration)
                time.sleep(self.fade_duration / 1000.0)
            except pygame.error as e:
                self.logger.warning(f"Error fading out music: {e}")

        tracks = self.music_library.get(emotion, [])
        if not tracks:
            self.logger.warning(f"No tracks available for emotion: {emotion}")
            return

        # Smart track selection: avoid recently played tracks
        available_tracks = [
            track for track in tracks 
            if track not in self.play_history[-self.max_history:]
        ]

        # If all tracks have been played recently, reset the pool
        if not available_tracks:
            available_tracks = tracks
            self.logger.debug("All tracks played recently, resetting pool")

        # Select a random track from available tracks
        track_path = random.choice(available_tracks)

        if track_path and os.path.exists(track_path):
            try:
                pygame.mixer.music.load(track_path)
                pygame.mixer.music.play(loops=0)
                
                # Update tracking variables
                self.last_track = self.current_track
                self.current_track = track_path
                self.is_playing = True
                self.is_paused = False
                
                # Add to play history
                self.play_history.append(track_path)
                if len(self.play_history) > self.max_history * 2:
                    self.play_history = self.play_history[-self.max_history:]
                
                self.logger.track_playing(emotion, track_path)

            except pygame.error as e:
                self.logger.track_error(track_path, str(e))
        else:
            self.logger.error(f"Track not found or does not exist: {track_path}")

    def stop_music(self) -> None:
        """Stop music playback."""
        try:
            pygame.mixer.music.stop()
            self.is_playing = False
            self.is_paused = False
            self.current_emotion = None
            self.current_track = None
            self.logger.info("Music stopped")
        except pygame.error as e:
            self.logger.error(f"Error stopping music: {e}")

    def pause_music(self) -> None:
        """Pause music playback."""
        if self.is_playing and not self.is_paused:
            try:
                pygame.mixer.music.pause()
                self.is_paused = True
                self.logger.info("Music paused")
            except pygame.error as e:
                self.logger.error(f"Error pausing music: {e}")

    def resume_music(self) -> None:
        """Resume music playback."""
        if self.is_paused:
            try:
                pygame.mixer.music.unpause()
                self.is_paused = False
                self.logger.info("Music resumed")
            except pygame.error as e:
                self.logger.error(f"Error resuming music: {e}")

    def next_track(self) -> None:
        """Play the next track in the current emotion."""
        if self.current_emotion:
            self.switch_track(self.current_emotion)
        else:
            self.logger.warning("No current emotion set, cannot play next track")

    def set_volume(self, volume: float) -> None:
        """
        Set playback volume.
        
        Args:
            volume: Volume level (0.0 to 1.0)
        """
        volume = max(0.0, min(1.0, volume))
        try:
            pygame.mixer.music.set_volume(volume)
            self.logger.debug(f"Volume set to {volume:.2f}")
        except pygame.error as e:
            self.logger.error(f"Error setting volume: {e}")

    def get_volume(self) -> float:
        """
        Get current volume.
        
        Returns:
            Current volume (0.0 to 1.0)
        """
        try:
            return pygame.mixer.music.get_volume()
        except pygame.error:
            return 0.0

    def is_track_playing(self) -> bool:
        """
        Check if a track is currently playing.
        
        Returns:
            True if playing, False otherwise
        """
        try:
            return pygame.mixer.music.get_busy()
        except pygame.error:
            return False

    def quit_player(self) -> None:
        """Quit the music player and cleanup resources."""
        try:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
            self.logger.info("Music player quit successfully")
        except pygame.error as e:
            self.logger.error(f"Error quitting player: {e}")


def get_file_paths(folder_path: str, supported_formats: List[str] = None) -> List[str]:
    """
    Get all audio file paths from a folder.
    
    Args:
        folder_path: Path to the folder
        supported_formats: List of supported file extensions (e.g., ['.mp3', '.wav'])
        
    Returns:
        List of audio file paths
    """
    if supported_formats is None:
        supported_formats = ['.mp3', '.wav', '.ogg', '.flac']
    
    file_paths = []
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        logger = get_logger()
        logger.warning(f"Music folder not found: {folder_path}")
        return file_paths
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in supported_formats):
                file_paths.append(os.path.join(root, file))
    
    return file_paths


# Emotion dictionary
emotion_dict = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

# Get current working directory
path = os.getcwd()

# Cross-platform music library paths using os.path.join
music_folders = {
    0: os.path.join(path, 'Music', 'Angry'),
    1: os.path.join(path, 'Music', 'Disgust'),
    2: os.path.join(path, 'Music', 'Fear'),
    3: os.path.join(path, 'Music', 'Happy'),
    4: os.path.join(path, 'Music', 'Sad'),
    5: os.path.join(path, 'Music', 'Surprise'),
    6: os.path.join(path, 'Music', 'Neutral')
}

# Create the music library with support for multiple audio formats
music_library = {emotion_dict[i]: get_file_paths(folder) for i, folder in music_folders.items()}

# Log music library statistics
logger = get_logger()
for emotion, tracks in music_library.items():
    logger.info(f"Loaded {len(tracks)} tracks for {emotion}")
