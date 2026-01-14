"""
Emotion Smoother Module
Prevents rapid emotion changes by implementing confidence-based smoothing.
"""

import time
from collections import deque, Counter
from typing import Optional, Dict, List, Tuple

class EmotionSmoother:
    """
    Smooths emotion detection to prevent rapid jumping between emotions.
    Uses a rolling window and confidence threshold to validate emotion changes.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.6,
        minimum_duration: float = 5.0,
        window_size: int = 15
    ):
        """
        Initialize the emotion smoother.
        
        Args:
            confidence_threshold: Minimum confidence (0-1) required to change emotion
            minimum_duration: Minimum seconds an emotion must persist before changing
            window_size: Number of recent detections to consider for confidence
        """
        self.confidence_threshold = confidence_threshold
        self.minimum_duration = minimum_duration
        self.window_size = window_size
        
        # Rolling window of recent emotion detections
        self.detection_window = deque(maxlen=window_size)
        
        # Current stable emotion
        self.current_emotion: Optional[int] = None
        
        # Time when current emotion was established
        self.emotion_start_time: Optional[float] = None
        
        # History of emotion changes for analytics
        self.emotion_history: List[Tuple[int, float]] = []
    
    def add_detection(self, emotion: int) -> Optional[int]:
        """
        Add a new emotion detection and determine if emotion should change.
        
        Args:
            emotion: Detected emotion index
            
        Returns:
            Stable emotion index if validated, None if no change
        """
        current_time = time.time()
        
        # Add to rolling window
        self.detection_window.append(emotion)
        
        # If this is the first detection, set it immediately
        if self.current_emotion is None:
            self.current_emotion = emotion
            self.emotion_start_time = current_time
            self.emotion_history.append((emotion, current_time))
            return emotion
        
        # Check if enough time has passed since last emotion change
        time_since_change = current_time - self.emotion_start_time
        if time_since_change < self.minimum_duration:
            # Not enough time has passed, keep current emotion
            return None
        
        # Calculate confidence for the new emotion
        confidence = self._calculate_confidence(emotion)
        
        # Check if confidence threshold is met and emotion is different
        if confidence >= self.confidence_threshold and emotion != self.current_emotion:
            # Change to new emotion
            self.current_emotion = emotion
            self.emotion_start_time = current_time
            self.emotion_history.append((emotion, current_time))
            return emotion
        
        # No change
        return None
    
    def _calculate_confidence(self, emotion: int) -> float:
        """
        Calculate confidence score for an emotion based on recent detections.
        
        Args:
            emotion: Emotion to calculate confidence for
            
        Returns:
            Confidence score (0-1)
        """
        if not self.detection_window:
            return 0.0
        
        # Count occurrences of the emotion in the window
        emotion_count = sum(1 for e in self.detection_window if e == emotion)
        
        # Calculate confidence as percentage of window
        confidence = emotion_count / len(self.detection_window)
        
        return confidence
    
    def get_current_emotion(self) -> Optional[int]:
        """
        Get the current stable emotion.
        
        Returns:
            Current emotion index or None
        """
        return self.current_emotion
    
    def get_confidence(self, emotion: Optional[int] = None) -> float:
        """
        Get confidence score for an emotion.
        
        Args:
            emotion: Emotion to get confidence for (current emotion if None)
            
        Returns:
            Confidence score (0-1)
        """
        if emotion is None:
            emotion = self.current_emotion
        
        if emotion is None:
            return 0.0
        
        return self._calculate_confidence(emotion)
    
    def get_window_distribution(self) -> Dict[int, int]:
        """
        Get distribution of emotions in the current window.
        
        Returns:
            Dictionary mapping emotion indices to counts
        """
        if not self.detection_window:
            return {}
        
        return dict(Counter(self.detection_window))
    
    def get_time_in_current_emotion(self) -> float:
        """
        Get time spent in current emotion.
        
        Returns:
            Seconds in current emotion
        """
        if self.emotion_start_time is None:
            return 0.0
        
        return time.time() - self.emotion_start_time
    
    def reset(self) -> None:
        """Reset the smoother to initial state."""
        self.detection_window.clear()
        self.current_emotion = None
        self.emotion_start_time = None
    
    def get_history(self) -> List[Tuple[int, float]]:
        """
        Get emotion change history.
        
        Returns:
            List of (emotion, timestamp) tuples
        """
        return self.emotion_history.copy()
    
    def update_parameters(
        self,
        confidence_threshold: Optional[float] = None,
        minimum_duration: Optional[float] = None,
        window_size: Optional[int] = None
    ) -> None:
        """
        Update smoother parameters at runtime.
        
        Args:
            confidence_threshold: New confidence threshold
            minimum_duration: New minimum duration
            window_size: New window size
        """
        if confidence_threshold is not None:
            self.confidence_threshold = max(0.0, min(1.0, confidence_threshold))
        
        if minimum_duration is not None:
            self.minimum_duration = max(0.0, minimum_duration)
        
        if window_size is not None and window_size > 0:
            # Create new deque with new size, preserving recent data
            old_data = list(self.detection_window)
            self.window_size = window_size
            self.detection_window = deque(old_data[-window_size:], maxlen=window_size)
