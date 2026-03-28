"""
Win Prediction Model Interface
==============================
Defines the data contract between the CV pipeline and the win prediction model.

Your partner should:
  1. Subclass WinPredictionModel
  2. Implement predict(packet) -> PredictionResult
  3. Optionally override on_point_scored() and reset()

The CV pipeline calls predict() every frame (or at --predict-interval).
predict() MUST return quickly (<5 ms) to avoid blocking the video feed.

Example usage:
    from prediction_model_base import WinPredictionModel, PredictionDataPacket, PredictionResult

    class MyModel(WinPredictionModel):
        def __init__(self):
            self.model = load_my_model(...)

        def predict(self, packet):
            features = self._extract(packet)
            prob = self.model(features)
            return PredictionResult(
                player1_win_prob=prob,
                player2_win_prob=1 - prob,
                confidence=0.8,
                model_ready=True,
            )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# DATA PACKET: CV pipeline -> Prediction model
# =============================================================================

@dataclass
class BallState:
    """Ball tracking state for the current frame."""
    detected: bool = False
    position_px: Optional[tuple] = None        # (x, y) pixel centre
    position_m: Optional[tuple] = None         # (x_m, y_m) on table plane (metres)
    speed_mps: Optional[float] = None          # metres per second (smoothed)
    speed_pps: Optional[float] = None          # pixels per second (smoothed)
    trajectory_px: list = field(default_factory=list)  # last N (x,y) pixel positions


@dataclass
class ScoreState:
    """Current match score."""
    player1_score: Optional[int] = None
    player2_score: Optional[int] = None
    player1_sets: int = 0
    player2_sets: int = 0
    score_reliable: bool = True                # False when OCR is obscured


@dataclass
class RallyState:
    """State of the current rally / point."""
    is_active: bool = False                    # True when ball is confirmed in play
    rally_id: int = 0
    rally_duration_sec: float = 0.0
    mean_speed_mps: Optional[float] = None
    max_speed_mps: Optional[float] = None
    landing_zones: list = field(default_factory=lambda: [0] * 9)  # 3x3 histogram


@dataclass
class PoseState:
    """Per-player pose features. None if not computed this frame (~6 Hz)."""
    player1: Optional[dict] = None             # 22-feature dict from PoseFeatureExtractor
    player2: Optional[dict] = None


@dataclass
class PredictionDataPacket:
    """
    Complete feature packet sent to the prediction model each frame.

    Fields
    ------
    frame_idx : int
        Current frame number.
    timestamp_sec : float
        Seconds elapsed since video start.
    ball : BallState
        Ball detection and tracking data.
    score : ScoreState
        Current score and set counts.
    rally : RallyState
        Rally-level aggregated statistics.
    pose : PoseState
        Per-player body pose features (may be None on frames where pose
        extraction did not run).
    match_elapsed_sec : float
        Total elapsed time since the match / video started.
    """
    frame_idx: int = 0
    timestamp_sec: float = 0.0
    ball: BallState = field(default_factory=BallState)
    score: ScoreState = field(default_factory=ScoreState)
    rally: RallyState = field(default_factory=RallyState)
    pose: PoseState = field(default_factory=PoseState)
    match_elapsed_sec: float = 0.0


# =============================================================================
# PREDICTION RESULT: Prediction model -> CV pipeline
# =============================================================================

@dataclass
class PredictionResult:
    """
    Returned by the prediction model each time predict() is called.

    Fields
    ------
    player1_win_prob : float
        Probability (0.0–1.0) that player 1 wins the match.
    player2_win_prob : float
        Probability (0.0–1.0) that player 2 wins (should equal 1 - player1_win_prob).
    confidence : float
        Model's confidence in this prediction (0.0–1.0). Used for display only.
    model_ready : bool
        False until the model has seen enough data to make meaningful predictions.
        The overlay will show "Warming up…" while this is False.
    """
    player1_win_prob: float = 0.5
    player2_win_prob: float = 0.5
    confidence: float = 0.0
    model_ready: bool = False


# =============================================================================
# ABSTRACT BASE CLASS: Partner implements this
# =============================================================================

class WinPredictionModel(ABC):
    """
    Abstract interface for the win prediction model.

    Subclass this and implement predict(). The CV pipeline will:
      - Call predict(packet) every frame (or every --predict-interval frames)
      - Call on_point_scored(packet) when a stable score change is detected
      - Call reset() when the user starts a new session
    """

    @abstractmethod
    def predict(self, packet: PredictionDataPacket) -> PredictionResult:
        """
        Produce a win probability prediction from the current CV state.

        MUST return within ~5 ms to avoid dropping frames.
        If the model needs more time, run inference in a background thread
        and return the most recent cached result here.
        """
        ...

    def on_point_scored(self, packet: PredictionDataPacket) -> None:
        """
        Optional hook called when a point is confirmed (stable score change).
        Good place to trigger model weight updates or log training data.
        """
        pass

    def reset(self) -> None:
        """Called when a new match / session starts. Reset internal state."""
        pass


# =============================================================================
# DUMMY MODEL: 50/50 baseline (ships with repo so pipeline runs standalone)
# =============================================================================

class DummyPredictionModel(WinPredictionModel):
    """
    Baseline model that always returns 50/50. Use for testing the pipeline
    without a trained prediction model.
    """

    def predict(self, packet: PredictionDataPacket) -> PredictionResult:
        return PredictionResult(
            player1_win_prob=0.5,
            player2_win_prob=0.5,
            confidence=0.0,
            model_ready=False,
        )
