"""Runtime configuration for the person re-identification pipeline."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / '.env')


def _setting(name: str, default: str) -> str:
    """Read a setting from environment, Streamlit secrets, or a default."""
    value = os.getenv(name)
    if value is not None:
        return value
    try:
        import streamlit as st
        if name in st.secrets:
            return str(st.secrets[name])
        mysql_secrets = st.secrets.get('mysql', {})
        if name in mysql_secrets:
            return str(mysql_secrets[name])
    except Exception:
        pass
    return default


def database_timestamp() -> datetime:
    """Return the current UTC time as a naive datetime for MySQL DATETIME."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _resolve_device(device: str) -> str:
    """Resolve the project-level ``auto`` device to a PyTorch device string."""
    normalized = str(device or 'auto').strip().lower()
    if normalized != 'auto':
        return normalized
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        return 'cpu'


@dataclass(frozen=True)
class TrackingConfig:
    """Settings controlling frame scheduling and expensive inference."""
    detection_interval: int = 1
    reid_interval: int = 5
    reid_batch_size: int = 4
    motion_gate: float = 4.0
    iou_threshold: float = 0.30
    reid_threshold: float = 0.72
    reid_padding: float = 0.05
    use_segmentation_for_reid: bool = True
    async_database: bool = True
    drop_old_frames: bool = True
    live_mode: bool = True
    offline_mode: bool = False


@dataclass
class Settings:
    """Complete runtime configuration for detection, tracking, and storage."""
    project_root: Path = Path(_setting('REID_PROJECT_ROOT', str(PROJECT_ROOT)))
    detector_model: str | Path = Path('Models') / 'best.pt'
    reid_model: str = 'osnet_x1_0'
    reid_weights: str | Path = Path('Models') / 'osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth'
    device: str = _setting('REID_DEVICE', 'auto')
    confidence: float = 0.1
    match_threshold: float = 0.55
    known_retention_threshold: float = float(_setting('REID_KNOWN_RETENTION_THRESHOLD', '0.45'))
    gallery_dir: Path | None = None
    output_dir: Path | None = None
    tracking_config: TrackingConfig = field(default_factory=TrackingConfig)

    hybrid_weight_body: float = 0.5
    hybrid_weight_face: float = 0.5

    tracker_max_lost_frames: int = int(_setting('REID_TRACKER_MAX_LOST_FRAMES', '8'))
    tracker_max_occlusion_frames: int = int(_setting('REID_TRACKER_MAX_OCCLUSION_FRAMES', '6'))
    tracker_max_recovery_frames: int = int(_setting('REID_TRACKER_MAX_RECOVERY_FRAMES', '24'))
    tracker_max_trajectory_history: int = int(_setting('REID_TRACKER_MAX_TRAJECTORY_HISTORY', '30'))
    tracker_max_body_gallery: int = int(_setting('REID_TRACKER_MAX_BODY_GALLERY', '12'))
    tracker_max_face_gallery: int = int(_setting('REID_TRACKER_MAX_FACE_GALLERY', '6'))
    tracker_occlusion_iou_threshold: float = float(_setting('REID_TRACKER_OCCLUSION_IOU_THRESHOLD', '0.15'))
    tracker_recovery_reid_threshold: float = float(_setting('REID_TRACKER_RECOVERY_REID_THRESHOLD', '0.60'))
    tracker_motion_gate_threshold: float = float(_setting('REID_TRACKER_MOTION_GATE_THRESHOLD', '4.0'))
    tracker_turn_threshold: float = float(_setting('REID_TRACKER_TURN_THRESHOLD', '0.75'))
    tracker_motion_confidence_threshold: float = float(_setting('REID_TRACKER_MOTION_CONFIDENCE_THRESHOLD', '0.45'))
    tracker_gallery_update_threshold: float = float(_setting('REID_TRACKER_GALLERY_UPDATE_THRESHOLD', '0.72'))
    tracker_normal_reid_gate: float = float(_setting('REID_TRACKER_NORMAL_REID_GATE', '0.30'))
    tracker_strong_reid_gate: float = float(_setting('REID_TRACKER_STRONG_REID_GATE', '0.55'))
    tracker_normal_iou_gate: float = float(_setting('REID_TRACKER_NORMAL_IOU_GATE', '0.03'))
    tracker_strong_reid_iou_bypass: float = float(_setting('REID_TRACKER_STRONG_REID_IOU_BYPASS', '0.70'))
    tracker_low_motion_reid_gate: float = float(_setting('REID_TRACKER_LOW_MOTION_REID_GATE', '0.45'))
    tracker_weak_motion_distance: float = float(_setting('REID_TRACKER_WEAK_MOTION_DISTANCE', '1.0'))
    tracker_prediction_velocity_decay: float = float(_setting('REID_TRACKER_PREDICTION_VELOCITY_DECAY', '0.70'))
    tracker_max_prediction_displacement_per_frame: float = float(_setting('REID_TRACKER_MAX_PREDICTION_DISPLACEMENT_PER_FRAME', '80.0'))
    tracker_max_total_prediction_displacement: float = float(_setting('REID_TRACKER_MAX_TOTAL_PREDICTION_DISPLACEMENT', '240.0'))
    tracker_prediction_uncertainty_growth: float = float(_setting('REID_TRACKER_PREDICTION_UNCERTAINTY_GROWTH', '0.25'))
    tracker_face_person_min_match_score: float = float(_setting('REID_TRACKER_FACE_PERSON_MIN_MATCH_SCORE', '0.35'))
    tracker_face_person_ambiguity_margin: float = float(_setting('REID_TRACKER_FACE_PERSON_AMBIGUITY_MARGIN', '0.10'))
    tracker_recovery_motion_gate: float = float(_setting('REID_TRACKER_RECOVERY_MOTION_GATE', '6.0'))
    tracker_detection_dedup_iou: float = float(_setting('REID_TRACKER_DETECTION_DEDUP_IOU', '0.75'))
    tracker_min_detection_confidence: float = float(_setting('REID_TRACKER_MIN_DETECTION_CONFIDENCE', '0.35'))
    tracker_min_detection_width: int = int(_setting('REID_TRACKER_MIN_DETECTION_WIDTH', '20'))
    tracker_min_detection_height: int = int(_setting('REID_TRACKER_MIN_DETECTION_HEIGHT', '40'))
    tracker_min_detection_area: int = int(_setting('REID_TRACKER_MIN_DETECTION_AREA', '1200'))
    tracker_duplicate_track_iou: float = float(_setting('REID_TRACKER_DUPLICATE_TRACK_IOU', '0.70'))
    tracker_duplicate_appearance_threshold: float = float(_setting('REID_TRACKER_DUPLICATE_APPEARANCE_THRESHOLD', '0.80'))
    tracker_max_visual_occlusion_frames: int = int(_setting('REID_TRACKER_MAX_VISUAL_OCCLUSION_FRAMES', '6'))
    tracker_identity_face_threshold: float = float(_setting('REID_TRACKER_IDENTITY_FACE_THRESHOLD', '0.72'))
    tracker_identity_body_candidate_threshold: float = float(_setting('REID_TRACKER_IDENTITY_BODY_CANDIDATE_THRESHOLD', '0.72'))
    tracker_identity_body_confirm_threshold: float = float(_setting('REID_TRACKER_IDENTITY_BODY_CONFIRM_THRESHOLD', '0.78'))
    tracker_identity_margin: float = float(_setting('REID_TRACKER_IDENTITY_MARGIN', '0.06'))
    tracker_identity_candidate_min_frames: int = int(_setting('REID_TRACKER_IDENTITY_CANDIDATE_MIN_FRAMES', '3'))
    tracker_identity_retention_frames: int = int(_setting('REID_TRACKER_IDENTITY_RETENTION_FRAMES', '60'))
    tracker_identity_lock_timeout: int = int(_setting('REID_TRACKER_IDENTITY_LOCK_TIMEOUT', '60'))
    tracker_recovery_margin: float = float(_setting('REID_TRACKER_RECOVERY_MARGIN', '0.06'))
    tracker_association_min_iou: float = float(_setting('REID_TRACKER_ASSOCIATION_MIN_IOU', '0.01'))
    tracker_association_max_scale_change: float = float(_setting('REID_TRACKER_ASSOCIATION_MAX_SCALE_CHANGE', '2.50'))
    tracker_gallery_min_detection_confidence: float = float(_setting('REID_TRACKER_GALLERY_MIN_DETECTION_CONFIDENCE', '0.70'))

    metrics_enabled: bool = _setting('METRICS_ENABLED', 'true').lower() == 'true'
    metrics_event_logging: bool = _setting('METRICS_EVENT_LOGGING', 'true').lower() == 'true'
    metrics_output_dir: Path = Path(_setting('METRICS_OUTPUT_DIR', 'results'))
    metrics_log_detections: bool = _setting('METRICS_LOG_DETECTIONS', 'false').lower() == 'true'
    metrics_flush_size: int = int(_setting('METRICS_FLUSH_SIZE', '500'))

    face_model: str = 'buffalo_l'
    face_det_size: tuple[int, int] = (640, 640)
    face_det_threshold: float = 0.5

    mysql_host: str = _setting('REID_MYSQL_HOST', '')
    mysql_port: int = int(_setting('REID_MYSQL_PORT', '3306'))
    mysql_user: str = _setting('REID_MYSQL_USER', '')
    mysql_password: str = _setting('REID_MYSQL_PASSWORD', '')
    mysql_database: str = _setting('REID_MYSQL_DATABASE', '')

    def __post_init__(self) -> None:
        """Normalize derived paths and nested configuration."""
        self.device = _resolve_device(self.device)
        self.gallery_dir = self.gallery_dir or self.project_root / 'known_people'
        self.output_dir = self.output_dir or self.project_root / 'output'
        detector_path = Path(self.detector_model)
        weights_path = Path(self.reid_weights)
        if not detector_path.is_absolute(): self.detector_model = str(self.project_root / detector_path)
        if not weights_path.is_absolute(): self.reid_weights = str(self.project_root / weights_path)
        if not self.metrics_output_dir.is_absolute(): self.metrics_output_dir = self.project_root / self.metrics_output_dir
        if not isinstance(self.tracking_config, TrackingConfig): self.tracking_config = TrackingConfig(**self.tracking_config)
