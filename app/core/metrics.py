"""Low-overhead JSON/CSV audit metrics for V06 tracker runs (no database storage)."""
from __future__ import annotations

import csv
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.utils.pipeline_logging import get_pipeline_logger

logger = get_pipeline_logger()

CSV_COLUMNS = ('session_id frame timestamp event_type track_id identity track_state identity_state '
               'detection_id detection_confidence face_score body_score face_margin body_margin iou_score motion_score '
               'association_score reid_margin recovery_result rejection_reason identity_owner previous_identity '
               'current_identity previous_track_state current_track_state gallery_action presence_event '
               'face_index person_index best_face_person_score second_best_face_person_score face_person_margin '
               'face_person_result face_count inference_time_ms').split()
CSV_COLUMNS += ('best_score second_best_score association_margin recovery_margin').split()

TRACK_STATE_NAMES = {0: 'TENTATIVE', 1: 'CONFIRMED', 2: 'OCCLUDED', 3: 'RECOVERED', 4: 'LOST', 5: 'DELETED'}
IDENTITY_STATE_NAMES = {0: 'UNKNOWN', 1: 'CANDIDATE', 2: 'CONFIRMED', 3: 'RETAINED'}


class MetricsCollector:
    """Buffered, best-effort metrics collector; it never changes tracker decisions."""
    def __init__(self, enabled: bool = True, event_logging: bool = True, output_dir: Path | str = 'results',
                 log_detections: bool = False, flush_size: int = 500):
        self.enabled, self.event_logging = enabled, event_logging
        self.output_dir, self.log_detections, self.flush_size = Path(output_dir), log_detections, flush_size
        self.events: list[dict[str, Any]] = []; self.counters = Counter(); self.rejections = Counter()
        self.track_data: dict[int, dict[str, Any]] = {}; self.identity_data = defaultdict(lambda: {'track_ids': set(), 'confirmed': False, 'identity_switches': 0, 'recovery_count': 0, 'in_events': 0, 'out_events': 0})
        self.session: dict[str, Any] = {}; self.run_dir: Path | None = None; self.csv_path: Path | None = None
        self.started_perf = 0.; self.events_write_success = self.metrics_write_success = False

    def start_session(self, video_name: str, fps: float, configuration: dict[str, Any]) -> None:
        if not self.enabled: return
        stamp = datetime.now(timezone.utc); base = Path(video_name).stem or 'camera'
        root = self.output_dir / base; root.mkdir(parents=True, exist_ok=True)
        index = 1
        while (root / f'run_{index:03d}').exists(): index += 1
        self.run_dir = root / f'run_{index:03d}'; self.run_dir.mkdir()
        self.csv_path = self.run_dir / 'events.csv'
        self.session = {'session_id': f'{stamp:%Y-%m-%d}_{base}_{index:03d}', 'video_name': base,
                        'started_at': stamp.isoformat(), 'ended_at': None, 'duration_seconds': 0., 'fps': fps,
                        'total_frames': 0, 'tracker_version': 'V06'}
        self.configuration = configuration; self.started_perf = time.perf_counter()
        if self.event_logging:
            try:
                with self.csv_path.open('w', newline='', encoding='utf-8') as handle: csv.DictWriter(handle, fieldnames=CSV_COLUMNS).writeheader()
                self.events_write_success = True
            except OSError: logger.exception('Metrics CSV initialization failed.')

    def record(self, event_type: str, frame: int, timestamp: float = 0., track=None, **values: Any) -> None:
        if not self.enabled: return
        increment = values.pop('count', 1)
        self.counters[event_type] += increment
        if event_type == 'FACE_INFERENCE':
            self.counters['FACE_INFERENCE_TIME_MS'] += float(values.get('inference_time_ms', 0.))
        track_id = values.pop('track_id', getattr(track, 'tracker_id', ''))
        identity = values.pop('identity', getattr(track, 'name', ''))
        if track_id != '':
            data = self.track_data.setdefault(int(track_id), {'track_id': int(track_id), 'first_frame': frame, 'last_frame': frame, 'observations': 0, 'missed_frames': 0, 'occlusion_count': 0, 'occlusion_frames': 0, 'recovery_attempts': 0, 'successful_recoveries': 0, 'failed_recoveries': 0, 'ambiguous_recoveries': 0, 'identity_changes': 0, 'gallery_update_attempts': 0, 'gallery_updates_accepted': 0, 'gallery_updates_rejected': 0})
            data['last_frame'] = frame; data['missed_frames'] = max(data['missed_frames'], getattr(track, 'missed_frames', 0))
            if event_type in ('TRACK_CREATED', 'TRACK_MATCHED', 'TRACK_RECOVERED'): data['observations'] += 1
            if event_type == 'TRACK_OCCLUDED': data['occlusion_count'] += 1
            if event_type == 'RECOVERY_ATTEMPT': data['recovery_attempts'] += 1
            if event_type == 'RECOVERY_SUCCESS': data['successful_recoveries'] += 1
            if event_type == 'RECOVERY_FAILED': data['failed_recoveries'] += 1
            if event_type == 'RECOVERY_AMBIGUOUS': data['ambiguous_recoveries'] += 1
            if event_type == 'IDENTITY_SWITCH': data['identity_changes'] += 1
            if event_type.startswith('GALLERY_UPDATE_'): data['gallery_update_attempts'] += event_type == 'GALLERY_UPDATE_ATTEMPT'; data['gallery_updates_accepted'] += event_type == 'GALLERY_UPDATE_ACCEPTED'; data['gallery_updates_rejected'] += event_type == 'GALLERY_UPDATE_REJECTED'
        if event_type == 'IDENTITY_CONFIRMED' and identity and identity != 'Unknown':
            identity_metrics = self.identity_data[identity]
            if identity_metrics['confirmed'] and track_id not in identity_metrics['track_ids']:
                self.counters['TRACK_FRAGMENT'] += 1
                self.record('TRACK_FRAGMENT', frame, timestamp, track, identity=identity)
            identity_metrics['confirmed'] = True
        if identity and identity != 'Unknown': self.identity_data[identity]['track_ids'].add(track_id)
        if identity and identity != 'Unknown' and event_type in ('PRESENCE_IN', 'PRESENCE_OUT'):
            self.identity_data[identity]['in_events' if event_type == 'PRESENCE_IN' else 'out_events'] += 1
        if identity and identity != 'Unknown' and event_type == 'IDENTITY_SWITCH':
            self.identity_data[identity]['identity_switches'] += 1
        if event_type == 'ASSOCIATION_REJECTED':
            self.rejections[values.get('rejection_reason', 'NO_SAFE_MATCH')] += 1
            reason = values.get('rejection_reason', 'NO_SAFE_MATCH')
            category = {'REID_GATE_FAIL': 'association_reject_reid', 'RECOVERY_REID_GATE_FAIL': 'association_reject_reid', 'IOU_GATE_FAIL': 'association_reject_iou', 'MOTION_GATE_FAIL': 'association_reject_motion', 'SCALE_GATE_FAIL': 'association_reject_scale', 'COMBINED_GATE_FAIL': 'association_reject_combined', 'LOW_DETECTION_CONFIDENCE': 'association_reject_confidence', 'NO_SAFE_MATCH': 'association_reject_no_safe_match'}.get(reason)
            if category: self.counters[category] += 1
        if event_type == 'IDENTITY_OWNER_CONFLICT': self.rejections['IDENTITY_OWNER_CONFLICT'] += 1
        if event_type == 'GALLERY_UPDATE_REJECTED': self.rejections[values.get('rejection_reason', 'GALLERY_REJECT_OTHER')] += 1
        if not self.event_logging or (event_type == 'DETECTION' and not self.log_detections): return
        row = {key: '' for key in CSV_COLUMNS}; row.update({key: value for key, value in values.items() if key in CSV_COLUMNS})
        track_state = getattr(track, 'state', '')
        identity_state = getattr(track, 'identity_state', '')
        row.update(session_id=self.session.get('session_id', ''), frame=frame, timestamp=f'{timestamp:.3f}', event_type=event_type, track_id=track_id, identity=identity,
               track_state=TRACK_STATE_NAMES.get(track_state, track_state), identity_state=IDENTITY_STATE_NAMES.get(identity_state, identity_state))
        self.events.append(row)
        if len(self.events) >= self.flush_size: self._flush()

    def _flush(self) -> None:
        if not self.events or not self.csv_path: return
        try:
            with self.csv_path.open('a', newline='', encoding='utf-8') as handle: csv.DictWriter(handle, fieldnames=CSV_COLUMNS).writerows(self.events)
            self.events.clear(); self.events_write_success = True
        except OSError: logger.exception('Metrics CSV write failed; dropping buffered audit events.'); self.events.clear()

    def finalize(self, total_frames: int, presence, tracks: list, fps: float) -> None:
        if not self.enabled: return
        self._flush(); elapsed = time.perf_counter() - self.started_perf
        self.session.update(ended_at=datetime.now(timezone.utc).isoformat(), total_frames=total_frames, duration_seconds=total_frames / fps if fps else 0.)
        for track in tracks:
            data = self.track_data.setdefault(track.tracker_id, {'track_id': track.tracker_id, 'first_frame': 0, 'last_frame': total_frames, 'observations': 0, 'missed_frames': 0, 'occlusion_count': 0, 'occlusion_frames': 0, 'recovery_attempts': 0, 'successful_recoveries': 0, 'failed_recoveries': 0, 'ambiguous_recoveries': 0, 'identity_changes': 0, 'gallery_update_attempts': 0, 'gallery_updates_accepted': 0, 'gallery_updates_rejected': 0})
            data.update(identity=track.name, final_track_state=TRACK_STATE_NAMES.get(track.state, track.state), final_track_state_id=track.state,
                        final_identity_state=IDENTITY_STATE_NAMES.get(track.identity_state, track.identity_state), final_identity_state_id=track.identity_state,
                        missed_frames=track.missed_frames, occlusion_frames=track.occlusion_frames)
        identities = []
        for name, data in self.identity_data.items():
            data = dict(data); data['track_ids'] = sorted(int(x) for x in data['track_ids'] if x != ''); identities.append({'identity': name, **data})
        attempts = self.counters['RECOVERY_ATTEMPT']; success = self.counters['RECOVERY_SUCCESS']; failed = self.counters['RECOVERY_FAILED']; ambiguous = self.counters['RECOVERY_AMBIGUOUS']
        metrics = {'detections_raw': self.counters['DETECTION_RAW'] or self.counters['DETECTION'], 'detections_after_confidence_filter': self.counters['DETECTIONS_AFTER_CONFIDENCE_FILTER'], 'detections_after_bbox_filter': self.counters['DETECTIONS_AFTER_BBOX_FILTER'], 'detections_after_deduplication': self.counters['DETECTIONS_AFTER_DEDUPLICATION'], 'total_detections_raw': self.counters['DETECTION_RAW'] or self.counters['DETECTION'], 'total_detections_valid': self.counters['DETECTIONS_AFTER_BBOX_FILTER'], 'total_detections_deduplicated': self.counters['DETECTIONS_AFTER_DEDUPLICATION'], 'total_detections': self.counters['DETECTIONS_AFTER_DEDUPLICATION'] or self.counters['DETECTION'], 'face_inference_calls': self.counters['FACE_INFERENCE'], 'face_person_match_attempts': self.counters['FACE_PERSON_MATCH_ATTEMPT'], 'face_person_matches': self.counters['FACE_PERSON_MATCH'], 'face_person_ambiguous': self.counters['FACE_PERSON_AMBIGUOUS'], 'face_person_low_score': self.counters['FACE_PERSON_LOW_SCORE'], 'face_person_no_candidate': self.counters['FACE_PERSON_NO_CANDIDATE'], 'face_person_person_conflict': self.counters['FACE_PERSON_PERSON_CONFLICT'], 'face_person_unassigned': self.counters['FACE_PERSON_UNASSIGNED'], 'face_inference_time_ms': self.counters['FACE_INFERENCE_TIME_MS'], 'prediction_used': self.counters['PREDICTION_USED'], 'prediction_damped': self.counters['PREDICTION_DAMPED'], 'prediction_clamped': self.counters['PREDICTION_CLAMPED'], 'prediction_hidden': self.counters['PREDICTION_HIDDEN'], 'prediction_expired': self.counters['PREDICTION_EXPIRED'], 'unique_tracker_ids': len(self.track_data), 'confirmed_identities': sum(d['confirmed'] for d in self.identity_data.values()), 'track_fragments': self.counters['TRACK_FRAGMENT'], 'fragments_by_identity': {name: max(0, len(data['track_ids']) - 1) for name, data in self.identity_data.items()}, 'id_switches': self.counters['IDENTITY_SWITCH'], 'identity_confirmations': self.counters['IDENTITY_CONFIRMED'], 'identity_retentions': self.counters['IDENTITY_RETAINED'], 'identity_owner_conflicts': self.counters['IDENTITY_OWNER_CONFLICT'], 'false_identity_claims': None, 'recovery_attempts': attempts, 'successful_recoveries': success, 'failed_recoveries': failed, 'ambiguous_recoveries': ambiguous, 'recovery_rejected': self.counters['RECOVERY_REJECTED'], 'in_events': self.counters['PRESENCE_IN'], 'out_events': self.counters['PRESENCE_OUT'], 'total_in_events': self.counters['PRESENCE_IN'], 'total_out_events': self.counters['PRESENCE_OUT'], 'false_in_events': None, 'false_out_events': None, 'gallery_update_attempts': self.counters['GALLERY_UPDATE_ATTEMPT'], 'gallery_updates_accepted': self.counters['GALLERY_UPDATE_ACCEPTED'], 'gallery_updates_rejected': self.counters['GALLERY_UPDATE_REJECTED'], 'association_attempts': self.counters['ASSOCIATION_ATTEMPT'], 'association_successes': self.counters['TRACK_MATCHED'], 'association_rejections': self.counters['ASSOCIATION_REJECTED'], 'association_reject_reid': self.counters['association_reject_reid'], 'association_reject_iou': self.counters['association_reject_iou'], 'association_reject_motion': self.counters['association_reject_motion'], 'association_reject_scale': self.counters['association_reject_scale'], 'association_reject_combined': self.counters['association_reject_combined'], 'association_reject_confidence': self.counters['association_reject_confidence'], 'association_reject_no_safe_match': self.counters['association_reject_no_safe_match'], 'processing_time_seconds': elapsed, 'processing_fps': total_frames / elapsed if elapsed else 0.}
        payload = {'session': self.session, 'metrics': metrics, 'rejection_reasons': dict(self.rejections), 'recovery': {'attempts': attempts, 'successful': success, 'failed': failed, 'ambiguous': ambiguous, 'recovery_success_rate': success / attempts if attempts else 0., 'recovery_failure_rate': failed / attempts if attempts else 0., 'recovery_ambiguity_rate': ambiguous / attempts if attempts else 0.}, 'tracks': sorted(self.track_data.values(), key=lambda x: x['track_id']), 'identities': identities, 'configuration': self.configuration, 'id_switch_ground_truth_metrics': None, 'metrics_write_success': True, 'events_write_success': self.events_write_success}
        try:
            (self.run_dir / 'metrics.json').write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8'); self.metrics_write_success = True
        except OSError: logger.exception('Metrics JSON write failed.')
        self.summary = metrics
