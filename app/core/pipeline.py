"""High-level orchestration of detection, tracking, presence accounting, and storage."""
from __future__ import annotations
import asyncio
import cv2
import queue
import threading
import time
from pathlib import Path
from app.config import Settings, database_timestamp
from app.core.detector import PersonDetector
from app.core.reid import FeatureExtractor
from app.core.face import FaceProcessor
from app.core.gallery import Gallery
from app.core.tracker import IdentityState, ReIDTracker, TrackState
from app.core.presence import PresenceTracker
from app.core.metrics import MetricsCollector
from app.storage.mysql_store import MySQLPresenceStore
from app.utils.utils import draw_label, draw_presence_panel
from app.utils.pipeline_logging import get_pipeline_logger

logger = get_pipeline_logger()


class AsyncDatabaseWriter:
    """Persist completed runs on a worker thread instead of the frame thread.

    The queue is bounded deliberately. When persistence falls behind, the
    oldest queued payload is discarded so database latency cannot accumulate
    unbounded tracking latency.
    """
    def __init__(self, store, maxsize: int = 32):
        self.store = store
        self.queue = queue.Queue(maxsize=maxsize)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def put(self, payload):
        """Queue a payload without blocking the tracking loop."""
        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            self.queue.put_nowait(payload)

    def _worker(self):
        """Consume queued run payloads until the sentinel is received."""
        while True:
            payload = self.queue.get()
            if payload is None:
                return
            try:
                self.store.save_run(**payload)
            except Exception:
                logger.exception('Async database write failed; dropping queued run.')

    def flush(self):
        """Discard queued payloads that are no longer useful to live tracking."""
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

    def stop(self):
        """Request worker shutdown and wait briefly for a clean exit."""
        self.put(None)
        self._thread.join(timeout=2.0)


class ReIDPipeline:
    """Construct and run the detector, tracker, presence, and storage pipeline.

    Models and galleries are loaded once during construction. Per-frame work is
    then limited to detection, tracking, rendering, metrics, and presence
    updates; database persistence is kept outside that critical path.
    """
    def __init__(self, settings: Settings):
        """Create every long-lived pipeline dependency from ``settings``.

        The tracker receives all identity and recovery thresholds explicitly so
        its behavior is reproducible and can be recorded with benchmark data.
        """
        self.settings = settings
        self.extractor = FeatureExtractor(
            model_name=settings.reid_model,
            model_path=settings.reid_weights,
            device=settings.device,
        )
        self.face_processor = FaceProcessor(
            model_name=settings.face_model,
            det_size=settings.face_det_size,
        )
        self.gallery = Gallery.load(
            settings.gallery_dir,
            body_extractor=self.extractor,
            face_processor=self.face_processor
        )
        self.detector = PersonDetector(
            model_name=str(settings.detector_model),
            confidence=settings.confidence,
            device=settings.device,
        )
        self.tracker = ReIDTracker(
            gallery=self.gallery,
            extractor=self.extractor,
            face_processor=self.face_processor,
            threshold=settings.match_threshold,
            w_body=settings.hybrid_weight_body,
            w_face=settings.hybrid_weight_face,
            max_lost_frames=settings.tracker_max_lost_frames,
            max_occlusion_frames=settings.tracker_max_occlusion_frames,
            max_recovery_frames=settings.tracker_max_recovery_frames,
            max_trajectory_history=settings.tracker_max_trajectory_history,
            max_body_gallery=settings.tracker_max_body_gallery,
            max_face_gallery=settings.tracker_max_face_gallery,
            occlusion_iou_threshold=settings.tracker_occlusion_iou_threshold,
            recovery_reid_threshold=settings.tracker_recovery_reid_threshold,
            motion_gate_threshold=settings.tracker_motion_gate_threshold,
            recovery_motion_gate=settings.tracker_recovery_motion_gate,
            turn_threshold=settings.tracker_turn_threshold,
            motion_confidence_threshold=settings.tracker_motion_confidence_threshold,
            gallery_update_threshold=settings.tracker_gallery_update_threshold,
            normal_reid_gate=settings.tracker_normal_reid_gate,
            strong_reid_gate=settings.tracker_strong_reid_gate,
            normal_iou_gate=settings.tracker_normal_iou_gate,
            strong_reid_iou_bypass=settings.tracker_strong_reid_iou_bypass,
            low_motion_reid_gate=settings.tracker_low_motion_reid_gate,
            weak_motion_distance=settings.tracker_weak_motion_distance,
            prediction_velocity_decay=settings.tracker_prediction_velocity_decay,
            max_prediction_displacement_per_frame=settings.tracker_max_prediction_displacement_per_frame,
            max_total_prediction_displacement=settings.tracker_max_total_prediction_displacement,
            prediction_uncertainty_growth=settings.tracker_prediction_uncertainty_growth,
            face_person_min_match_score=settings.tracker_face_person_min_match_score,
            face_person_ambiguity_margin=settings.tracker_face_person_ambiguity_margin,
            detection_dedup_iou=settings.tracker_detection_dedup_iou,
            min_detection_confidence=settings.tracker_min_detection_confidence,
            min_detection_width=settings.tracker_min_detection_width,
            min_detection_height=settings.tracker_min_detection_height,
            min_detection_area=settings.tracker_min_detection_area,
            duplicate_track_iou=settings.tracker_duplicate_track_iou,
            duplicate_appearance_threshold=settings.tracker_duplicate_appearance_threshold,
            identity_face_threshold=settings.tracker_identity_face_threshold,
            identity_body_candidate_threshold=settings.tracker_identity_body_candidate_threshold,
            identity_body_confirm_threshold=settings.tracker_identity_body_confirm_threshold,
            identity_margin=settings.tracker_identity_margin,
            identity_candidate_min_frames=settings.tracker_identity_candidate_min_frames,
            identity_retention_frames=settings.tracker_identity_retention_frames,
            identity_lock_timeout=settings.tracker_identity_lock_timeout,
            recovery_margin=settings.tracker_recovery_margin,
            association_min_iou=settings.tracker_association_min_iou,
            association_max_scale_change=settings.tracker_association_max_scale_change,
            gallery_min_detection_confidence=settings.tracker_gallery_min_detection_confidence,
            reid_interval=settings.tracking_config.reid_interval,
        )
        self.presence = PresenceTracker()
        self.metrics = MetricsCollector(settings.metrics_enabled, settings.metrics_event_logging,
                                        settings.metrics_output_dir, settings.metrics_log_detections,
                                        settings.metrics_flush_size)
        self.tracker.metrics = self.metrics
        self._metrics_presence_events = 0
        self.store = None
        self.async_writer = None
        self.perf_samples: list[dict[str, float]] = []
        if settings.mysql_host:
            self.store = MySQLPresenceStore(
                host=settings.mysql_host,
                port=settings.mysql_port,
                user=settings.mysql_user,
                password=settings.mysql_password,
                database=settings.mysql_database,
            )
            if settings.tracking_config.async_database:
                self.async_writer = AsyncDatabaseWriter(self.store, maxsize=8)
            logger.info('Pipeline initialized: device=%s, database=%s', settings.device, bool(self.store))

    def benchmark_run(self, source: str | int, max_frames: int = 120, output_video: Path | None = None) -> dict[str, float]:
        """Run a bounded offline sample and aggregate stage timings.

        Latency percentiles are calculated from per-frame samples rather than
        treating all tracker time as Re-ID time. This keeps optimization claims
        tied to measured stages.
        """
        self.perf_samples.clear()
        self.run_on_video(source, save_to_db=False, output_video=output_video, max_frames=max_frames, preview=False)
        if not self.perf_samples:
            return {'fps': 0.0, 'avg_latency_ms': 0.0, 'p95_latency_ms': 0.0}
        latencies = [float(sample['total_ms']) for sample in self.perf_samples]
        latencies.sort()
        p95 = latencies[max(0, min(len(latencies) - 1, int(len(latencies) * 0.95)))]
        total_fps = 1.0 / max(sum(sample['total_ms'] for sample in self.perf_samples) / max(len(self.perf_samples), 1) / 1000.0, 1e-6)
        return {
            'fps': total_fps,
            'avg_latency_ms': sum(latencies) / max(len(latencies), 1),
            'p95_latency_ms': p95,
            'detection_ms': sum(sample.get('detection_ms', 0.0) for sample in self.perf_samples) / max(len(self.perf_samples), 1),
            'YOLO_ms': sum(sample.get('YOLO_ms', 0.0) for sample in self.perf_samples) / max(len(self.perf_samples), 1),
            'mask_transfer_ms': sum(sample.get('mask_transfer_ms', 0.0) for sample in self.perf_samples) / max(len(self.perf_samples), 1),
            'mask_resize_ms': sum(sample.get('mask_resize_ms', 0.0) for sample in self.perf_samples) / max(len(self.perf_samples), 1),
            'mask_application_ms': sum(sample.get('mask_application_ms', 0.0) for sample in self.perf_samples) / max(len(self.perf_samples), 1),
            'crop_ms': sum(sample.get('crop_ms', 0.0) for sample in self.perf_samples) / max(len(self.perf_samples), 1),
            'face_detection_ms': sum(sample.get('face_detection_ms', 0.0) for sample in self.perf_samples) / max(len(self.perf_samples), 1),
            'reid_preprocessing_ms': sum(sample.get('reid_preprocessing_ms', 0.0) for sample in self.perf_samples) / max(len(self.perf_samples), 1),
            'reid_inference_ms': sum(sample.get('reid_inference_ms', 0.0) for sample in self.perf_samples) / max(len(self.perf_samples), 1),
            'reid_calls_per_frame': sum(sample.get('reid_calls', 0.0) for sample in self.perf_samples) / max(len(self.perf_samples), 1),
            'reid_crops_per_frame': sum(sample.get('reid_crops', 0.0) for sample in self.perf_samples) / max(len(self.perf_samples), 1),
            'face_calls_per_frame': sum(sample.get('face_calls', 0.0) for sample in self.perf_samples) / max(len(self.perf_samples), 1),
            'association_ms': sum(sample.get('association_ms', 0.0) for sample in self.perf_samples) / max(len(self.perf_samples), 1),
            'database_ms': sum(sample.get('database_ms', 0.0) for sample in self.perf_samples) / max(len(self.perf_samples), 1),
            'render_ms': sum(sample.get('render_ms', 0.0) for sample in self.perf_samples) / max(len(self.perf_samples), 1),
        }

    def run_on_video(self, source: str | int, save_to_db: bool = True,
                     output_video: Path | None = None, max_frames: int = -1,
                     preview: bool = True, live_mode: bool | None = None) -> None:
        """Process a camera, file, or stream until completion or user stop.

        Integer camera sources use a one-slot producer/consumer buffer that
        drops stale frames. File sources remain sequential so offline processing
        does not skip frames and benchmark results remain deterministic.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error('Unable to open source: %s', source)
            source_path = Path(source) if isinstance(source, (str, Path)) else None
            if source_path is not None and source_path.is_file() and source_path.suffix.lower() == '.mp4':
                try:
                    has_moov = b'moov' in source_path.read_bytes()
                except OSError:
                    has_moov = True
                if not has_moov:
                    raise RuntimeError(
                        f'Cannot decode source: {source}. The MP4 is incomplete and missing its moov atom; '
                        're-export or repair the recording before running the pipeline.'
                    )
            raise RuntimeError(f'Cannot open source: {source}')
        logger.info('Pipeline started: source=%s, save_to_db=%s, output=%s', source, save_to_db, output_video)
        started_at = database_timestamp()
        prev_time = database_timestamp()
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        live_mode = (self.settings.tracking_config.live_mode and isinstance(source, int)
                     if live_mode is None else live_mode)
        # A single-slot queue bounds live latency: the worker always receives
        # the newest available frame instead of processing an ever-growing backlog.
        capture_queue = queue.Queue(maxsize=1)
        capture_stop = threading.Event()
        capture_done = threading.Event()
        capture_thread = None
        if live_mode:
            def capture_latest():
                """Continuously capture frames and replace stale queued data."""
                try:
                    while not capture_stop.is_set():
                        captured, captured_frame = cap.read()
                        if not captured:
                            break
                        item = (time.perf_counter(), captured_frame)
                        try:
                            capture_queue.put_nowait(item)
                        except queue.Full:
                            try:
                                capture_queue.get_nowait()
                            except queue.Empty:
                                pass
                            capture_queue.put_nowait(item)
                finally:
                    capture_done.set()
            capture_thread = threading.Thread(target=capture_latest, name='camera-capture', daemon=True)
            capture_thread.start()
        self.metrics.start_session(str(source), fps, self._metrics_configuration())
        out = None
        processing_failed = False
        latest_frame = None
        last_processed_frame = 0
        try:
            if output_video:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(output_video), fourcc, fps,
                                      (int(cap.get(3)), int(cap.get(4))))
                if not out.isOpened():
                    logger.error('Unable to create output video: %s', output_video)
                    raise RuntimeError(f'Cannot create output video: {output_video}')
            while True:
                if live_mode:
                    try:
                        captured_at, latest_frame = capture_queue.get(timeout=.5)
                    except queue.Empty:
                        if capture_done.is_set(): break
                        continue
                    capture_ms = (time.perf_counter() - captured_at) * 1000.0
                else:
                    ret, frame = cap.read()
                    if not ret: break
                    latest_frame = frame
                    capture_ms = 0.0
                if max_frames > 0 and frame_count >= max_frames:
                    break
                frame_count += 1
                if max_frames > 0 and frame_count > max_frames:
                    break
                current_time = database_timestamp()
                elapsed = (current_time - prev_time).total_seconds()
                prev_time = current_time
                start_total = time.perf_counter()
                detection_started = time.perf_counter()
                detections = self.detector.detect(latest_frame)
                detection_ms = (time.perf_counter() - detection_started) * 1000.0
                for detection in detections:
                    self.metrics.record('DETECTION', frame_count, frame_count / fps,
                                        detection_confidence=float(getattr(detection, 'confidence', 0.)))
                reid_started = time.perf_counter()
                tracks = self.tracker.update(latest_frame, detections)
                reid_ms = (time.perf_counter() - reid_started) * 1000.0
                detector_profile = getattr(self.detector, 'last_profile', {})
                tracker_profile = getattr(self.tracker, 'last_profile', {})
                render_started = time.perf_counter()
                # Only confirmed gallery matches contribute to attendance
                # statistics. Predicted or stale tracks must not extend a
                # person's presence interval after they leave the scene.
                # Predicted/stale tracks are not currently visible; counting them here
                # would keep a departed person's timer running until track expiry.
                visible_names = {
                    t.name for t in tracks
                    if t.name != 'Unknown' and (
                        (t.time_since_update == 0 and t.state in (TrackState.CONFIRMED, TrackState.RECOVERED))
                        or (t.state == TrackState.OCCLUDED and t.missed_frames <= self.settings.tracker_max_occlusion_frames)
                    )
                }
                self.presence.update(visible_names, elapsed, current_time)
                # Presence remains unchanged; observe only newly emitted events.
                while getattr(self, '_metrics_presence_events', 0) < len(self.presence.events):
                    event = self.presence.events[self._metrics_presence_events]; self._metrics_presence_events += 1
                    self.metrics.record(f'PRESENCE_{event.event_type}', frame_count, frame_count / fps, identity=event.person_name, presence_event=event.event_type)
                state_names = {
                    TrackState.TENTATIVE: 'TENTATIVE', TrackState.CONFIRMED: 'CONFIRMED',
                    TrackState.OCCLUDED: 'OCCLUDED', TrackState.RECOVERED: 'RECOVERED',
                    TrackState.LOST: 'LOST', TrackState.DELETED: 'DELETED',
                }
                identity_names = {
                    IdentityState.UNKNOWN: 'UNKNOWN', IdentityState.CANDIDATE: 'CANDIDATE',
                    IdentityState.CONFIRMED: 'IDENTITY', IdentityState.RETAINED: 'RETAINED',
                }
                for track in tracks:
                    # A predicted box is useful visual feedback during an
                    # occlusion, but stale normal/lost tracks should not be
                    # drawn as currently visible.
                    if track.state == TrackState.LOST or (track.time_since_update != 0 and (track.state != TrackState.OCCLUDED or track.missed_frames > self.settings.tracker_max_visual_occlusion_frames or not track.prediction_visible)):
                        continue
                    known = track.state in (TrackState.CONFIRMED, TrackState.RECOVERED) and track.name != 'Unknown'
                    label = track.name if known else f'Person {track.tracker_id}'
                    state = state_names[track.state]
                    detail = f'ID {track.tracker_id} | {state} | conf {track.association_confidence:.2f}'
                    if track.identity_state != IdentityState.UNKNOWN:
                        detail += f' | {identity_names[track.identity_state]}:{track.identity_source or track.identity_candidate}'
                    if track.state == TrackState.OCCLUDED and track.occluded_by is not None:
                        detail += f' | by {track.occluded_by}'
                    draw_label(latest_frame, tuple(track.bbox.astype(int)), label, known, detail)
                draw_presence_panel(latest_frame, self.presence.records, self.presence.active_names)
                render_ms = (time.perf_counter() - render_started) * 1000.0
                if out is not None:
                    out.write(latest_frame)

                if preview:
                    cv2.imshow("ReID Tracking", latest_frame)
                total_ms = (time.perf_counter() - start_total) * 1000.0
                self.perf_samples.append({
                    'capture_ms': capture_ms,
                    'detection_ms': detection_ms,
                    'YOLO_ms': detector_profile.get('yolo_ms', detection_ms),
                    'mask_transfer_ms': detector_profile.get('mask_transfer_ms', 0.0),
                    'mask_resize_ms': detector_profile.get('mask_resize_ms', 0.0),
                    'mask_application_ms': tracker_profile.get('reid_preprocessing_ms', 0.0),
                    'crop_ms': tracker_profile.get('reid_preprocessing_ms', 0.0),
                    'face_detection_ms': tracker_profile.get('face_detection_ms', 0.0),
                    'reid_preprocessing_ms': tracker_profile.get('reid_preprocessing_ms', 0.0),
                    'reid_inference_ms': tracker_profile.get('reid_inference_ms', 0.0),
                    'reid_ms': reid_ms,
                    'association_ms': tracker_profile.get('association_ms', 0.0),
                    'identity_ms': tracker_profile.get('identity_ms', 0.0),
                    'database_ms': 0.0,
                    'render_ms': render_ms,
                    'total_ms': total_ms,
                    'fps': 1000.0 / max(total_ms, 1e-6),
                    'reid_calls': tracker_profile.get('reid_calls', 0),
                    'reid_crops': tracker_profile.get('reid_crops', 0),
                    'face_calls': tracker_profile.get('face_calls', 0),
                    'face_detections': tracker_profile.get('face_detections', 0),
                })
                if preview and cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info('User requested pipeline stop.')
                    break
        except Exception:
            processing_failed = True
            logger.exception('Pipeline failed: source=%s, frames=%s', source, frame_count)
            raise
        finally:
            # Always release capture, output, worker threads, and database
            # resources, including when model inference raises an exception.
            capture_stop.set()
            if capture_thread is not None:
                capture_thread.join(timeout=2.0)
            cap.release()
            if out is not None:
                out.release()
            cv2.destroyAllWindows()
            completed_at = database_timestamp()
            self.presence.finalize(completed_at)
            while getattr(self, '_metrics_presence_events', 0) < len(self.presence.events):
                event = self.presence.events[self._metrics_presence_events]; self._metrics_presence_events += 1
                self.metrics.record(f'PRESENCE_{event.event_type}', frame_count, frame_count / fps, identity=event.person_name, presence_event=event.event_type)
            self.tracker.cleanup(final=True)
            self.metrics.finalize(frame_count, self.presence, self.tracker.tracks, fps)
            if self.metrics.enabled:
                summary = self.metrics.summary
                print('\nV06 TRACKING METRICS\n'
                      f'Video: {source}\nFrames: {frame_count}\nFPS: {fps:.2f}\n'
                      f"Total detections: {summary['total_detections']}\nUnique tracker IDs: {summary['unique_tracker_ids']}\n"
                      f"Confirmed identities: {summary['confirmed_identities']}\nTrack fragments: {summary['track_fragments']}\n"
                      f"ID switches: {summary['id_switches']}\nFalse identity claims: {summary['false_identity_claims']}\n"
                      f"Recovery attempts: {summary['recovery_attempts']}\nSuccessful recoveries: {summary['successful_recoveries']}\n"
                      f"Failed recoveries: {summary['failed_recoveries']}\nAmbiguous recoveries: {summary['ambiguous_recoveries']}\n"
                      f"IN events: {summary['in_events']}\nOUT events: {summary['out_events']}\n"
                      f"Gallery updates accepted/rejected: {summary['gallery_updates_accepted']}/{summary['gallery_updates_rejected']}\n"
                      f"Association rejections: {summary['association_rejections']}\nProcessing FPS: {summary['processing_fps']:.2f}\n"
                      f'Metrics JSON: {self.metrics.run_dir / "metrics.json"}\nEvents CSV: {self.metrics.csv_path}')
            if self.store and save_to_db:
                payload = dict(
                    source=str(source),
                    started_at=started_at,
                    completed_at=completed_at,
                    records=self.presence.records,
                    events=self.presence.events,
                )
                try:
                    if self.async_writer is not None:
                        self.async_writer.put(payload)
                    else:
                        self.store.save_run(**payload)
                    logger.info('Run queued for persistence: source=%s, frames=%s, people=%s, events=%s',
                                source, frame_count, len(self.presence.records), len(self.presence.events))
                except Exception as error:
                    logger.exception('Database save failed for source=%s', source)
                    if not processing_failed:
                        raise RuntimeError('Pipeline finished, but the run could not be saved to MySQL.') from error
            if self.store:
                try:
                    if self.async_writer is not None:
                        self.async_writer.stop()
                    self.store.close()
                except Exception:
                    logger.exception('Failed to close the database connection.')
            logger.info('Pipeline finished: source=%s, frames=%s, failed=%s', source, frame_count, processing_failed)

    def _metrics_configuration(self) -> dict[str, object]:
        """Non-secret settings needed to reproduce a metrics run."""
        keys = ('tracker_motion_gate_threshold', 'tracker_association_min_iou', 'tracker_association_max_scale_change',
                'tracker_recovery_reid_threshold', 'tracker_recovery_margin', 'tracker_identity_lock_timeout',
            'tracker_gallery_update_threshold', 'tracker_identity_candidate_min_frames', 'tracker_max_occlusion_frames',
            'tracker_normal_reid_gate', 'tracker_strong_reid_gate', 'tracker_normal_iou_gate',
                'tracker_strong_reid_iou_bypass', 'tracker_face_person_min_match_score',
                'tracker_face_person_ambiguity_margin', 'tracker_prediction_velocity_decay',
                'tracker_max_prediction_displacement_per_frame', 'tracker_max_total_prediction_displacement',
                'tracker_prediction_uncertainty_growth')
        return {key: getattr(self.settings, key) for key in keys}
