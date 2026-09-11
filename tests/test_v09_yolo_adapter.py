from app.core.detector import Detection


def test_detection_keeps_yolo_track_id():
    detection = Detection((10, 20, 50, 100), 0.9, None, 17)
    assert detection.yolo_track_id == 17


def test_detection_supports_uninitialized_yolo_id():
    detection = Detection((10, 20, 50, 100), 0.9)
    assert detection.yolo_track_id is None
