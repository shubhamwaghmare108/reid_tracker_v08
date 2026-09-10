"""Small image and drawing helpers shared by enrollment and the live pipeline."""
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import cv2
import numpy as np

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def image_files(folder: Path) -> Iterable[Path]:
    """Return supported image files below ``folder`` in a repeatable order."""
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob('*') if p.suffix.lower() in IMAGE_SUFFIXES)


def crop_box(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray | None:
    """Safely crop an ``(x1, y1, x2, y2)`` box, clipping it to image boundaries."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def draw_label(frame: np.ndarray, box: tuple[int, int, int, int], label: str, known: bool,
               detail: str | None = None):
    """Draw a track label and optional compact diagnostic detail on a frame."""
    x1, y1, x2, y2 = box
    color = (40, 190, 40) if known else (25, 80, 235)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    lines = [label] + ([detail] if detail else [])
    metrics = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0] for line in lines]
    width = max(width for width, _ in metrics)
    line_height = 19
    top = max(0, y1 - len(lines) * line_height - 6)
    cv2.rectangle(frame, (x1, top), (x1 + width + 8, y1), color, -1)
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (x1 + 4, top + 15 + index * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)


def draw_presence_panel(frame: np.ndarray, records, active_names: set[str]):
    """Draw the running presence totals directly onto the OpenCV preview frame."""
    if not records:
        return
    rows = sorted(records.items())
    x, y, width, row_height = 12, 12, 345, 25
    height = 38 + len(rows) * row_height
    cv2.rectangle(frame, (x, y), (x + width, y + height), (20, 20, 20), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (110, 110, 110), 1)
    cv2.putText(frame, 'PERSON', (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1)
    cv2.putText(frame, 'TIME', (x + 155, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1)
    cv2.putText(frame, 'IN', (x + 240, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1)
    cv2.putText(frame, 'OUT', (x + 290, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1)
    for index, (name, stats) in enumerate(rows, start=1):
        baseline = y + 25 + index * row_height
        color = (50, 210, 50) if name in active_names else (190, 190, 190)
        cv2.putText(frame, name[:18], (x + 10, baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)
        cv2.putText(frame, f'{stats.total_seconds:.1f}s', (x + 155, baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)
        cv2.putText(frame, str(stats.entries), (x + 240, baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)
        cv2.putText(frame, str(stats.exits), (x + 290, baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)
