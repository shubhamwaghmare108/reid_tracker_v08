"""Face detection and face-embedding extraction using InsightFace."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2
from insightface.app import FaceAnalysis


@dataclass
class FaceDetection:
    """Frame-level face observation passed to the person association stage."""
    bbox: np.ndarray
    embedding: np.ndarray
    confidence: float = 1.0
    identity: str | None = None
    similarity: float | None = None

class FaceProcessor:
    """Choose an available ONNX runtime provider and produce normalized face vectors."""
    def __init__(self, model_name: str = 'buffalo_l', det_size: tuple[int, int] = (640, 640),
                 providers: list[str] | None = None):
        self.model_name = model_name
        self.det_size = det_size
        self.providers = providers
        self.app = None

    def _ensure_loaded(self):
        """Load the InsightFace model only when the first face operation actually needs it."""
        if self.app is not None:
            return
        providers = self.providers
        if providers is None:
            import onnxruntime
            available = onnxruntime.get_available_providers()
            providers = [
                provider for provider in ('CUDAExecutionProvider', 'CPUExecutionProvider')
                if provider in available
            ]
            if not providers:
                providers = available
        self.app = FaceAnalysis(name=self.model_name, providers=providers)
        self.app.prepare(ctx_id=0 if 'CUDAExecutionProvider' in providers else -1, det_size=self.det_size)

    def extract(self, image: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Detect faces and return the largest face's box and unit-length embedding."""
        self._ensure_loaded()
        faces = self._faces(image)
        if not faces:
            return None, None
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        bbox = face.bbox.astype(int)
        # Unit vectors make a dot product equivalent to cosine similarity.
        emb = face.embedding / np.linalg.norm(face.embedding)
        return bbox, emb

    def _faces(self, image: np.ndarray):
        self._ensure_loaded()
        return self.app.get(image)

    def extract_faces(self, image: np.ndarray) -> list[FaceDetection]:
        """Detect all faces once for a frame so person crops can reuse the results."""
        self._ensure_loaded()
        results = []
        for face in self._faces(image):
            embedding = face.embedding.astype(np.float32, copy=False)
            norm = np.linalg.norm(embedding)
            if norm > 1e-12:
                results.append(FaceDetection(face.bbox.astype(np.float32), embedding / norm,
                                              float(getattr(face, 'det_score', 1.0))))
        return results

    def extract_from_face_crop(self, face_image: np.ndarray) -> np.ndarray | None:
        """
        Extract embedding from a pre‑cropped face image.
        The image should be in BGR format (as read by OpenCV).
        It will be resized to 112x112 if needed.
        """
        self._ensure_loaded()
        if face_image is None or face_image.size == 0:
            return None
        if face_image.shape[:2] != (112, 112):
            face_image = cv2.resize(face_image, (112, 112))
        # The recognition model expects BGR input
        emb = self.app.models['recognition'].get_feat(face_image).flatten()
        norm = np.linalg.norm(emb)
        if norm > 1e-12:
            emb /= norm
        return emb
