"""Load the reference gallery: one averaged body/face embedding per enrolled person."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2
from app.utils.utils import image_files


@dataclass
class Gallery:
    """Parallel arrays holding names and their representative embedding vectors."""
    names: list[str]
    body_means: np.ndarray
    face_means: np.ndarray

    @classmethod
    def load(cls, root: Path, body_extractor, face_processor=None):
        """Load cached means or calculate and cache them from enrollment images."""
        names, body_embs, face_embs = [], [], []
        for identity_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            # Averaging several photos makes the reference less sensitive to one pose or frame.
            body_path = identity_dir / 'embedding_mean.npy'
            if body_path.exists():
                body_mean = np.load(body_path)
            else:
                body_embeddings = []
                for img_path in image_files(identity_dir / 'images'):
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        emb = body_extractor.extract(img)
                        if emb is not None:
                            body_embeddings.append(emb)
                if body_embeddings:
                    body_mean = np.asarray(np.mean(body_embeddings, axis=0), dtype=np.float32).reshape(-1)
                    body_mean /= np.linalg.norm(body_mean) + 1e-12
                    np.save(identity_dir / 'embedding_mean.npy', body_mean)
                else:
                    body_mean = None

            # Face references are optional; body matching still works without them.
            face_path = identity_dir / 'face_embedding_mean.npy'
            if face_path.exists():
                face_mean = np.load(face_path)
            elif face_processor is not None:
                face_embeddings = []
                for img_path in image_files(identity_dir / 'faces'):
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        _, emb = face_processor.extract(img)
                        if emb is not None:
                            face_embeddings.append(emb)
                if face_embeddings:
                    face_mean = np.asarray(np.mean(face_embeddings, axis=0), dtype=np.float32).reshape(-1)
                    face_mean /= np.linalg.norm(face_mean) + 1e-12
                    np.save(identity_dir / 'face_embedding_mean.npy', face_mean)
                else:
                    face_mean = None
            else:
                face_mean = None

            if body_mean is not None:
                names.append(identity_dir.name)
                body_embs.append(body_mean)
                # NaN is a sentinel for identities without a face reference.  A zero
                # vector would be treated as a real embedding and incorrectly dilute
                # the body-match score.
                face_embs.append(face_mean if face_mean is not None else np.full_like(body_mean, np.nan))

        return cls(
            names=names,
            body_means=np.stack(body_embs) if body_embs else np.empty((0, 0)),
            face_means=np.stack(face_embs) if face_embs else np.empty((0, 0)),
        )

    def has_body(self) -> bool:
        """Whether at least one enrolled identity has a body embedding."""
        return self.body_means.shape[0] > 0

    def has_face(self) -> bool:
        """Whether any enrolled identity has a real (non-NaN) face embedding."""
        return self.face_means.size > 0 and np.isfinite(self.face_means).any()
