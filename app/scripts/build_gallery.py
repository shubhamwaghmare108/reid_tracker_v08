"""Build cached reference embeddings from the images collected during enrollment."""
import os
import cv2
import numpy as np
from pathlib import Path
from app.config import Settings
from app.core.reid import FeatureExtractor
from app.core.face import FaceProcessor


def main():
    """Average each person's body and face vectors, then save those compact gallery files."""
    settings = Settings()
    body_extractor = FeatureExtractor(
        model_name=settings.reid_model,
        model_path=settings.reid_weights,
        device=settings.device,
    )
    face_processor = FaceProcessor(
        model_name=settings.face_model,
        det_size=settings.face_det_size,
    )

    gallery_root = settings.gallery_dir
    for person_dir in gallery_root.iterdir():
        if not person_dir.is_dir():
            continue

        # Body embeddings describe clothing and shape; each image supplies one vector.
        body_embeddings = []
        body_img_dir = person_dir / 'images'
        if body_img_dir.exists():
            for img_path in sorted(body_img_dir.glob('*.[jJ][pP][gG]')):
                img = cv2.imread(str(img_path))
                if img is not None:
                    emb = body_extractor.extract(img)
                    if emb is not None:
                        body_embeddings.append(emb)
        if body_embeddings:
            # The normalized mean is a more stable reference than any single image.
            body_mean = np.mean(body_embeddings, axis=0)
            body_mean /= np.linalg.norm(body_mean) + 1e-12
            np.save(person_dir / 'embedding_mean.npy', body_mean)

        # Face embeddings are stored separately because they may not exist for every person.
        face_embeddings = []
        face_img_dir = person_dir / 'faces'
        if face_img_dir.exists():
            for img_path in sorted(face_img_dir.glob('*.[jJ][pP][gG]')):
                img = cv2.imread(str(img_path))
                if img is not None:
                    emb = face_processor.extract_from_face_crop(img)
                    if emb is not None:
                        face_embeddings.append(emb)
        if face_embeddings:
            face_mean = np.mean(face_embeddings, axis=0)
            face_mean /= np.linalg.norm(face_mean) + 1e-12
            np.save(person_dir / 'face_embedding_mean.npy', face_mean)

        print(f'{person_dir.name}: {len(body_embeddings)} body, {len(face_embeddings)} face')


if __name__ == '__main__':
    main()
