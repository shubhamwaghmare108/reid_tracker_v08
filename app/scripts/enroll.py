"""Interactive webcam tool for collecting good body and optional face references."""
import os
import cv2
import numpy as np
from ultralytics import YOLO
from app.config import Settings
from app.core.face import FaceProcessor


def main():
    """Show the camera feed and save a sample only when the user presses Space."""
    settings = Settings()
    person_name = input('Person Name: ').strip()
    
    # Store every identity in its own folder so the gallery loader can discover it.
    body_dir = settings.gallery_dir / person_name / 'images'
    face_dir = settings.gallery_dir / person_name / 'faces'
    body_dir.mkdir(parents=True, exist_ok=True)
    face_dir.mkdir(parents=True, exist_ok=True)

    max_images = 10
    conf_threshold = 0.7
    min_width, min_height = 80, 160
    blur_threshold = 100

    # Segmentation makes the body crop contain less distracting background.
    model = YOLO(str(settings.detector_model))
    
    # Load face processor (InsightFace)
    face_processor = FaceProcessor(
        model_name=settings.face_model,
        det_size=settings.face_det_size,
    )

    cap = cv2.VideoCapture(0)
    count = len([f for f in body_dir.iterdir() if f.suffix.lower() in ('.jpg', '.png')])
    if count >= max_images:
        print('Enrollment already completed.')
        return

    save_pressed = False
    print('\nSPACE : Save image\nESC   : Exit\n')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        result = model(frame, verbose=False)[0]
        body_crop = None
        face_crop = None

        if result.masks is not None:
            # Enrollment expects one subject, so choose the largest confident person detection.
            best_idx = -1
            best_area = 0
            for i, box in enumerate(result.boxes):
                if int(box.cls) != 0 or float(box.conf) < conf_threshold:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2-x1)*(y2-y1)
                if area > best_area:
                    best_area = area
                    best_idx = i

            if best_idx != -1:
                box = result.boxes[best_idx]
                mask = result.masks.data[best_idx].cpu().numpy()
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if (x2-x1) >= min_width and (y2-y1) >= min_height:
                    # Resize mask to frame size and apply
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    binary = (mask > 0.5).astype(np.uint8)
                    segmented = frame.copy()
                    segmented[binary == 0] = 0
                    body_crop = segmented[y1:y2, x1:x2]

                    if body_crop.size != 0:
                        body_crop = cv2.resize(body_crop, (128, 256))
                        # Laplacian variance is a simple sharpness check; blurry samples are rejected.
                        blur = cv2.Laplacian(cv2.cvtColor(body_crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                        if blur >= blur_threshold:
                            # Try to detect a face inside the person ROI
                            person_roi = frame[y1:y2, x1:x2]
                            if person_roi.size != 0:
                                _, face_emb = face_processor.extract(person_roi)
                                if face_emb is not None:
                                    # We need the face crop; we can re-detect and crop
                                    # But face_processor.extract only returns embedding and bbox relative to the input image.
                                    # Let's use the face detection from the processor directly.
                                    # We'll get the detection again from the processor's internal method.
                                    # Simpler: we can use the face_processor.app.get() to get faces and crop.
                                    faces = face_processor.app.get(person_roi)
                                    if faces:
                                        # Take the largest face
                                        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                                        fx1, fy1, fx2, fy2 = face.bbox.astype(int)
                                        # Ensure within bounds
                                        fx1 = max(0, fx1)
                                        fy1 = max(0, fy1)
                                        fx2 = min(person_roi.shape[1], fx2)
                                        fy2 = min(person_roi.shape[0], fy2)
                                        face_crop = person_roi[fy1:fy2, fx1:fx2]
                                        if face_crop.size != 0:
                                            face_crop = cv2.resize(face_crop, (112, 112))
                                            cv2.imshow('Face', face_crop)

                            cv2.rectangle(display, (x1, y1), (x2, y2), (0,255,0), 2)
                            cv2.putText(display, f'{count}/{max_images}', (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                            cv2.imshow('Person', body_crop)
                        else:
                            body_crop = None

        cv2.imshow('Enrollment', display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord(' '):
            if not save_pressed and body_crop is not None:
                # Save body image
                body_path = body_dir / f'{count:04d}.jpg'
                cv2.imwrite(str(body_path), body_crop)
                print(f'Saved body {count:04d}.jpg')

                # Save face image if available
                if face_crop is not None:
                    face_path = face_dir / f'{count:04d}.jpg'
                    cv2.imwrite(str(face_path), face_crop)
                    print(f'Saved face {count:04d}.jpg')

                count += 1
                save_pressed = True
                if count >= max_images:
                    break
        else:
            save_pressed = False

    cap.release()
    cv2.destroyAllWindows()
    print(f'Enrollment completed with {count} images.')


if __name__ == '__main__':
    main()
