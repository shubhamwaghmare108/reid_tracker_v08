"""Body appearance embeddings used to re-identify a person across frames."""
from __future__ import annotations
import numpy as np
import torch


class FeatureExtractor:
    """Adapter around TorchReID with validated, normalized descriptors."""
    def __init__(self, model_name: str = 'osnet_x1_0', model_path: str | None = '', device: str = 'cpu'):
        try:
            from torchreid.utils import FeatureExtractor as TorchExtractor
        except ModuleNotFoundError:
            from torchreid.reid.utils import FeatureExtractor as TorchExtractor
        self.extractor = TorchExtractor(model_name=model_name, model_path=model_path if model_path else None,
                                        device=device, verbose=False)
        self.device = device
        self.feature_dim: int | None = None

    @staticmethod
    def _normalize(feature: np.ndarray) -> np.ndarray:
        feature = np.asarray(feature, dtype=np.float32).reshape(-1)
        if feature.size == 0 or not np.isfinite(feature).all():
            raise ValueError('Invalid Re-ID feature: empty or non-finite')
        norm = float(np.linalg.norm(feature))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError('Invalid Re-ID feature: zero norm')
        return feature / norm

    def _validate_rows(self, result, expected_count: int | None = None):
        result = np.asarray(result, dtype=np.float32)
        if result.size == 0:
            raise ValueError('Re-ID model returned no features')
        if result.ndim == 1:
            result = result.reshape(1, -1)
        elif result.ndim != 2:
            raise ValueError(f'Re-ID model returned unexpected shape: {result.shape}')
        if expected_count is not None and result.shape[0] != expected_count:
            raise ValueError(f'Re-ID output count mismatch: expected {expected_count}, got {result.shape[0]}')
        if self.feature_dim is None:
            self.feature_dim = int(result.shape[1])
        if result.shape[1] != self.feature_dim:
            raise ValueError(f'Re-ID dimension mismatch: expected {self.feature_dim}, got {result.shape[1]}')
        return np.stack([self._normalize(row) for row in result], axis=0)

    def _run_model(self, rgb_batch, expected_count: int | None = None):
        invoke = self.extractor if callable(self.extractor) else getattr(self.extractor, '__call__', None)
        if invoke is None:
            raise TypeError('Re-ID extractor is not callable')
        with torch.no_grad():
            feature = invoke(rgb_batch)
        result = feature.detach().cpu().numpy() if isinstance(feature, torch.Tensor) else np.asarray(feature)
        return self._validate_rows(result, expected_count=expected_count)

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract one descriptor from an OpenCV BGR crop."""
        if image is None or image.size == 0:
            raise ValueError('Empty crop')
        rgb = image[:, :, ::-1].copy() if image.ndim == 3 and image.shape[2] == 3 else image
        return self._run_model(rgb, expected_count=1)[0]

    def batch_extract(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Extract one descriptor per valid crop without shifting detection order."""
        if not images:
            return []
        rgb_batch = []
        for image in images:
            if image is None or image.size == 0:
                continue
            rgb_batch.append(image[:, :, ::-1].copy() if image.ndim == 3 and image.shape[2] == 3 else image)
        if not rgb_batch:
            return []
        features = self._run_model(rgb_batch, expected_count=len(rgb_batch))
        return [feature for feature in features]
