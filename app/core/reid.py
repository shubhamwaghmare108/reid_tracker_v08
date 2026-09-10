"""Body appearance embeddings used to re-identify a person across frames."""
from __future__ import annotations
import numpy as np
import torch


class FeatureExtractor:
    """Adapter around TorchReID with one normalized vector per person crop.

    OpenCV supplies BGR ``H x W x 3`` arrays, while TorchReID expects RGB
    images and performs its own resize, tensor conversion, and batching. This
    class keeps that library-specific boundary in one place.
    """
    def __init__(self, model_name: str = 'osnet_x1_0', model_path: str | None = '', device: str = 'cpu'):
        """Load TorchReID while supporting both package layouts in use.

        The project has been run with releases exposing the extractor from
        either ``torchreid.utils`` or ``torchreid.reid.utils``. The fallback
        keeps the application independent of that packaging difference.
        """
        try:
            from torchreid.utils import FeatureExtractor as TorchExtractor
        except ModuleNotFoundError:
            from torchreid.reid.utils import FeatureExtractor as TorchExtractor
        self.extractor = TorchExtractor(
            model_name=model_name,
            model_path=model_path if model_path else None,
            device=device,
            verbose=False
        )
        self.device = device

    @staticmethod
    def _normalize(feature: np.ndarray) -> np.ndarray:
        """Convert one feature to float32 and unit length for cosine scoring."""
        feature = np.asarray(feature, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(feature)
        if norm > 1e-12:
            feature = feature / norm
        return feature

    def _run_model(self, rgb_batch):
        """Invoke TorchReID without gradients and normalize its returned rows.

        ``rgb_batch`` may be one image or a list of images. The underlying
        TorchReID extractor owns preprocessing, so this method intentionally
        does not convert a list into a NumPy 4D array.
        """
        if callable(self.extractor):
            invoke = self.extractor
        elif hasattr(self.extractor, '__call__'):
            invoke = self.extractor.__call__
        else:
            raise TypeError('Re-ID extractor is not callable')
        with torch.no_grad():
            feature = invoke(rgb_batch)
        if isinstance(feature, torch.Tensor):
            result = feature.detach().cpu().numpy()
        else:
            result = np.asarray(feature)
        if result.size == 0:
            return np.empty((0,), dtype=np.float32)
        result = np.asarray(result, dtype=np.float32)
        if result.ndim == 1:
            return self._normalize(result)
        if result.ndim == 2 and result.shape[0] == 1:
            return self._normalize(result[0])
        return np.stack([self._normalize(item) for item in result], axis=0)

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract one descriptor from an OpenCV BGR crop."""
        if image is None or image.size == 0:
            raise ValueError('Empty crop')
        if image.ndim == 3 and image.shape[2] == 3:
            rgb = image[:, :, ::-1].copy()
        else:
            rgb = image
        return self._run_model(rgb)

    def batch_extract(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Extract descriptors for several crops with one TorchReID call.

        Empty crops are omitted because TorchReID cannot transform them. The
        tracker owns the original detection-to-crop index mapping, so this
        method only returns descriptors for valid inputs in list order.
        """
        if not images:
            return []
        rgb_batch = []
        for image in images:
            if image is None or image.size == 0:
                continue
            if image.ndim == 3 and image.shape[2] == 3:
                rgb_batch.append(image[:, :, ::-1].copy())
            else:
                rgb_batch.append(image)
        if not rgb_batch:
            return []
        # TorchReID accepts a list of HWC images and stacks them after applying
        # its per-image torchvision transform. Pre-stacking here would make
        # torchvision treat the whole batch as one image and raise the 4D
        # ``pic should be 2/3 dimensional`` error.
        features = self._run_model(rgb_batch)
        if features.ndim == 1:
            return [self._normalize(features)]
        return [self._normalize(feature) for feature in features]
