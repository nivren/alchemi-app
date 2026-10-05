"""Small analytic model fixtures for application lifecycle tests."""

import torch

from nvalchemi.models.base import BaseModelMixin, ModelConfig


class CellHarmonic(BaseModelMixin):
    """Analytic E(H) with exact row-cell stress and zero Cartesian forces."""

    def __init__(self):
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces", "stress"}), supports_pbc=True
        )

    @property
    def embedding_shapes(self):
        return {}

    def compute_embeddings(self, data, **kwargs):
        return data

    def __call__(self, batch):
        h = batch.cell
        target = torch.eye(3, dtype=h.dtype, device=h.device).expand_as(h) * 2
        gradient = h - target
        volume = torch.linalg.det(h).abs()
        return {
            "energy": 0.5 * gradient.square().sum((-1, -2)),
            "forces": torch.zeros_like(batch.positions),
            "stress": h.mT @ gradient / volume[:, None, None],
        }
