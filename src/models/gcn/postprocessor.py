# src/models/gcn/postprocessor.py: take the final refinement step and clamp to standard corners

from src.models.base.base_postprocessor import BasePostprocessor


class GCNPostprocessor(BasePostprocessor):
    """Selects the last refinement output from (N, T+1, 4, 2) and clamps it to [0, 1] as (N, 4, 2) corners."""

    def __call__(self, raw_output):
        return raw_output[:, -1].clamp(0.0, 1.0)
