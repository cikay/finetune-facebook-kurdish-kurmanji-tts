import operator as op
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torchaudio
from df.enhance import enhance, load_audio

SAMPLE_RATE = 16000

OPS = {
    "lt": op.lt,
    "lte": op.le,
    "gt": op.gt,
    "gte": op.ge,
    "eq": op.eq,
}


@lru_cache(maxsize=1)
def _load_model():
    from df.enhance import init_df
    return init_df()


class DeepFilterNetTool:
    def __init__(self, config: dict) -> None:
        self.run_if = config.get("run_if", {})
        self.replace_if = config.get("replace_if", {})

    def should_run(self, metrics: dict) -> bool:
        return all(
            self._check_condition(key, threshold, metrics)
            for key, threshold in self.run_if.items()
        )

    def should_replace(self, original_metrics: dict, enhanced_metrics: dict) -> bool:
        if not self.replace_if:
            return True
        return all(
            self._check_damage_condition(key, threshold, original_metrics, enhanced_metrics)
            for key, threshold in self.replace_if.items()
        )

    def enhance(self, input_path: Path) -> np.ndarray:
        model, df_state, _ = _load_model()
        audio, _ = load_audio(str(input_path), sr=df_state.sr())
        enhanced = enhance(model, df_state, audio)

        enhanced_tensor = torch.from_numpy(np.array(enhanced)).float()
        if enhanced_tensor.dim() == 1:
            enhanced_tensor = enhanced_tensor.unsqueeze(0)

        resampled = torchaudio.functional.resample(enhanced_tensor, df_state.sr(), SAMPLE_RATE)
        return resampled.squeeze().numpy()

    def _check_condition(self, key: str, threshold, metrics: dict) -> bool:
        *path_parts, operator_name = key.split("__")
        value = metrics
        for part in path_parts:
            value = value[part]
        return OPS[operator_name](value, threshold)

    def _check_damage_condition(
        self, key: str, threshold, original: dict, enhanced: dict
    ) -> bool:
        # key format: (enhance|damage)__<metric_path...>__<operator>
        parts = key.split("__")
        prefix = parts[0]
        metric_path = parts[1:-1]
        operator_name = parts[-1]

        orig_value = original
        enh_value = enhanced
        for part in metric_path:
            orig_value = orig_value[part]
            enh_value = enh_value[part]

        change = (enh_value - orig_value) if prefix == "enhance" else (orig_value - enh_value)
        return OPS[operator_name](change, threshold)
