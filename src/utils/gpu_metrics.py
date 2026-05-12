"""Per-step GPU utilization sampler for wandb logging.

wandb's built-in system metrics sample every ~30 s on a separate timeline
and don't align with training steps; that's too coarse to spot dataloader
stalls. This wrapper polls pynvml on demand so the trainer can log GPU
util / memory / power on the same `step` axis as `train/loss_avg50`.
"""
from __future__ import annotations

import logging

LOG = logging.getLogger(__name__)


class GpuSampler:
    """Per-call NVML poll for GPU 0. Cheap (<1 ms per sample).

    On nvml init failure (no GPU, no driver, headless dev box), `sample()`
    returns an empty dict so the trainer keeps running without spamming
    errors. Set `device_index` to target a different GPU under DDP.
    """

    def __init__(self, device_index: int = 0):
        self._handle = None
        self._available = False
        self._mem_total_gb: float | None = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self._mem_total_gb = mem.total / (1024 ** 3)
            self._available = True
            LOG.info(
                "GpuSampler ready on cuda:%d (total=%.1f GB)",
                device_index, self._mem_total_gb,
            )
        except Exception as e:
            LOG.warning("GpuSampler disabled: %r", e)

    @property
    def available(self) -> bool:
        return self._available

    def sample(self) -> dict[str, float]:
        if not self._available:
            return {}
        try:
            util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            power_mw = self._pynvml.nvmlDeviceGetPowerUsage(self._handle)
            temp_c = self._pynvml.nvmlDeviceGetTemperature(
                self._handle, self._pynvml.NVML_TEMPERATURE_GPU
            )
            return {
                "system/gpu_util_pct": float(util.gpu),
                "system/gpu_mem_used_gb": mem.used / (1024 ** 3),
                "system/gpu_mem_total_gb": self._mem_total_gb or 0.0,
                "system/gpu_mem_pct": 100.0 * mem.used / mem.total,
                "system/gpu_power_w": power_mw / 1000.0,
                "system/gpu_temp_c": float(temp_c),
            }
        except Exception as e:
            LOG.warning("GpuSampler.sample() failed: %r", e)
            return {}

    def shutdown(self) -> None:
        if self._available:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
            self._available = False
