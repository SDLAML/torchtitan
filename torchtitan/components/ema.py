# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Configurable

__all__ = ["EMAOptimizersContainer"]


class _EMAParamOptimizer(Optimizer):
    """One per model part, mirroring ``OptimizersContainer.optimizers: list[T]``.

    Never stepped as a real optimizer -- exists solely to hold
    ``state[p]["ema_params"]`` per parameter, reusing ``torch.optim.Optimizer``'s
    per-param state dict + DCP flatten machinery instead of a bespoke DTensor
    state-dict format. Always built over the real parameter list even when EMA
    is disabled; ``enable`` only controls whether the EMA tensor is allocated.
    """

    def __init__(self, params: list[nn.Parameter], *, enable: bool) -> None:
        super().__init__(params, defaults={})
        if enable:
            for group in self.param_groups:
                for p in group["params"]:
                    self.state[p]["ema_params"] = p.detach().clone()

    def step(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "_EMAParamOptimizer must not be step()-ed; call "
            "EMAOptimizersContainer.step(current_step) instead."
        )


class EMAOptimizersContainer(OptimizersContainer):
    """Pseudo-optimizer container maintaining an online EMA of model weights.

    Subclasses ``OptimizersContainer`` like ``OptimizersInBackwardContainer``
    does: override ``__init__``/``step()``/``zero_grad()``, reuse
    ``state_dict()``/``load_state_dict()`` (with an ``enable`` short-circuit).
    Never merged into ``Trainer.optimizers`` or passed to
    ``LRSchedulersContainer`` -- it is a sibling object, always built and
    wired unconditionally into ``CheckpointManager`` and into a
    ``register_step_post_hook`` on the real optimizer; ``enable`` decides
    whether that amounts to anything.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        enable: bool = False
        """Whether EMA tracking is active. Always built regardless (so trainer.py
        needs no conditional wiring), but holds no EMA tensors and contributes
        nothing to the checkpoint when False."""

        decay: float | None = None
        """ema_params = decay * ema_params + (1 - decay) * param, applied once
        per firing. If None (default), decay is instead computed dynamically
        from half_life_fraction -- see below."""

        half_life_fraction: float = 0.05
        """Used when decay is None: decay = 2 ** (-1 / (half_life_fraction * t)),
        where t is elapsed steps since start_step. Keeps roughly the most
        recent half_life_fraction share of steps dominant. Default 0.05
        matches the common decay = 2 ** (-20 / t) rule of thumb."""

        start_step: int = 0
        """First training step (matching Trainer.step) at which EMA tracking
        begins. Intentionally decoupled from the LR scheduler's WSD phases."""

        step_bias: int = 0
        """Added to (current_step - start_step) when computing t (only relevant
        when decay is None). A normal resume already gets correct continuity
        for free since current_step is restored as usual; step_bias is only
        for deliberately renumbering Trainer.step (e.g. a new training phase)
        while wanting the EMA to keep aging as if uninterrupted."""

        update_every_n_steps: int = 1
        """Only fire the EMA update every N real optimizer steps."""

        offload_to_cpu: bool = False
        """Keep EMA weights in pinned CPU memory, updated via an async
        side-stream H2D/D2H pipeline, for GH200's NVLink-C2C interconnect."""

    def __init__(self, config: Config, *, model_parts: list[nn.Module]) -> None:
        self.enable = config.enable
        self.decay = config.decay
        self.half_life_fraction = config.half_life_fraction
        self.start_step = config.start_step
        self.step_bias = config.step_bias
        self.update_every_n_steps = config.update_every_n_steps
        self.offload_to_cpu = config.offload_to_cpu
        self.model_parts = model_parts
        # We override __init__ entirely rather than calling
        # OptimizersContainer's, so attributes it would normally set (and that
        # load_state_dict() reads) must be set explicitly here.
        self.preserve_lrs_when_loading = False
        self.norms_to_log: list[str] | None = None
        self.log_queue = None
        self.log_thread = None
        self.optimizers: list[_EMAParamOptimizer] = []
        all_params: list[nn.Parameter] = []
        for model in model_parts:
            params = [p for p in model.parameters() if p.requires_grad]
            self.optimizers.append(_EMAParamOptimizer(params, enable=self.enable))
            all_params.extend(params)
        self._validate_length(len(self.model_parts))
        self._post_init(all_params, {})

        self._offload_stream: torch.cuda.Stream | None = None
        self._offload_scratch: dict[int, list[torch.Tensor]] = {}
        self._pending_event: torch.cuda.Event | None = None
        if self.enable and self.offload_to_cpu:
            self._init_cpu_offload()

    def zero_grad(self, *args, **kwargs) -> None:
        pass  # never called by the training loop; no-op for safety

    def step(self, current_step: int) -> None:
        """Call directly with the trainer's global step count -- this object
        is never merged into Trainer.optimizers, so there's no zero-arg/closure
        step() convention to honor here."""
        if not self.enable or current_step < self.start_step:
            return
        if (current_step - self.start_step) % self.update_every_n_steps != 0:
            return
        # Clamped to >= 1 to avoid dividing by zero when current_step happens
        # to equal start_step exactly.
        t = max(current_step - self.start_step + self.step_bias, 1)
        self._update(t)

    def _decay_at(self, t: int) -> float:
        if self.decay is not None:
            return self.decay
        return 2.0 ** (-1.0 / (self.half_life_fraction * t))

    def _update(self, t: int) -> None:
        decay = self._decay_at(t)
        for ema_opt, model in zip(self.optimizers, self.model_parts):
            params = [p for p in model.parameters() if p.requires_grad]
            if not params:
                continue
            ema_params = [ema_opt.state[p]["ema_params"] for p in params]
            if self.offload_to_cpu:
                self._update_offloaded(params, ema_params, decay)
            elif torch.is_floating_point(ema_params[0]) or torch.is_complex(
                ema_params[0]
            ):
                torch._foreach_lerp_(ema_params, params, 1.0 - decay)
            else:
                for e, p in zip(ema_params, params):
                    e.copy_(e * decay + p * (1.0 - decay))

    # --- CPU offload path (GH200-optimized: async side-stream, pinned memory) ---

    def _init_cpu_offload(self) -> None:
        self._offload_stream = torch.cuda.Stream()
        for ema_opt in self.optimizers:
            for st in ema_opt.state.values():
                st["ema_params"] = st["ema_params"].cpu().pin_memory()

    def _get_scratch(self, key: int, params: list[torch.Tensor]) -> list[torch.Tensor]:
        scratch = self._offload_scratch.get(key)
        if scratch is None:
            scratch = [torch.empty_like(p) for p in params]
            self._offload_scratch[key] = scratch
        return scratch

    def _maybe_wait_pending(self) -> None:
        if self._pending_event is not None:
            self._pending_event.synchronize()
            self._pending_event = None

    def _update_offloaded(
        self,
        params: list[torch.Tensor],
        ema_params: list[torch.Tensor],
        decay: float,
    ) -> None:
        self._maybe_wait_pending()
        scratch = self._get_scratch(id(ema_params), params)
        stream = self._offload_stream
        assert stream is not None
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            torch._foreach_copy_(scratch, ema_params, non_blocking=True)  # H2D
            torch._foreach_lerp_(scratch, params, 1.0 - decay)
            torch._foreach_copy_(ema_params, scratch, non_blocking=True)  # D2H
            self._pending_event = torch.cuda.Event()
            self._pending_event.record(stream)
        # Don't synchronize here -- the event is waited on lazily, next call
        # or at state_dict() (checkpoint save).

    # --- checkpointing ---

    def state_dict(self) -> dict[str, Any]:
        if not self.enable:
            return {}
        if self.offload_to_cpu:
            self._maybe_wait_pending()
        return super().state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not self.enable:
            return
        if not state_dict:
            # Checkpoint had no EMA data (saved with EMA disabled, or predates
            # this feature) -- cold-start from the just-loaded model weights,
            # since our clone at __init__ predates CheckpointManager.load().
            for ema_opt, model in zip(self.optimizers, self.model_parts):
                for p in (p for p in model.parameters() if p.requires_grad):
                    ema_opt.state[p]["ema_params"].copy_(p.detach())
            return
        super().load_state_dict(state_dict)
