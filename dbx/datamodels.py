"""dbx.datamodels — Base evaluators and factories for model feature extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
import gc
from typing import Any

try:
    import torch
except ImportError:
    torch = None

import dbx
from dbx.datablocks import Datablock
from dbx.dataparts import Logger


class DatamodelEvaluator:
    """Generic base for hook-based layer activation capture evaluators.

    Subclasses must implement:

    * :meth:`layer_names` — ordered list of capture keys produced by ``__call__``.
    * :meth:`__call__` — run a forward pass and return a ``dict[str, Tensor]``
      mapping capture keys to activation tensors, plus ``"final"`` (if requested)
      for the model's own output.
    * :meth:`clear` — release captured tensors and free accelerator memory.

    Optionally override :meth:`__pre_call__` to lazily register hooks.
    """

    def __init__(self, *, device: str = "cuda", log: Logger | None = None):
        self.device = device
        self.log = log or Logger(stack_depth=3)

    @property
    def layer_names(self) -> list[str]:
        """Return the ordered list of capture keys that ``__call__`` produces.

        Must be overridden by subclasses.
        """
        raise NotImplementedError

    def __pre_call__(self):
        """Hook called before each forward pass (e.g. to lazily register hooks)."""
        pass

    def __call__(self, x: Any) -> dict[str, Any]:
        """Run forward pass on batch *x* and return captured activations.

        Returns
        -------
        dict[str, Tensor]
            Mapping from capture-key to activation tensor, plus
            ``"final"`` for the backbone's own output.

        Must be overridden by subclasses.
        """
        raise NotImplementedError

    def clear(self):
        """Release captured tensors and free accelerator memory."""
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self

    @property
    def layer_features(self) -> dict[str, Any]:
        """Most recently captured activations (read-only snapshot).

        Must be overridden by subclasses.
        """
        raise NotImplementedError


class DatamodelEvaluatorFactory(Datablock):
    """Abstract Datablock used for spec-based evaluator dependency tracking.

    This Datablock does **not** build or persist anything itself.
    It exists so that feature blocks/clips can declare their evaluator configuration
    as a spec dependency, enabling deterministic hashing and lineage tracking.

    Concrete subclasses may extend ``VAR`` with model-specific fields
    and override :meth:`evaluator` to return a ready-to-use
    :class:`DatamodelEvaluator`.
    """

    VERSION = 1
    Evaluator: type[DatamodelEvaluator] = DatamodelEvaluator

    @dataclass
    class VAR(Datablock.VAR):
        capture_layers: list = field(default_factory=list)  # list[str] — named layers
        capture_final: bool = True  # capture model output as 'features_final'

    def evaluator(self, *, device: str = "cuda", log: Logger | None = None) -> DatamodelEvaluator:
        """Create a live :class:`DatamodelEvaluator`.

        The result is cached per device so that repeated calls with
        the same device return the same instance.
        """
        if not hasattr(self, '_evaluators'):
            self._evaluators = {}
        if device not in self._evaluators:
            log = log or self.log
            self._evaluators[device] = self.Evaluator(
                backbone=None,
                capture_layers=self.var.capture_layers,
                capture_final=self.var.capture_final,
                device=device,
                log=log,
            )
        return self._evaluators[device]


class DataformerEvaluator(DatamodelEvaluator):
    """Transformer-specific activation-capturing backbone evaluator.

    Parameters
    ----------
    backbone
        A pre-loaded model, a lazy-eval string (e.g. ``"$module.func()"``), or ``None``.
    capture_blocks : list[int] | str
        Transformer block indices to capture, or ``'all'``.
    capture_layers : list[str]
        Named model layers to capture.
    capture_final : bool
        Whether to capture the model's final output tensor.
    cls_token_only : bool
        When ``True``, hooks capture only the CLS token activation (index 0 of
        the sequence dimension).
    transform : callable, optional
        Preprocessing transform applied to inputs before forward pass.
    device : str
        Target device string (default ``"cuda"``).
    log : Logger
        Logger instance.
    """

    DEFAULT_BACKBONE: Any = None
    DEFAULT_TRANSFORM: Any = None

    def __init__(
        self,
        backbone=None,
        *,
        capture_blocks: list[int] | str | None = None,
        capture_layers: list[str] | None = None,
        capture_final: bool = True,
        cls_token_only: bool = False,
        transform=None,
        device: str = "cuda",
        log: Logger | None = None,
    ):
        super().__init__(device=device, log=log)
        self._backbone = backbone if backbone is not None else self.DEFAULT_BACKBONE
        self.transform = transform if transform is not None else self.DEFAULT_TRANSFORM
        if self.transform is None:
            self.transform = lambda x: x

        self.capture_blocks_raw = capture_blocks
        self.capture_blocks = list(capture_blocks) if isinstance(capture_blocks, (list, tuple)) else []
        self.capture_layers = list(capture_layers or [])
        self.capture_final = capture_final
        self.cls_token_only = cls_token_only
        self._captured: dict[str, Any] = {}
        self._hooks_registered = False

    @property
    def backbone(self):
        """Lazy-load the backbone model on first access."""
        if isinstance(self._backbone, str):
            self.log.verbose(f"Evaluating {self._backbone} on {self.device}")
            self._backbone = dbx.eval(self._backbone).to(self.device)
        return self._backbone

    def _get_blocks(self, model):
        """Extract transformer block container from model."""
        if hasattr(model, 'blocks'):
            return model.blocks
        if hasattr(model, 'layers'):
            return model.layers
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'blocks'):
            return model.transformer.blocks
        raise AttributeError(f"Could not find blocks/layers on model {type(model).__name__}")

    def _make_capture_hook(self, name: str):
        cls_only = self.cls_token_only

        def hook(module, input, output):
            t = output
            if isinstance(t, (tuple, list)):
                t = t[0]
            if hasattr(t, 'cpu'):
                t = t.cpu().detach()
            if cls_only and hasattr(t, 'dim') and t.dim() == 3:
                t = t[:, 0]
            self._captured[name] = t
        return hook

    @staticmethod
    def _capture_key(layer) -> str:
        return f"block.{layer}" if isinstance(layer, int) else str(layer)

    def _register_capture_hooks(self):
        if self._hooks_registered:
            return

        if self.capture_blocks_raw == 'all':
            blocks = self._get_blocks(self.backbone)
            self.capture_blocks = list(range(len(blocks)))

        if self.capture_blocks:
            blocks = self._get_blocks(self.backbone)
            for idx in self.capture_blocks:
                if idx < 0:
                    idx = len(blocks) + idx
                assert 0 <= idx < len(blocks), (
                    f"Block index {idx} out of range [0, {len(blocks)})"
                )
                key = self._capture_key(idx)
                blocks[idx].register_forward_hook(self._make_capture_hook(key))
                self.log.debug(f"Registered capture hook: {key}")

        for layer in self.capture_layers:
            key = self._capture_key(layer)
            if layer == "backbone":
                self.backbone.register_forward_hook(self._make_capture_hook(key))
            else:
                getattr(self.backbone, layer).register_forward_hook(self._make_capture_hook(key))
            self.log.debug(f"Registered capture hook: {key}")

        self._hooks_registered = True

    @property
    def layer_names(self) -> list[str]:
        names = [self._capture_key(b) for b in self.capture_blocks]
        names += [self._capture_key(l) for l in self.capture_layers]
        if self.capture_final:
            names.append('final')
        return names

    def __pre_call__(self):
        self._register_capture_hooks()

    def __call__(self, x) -> dict[str, Any]:
        self._captured.clear()
        self.__pre_call__()
        if torch is not None and hasattr(x, 'to'):
            with torch.no_grad():
                y = self.transform(x.to(self.device))
                z = self.backbone(y)
                if hasattr(z, 'cpu'):
                    z = z.cpu().detach()
                del y
        else:
            y = self.transform(x)
            z = self.backbone(y)

        result = dict(self._captured)
        if self.capture_final:
            out = z
            if self.cls_token_only and hasattr(out, 'dim') and out.dim() == 3:
                out = out[:, 0]
            result['final'] = out
        return result

    @property
    def layer_features(self) -> dict[str, Any]:
        return dict(self._captured)

    def clear(self):
        self._captured.clear()
        return super().clear()


class DataformerEvaluatorFactory(DatamodelEvaluatorFactory):
    """Datablock factory for Transformer-specific evaluators (Dataformer)."""

    VERSION = 1
    Evaluator: type[DataformerEvaluator] = DataformerEvaluator

    @dataclass
    class VAR(DatamodelEvaluatorFactory.VAR):
        capture_blocks: list = field(default_factory=list)  # list[int] — transformer block indices
        cls_token_only: bool = False  # capture only CLS token activations

    def evaluator(self, *, device: str = "cuda", log: Logger | None = None) -> DataformerEvaluator:
        """Create a live :class:`DataformerEvaluator`.

        The result is cached per device so that repeated calls with
        the same device return the same instance.
        """
        if not hasattr(self, '_evaluators'):
            self._evaluators = {}
        if device not in self._evaluators:
            log = log or self.log
            self._evaluators[device] = self.Evaluator(
                backbone=None,
                capture_blocks=self.var.capture_blocks,
                capture_layers=self.var.capture_layers,
                capture_final=self.var.capture_final,
                cls_token_only=self.var.cls_token_only,
                device=device,
                log=log,
            )
        return self._evaluators[device]
