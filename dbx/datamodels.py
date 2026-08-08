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
    """Generic hook-based layer activation capture evaluator for any model.

    Parameters
    ----------
    model
        A pre-loaded model, a lazy-eval string (e.g. ``"$module.func()"``), or ``None``.
    capture_layers : list[str]
        Named model layers to capture.
    capture_final : bool
        Whether to capture the model's final output tensor.
    transform : callable, optional
        Preprocessing transform applied to inputs before forward pass.
    device : str
        Target device string (default ``"cuda"``).
    log : Logger
        Logger instance.
    """

    DEFAULT_MODEL: Any = None
    DEFAULT_BACKBONE: Any = None
    DEFAULT_TRANSFORM: Any = None

    def __init__(
        self,
        model=None,
        *,
        backbone=None,
        capture_layers: list[str] | None = None,
        capture_final: bool = True,
        transform=None,
        device: str = "cuda",
        log: Logger | None = None,
    ):
        self.device = device
        self.log = log or Logger(stack_depth=3)
        model_val = model if model is not None else backbone
        if model_val is None:
            model_val = self.DEFAULT_MODEL if self.DEFAULT_MODEL is not None else self.DEFAULT_BACKBONE
        self._model = model_val
        self.transform = transform if transform is not None else self.DEFAULT_TRANSFORM
        if self.transform is None:
            self.transform = lambda x: x

        self.capture_layers = list(capture_layers or [])
        self.capture_final = capture_final
        self._captured: dict[str, Any] = {}
        self._hooks_registered = False

    @property
    def model(self):
        """Lazy-load the model on first access."""
        if isinstance(self._model, str):
            self.log.verbose(f"Evaluating {self._model} on {self.device}")
            model_obj = dbx.eval(self._model)
            if hasattr(model_obj, 'to'):
                model_obj = model_obj.to(self.device)
            self._model = model_obj
        return self._model

    @property
    def backbone(self):
        """Alias for model."""
        return self.model

    def _make_capture_hook(self, name: str):
        def hook(module, input, output):
            t = output
            if isinstance(t, (tuple, list)):
                t = t[0]
            if hasattr(t, 'cpu'):
                t = t.cpu().detach()
            self._captured[name] = t
        return hook

    def _register_capture_hooks(self):
        if self._hooks_registered:
            return

        for layer in self.capture_layers:
            if layer in ("backbone", "model"):
                self.model.register_forward_hook(self._make_capture_hook(layer))
            else:
                getattr(self.model, layer).register_forward_hook(self._make_capture_hook(layer))
            self.log.debug(f"Registered capture hook: {layer}")

        self._hooks_registered = True

    @property
    def layer_names(self) -> list[str]:
        """Return the ordered list of capture keys that ``__call__`` produces."""
        names = list(self.capture_layers)
        if self.capture_final:
            names.append('final')
        return names

    def __pre_call__(self):
        """Hook called before each forward pass (e.g. to lazily register hooks)."""
        self._register_capture_hooks()

    def __call__(self, x: Any) -> dict[str, Any]:
        """Run forward pass on batch *x* and return captured activations."""
        self._captured.clear()
        self.__pre_call__()
        if torch is not None and hasattr(x, 'to'):
            with torch.no_grad():
                y = self.transform(x.to(self.device))
                z = self.model(y)
                if hasattr(z, 'cpu'):
                    z = z.cpu().detach()
                del y
        else:
            y = self.transform(x)
            z = self.model(y)

        result = dict(self._captured)
        if self.capture_final:
            result['final'] = z
        return result

    def clear(self):
        """Release captured tensors and free accelerator memory."""
        self._captured.clear()
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self

    @property
    def layer_features(self) -> dict[str, Any]:
        """Most recently captured activations (read-only snapshot)."""
        return dict(self._captured)


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
                model=None,
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
    model
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

    def __init__(
        self,
        model=None,
        *,
        backbone=None,
        capture_blocks: list[int] | str | None = None,
        capture_layers: list[str] | None = None,
        capture_final: bool = True,
        cls_token_only: bool = False,
        transform=None,
        device: str = "cuda",
        log: Logger | None = None,
    ):
        model_val = model if model is not None else backbone
        super().__init__(
            model=model_val,
            capture_layers=capture_layers,
            capture_final=capture_final,
            transform=transform,
            device=device,
            log=log,
        )
        self.capture_blocks_raw = capture_blocks
        self.capture_blocks = list(capture_blocks) if isinstance(capture_blocks, (list, tuple)) else []
        self.cls_token_only = cls_token_only

    @property
    def backbone(self):
        """Alias for model."""
        return self.model

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
            blocks = self._get_blocks(self.model)
            self.capture_blocks = list(range(len(blocks)))

        if self.capture_blocks:
            blocks = self._get_blocks(self.model)
            for idx in self.capture_blocks:
                if idx < 0:
                    idx = len(blocks) + idx
                assert 0 <= idx < len(blocks), (
                    f"Block index {idx} out of range [0, {len(blocks)})"
                )
                key = self._capture_key(idx)
                blocks[idx].register_forward_hook(self._make_capture_hook(key))
                self.log.debug(f"Registered capture hook: {key}")

        super()._register_capture_hooks()

    @property
    def layer_names(self) -> list[str]:
        names = [self._capture_key(b) for b in self.capture_blocks]
        names += [self._capture_key(l) for l in self.capture_layers]
        if self.capture_final:
            names.append('final')
        return names

    def __call__(self, x) -> dict[str, Any]:
        result = super().__call__(x)
        if self.capture_final and self.cls_token_only:
            out = result['final']
            if hasattr(out, 'dim') and out.dim() == 3:
                result['final'] = out[:, 0]
        return result


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
                model=None,
                capture_blocks=self.var.capture_blocks,
                capture_layers=self.var.capture_layers,
                capture_final=self.var.capture_final,
                cls_token_only=self.var.cls_token_only,
                device=device,
                log=log,
            )
        return self._evaluators[device]
