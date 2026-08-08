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

    @dataclass
    class VAR(Datablock.VAR):
        capture_layers: list = field(default_factory=list)  # list[str] — named layers
        capture_final: bool = True  # capture model output as 'features_final'

    def evaluator(self, *, device: str | None = None, log: Logger | None = None) -> DatamodelEvaluator:
        """Create a live :class:`DatamodelEvaluator`.

        Must be overridden by subclasses.
        """
        raise NotImplementedError


class DataformerEvaluator(DatamodelEvaluator):
    """Transformer-specific evaluator base for block and layer activation capture."""
    pass


class DataformerEvaluatorFactory(DatamodelEvaluatorFactory):
    """Datablock factory for Transformer-specific evaluators (Dataformer).

    Concrete subclasses may extend ``VAR`` with Transformer-specific fields
    and override :meth:`evaluator` to return a ready-to-use
    :class:`DataformerEvaluator`.
    """

    VERSION = 1

    @dataclass
    class VAR(DatamodelEvaluatorFactory.VAR):
        capture_blocks: list = field(default_factory=list)  # list[int] — transformer block indices
        cls_token_only: bool = False  # capture only CLS token activations

    def evaluator(self, *, device: str | None = None, log: Logger | None = None) -> DataformerEvaluator:
        """Create a live :class:`DataformerEvaluator`.

        Must be overridden by subclasses.
        """
        raise NotImplementedError
