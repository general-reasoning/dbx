"""dbx.datamodels — Base evaluators and factories for model feature extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
import gc
import math
from typing import Any

import numpy as np

try:
    import torch
except ImportError:
    torch = None

import dbx
from dbx.datablocks import Datablock, Datastack
from dbx.dataparts import Logger
from dbx.datastreams import (
    DatastreamTab,
    DatastreamTable,
    ZipStreamingDataset,
    ZipIterableStreamingDatasets,
)
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
    DEFAULT_TRANSFORM: Any = None

    # 1. Datablock / Evaluator Protocol Methods ─────────────────────

    def __init__(
        self,
        model=None,
        *,
        capture_layers: list[str] | None = None,
        capture_final: bool = True,
        transform=None,
        device: str = "cuda",
        log: Logger | None = None,
    ):
        self.device = device
        self.log = log or Logger(stack_depth=3)
        self._model = model if model is not None else self.DEFAULT_MODEL
        self.transform = transform if transform is not None else self.DEFAULT_TRANSFORM
        if self.transform is None:
            self.transform = lambda x: x

        self.capture_layers = list(capture_layers or [])
        self.capture_final = capture_final
        self._captured: dict[str, Any] = {}
        self._hooks_registered = False

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

    # 2. Properties and Accessors ───────────────────────────────────

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
    def layer_names(self) -> list[str]:
        """Return the ordered list of capture keys that ``__call__`` produces."""
        names = list(self.capture_layers)
        if self.capture_final:
            names.append('final')
        return names

    @property
    def layer_features(self) -> dict[str, Any]:
        """Most recently captured activations (read-only snapshot)."""
        return dict(self._captured)

    # 3. Private and Utility Methods ────────────────────────────────

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


class DatamodelEvaluatorFactory(Datablock):
    """Abstract Datablock used for spec-based evaluator dependency tracking.

    This Datablock does **not** build or persist anything itself.
    It exists so that feature blocks/clips can declare their evaluator configuration
    as a spec dependency, enabling deterministic hashing and lineage tracking.

    Concrete subclasses may extend ``VAR`` with model-specific fields
    and override `evaluator()` to return a ready-to-use
    `DatamodelEvaluator`.
    """

    VERSION = 1
    Evaluator: type[DatamodelEvaluator] = DatamodelEvaluator

    @dataclass
    class VAR(Datablock.VAR):
        capture_layers: list = field(default_factory=list)  # list[str] — named layers
        capture_final: bool = True  # capture model output as 'features_final'

    # 1. Datablock Protocol Methods ─────────────────────────────────

    def __init__(self, *, capture_layers=None, capture_final=True, spec=None, **kwargs):
        spec = dict(spec) if spec is not None else {}
        if capture_layers is not None:
            spec['capture_layers'] = capture_layers
        if capture_final is not True:
            spec['capture_final'] = capture_final
        super().__init__(spec=spec, **kwargs)

    def evaluator(self, *, device: str = "cuda", log: Logger | None = None) -> DatamodelEvaluator:
        """Create a live `DatamodelEvaluator`.

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
    """Transformer-specific activation-capturing model evaluator.

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

    # 1. Datablock / Evaluator Protocol Methods ─────────────────────

    def __init__(
        self,
        model=None,
        *,
        capture_blocks: list[int] | str | None = None,
        capture_layers: list[str] | None = None,
        capture_final: bool = True,
        cls_token_only: bool = False,
        transform=None,
        device: str = "cuda",
        log: Logger | None = None,
    ):
        super().__init__(
            model=model,
            capture_layers=capture_layers,
            capture_final=capture_final,
            transform=transform,
            device=device,
            log=log,
        )
        self.capture_blocks_raw = capture_blocks
        self.capture_blocks = list(capture_blocks) if isinstance(capture_blocks, (list, tuple)) else ([] if capture_blocks != 'all' else 'all')
        self.cls_token_only = cls_token_only

    def __call__(self, x) -> dict[str, Any]:
        result = super().__call__(x)
        if self.capture_final and self.cls_token_only:
            out = result['final']
            if hasattr(out, 'dim') and out.dim() == 3:
                result['final'] = out[:, 0]
        return result

    # 2. Properties and Accessors ───────────────────────────────────

    @property
    def layer_names(self) -> list[str]:
        blocks_spec = self.capture_blocks
        if blocks_spec == 'all' or self.capture_blocks_raw == 'all':
            blocks = self._get_blocks(self.model)
            blocks_list = list(range(len(blocks)))
        elif isinstance(blocks_spec, (list, tuple)):
            if any(isinstance(b, int) and b < 0 for b in blocks_spec):
                blocks = self._get_blocks(self.model)
                blocks_list = [len(blocks) + b if isinstance(b, int) and b < 0 else b for b in blocks_spec]
            else:
                blocks_list = list(blocks_spec)
        else:
            blocks_list = []

        names = [self._capture_key(b) for b in blocks_list]
        names += [self._capture_key(l) for l in self.capture_layers]
        if self.capture_final:
            names.append('final')
        return names

    # 3. Private and Utility Methods ────────────────────────────────

    def _get_blocks(self, model):
        """Extract transformer block container from model."""
        if hasattr(model, 'blocks'):
            return model.blocks
        if hasattr(model, 'layers'):
            return model.layers
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'blocks'):
            return model.transformer.blocks
        raise AttributeError(f"Could not find blocks/layers on model {type(model).__name__}")

    @staticmethod
    def _capture_key(layer) -> str:
        return f"block.{layer}" if isinstance(layer, int) else str(layer)

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

    def _register_capture_hooks(self):
        if self._hooks_registered:
            return

        blocks = self._get_blocks(self.model)

        if self.capture_blocks_raw == 'all' or self.capture_blocks == 'all':
            self.capture_blocks = list(range(len(blocks)))
        elif self.capture_blocks:
            resolved = []
            for idx in self.capture_blocks:
                if idx < 0:
                    idx = len(blocks) + idx
                assert 0 <= idx < len(blocks), (
                    f"Block index {idx} out of range [0, {len(blocks)})"
                )
                resolved.append(idx)
            self.capture_blocks = resolved

        if isinstance(self.capture_blocks, list):
            for idx in self.capture_blocks:
                key = self._capture_key(idx)
                blocks[idx].register_forward_hook(self._make_capture_hook(key))
                self.log.debug(f"Registered capture hook: {key}")

        super()._register_capture_hooks()


class DataformerEvaluatorFactory(DatamodelEvaluatorFactory):
    """Datablock factory for Transformer-specific evaluators (Dataformer)."""

    VERSION = 1
    Evaluator: type[DataformerEvaluator] = DataformerEvaluator

    @dataclass
    class VAR(DatamodelEvaluatorFactory.VAR):
        capture_blocks: list = field(default_factory=list)  # list[int] — transformer block indices
        cls_token_only: bool = False  # capture only CLS token activations

    # 1. Datablock Protocol Methods ─────────────────────────────────

    def __init__(self, *, capture_blocks=None, capture_layers=None, capture_final=True, cls_token_only=False, spec=None, **kwargs):
        spec = dict(spec) if spec is not None else {}
        if capture_blocks is not None:
            spec['capture_blocks'] = capture_blocks
        if capture_layers is not None:
            spec['capture_layers'] = capture_layers
        if capture_final is not True:
            spec['capture_final'] = capture_final
        if cls_token_only:
            spec['cls_token_only'] = cls_token_only
        super().__init__(spec=spec, **kwargs)

    def evaluator(self, *, device: str = "cuda", log: Logger | None = None) -> DataformerEvaluator:
        """Create a live `DataformerEvaluator`.

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


class DatafeatureTab(DatastreamTab):
    """A tab storing multi-layer feature activations captured by an evaluator.

    Inherits access to the slices of the upstream `sampletab`. Calling `dataset()`
    or `data()` with slice names present in `sampletab` seamlessly zips them in
    using the `ZipStreamingDataset` mechanism.
    """

    VERSION = 1
    LEGACY_NORM = False

    @dataclass
    class VAR(DatastreamTab.VAR):
        sampletab: DatastreamTab
        evaluator_factory: DatamodelEvaluatorFactory
        shard_size: int = 1024

    # 1. Datablock / Datastream Protocol Methods ─────────────────────

    def __init__(self, *args, device_batch_size: int = 64, device: str = "cuda", **kwargs):
        self.device = device
        self.device_batch_size = device_batch_size
        super().__init__(*args, **kwargs)

    def __post_init__(self):
        super().__post_init__()
        factory = getattr(self.var, 'evaluator_factory', None)
        if hasattr(factory, 'layer_names'):
            self._feature_names = list(factory.layer_names)
        elif factory is not None and hasattr(factory, 'var'):
            names = []
            for block in getattr(factory.var, 'capture_blocks', []):
                names.append(f"block.{block}")
            for layer in getattr(factory.var, 'capture_layers', []):
                names.append(layer)
            if getattr(factory.var, 'capture_final', True):
                names.append('final')
            self._feature_names = names
        else:
            self._feature_names = []

        if not getattr(self, 'SLICES', None):
            self.SLICES = tuple(f"features_{name.replace('.', '_')}" for name in self._feature_names)

    def __build__(self, evaluator=None, sampletab=None):
        if evaluator is None:
            evaluator = self.var.evaluator_factory.evaluator(device=self.device, log=self.log)
        if sampletab is None:
            sampletab = self.var.sampletab

        feature_names = evaluator.layer_names
        self._feature_names = feature_names
        self.SLICES = tuple(f"features_{name.replace('.', '_')}" for name in feature_names)

        slice_specs = {
            f"features_{name.replace('.', '_')}": (f"features_{name.replace('.', '_')}.mds", "ndarray:float32")
            for name in feature_names
        }

        writers = self.slice_writers(slice_specs, shard_size=self.var.shard_size)
        try:
            sample_data = sampletab.data(concat=True)
            input_key = next(iter(sample_data.keys()))
            inputs = sample_data[input_key]
            if isinstance(inputs, dict):
                input_key_inner = next(iter(inputs.keys()))
                inputs = inputs[input_key_inner]

            if not hasattr(inputs, 'shape') or not hasattr(inputs, 'to'):
                inputs = torch.tensor(np.array(inputs))

            n_samples = len(inputs)
            n_batches = math.ceil(n_samples / self.device_batch_size)

            for k in range(n_batches):
                m = k * self.device_batch_size
                n = min((k + 1) * self.device_batch_size, n_samples)
                batch = inputs[m:n].to(self.device)
                result = evaluator(batch)

                for name in feature_names:
                    col_name = f"features_{name.replace('.', '_')}"
                    if name in result:
                        arr = result[name].cpu().numpy().astype(np.float32)
                        for i in range(len(arr)):
                            writers[col_name].write({col_name: arr[i]})
                evaluator.clear()
        finally:
            for w in writers.values():
                w.finish()
        return self

    def dataset(
        self,
        *slices,
        mode='map',
        columns=None,
        shared=None,
        validate_shared=False,
        on_conflict='last',
        skip_none=True,
        zip_validator=None,
        **kwargs,
    ):
        if mode not in ('map', 'iter'):
            raise ValueError(f"{self.__class__.__name__}.dataset: mode must be 'map' or 'iter', got {mode!r}")

        requested = list(slices)
        if len(requested) == 1 and isinstance(requested[0], (tuple, list)):
            requested = list(requested[0])

        if not requested:
            requested = list(self.slices)

        datasets = []
        for s in requested:
            if s in self.slices:
                datasets.append(self.datastream(s, **kwargs))
            elif self.sampletab is not None and s in self.sampletab.slices:
                datasets.append(self.sampletab.datastream(s, **kwargs))
            else:
                avail = list(self.slices) + (list(self.sampletab.slices) if self.sampletab else [])
                raise KeyError(
                    f"{self.__class__.__name__}: unknown slice {s!r}; available slices are {avail}"
                )

        zip_cls = ZipStreamingDataset if mode == 'map' else ZipIterableStreamingDatasets
        return zip_cls(
            *datasets,
            columns=columns,
            shared=shared,
            validate_shared=validate_shared,
            on_conflict=on_conflict,
            skip_none=skip_none,
            zip_validator=zip_validator,
        )

    def data(self, *slices, concat=True):
        requested = list(slices)
        if len(requested) == 1 and isinstance(requested[0], (tuple, list)):
            requested = list(requested[0])

        if not requested:
            requested = list(self.slices)

        result = {}
        for s in requested:
            if s in self.slices:
                result[s] = super().data(s, concat=concat)[s]
            elif self.sampletab is not None and s in self.sampletab.slices:
                result[s] = self.sampletab.data(s, concat=concat)[s]
            else:
                avail = list(self.slices) + (list(self.sampletab.slices) if self.sampletab else [])
                raise KeyError(
                    f"{self.__class__.__name__}: unknown slice {s!r}; available slices are {avail}"
                )
        return result

    # 2. Properties and Accessors ───────────────────────────────────

    @property
    def sampletab(self) -> DatastreamTab:
        return self.var.sampletab

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    @property
    def available_slices(self) -> tuple[str, ...]:
        own = tuple(self.slices)
        upstream = tuple(self.sampletab.slices) if self.sampletab is not None else ()
        return own + upstream

    def __len__(self) -> int:
        return len(self.sampletab)


class DatafeatureTable(DatastreamTable):
    """A table of `DatafeatureTab` blocks built across a `DatastreamTable`."""

    TAB = DatafeatureTab
    VERSION = 1
    LEGACY_NORM = False

    @dataclass
    class VAR(DatastreamTable.VAR):
        sampletable: DatastreamTable
        evaluator_factory: DatamodelEvaluatorFactory
        shard_size: int = 1024

    # 1. Datablock / Datastack Protocol Methods ─────────────────────

    def __init__(self, *args, device_batch_size: int = 64, devices: list | None = None, **kwargs):
        self._devices = devices or ["cuda"]
        self.device_batch_size = device_batch_size
        super().__init__(*args, **kwargs)

    def __block__(self, idx: int, sampletable=None, device: str = "cuda", sampletab=None) -> DatafeatureTab:
        if sampletab is None:
            if sampletable is None:
                sampletable = self.var.sampletable
            sampletab = sampletable.tab(idx)
        return self.TAB(
            url=self.url,
            spec=dict(
                sampletab=dbx.quote(sampletab),
                evaluator_factory=self.spec['evaluator_factory'],
                shard_size=self.var.shard_size,
            ),
            device_batch_size=self.device_batch_size,
            device=device,
            revision=self.revision,
            tag=sampletab.tag,
        )

    def __split__(self, *args, **kwargs):
        devices = self._devices
        sampletable = self.var.sampletable
        sampletabs = [sampletable.tab(idx) for idx in range(self.n_tabs)]

        callable_kwargs = dict(
            build=True,
            sampletabs=sampletabs,
            evaluator_factory=self.var.evaluator_factory,
        )
        n_workers = len(devices)
        chunk_boundaries = np.array_split(range(self.n_tabs), n_workers)
        block_device = {}
        for worker_idx, chunk in enumerate(chunk_boundaries):
            dev = devices[worker_idx % len(devices)]
            for idx in chunk:
                block_device[idx] = dev
        makers = [
            self.BlockMaker(idx, device=block_device[idx])
            for idx in range(self.n_tabs)
        ]
        return makers, callable_kwargs

    def dataset(
        self,
        *slices,
        mode='map',
        columns=None,
        shared=None,
        validate_shared=False,
        on_conflict='last',
        skip_none=True,
        zip_validator=None,
        **kwargs,
    ):
        if mode not in ('map', 'iter'):
            raise ValueError(f"{self.__class__.__name__}.dataset: mode must be 'map' or 'iter', got {mode!r}")

        requested = list(slices)
        if len(requested) == 1 and isinstance(requested[0], (tuple, list)):
            requested = list(requested[0])

        if not requested:
            requested = list(self.slices)

        datasets = []
        for s in requested:
            if s in self.slices:
                datasets.append(self.datastream(s, **kwargs))
            elif self.sampletable is not None and s in self.sampletable.slices:
                datasets.append(self.sampletable.datastream(s, **kwargs))
            else:
                avail = list(self.slices) + (list(self.sampletable.slices) if self.sampletable else [])
                raise KeyError(
                    f"{self.__class__.__name__}: unknown slice {s!r}; available slices are {avail}"
                )

        zip_cls = ZipStreamingDataset if mode == 'map' else ZipIterableStreamingDatasets
        return zip_cls(
            *datasets,
            columns=columns,
            shared=shared,
            validate_shared=validate_shared,
            on_conflict=on_conflict,
            skip_none=skip_none,
            zip_validator=zip_validator,
        )

    def data(self, *slices, concat=True):
        requested = list(slices)
        if len(requested) == 1 and isinstance(requested[0], (tuple, list)):
            requested = list(requested[0])

        if not requested:
            requested = list(self.slices)

        result = {}
        for s in requested:
            if s in self.slices:
                result[s] = super().data(s, concat=concat)[s]
            elif self.sampletable is not None and s in self.sampletable.slices:
                result[s] = self.sampletable.data(s, concat=concat)[s]
            else:
                avail = list(self.slices) + (list(self.sampletable.slices) if self.sampletable else [])
                raise KeyError(
                    f"{self.__class__.__name__}: unknown slice {s!r}; available slices are {avail}"
                )
        return result

    # 2. Properties and Accessors ───────────────────────────────────

    @property
    def sampletable(self) -> DatastreamTable:
        return self.var.sampletable

    @property
    def n_tabs(self) -> int:
        return self.sampletable.n_tabs

    @property
    def available_slices(self) -> tuple[str, ...]:
        own = tuple(self.slices)
        upstream = tuple(self.sampletable.slices) if self.sampletable is not None else ()
        return own + upstream


class BipolarDatafeatureTab(DatastreamTab):
    """Bipolar (median-thresholded) encoding of a `DatafeatureTab`.

    Maps continuous features to ``{-1, +1}^d`` via ``sign(features - median)``,
    and computes a tab-level bipolar signature ``{-1, 0, +1}^d`` by thresholding the mean.
    """

    VERSION = 1
    LEGACY_NORM = False
    SLICES = ('bipolar_features', 'tab_bipolar_features')

    @dataclass
    class VAR(DatastreamTab.VAR):
        featuretab: DatafeatureTab
        layer: str = 'final'
        threshold: float = 0.5
        ternarize: bool = False

    # 1. Datablock / Datastream Protocol Methods ─────────────────────

    def __build__(self, median=None):
        layer = self.var.layer
        col_feature = f"features_{layer.replace('.', '_')}"
        if col_feature in self.featuretab.slices:
            raw_data = self.featuretab.data(col_feature, concat=True)[col_feature]
        else:
            raw_data = self.featuretab.data(concat=True)
            raw_data = raw_data[next(iter(raw_data.keys()))]

        if hasattr(raw_data, 'numpy'):
            features = raw_data.numpy()
        else:
            features = np.array(raw_data)

        if median is None:
            median = np.median(features, axis=0)

        tile_bipolar = np.sign(features - median).astype(np.int8)
        tile_bipolar[tile_bipolar == 0] = 1

        if self.var.ternarize:
            tab_mean = tile_bipolar.astype(np.float32).mean(axis=0)
            uncertain = (np.round(tab_mean).astype(np.int8) == 0)
            tile_bipolar[:, uncertain] = 0

        tab_mean = tile_bipolar.astype(np.float32).mean(axis=0)
        thresh = self.var.threshold
        tab_bipolar = np.where(np.abs(tab_mean) >= thresh, np.sign(tab_mean), 0).astype(np.int8)

        slice_specs = {
            'bipolar_features': ('bipolar_features.mds', 'ndarray:int8'),
            'tab_bipolar_features': ('tab_bipolar_features.mds', 'ndarray:int8'),
        }
        writers = self.slice_writers(slice_specs)
        try:
            for i in range(len(tile_bipolar)):
                writers['bipolar_features'].write({'bipolar_features': tile_bipolar[i]})
                writers['tab_bipolar_features'].write({'tab_bipolar_features': tab_bipolar})
        finally:
            for w in writers.values():
                w.finish()
        return self

    def dataset(
        self,
        *slices,
        mode='map',
        columns=None,
        shared=None,
        validate_shared=False,
        on_conflict='last',
        skip_none=True,
        zip_validator=None,
        **kwargs,
    ):
        if mode not in ('map', 'iter'):
            raise ValueError(f"{self.__class__.__name__}.dataset: mode must be 'map' or 'iter', got {mode!r}")

        requested = list(slices)
        if len(requested) == 1 and isinstance(requested[0], (tuple, list)):
            requested = list(requested[0])

        if not requested:
            requested = list(self.slices)

        datasets = []
        for s in requested:
            if s in self.slices:
                datasets.append(self.datastream(s, **kwargs))
            elif self.featuretab is not None and s in self.featuretab.available_slices:
                datasets.append(self.featuretab.dataset(s, mode=mode, **kwargs))
            else:
                avail = list(self.slices) + (list(self.featuretab.available_slices) if self.featuretab else [])
                raise KeyError(
                    f"{self.__class__.__name__}: unknown slice {s!r}; available slices are {avail}"
                )

        zip_cls = ZipStreamingDataset if mode == 'map' else ZipIterableStreamingDatasets
        return zip_cls(
            *datasets,
            columns=columns,
            shared=shared,
            validate_shared=validate_shared,
            on_conflict=on_conflict,
            skip_none=skip_none,
            zip_validator=zip_validator,
        )

    def data(self, *slices, concat=True):
        requested = list(slices)
        if len(requested) == 1 and isinstance(requested[0], (tuple, list)):
            requested = list(requested[0])

        if not requested:
            requested = list(self.slices)

        result = {}
        for s in requested:
            if s in self.slices:
                result[s] = super().data(s, concat=concat)[s]
            elif self.featuretab is not None and s in self.featuretab.available_slices:
                result[s] = self.featuretab.data(s, concat=concat)[s]
            else:
                avail = list(self.slices) + (list(self.featuretab.available_slices) if self.featuretab else [])
                raise KeyError(
                    f"{self.__class__.__name__}: unknown slice {s!r}; available slices are {avail}"
                )
        return result

    # 2. Properties and Accessors ───────────────────────────────────

    @property
    def featuretab(self) -> DatafeatureTab:
        return self.var.featuretab

    @property
    def sampletab(self) -> DatastreamTab | None:
        return getattr(self.featuretab, 'sampletab', None)

    @property
    def available_slices(self) -> tuple[str, ...]:
        own = tuple(self.slices)
        upstream = tuple(self.featuretab.available_slices) if self.featuretab is not None else ()
        return own + upstream

    def __len__(self) -> int:
        return len(self.featuretab)


class BipolarDatafeatureTable(DatastreamTable):
    """A table of `BipolarDatafeatureTab` blocks built over a `DatafeatureTable`."""

    TAB = BipolarDatafeatureTab
    VERSION = 1
    LEGACY_NORM = False

    @dataclass
    class VAR(DatastreamTable.VAR):
        featuretable: DatafeatureTable
        layer: str = 'final'
        threshold: float = 0.5
        ternarize: bool = False

    # 1. Datablock / Datastack Protocol Methods ─────────────────────

    def __block__(self, idx: int, featuretable=None, featuretab=None) -> BipolarDatafeatureTab:
        if featuretab is None:
            if featuretable is None:
                featuretable = self.var.featuretable
            featuretab = featuretable.tab(idx)
        return self.TAB(
            url=self.url,
            spec=dict(
                featuretab=dbx.quote(featuretab),
                layer=self.var.layer,
                threshold=self.var.threshold,
                ternarize=self.var.ternarize,
            ),
            revision=self.revision,
            tag=featuretab.tag,
        )

    def dataset(
        self,
        *slices,
        mode='map',
        columns=None,
        shared=None,
        validate_shared=False,
        on_conflict='last',
        skip_none=True,
        zip_validator=None,
        **kwargs,
    ):
        if mode not in ('map', 'iter'):
            raise ValueError(f"{self.__class__.__name__}.dataset: mode must be 'map' or 'iter', got {mode!r}")

        requested = list(slices)
        if len(requested) == 1 and isinstance(requested[0], (tuple, list)):
            requested = list(requested[0])

        if not requested:
            requested = list(self.slices)

        datasets = []
        for s in requested:
            if s in self.slices:
                datasets.append(self.datastream(s, **kwargs))
            elif self.featuretable is not None and s in self.featuretable.available_slices:
                datasets.append(self.featuretable.dataset(s, mode=mode, **kwargs))
            else:
                avail = list(self.slices) + (list(self.featuretable.available_slices) if self.featuretable else [])
                raise KeyError(
                    f"{self.__class__.__name__}: unknown slice {s!r}; available slices are {avail}"
                )

        zip_cls = ZipStreamingDataset if mode == 'map' else ZipIterableStreamingDatasets
        return zip_cls(
            *datasets,
            columns=columns,
            shared=shared,
            validate_shared=validate_shared,
            on_conflict=on_conflict,
            skip_none=skip_none,
            zip_validator=zip_validator,
        )

    def data(self, *slices, concat=True):
        requested = list(slices)
        if len(requested) == 1 and isinstance(requested[0], (tuple, list)):
            requested = list(requested[0])

        if not requested:
            requested = list(self.slices)

        result = {}
        for s in requested:
            if s in self.slices:
                result[s] = super().data(s, concat=concat)[s]
            elif self.featuretable is not None and s in self.featuretable.available_slices:
                result[s] = self.featuretable.data(s, concat=concat)[s]
            else:
                avail = list(self.slices) + (list(self.featuretable.available_slices) if self.featuretable else [])
                raise KeyError(
                    f"{self.__class__.__name__}: unknown slice {s!r}; available slices are {avail}"
                )
        return result

    # 2. Properties and Accessors ───────────────────────────────────

    @property
    def featuretable(self) -> DatafeatureTable:
        return self.var.featuretable

    @property
    def sampletable(self) -> DatastreamTable | None:
        return getattr(self.featuretable, 'sampletable', None)

    @property
    def n_tabs(self) -> int:
        return self.featuretable.n_tabs

    @property
    def available_slices(self) -> tuple[str, ...]:
        own = tuple(self.slices)
        upstream = tuple(self.featuretable.available_slices) if self.featuretable is not None else ()
        return own + upstream

