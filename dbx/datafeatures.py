"""dbx.datafeatures — Datablock / Datastack feature tables and bipolar encodings."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

try:
    import torch
except ImportError:
    torch = None

import dbx
from dbx.datamodels import DatamodelEvaluatorFactory
from dbx.datastreams import (
    DatastreamTab,
    DatastreamTable,
    ZipStreamingDataset,
    ZipIterableStreamingDatasets,
    DIRTOPIC,
)


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

        topics = dict(getattr(self, 'TOPICS', {}))
        topics[self.DATA] = {name: DIRTOPIC for name in self.SLICES}
        self.TOPICS = topics

    def __build__(self, evaluator=None, sampletab=None):
        if evaluator is None:
            evaluator = self.var.evaluator_factory.evaluator(device=self.device, log=self.log)
        if sampletab is None:
            sampletab = self.var.sampletab

        feature_names = evaluator.layer_names
        self._feature_names = feature_names
        self.SLICES = tuple(f"features_{name.replace('.', '_')}" for name in feature_names)
        topics = dict(getattr(self, 'TOPICS', {}))
        topics[self.DATA] = {name: DIRTOPIC for name in self.SLICES}
        self.TOPICS = topics

        slice_specs = {
            f"features_{name.replace('.', '_')}": {f"features_{name.replace('.', '_')}": "ndarray:float32"}
            for name in feature_names
        }

        with self.slice_writers(slice_specs, shard_size=self.var.shard_size) as writers:
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

    def __post_init__(self):
        super().__post_init__()
        factory = getattr(self.var, 'evaluator_factory', None)
        if hasattr(factory, 'layer_names'):
            feature_names = list(factory.layer_names)
        elif factory is not None and hasattr(factory, 'var'):
            names = []
            for block in getattr(factory.var, 'capture_blocks', []):
                names.append(f"block.{block}")
            for layer in getattr(factory.var, 'capture_layers', []):
                names.append(layer)
            if getattr(factory.var, 'capture_final', True):
                names.append('final')
            feature_names = names
        else:
            feature_names = []

        if not getattr(self, 'SLICES', None):
            self.SLICES = tuple(f"features_{name.replace('.', '_')}" for name in feature_names)

        topics = dict(getattr(self, 'TOPICS', {}))
        topics[self.DATA] = {name: DIRTOPIC for name in self.SLICES}
        self.TOPICS = topics

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
            'bipolar_features': {'bipolar_features': 'ndarray:int8'},
            'tab_bipolar_features': {'tab_bipolar_features': 'ndarray:int8'},
        }
        with self.slice_writers(slice_specs) as writers:
            for i in range(len(tile_bipolar)):
                writers['bipolar_features'].write({'bipolar_features': tile_bipolar[i]})
                writers['tab_bipolar_features'].write({'tab_bipolar_features': tab_bipolar})
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
