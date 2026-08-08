"""dbx.datafeatures — Datablock / Datastack feature tables and bipolar encodings."""

from __future__ import annotations

from dataclasses import dataclass
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
from dbx.datamodels import DatamodelEvaluatorFactory
from dbx.datapoints import (
    DatapointTab,
    DatapointTable,
    DIRTOPIC,
)
from dbx.datastreams import (
    ZipStreamingDataset,
    ZipIterableStreamingDatasets,
    concat_data,
)


def _extract_slice_data(res, slice_name):
    if isinstance(res, dict) and slice_name in res:
        return res[slice_name]
    return res


class DatafeatureTab(DatapointTab):
    """A tab storing multi-layer feature activations captured by an evaluator.

    Inherits access to the slices of the upstream `sampletab`. Calling `dataset()`
    or `data()` with slice names present in `sampletab` seamlessly zips them in
    using the `ZipStreamingDataset` mechanism.
    """

    VERSION = 1

    @dataclass
    class VAR(DatapointTab.VAR):
        sampletab: DatapointTab
        evaluator_factory: DatamodelEvaluatorFactory
        signal: tuple[str, str] | None = None
        features: dict | None = None
        shard_size_limit_bytes: int = 1 << 26  # 64 MiB default, in bytes

    # 1. Datablock / Datastream Protocol Methods ─────────────────────

    def __init__(self, *args, device_batch_size: int = 64, device: str = "cuda", **kwargs):
        self.device = device
        self.device_batch_size = device_batch_size
        super().__init__(*args, **kwargs)

    def __post_init__(self):
        super().__post_init__()
        factory = self.var.evaluator_factory
        layer_names = factory.layer_names
        if self.var.features is not None:
            self._feature_map = dict(self.var.features)
        else:
            self._feature_map = {
                f"features_{name.replace('.', '_')}" if not name.startswith("features_") else name: name
                for name in layer_names
            }
        self.SLICES = tuple(self._feature_map.keys())
        self.SLICE_DTYPES = {name: "ndarray:float32" for name in self.SLICES}
        topics = dict(self.TOPICS)
        topics[self.DATA] = {name: DIRTOPIC for name in self.SLICES}
        self.TOPICS = topics

    def __build__(self):
        evaluator = self.var.evaluator_factory.evaluator(device=self.device, log=self.log)
        sampletab = self.var.sampletab

        slice_specs = {
            col_name: {col_name: "ndarray:float32"}
            for col_name in self.SLICES
        }

        with self.slice_writers(slice_specs, size_limit=self.var.shard_size_limit_bytes) as writers:
            if self.var.signal is not None:
                if isinstance(self.var.signal, (list, tuple)):
                    slice_name = self.var.signal[0]
                    col_name = self.var.signal[1] if len(self.var.signal) > 1 else self.var.signal[0]
                else:
                    slice_name = str(self.var.signal)
                    col_name = str(self.var.signal)

                sample_data = sampletab.data(slice_name, concat=True)
                slice_data = sample_data.get(slice_name, next(iter(sample_data.values())))
                if isinstance(slice_data, dict):
                    inputs = slice_data.get(col_name, next(iter(slice_data.values())))
                else:
                    inputs = slice_data
            else:
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

                for col_name, layer_name in self._feature_map.items():
                    if layer_name in result:
                        arr = result[layer_name].cpu().numpy().astype(np.float32)
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
                result[s] = _extract_slice_data(super().data(s, concat=concat), s)
            elif self.sampletab is not None and s in self.sampletab.slices:
                result[s] = _extract_slice_data(self.sampletab.data(s, concat=concat), s)
            else:
                avail = list(self.slices) + (list(self.sampletab.slices) if self.sampletab else [])
                raise KeyError(
                    f"{self.__class__.__name__}: unknown slice {s!r}; available slices are {avail}"
                )
        return result

    # 2. Properties and Accessors ───────────────────────────────────

    @property
    def sampletab(self) -> DatapointTab:
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


class DatafeatureTable(DatapointTable):
    """A table of `DatafeatureTab` blocks built across a `DatapointTable`."""

    TAB = DatafeatureTab
    VERSION = 1

    @dataclass
    class VAR(DatapointTable.VAR):
        sampletable: DatapointTable
        evaluator_factory: DatamodelEvaluatorFactory
        signal: tuple[str, str] | None = None
        features: dict | None = None
        shard_size_limit_bytes: int = 1 << 26  # 64 MiB default, in bytes

    # 1. Datablock / Datastack Protocol Methods ─────────────────────

    def __init__(self, *args, device_batch_size: int = 64, devices: list | None = None, **kwargs):
        self._devices = devices or ["cuda"]
        self.device_batch_size = device_batch_size
        super().__init__(*args, **kwargs)

    def __post_init__(self):
        super().__post_init__()
        factory = self.var.evaluator_factory
        layer_names = factory.layer_names
        if self.var.features is not None:
            self._feature_map = dict(self.var.features)
        else:
            self._feature_map = {
                f"features_{name.replace('.', '_')}" if not name.startswith("features_") else name: name
                for name in layer_names
            }
        self.SLICES = tuple(self._feature_map.keys())
        self.SLICE_DTYPES = {name: "ndarray:float32" for name in self.SLICES}
        topics = dict(self.TOPICS)
        topics[self.DATA] = {name: DIRTOPIC for name in self.SLICES}
        self.TOPICS = topics

    class BlockMaker(Datastack.BlockMaker):
        """Lightweight callable that forms and optionally builds a block."""
        def __init__(self, idx: int, *, device: str = "cuda"):
            super().__init__(idx)
            self.device = device

        def __call__(self, table, *, build=True):
            tab = table.__block__(self.idx, device=self.device)
            tab.keyby = table.keyby
            skipped = tab.valid()
            if build:
                tab.build()
            result = {'tab_idx': self.idx, 'tag': tab.tag, 'skipped': skipped}
            del tab
            gc.collect()
            return result

    def __tab__(self, idx: int, device: str = "cuda", tag=None) -> DatafeatureTab:
        sampletab = self.var.sampletable.tab(idx)
        spec = dict(
            sampletab=dbx.quote(sampletab),
            evaluator_factory=self.spec['evaluator_factory'],
        )
        if self.var.signal is not None:
            spec['signal'] = self.var.signal
        if self.var.features is not None:
            spec['features'] = self.var.features
        spec['shard_size_limit_bytes'] = self.var.shard_size_limit_bytes
        return self.TAB(
            url=self.path('tabs'),
            storage_options=self.storage_options,
            capture_output=self.capture_output,
            cache=getattr(self, 'cache', None),
            cache_limit=getattr(self, 'cache_limit', None),
            verbose=False,
            spec=spec,
            device_batch_size=self.device_batch_size,
            device=device,
            revision=self.revision,
            tag=tag if tag is not None else sampletab.tag,
        )

    def __block__(self, idx: int, **kwargs) -> DatafeatureTab:
        return self.__tab__(idx, **kwargs)

    def __split__(self, *args, **kwargs):
        devices = self._devices

        callable_kwargs = dict(build=True)
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
                tab_data = [self.tab(i).data(s, concat=concat)[s] for i in range(self.n_tabs)]
                if concat:
                    if isinstance(tab_data[0], np.ndarray):
                        result[s] = np.concatenate(tab_data, axis=0)
                    else:
                        result[s] = concat_data(tab_data, dtype=self.SLICE_DTYPES.get(s))
                else:
                    result[s] = tab_data
            elif self.sampletable is not None and s in self.sampletable.slices:
                result[s] = _extract_slice_data(self.sampletable.data(s, concat=concat), s)
            else:
                avail = list(self.slices) + (list(self.sampletable.slices) if self.sampletable else [])
                raise KeyError(
                    f"{self.__class__.__name__}: unknown slice {s!r}; available slices are {avail}"
                )
        return result

    # 2. Properties and Accessors ───────────────────────────────────

    @property
    def sampletable(self) -> DatapointTable:
        return self.var.sampletable

    @property
    def n_tabs(self) -> int:
        return self.sampletable.n_tabs

    @property
    def available_slices(self) -> tuple[str, ...]:
        own = tuple(self.slices)
        upstream = tuple(self.sampletable.slices) if self.sampletable is not None else ()
        return own + upstream


class BipolarDatafeatureTab(DatapointTab):
    """Bipolar (median-thresholded) encoding of a `DatafeatureTab`.

    Maps continuous features to ``{-1, +1}^d`` via ``sign(features - median)``,
    and computes a tab-level bipolar signature ``{-1, 0, +1}^d`` by thresholding the mean.
    """

    VERSION = 1
    SLICES = ('bipolar_features', 'tab_bipolar_features')

    @dataclass
    class VAR(DatapointTab.VAR):
        featuretab: DatafeatureTab
        layer: str = 'final'
        threshold: float = 0.5
        ternarize: bool = False

    # 1. Datablock / Datastream Protocol Methods ─────────────────────

    def __build__(self):
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
                result[s] = _extract_slice_data(super().data(s, concat=concat), s)
            elif self.featuretab is not None and s in self.featuretab.available_slices:
                result[s] = _extract_slice_data(self.featuretab.data(s, concat=concat), s)
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
    def sampletab(self) -> DatapointTab | None:
        return self.featuretab.sampletab

    @property
    def available_slices(self) -> tuple[str, ...]:
        own = tuple(self.slices)
        upstream = tuple(self.featuretab.available_slices) if self.featuretab is not None else ()
        return own + upstream

    def __len__(self) -> int:
        return len(self.featuretab)


class BipolarDatafeatureTable(DatapointTable):
    """A table of `BipolarDatafeatureTab` blocks built over a `DatafeatureTable`."""

    TAB = BipolarDatafeatureTab
    VERSION = 1

    @dataclass
    class VAR(DatapointTable.VAR):
        featuretable: DatafeatureTable
        layer: str = 'final'
        threshold: float = 0.5
        ternarize: bool = False

    # 1. Datablock / Datastack Protocol Methods ─────────────────────

    def __tab__(self, idx: int, tag=None) -> BipolarDatafeatureTab:
        featuretab = self.var.featuretable.tab(idx)
        return self.TAB(
            url=self.path('tabs'),
            storage_options=self.storage_options,
            capture_output=self.capture_output,
            cache=getattr(self, 'cache', None),
            cache_limit=getattr(self, 'cache_limit', None),
            verbose=False,
            spec=dict(
                featuretab=dbx.quote(featuretab),
                layer=self.var.layer,
                threshold=self.var.threshold,
                ternarize=self.var.ternarize,
            ),
            revision=self.revision,
            tag=tag if tag is not None else featuretab.tag,
        )

    def __block__(self, idx: int, **kwargs) -> BipolarDatafeatureTab:
        return self.__tab__(idx, **kwargs)

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
                tab_data = [self.tab(i).data(s, concat=concat)[s] for i in range(self.n_tabs)]
                if concat:
                    if isinstance(tab_data[0], np.ndarray):
                        result[s] = np.concatenate(tab_data, axis=0)
                    else:
                        result[s] = concat_data(tab_data, dtype=self.SLICE_DTYPES.get(s))
                else:
                    result[s] = tab_data
            elif self.featuretable is not None and s in self.featuretable.available_slices:
                result[s] = _extract_slice_data(self.featuretable.data(s, concat=concat), s)
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
    def sampletable(self) -> DatapointTable | None:
        return self.featuretable.sampletable

    @property
    def n_tabs(self) -> int:
        return self.featuretable.n_tabs

    @property
    def available_slices(self) -> tuple[str, ...]:
        own = tuple(self.slices)
        upstream = tuple(self.featuretable.available_slices) if self.featuretable is not None else ()
        return own + upstream


class Datacollator(Datablock):
    """Callable Datablock for collating batches of datapoint dicts into signal and label arrays.

    `Datacollator` has no `TOPICS` (it does not build or persist files).
    When invoked as `collator(datapoints)`, it extracts the specified `signals` and `labels`
    `(slice, column)` pairs from each datapoint dict in `datapoints`, concatenating/stacking
    signal tensors along a new dimension 1 for each datapoint, and concatenating datapoints
    along dimension 0 (batch dimension).
    """

    TOPICS = {}

    @dataclass
    class VAR(Datablock.VAR):
        signals: list[tuple[str, str]]
        labels: list[tuple[str, str]]

    def __call__(self, datapoints: list[dict]) -> dict[str, np.ndarray]:
        """Collate a batch of datapoint dicts into a dict with 'signal' and 'label' keys.

        Parameters
        ----------
        datapoints : list[dict]
            List of datapoint sample dicts containing slice/column features and labels.

        Returns
        -------
        dict[str, np.ndarray]
            Dict with keys `'signal'` and `'label'` mapping to collated arrays.
        """
        return {
            'signal': self._collate_pairs(datapoints, self.var.signals),
            'label': self._collate_pairs(datapoints, self.var.labels),
        }

    @staticmethod
    def _norm_pair(pair: Any) -> tuple[str, str]:
        if isinstance(pair, (list, tuple)):
            if len(pair) >= 2:
                return str(pair[0]), str(pair[1])
            elif len(pair) == 1:
                return str(pair[0]), str(pair[0])
        return str(pair), str(pair)

    def _collate_pairs(self, datapoints: list[dict], pairs: list[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.array([])

        norm_pairs = [self._norm_pair(p) for p in pairs]
        batch_items = []

        for dp in datapoints:
            dp_signals = []
            for s_name, c_name in norm_pairs:
                if isinstance(dp, dict) and s_name in dp:
                    val = dp[s_name]
                    if isinstance(val, dict):
                        val = val.get(c_name, next(iter(val.values())))
                else:
                    val = dp

                if torch is not None and isinstance(val, torch.Tensor):
                    val = val.detach().cpu().numpy()
                else:
                    val = np.array(val)

                dp_signals.append(val)

            norm_signals = []
            for sig in dp_signals:
                if sig.ndim == 0:
                    norm_signals.append(sig.reshape(1, 1))
                elif sig.ndim == 1:
                    norm_signals.append(sig.reshape(1, -1))
                else:
                    norm_signals.append(sig)

            if norm_signals[0].ndim == 2 and all(x.ndim == 2 for x in norm_signals):
                try:
                    dp_tensor = np.stack(norm_signals, axis=1)
                except ValueError:
                    dp_tensor = np.concatenate(norm_signals, axis=1)
            else:
                dp_tensor = np.stack(norm_signals, axis=1)

            batch_items.append(dp_tensor)

        try:
            return np.stack(batch_items, axis=0)
        except ValueError:
            return np.concatenate(batch_items, axis=0)
