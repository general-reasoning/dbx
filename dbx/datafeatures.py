"""dbx.datafeatures — Datablock / Datastack feature tables and bipolar encodings."""

from __future__ import annotations

from dataclasses import dataclass, field
import gc
import math
import warnings
from typing import Any

import numpy as np

# Suppress PyTorch non-writable NumPy array UserWarning for read-only streaming buffers
warnings.filterwarnings("ignore", category=UserWarning, message=".*given NumPy array is not writable.*")

try:
    import torch
except ImportError:
    torch = None

import dbx
from dbx.datablocks import Datablock, Datastack, DIRTOPIC
from dbx.datamodels import DatamodelEvaluatorFactory
from dbx.datapoints import (
    DatapointTab,
    DatapointTable,
    DIRTOPIC,
    SLICETOPIC,
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


def _flatten_item(x):
    while isinstance(x, dict) and len(x) > 0:
        x = next(iter(x.values()))
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    if isinstance(x, np.ndarray):
        if x.dtype == np.object_:
            return np.array([_flatten_item(item) for item in x], dtype=np.float32)
        return x.astype(np.float32) if x.dtype != np.float32 else x
    try:
        return np.asarray(x, dtype=np.float32)
    except Exception:
        return np.array(x)


def _to_tensor(inputs, device=None) -> torch.Tensor:
    if isinstance(inputs, torch.Tensor):
        return inputs.to(device) if device is not None else inputs

    if isinstance(inputs, dict) and len(inputs) > 0:
        inputs = next(iter(inputs.values()))
        return _to_tensor(inputs, device)

    if isinstance(inputs, np.ndarray):
        if inputs.dtype == np.object_:
            try:
                stacked = np.stack([_flatten_item(x) for x in inputs])
                t = torch.from_numpy(np.ascontiguousarray(stacked))
            except Exception:
                stacked = np.array([_flatten_item(x) for x in inputs], dtype=np.float32)
                t = torch.from_numpy(np.ascontiguousarray(stacked))
            return t.to(device) if device is not None else t
        t = torch.from_numpy(np.ascontiguousarray(inputs))
        return t.to(device) if device is not None else t

    if isinstance(inputs, (list, tuple)):
        if len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
            t = torch.stack(inputs)
            return t.to(device) if device is not None else t
        try:
            stacked = np.stack([_flatten_item(x) for x in inputs])
            t = torch.from_numpy(np.ascontiguousarray(stacked))
        except Exception:
            stacked = np.array([_flatten_item(x) for x in inputs], dtype=np.float32)
            t = torch.from_numpy(np.ascontiguousarray(stacked))
        return t.to(device) if device is not None else t

    try:
        t = torch.as_tensor(inputs)
    except Exception:
        t = torch.tensor(_flatten_item(inputs))
    return t.to(device) if device is not None else t


def _extract_pair_data(data_dict, pair: tuple[str, str]):
    s_name, c_name = pair[0], pair[1]
    if isinstance(data_dict, dict) and s_name in data_dict:
        val = data_dict[s_name]
        if isinstance(val, dict):
            if c_name in val:
                return val[c_name]
            col_key = c_name.replace('features_', '')
            if col_key in val:
                return val[col_key]
            return next(iter(val.values()))
        return val
    return data_dict


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
        length: int | None = None

    def __call__(self, datapoints: list[dict], *, strip_keys: bool = False, signal_only: bool = False) -> dict[str, np.ndarray] | tuple[np.ndarray, ...] | np.ndarray:
        """Collate a batch of datapoint dicts into collated arrays.

        Parameters
        ----------
        datapoints : list[dict]
            List of datapoint sample dicts containing slice/column features and labels.

        Returns
        -------
        dict[str, np.ndarray] | tuple[np.ndarray, ...] | np.ndarray
            Collated signals and labels, formatted according to VAR options (length, strip_keys, signal_only).
        """
        sig_arr = self._collate_pairs(datapoints, self.var.signals)
        lbl_arr = self._collate_pairs(datapoints, self.var.labels)

        length = self.var.length
        if length is not None:
            if hasattr(sig_arr, 'ndim') and sig_arr.ndim >= 1 and sig_arr.shape[-1] > length:
                sig_arr = sig_arr[..., :length]
            if hasattr(lbl_arr, 'ndim') and lbl_arr.ndim >= 1 and lbl_arr.shape[-1] > length:
                lbl_arr = lbl_arr[..., :length]

        if signal_only:
            if strip_keys:
                return sig_arr
            return {'signal': sig_arr}

        if strip_keys:
            if len(self.var.labels) > 0:
                return (sig_arr, lbl_arr)
            return (sig_arr,)

        res = {'signal': sig_arr}
        if len(self.var.labels) > 0:
            res['label'] = lbl_arr
        return res

    @property
    def slices(self):
        return list(set([p[0] for p in self.var.signals] + [p[0] for p in self.var.labels]))

    @property
    def signal_pairs(self) -> tuple[tuple[str, str], ...]:
        """The signal ``(slice, column)`` pairs, each in full two-part form.

        A pair may be declared as a bare name or a one-element sequence, both
        of which mean the column of the same name; this is what a caller that
        has to address the data itself -- a per-tab breakdown, a log line --
        reads, rather than normalizing ``var.signals`` again at each site.
        """
        return tuple(self._norm_pair(p) for p in self.var.signals)

    @property
    def label_pairs(self) -> tuple[tuple[str, str], ...]:
        """The label ``(slice, column)`` pairs, as :attr:`signal_pairs`."""
        return tuple(self._norm_pair(p) for p in self.var.labels)

    @staticmethod
    def _norm_pair(pair: Any) -> tuple[str, str]:
        if isinstance(pair, (list, tuple)):
            if len(pair) >= 2:
                return str(pair[0]), str(pair[1])
            elif len(pair) == 1:
                return str(pair[0]), str(pair[0])
        return str(pair), str(pair)

    @staticmethod
    def _as_array(val):
        if torch is not None and isinstance(val, torch.Tensor):
            return val.detach().cpu().numpy()
        arr = np.array(val)
        if hasattr(arr, 'flags') and not arr.flags.writeable:
            arr = np.copy(arr)
        return arr

    def _collate_batch(self, batch: dict, norm_pairs) -> np.ndarray:
        """Collate a ``{slice: data}`` mapping, in which the batch is ALREADY stacked.

        This is what ``data(*collator.slices, concat=True)`` hands back -- one
        entry per slice, each holding every sample of it -- as opposed to the
        list of per-sample dicts a DataLoader yields. Both reach __call__, and
        they are told apart by shape rather than by a flag, because the callers
        that pass a whole slice at a time (a feature build, a probe fit) are the
        same ones that pass batches elsewhere.

        One pair passes its array through untouched, so a single-signal collation
        keeps the shape the slice was written with -- which is what a model is
        then fed. Several are stacked along a new axis 1, mirroring the signals
        axis of the per-sample form.
        """
        arrays = []
        for s_name, c_name in norm_pairs:
            val = batch[s_name] if s_name in batch else batch
            if isinstance(val, dict):
                val = val.get(c_name, next(iter(val.values())))
            arrays.append(self._as_array(val))
        if len(arrays) == 1:
            return arrays[0]
        return np.stack(arrays, axis=1)

    def _collate_pairs(self, datapoints: list[dict], pairs: list[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.array([])

        norm_pairs = [self._norm_pair(p) for p in pairs]

        if isinstance(datapoints, dict):
            return self._collate_batch(datapoints, norm_pairs)

        batch_items = []

        for dp in datapoints:
            dp_signals = []
            for s_name, c_name in norm_pairs:
                val = dp.get(s_name, dp) if isinstance(dp, dict) else dp
                while isinstance(val, dict) and len(val) > 0:
                    if c_name and c_name in val:
                        val = val[c_name]
                    elif c_name and c_name.replace('features_', '') in val:
                        val = val[c_name.replace('features_', '')]
                    else:
                        val = next(iter(val.values()))

                dp_signals.append(self._as_array(val))

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


class DatafeatureTab(DatapointTab):
    """A tab storing multi-layer feature activations captured by an evaluator.

    Inherits access to the slices of the upstream `sampletab`. Calling `dataset()`
    or `data()` with slice names present in `sampletab` seamlessly zips them in
    using the `ZipStreamingDataset` mechanism.
    """

    VERSION = 1
    TOPICS = {'features': SLICETOPIC}

    @dataclass
    class VAR(Datablock.VAR):
        datapoint_tab: DatapointTab
        evaluator_factory: DatamodelEvaluatorFactory
        collator: Datacollator
        feature_namemap: dict[str, str] | None = None
        shard_size_limit_bytes: int = 1 << 26  # 64 MiB default, in bytes

    # 1. Datablock / Datastream Protocol Methods ─────────────────────

    def __init__(
        self,
        *args,
        device_batch_size: int = 64,
        device: str = "cuda",
        streaming: bool = False,
        dataloader_kwargs: dict | None = None,
        **kwargs,
    ):
        self.device_batch_size = device_batch_size
        self.device = device
        super().__init__(
            *args,
            streaming=streaming,
            dataloader_kwargs=dataloader_kwargs,
            **kwargs,
        )

    def __post_init__(self):
        super().__post_init__()
        self.streaming = getattr(self, 'streaming', False)
        self.dataloader_kwargs = getattr(self, 'dataloader_kwargs', None) or {}
        factory = self.var.evaluator_factory
        layer_names = factory.layer_names if factory is not None else []
        namemap = self.var.feature_namemap
        if namemap is not None:
            if isinstance(namemap, dict):
                self._feature_map = dict(namemap)
            elif isinstance(namemap, (list, tuple)):
                self._feature_map = {str(k): str(k) for k in namemap}
            else:
                self._feature_map = {str(namemap): str(namemap)}
        else:
            self._feature_map = {name: name for name in layer_names}

    def __build__(self):
        if self.streaming:
            return self.__build_streaming__()
        else:
            return self.__build_bulk__()

    def __build_bulk__(self):
        evaluator = self.var.evaluator_factory.evaluator(device=self.device, log=self.log)
        datapoint_tab = self.var.datapoint_tab

        columns_spec = {
            col_name: "ndarray:float32"
            for col_name in self._feature_map.keys()
        }
        slice_specs = {"features": columns_spec}

        collator = self.var.collator
        if collator is None:
            slice_name = 'tiles' if 'tiles' in datapoint_tab.slices else (datapoint_tab.slices[0] if datapoint_tab.slices else 'tiles')
            col_name = 'tile' if slice_name == 'tiles' else slice_name
            collator = Datacollator(spec=dict(signals=[(slice_name, col_name)], labels=[]))

        with self.slice_writers(slice_specs, size_limit=self.var.shard_size_limit_bytes) as writers:
            sample_data = datapoint_tab.data(*collator.slices, concat=True)
            inputs = collator(sample_data, signal_only=True, strip_keys=True)
            inputs = _to_tensor(inputs, "cpu")

            n_samples = len(inputs)
            n_batches = math.ceil(n_samples / self.device_batch_size)

            for k in range(n_batches):
                m = k * self.device_batch_size
                n = min((k + 1) * self.device_batch_size, n_samples)
                batch = inputs[m:n].to(self.device)
                result = evaluator(batch)

                batch_len = n - m
                batch_features = {
                    col_name: result[layer_name].cpu().numpy().astype(np.float32)
                    for col_name, layer_name in self._feature_map.items()
                    if layer_name in result
                }
                for i in range(batch_len):
                    writers['features'].write({col_name: arr[i] for col_name, arr in batch_features.items()})
                evaluator.clear()
                del batch, result, batch_features
                gc.collect()
            del inputs
            gc.collect()
        return self

    def __build_streaming__(self):
        warnings.filterwarnings("ignore", category=UserWarning, message=".*given NumPy array is not writable.*")
        evaluator = self.var.evaluator_factory.evaluator(device=self.device, log=self.log)
        datapoint_tab = self.var.datapoint_tab

        columns_spec = {
            col_name: "ndarray:float32"
            for col_name in self._feature_map.keys()
        }
        slice_specs = {"features": columns_spec}

        collator = self.var.collator
        if collator is None:
            slice_name = 'tiles' if 'tiles' in datapoint_tab.slices else (datapoint_tab.slices[0] if datapoint_tab.slices else 'tiles')
            col_name = 'tile' if slice_name == 'tiles' else slice_name
            collator = Datacollator(spec=dict(signals=[(slice_name, col_name)], labels=[]))

        dataset = datapoint_tab.dataset(*collator.slices)
        dl_kwargs = dict(self.dataloader_kwargs) if self.dataloader_kwargs else {}
        dl_kwargs.setdefault('batch_size', self.device_batch_size)
        dl_kwargs.setdefault('collate_fn', lambda batch: batch)

        dataloader = torch.utils.data.DataLoader(dataset, **dl_kwargs)

        with self.slice_writers(slice_specs, size_limit=self.var.shard_size_limit_bytes) as writers:
            for batch_data in dataloader:
                inputs = collator(batch_data, signal_only=True, strip_keys=True)
                batch = _to_tensor(inputs, self.device)
                if batch.ndim > 2 and batch.shape[1:3] == (1, 1):
                    batch = batch.squeeze(1).squeeze(1)
                result = evaluator(batch)

                batch_len = len(batch)
                batch_features = {
                    col_name: result[layer_name].cpu().numpy().astype(np.float32)
                    for col_name, layer_name in self._feature_map.items()
                    if layer_name in result
                }
                for i in range(batch_len):
                    writers['features'].write({col_name: arr[i] for col_name, arr in batch_features.items()})
                evaluator.clear()
                del batch_data, inputs, batch, result, batch_features
                gc.collect()
        return self

    def dataset(
        self,
        *slices,
        datapoint_slices: list | None = None,
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

        if datapoint_slices:
            for s in datapoint_slices:
                if s not in requested:
                    requested.append(s)

        dp_block = getattr(self.var, 'datapoint_tab', None) or getattr(self.var, 'datapoint_table', None)

        datasets = []
        for s in requested:
            s_name = s[0] if isinstance(s, (tuple, list)) else str(s)
            if s_name in self.slices:
                datasets.append(self.datastream(s_name, **kwargs))
            elif dp_block is not None and s_name in dp_block.slices:
                datasets.append(dp_block.datastream(s_name, **kwargs))
            else:
                avail = list(self.slices) + (list(dp_block.slices) if dp_block else [])
                raise KeyError(
                    f"{self.__class__.__name__}: unknown slice {s_name!r}; available slices are {avail}"
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

    def data(self, *slices, datapoint_slices: list | None = None, concat=True):
        requested = list(slices)
        if len(requested) == 1 and isinstance(requested[0], (tuple, list)):
            if len(requested[0]) > 0 and isinstance(requested[0][0], (tuple, list)):
                requested = list(requested[0])
            elif len(requested[0]) == 2 and isinstance(requested[0][0], str) and isinstance(requested[0][1], str):
                requested = [requested[0]]
            else:
                requested = list(requested[0])

        if not requested:
            requested = list(self.slices)

        if datapoint_slices:
            for s in datapoint_slices:
                if s not in requested:
                    requested.append(s)

        result = {}
        for item in requested:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                s_name, c_name = str(item[0]), str(item[1])
            else:
                s_name, c_name = str(item), None

            if s_name == 'features' or s_name in self.slices:
                raw_data = _extract_slice_data(super().data('features', concat=concat), 'features')
                if c_name is not None:
                    if isinstance(raw_data, dict):
                        target_col = c_name if c_name in raw_data else raw_data.get(c_name.replace('features_', ''), next(iter(raw_data.values())))
                        if isinstance(target_col, str) and target_col in raw_data:
                            result['features'] = {c_name: raw_data[target_col]}
                        else:
                            result['features'] = {c_name: target_col}
                    else:
                        result['features'] = {c_name: raw_data}
                else:
                    result['features'] = raw_data
            elif self.var.datapoint_tab is not None and s_name in self.var.datapoint_tab.slices:
                dp_data = _extract_slice_data(self.var.datapoint_tab.data(s_name, concat=concat), s_name)
                if c_name is not None and isinstance(dp_data, dict):
                    result[s_name] = {c_name: dp_data.get(c_name, next(iter(dp_data.values())))}
                else:
                    result[s_name] = dp_data
            else:
                col_key = s_name.replace('features_', '')
                raw_data = _extract_slice_data(super().data('features', concat=concat), 'features')
                if isinstance(raw_data, dict) and col_key in raw_data:
                    result[s_name] = raw_data[col_key]
                elif isinstance(raw_data, dict) and s_name in raw_data:
                    result[s_name] = raw_data[s_name]
                else:
                    avail = list(self.slices) + (list(self.var.datapoint_tab.slices) if self.var.datapoint_tab else [])
                    raise KeyError(
                        f"{self.__class__.__name__}: unknown slice {s_name!r}; available slices are {avail}"
                    )
        return result

    # 2. Properties and Accessors ───────────────────────────────────

    def __len__(self) -> int:
        return len(self.var.datapoint_tab)


class DatafeatureTable(DatapointTable):
    """A table of `DatafeatureTab` blocks built across a `DatapointTable`."""

    TAB = DatafeatureTab
    VERSION = 1

    @dataclass
    class VAR(Datablock.VAR):
        datapoint_table: DatapointTable
        evaluator_factory: DatamodelEvaluatorFactory
        collator: Datacollator
        feature_namemap: dict | None = None
        shard_size_limit_bytes: int = 1 << 26  # 64 MiB default, in bytes

    # 1. Datablock / Datastack Protocol Methods ─────────────────────

    def __init__(
        self,
        *args,
        device_batch_size: int = 64,
        devices: list | None = None,
        streaming: bool = False,
        dataloader_kwargs: dict | None = None,
        filter_built_tabs: bool = False,
        **kwargs,
    ):
        # Pass device_batch_size, devices, streaming, dataloader_kwargs, filter_built_tabs through Datastack.__init__ so they
        # survive multiprocessing pickling (Datablock folds **kwargs into the
        # state dict, which __getstate__/__setstate__ round-trips faithfully).
        super().__init__(
            *args,
            device_batch_size=device_batch_size,
            devices=devices or ["cuda"],
            streaming=streaming,
            dataloader_kwargs=dataloader_kwargs,
            filter_built_tabs=filter_built_tabs,
            **kwargs,
        )

    def __post_init__(self):
        super().__post_init__()
        # Read back from self — works whether we came through __init__ or __setstate__.
        self.streaming = getattr(self, 'streaming', False)
        self.dataloader_kwargs = getattr(self, 'dataloader_kwargs', None) or {}
        self.filter_built_tabs = getattr(self, 'filter_built_tabs', False)
        self._devices = getattr(self, 'devices', None)
        factory = self.var.evaluator_factory
        layer_names = factory.layer_names if factory is not None else []
        namemap = self.var.feature_namemap
        if namemap is not None:
            if isinstance(namemap, dict):
                self._feature_map = dict(namemap)
            elif isinstance(namemap, (list, tuple)):
                self._feature_map = {str(k): str(k) for k in namemap}
            else:
                self._feature_map = {str(namemap): str(namemap)}
        else:
            self._feature_map = {name: name for name in layer_names}

    def __tab__(self, idx: int, device: str | None = None, tag=None) -> DatafeatureTab:
        if device is None:
            devs = getattr(self, '_devices', None) or getattr(self, 'devices', None)
            if not devs and hasattr(self, 'var') and getattr(self.var, 'datapoint_table', None) is not None:
                dp_tbl = self.var.datapoint_table
                devs = getattr(dp_tbl, '_devices', None) or getattr(dp_tbl, 'devices', None)
            devs = devs or ["cuda"]
            device = devs[idx % len(devs)]
        datapoint_tab = self.var.datapoint_table.tab(idx)
        spec = dict(
            datapoint_tab=dbx.quote(datapoint_tab),
            evaluator_factory=self.spec['evaluator_factory'],
        )
        if self.var.collator is not None:
            spec['collator'] = self.var.collator
        if self.var.feature_namemap is not None:
            spec['feature_namemap'] = self.var.feature_namemap
        spec['shard_size_limit_bytes'] = self.var.shard_size_limit_bytes
        return self.TAB(
            url=self.url,
            storage_options=self.storage_options,
            capture_output=self.capture_output,
            cache=getattr(self, 'cache', None),
            cache_limit=getattr(self, 'cache_limit', None),
            verbose=False,
            spec=spec,
            device_batch_size=self.device_batch_size,
            device=device,
            streaming=self.streaming,
            dataloader_kwargs=self.dataloader_kwargs,
            revision=self.revision,
            tag=tag if tag is not None else datapoint_tab.tag,
        )

    def __block__(self, idx: int, **kwargs) -> DatafeatureTab:
        return self.__tab__(idx, **kwargs)

    def dataset(
        self,
        *slices,
        datapoint_slices: list | None = None,
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

        if datapoint_slices:
            for s in datapoint_slices:
                if s not in requested:
                    requested.append(s)

        dp_block = getattr(self.var, 'datapoint_table', None) or getattr(self.var, 'datapoint_tab', None)

        datasets = []
        for s in requested:
            s_name = s[0] if isinstance(s, (tuple, list)) else str(s)
            if s_name in self.slices:
                datasets.append(self.datastream(s_name, **kwargs))
            elif dp_block is not None and s_name in dp_block.slices:
                datasets.append(dp_block.datastream(s_name, **kwargs))
            else:
                avail = list(self.slices) + (list(dp_block.slices) if dp_block else [])
                raise KeyError(
                    f"{self.__class__.__name__}: unknown slice {s_name!r}; available slices are {avail}"
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

    def data(self, *slices, datapoint_slices: list | None = None, concat=True):
        requested = list(slices)
        if len(requested) == 1 and isinstance(requested[0], (tuple, list)):
            if len(requested[0]) > 0 and isinstance(requested[0][0], (tuple, list)):
                requested = list(requested[0])
            elif len(requested[0]) == 2 and isinstance(requested[0][0], str) and isinstance(requested[0][1], str):
                requested = [requested[0]]
            else:
                requested = list(requested[0])

        if not requested:
            requested = list(self.slices)

        if datapoint_slices:
            for s in datapoint_slices:
                if s not in requested:
                    requested.append(s)

        result = {}
        for item in requested:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                s_name, c_name = str(item[0]), str(item[1])
            else:
                s_name, c_name = str(item), None

            if s_name == 'features' or s_name in self.slices:
                tab_results = [self.tab(i).data((s_name, c_name) if c_name else s_name, concat=concat) for i in range(self.n_tabs)]
                feat_datas = [tr.get('features', tr) for tr in tab_results]
                if concat:
                    if isinstance(feat_datas[0], dict):
                        col_keys = feat_datas[0].keys()
                        result['features'] = {
                            k: np.concatenate([fd[k] for fd in feat_datas], axis=0) if isinstance(feat_datas[0][k], np.ndarray) else concat_data([fd[k] for fd in feat_datas])
                            for k in col_keys
                        }
                    elif isinstance(feat_datas[0], np.ndarray):
                        result['features'] = np.concatenate(feat_datas, axis=0)
                    else:
                        result['features'] = concat_data(feat_datas)
                else:
                    result['features'] = feat_datas
            elif self.var.datapoint_table is not None and s_name in self.var.datapoint_table.slices:
                dp_data = _extract_slice_data(self.var.datapoint_table.data(s_name, concat=concat), s_name)
                result[s_name] = dp_data
            else:
                col_key = s_name.replace('features_', '')
                tab_results = [self.tab(i).data(('features', col_key), concat=concat) for i in range(self.n_tabs)]
                feat_datas = [tr.get('features', tr)[col_key] for tr in tab_results if isinstance(tr.get('features', tr), dict) and col_key in tr.get('features', tr)]
                if concat and feat_datas:
                    if isinstance(feat_datas[0], np.ndarray):
                        result[s_name] = np.concatenate(feat_datas, axis=0)
                    else:
                        result[s_name] = concat_data(feat_datas)
                else:
                    result[s_name] = feat_datas
        return result

    # 2. Properties and Accessors ───────────────────────────────────

    @property
    def n_tabs(self) -> int:
        return self.var.datapoint_table.n_tabs


class BipolarDatafeatureTab(DatapointTab):
    """Bipolar (median-thresholded) encoding of a `DatafeatureTab`.

    Maps continuous features to ``{-1, +1}^d`` via ``sign(features - median)``,
    and computes a tab-level bipolar signature ``{-1, 0, +1}^d`` by thresholding the mean.
    """

    VERSION = 1
    TOPICS = {
        'bipolar_features': SLICETOPIC,
        'tab_bipolar_features': SLICETOPIC,
    }

    @dataclass
    class VAR(DatapointTab.VAR):
        featuretab: DatafeatureTab = None
        layer: str = 'final'
        threshold: float = 0.5
        ternarize: bool = False

    # 1. Datablock / Datastream Protocol Methods ─────────────────────

    def __build__(self):
        layer = self.var.layer
        res = self.featuretab.data(('features', layer), concat=True)
        raw_data = _extract_pair_data(res, ('features', layer))

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
            elif self.featuretab is not None and s in self.featuretab.slices:
                datasets.append(self.featuretab.dataset(s, mode=mode, **kwargs))
            else:
                avail = list(self.slices) + (list(self.featuretab.slices) if self.featuretab else [])
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
            if len(requested[0]) > 0 and isinstance(requested[0][0], (tuple, list)):
                requested = list(requested[0])
            elif len(requested[0]) == 2 and isinstance(requested[0][0], str) and (requested[0][0] in ('features', 'samples', 'labels') or requested[0][0] in self.slices):
                requested = [requested[0]]
            else:
                requested = list(requested[0])

        if not requested:
            requested = list(self.slices)

        result = {}
        for item in requested:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                s_name, c_name = str(item[0]), str(item[1])
            else:
                s_name, c_name = str(item), None

            if s_name in self.slices:
                result[s_name] = _extract_slice_data(super().data(s_name, concat=concat), s_name)
            elif self.featuretab is not None and s_name in self.featuretab.slices:
                result[s_name] = _extract_slice_data(self.featuretab.data(item, concat=concat), s_name)
            else:
                avail = list(self.slices) + (list(self.featuretab.slices) if self.featuretab else [])
                raise KeyError(
                    f"{self.__class__.__name__}: unknown slice {s_name!r}; available slices are {avail}"
                )
        return result

    # 2. Properties and Accessors ───────────────────────────────────

    @property
    def featuretab(self) -> DatafeatureTab:
        return self.var.featuretab

    @property
    def available_slices(self) -> tuple[str, ...]:
        own = tuple(self.slices)
        upstream = tuple(self.featuretab.slices) if self.featuretab is not None else ()
        return own + upstream

    def __len__(self) -> int:
        return len(self.featuretab)


class BipolarDatafeatureTable(DatapointTable):
    """A table of `BipolarDatafeatureTab` blocks built over a `DatafeatureTable`."""

    TAB = BipolarDatafeatureTab
    VERSION = 1

    @dataclass
    class VAR(DatapointTable.VAR):
        featuretable: DatafeatureTable = None
        layer: str = 'final'
        threshold: float = 0.5
        ternarize: bool = False

    # 1. Datablock / Datastack Protocol Methods ─────────────────────

    def __tab__(self, idx: int, tag=None) -> BipolarDatafeatureTab:
        featuretab = self.var.featuretable.tab(idx)
        return self.TAB(
            url=self.url,
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
            s_name = s[0] if isinstance(s, (tuple, list)) else str(s)
            if s_name in self.slices:
                datasets.append(self.datastream(s_name, **kwargs))
            elif self.featuretable is not None and s_name in self.featuretable.slices:
                datasets.append(self.featuretable.dataset(s_name, mode=mode, **kwargs))
            else:
                avail = list(self.slices) + (list(self.featuretable.slices) if self.featuretable else [])
                raise KeyError(
                    f"{self.__class__.__name__}: unknown slice {s_name!r}; available slices are {avail}"
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
            if len(requested[0]) > 0 and isinstance(requested[0][0], (tuple, list)):
                requested = list(requested[0])
            elif len(requested[0]) == 2 and isinstance(requested[0][0], str) and (requested[0][0] in ('features', 'samples', 'labels') or requested[0][0] in self.slices):
                requested = [requested[0]]
            else:
                requested = list(requested[0])

        if not requested:
            requested = list(self.slices)

        result = {}
        for item in requested:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                s_name, c_name = str(item[0]), str(item[1])
            else:
                s_name, c_name = str(item), None

            if s_name in self.slices:
                tab_data = [self.tab(i).data(s_name, concat=concat)[s_name] for i in range(self.n_tabs)]
                if concat:
                    if isinstance(tab_data[0], np.ndarray):
                        result[s_name] = np.concatenate(tab_data, axis=0)
                    else:
                        result[s_name] = concat_data(tab_data, dtype=self.SLICE_DTYPES.get(s_name))
                else:
                    result[s_name] = tab_data
            elif self.featuretable is not None and s_name in self.featuretable.slices:
                result[s_name] = _extract_slice_data(self.featuretable.data(item, concat=concat), s_name)
            else:
                avail = list(self.slices) + (list(self.featuretable.slices) if self.featuretable else [])
                raise KeyError(
                    f"{self.__class__.__name__}: unknown slice {s_name!r}; available slices are {avail}"
                )
        return result

    # 2. Properties and Accessors ───────────────────────────────────

    @property
    def featuretable(self) -> DatafeatureTable:
        return self.var.featuretable

    @property
    def n_tabs(self) -> int:
        return self.featuretable.n_tabs

    @property
    def available_slices(self) -> tuple[str, ...]:
        own = tuple(self.slices)
        upstream = tuple(self.featuretable.slices) if self.featuretable is not None else ()
        return own + upstream
