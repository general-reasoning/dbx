"""dbx.dataprobes — Generic feature probes and utilities for DatafeatureTables."""

from __future__ import annotations

from dataclasses import dataclass
import functools
import gc
from typing import Any

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import dbx
from dbx.datablocks import Datablock
from dbx.datafeatures import DatafeatureTable, DatafeatureTab
from dbx.dataparts import (
    Logger,
    read_npz,
    read_pickle,
    read_tensor,
    write_npz,
    write_pickle,
    write_tensor,
)

NORMALIZATION_MODES = {None, "l2", "corner-l1", "corner-l2", "corner-linfty"}


def normalize_features(features: Any, mode: str | None) -> Any:
    """Apply optional normalization to a feature tensor or array."""
    if mode is None:
        return features

    if torch is not None and isinstance(features, torch.Tensor):
        if mode == "l2":
            return torch.nn.functional.normalize(features.float(), p=2, dim=-1)
        elif mode in ("corner-l1", "corner-l2"):
            return torch.sign(features)
        elif mode == "corner-linfty":
            abs_f = features.abs()
            idx = abs_f.argmax(dim=-1, keepdim=True)
            result = torch.zeros_like(features)
            result.scatter_(-1, idx, features.gather(-1, idx).sign())
            return result
        else:
            raise ValueError(f"Unknown normalization mode {mode!r}")
    else:
        arr = np.asarray(features)
        if mode == "l2":
            norms = np.linalg.norm(arr, ord=2, axis=-1, keepdims=True)
            norms[norms == 0] = 1.0
            return arr / norms
        elif mode in ("corner-l1", "corner-l2"):
            return np.sign(arr)
        elif mode == "corner-linfty":
            abs_f = np.abs(arr)
            idx = np.argmax(abs_f, axis=-1)
            result = np.zeros_like(arr)
            if arr.ndim == 1:
                result[idx] = np.sign(arr[idx])
            else:
                rows = np.arange(len(arr))
                result[rows, idx] = np.sign(arr[rows, idx])
            return result
        else:
            raise ValueError(f"Unknown normalization mode {mode!r}")


class DatafeatureAffineLogisticProber:
    """Standalone logistic regression evaluator for data features.

    Provides static `evaluate_features` and `evaluate_features2`
    helpers that train/test a `LogisticRegression`
    and return `classification_report` strings.
    """

    def __init__(self, log: Logger | None = None):
        self.log = log or Logger()

    @staticmethod
    def ndarray(X: Any) -> np.ndarray:
        """Coerce X to a numpy ndarray."""
        if isinstance(X, list):
            return np.array(X)
        elif torch is not None and isinstance(X, torch.Tensor):
            return X.numpy()
        elif isinstance(X, np.ndarray):
            return X
        else:
            return np.array(X)

    @staticmethod
    def evaluate_features(Xy: tuple[Any, Any], *, fraction: float = 0.8, fit_intercept: bool = True) -> str:
        """Train/test a LogisticRegression and return the classification report."""
        features, labels = Xy
        features = DatafeatureAffineLogisticProber.ndarray(features)
        labels = DatafeatureAffineLogisticProber.ndarray(labels)
        N = len(labels)
        ntrain = int(N * fraction)
        perm = np.random.permutation(N)
        X_train, y_train = features[perm[:ntrain]], labels[perm[:ntrain]]
        X_test, y_test = features[perm[ntrain:]], labels[perm[ntrain:]]

        clf = LogisticRegression(fit_intercept=fit_intercept)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return classification_report(y_test, y_pred)

    @staticmethod
    def evaluate_features2(
        Xy1: tuple[Any, Any],
        Xy2: tuple[Any, Any],
        *,
        fraction: float = 0.8,
        fit_intercept: bool = True,
        tags: tuple[str, str] = ("(1)", "(2)"),
        log: Logger | None = None,
    ) -> tuple[str, str]:
        """Evaluate two feature sets side-by-side."""
        import datetime
        log = log or Logger()
        label1, label2 = tags
        log.verbose(f"EVALUATING features: {label1}: started at {datetime.datetime.now()}")
        report1 = DatafeatureAffineLogisticProber.evaluate_features(
            Xy1, fraction=fraction, fit_intercept=fit_intercept
        )
        log.verbose(f"EVALUATING features: {label1}: finished at {datetime.datetime.now()}")
        log.verbose(f"EVALUATING features: {label2}: started at {datetime.datetime.now()}")
        report2 = DatafeatureAffineLogisticProber.evaluate_features(
            Xy2, fraction=fraction, fit_intercept=fit_intercept
        )
        log.verbose(f"EVALUATING features: {label2}: finished at {datetime.datetime.now()}")

        rstr = f"---------- {label1} ------------\n{report1}\n---------- {label2} ------------\n{report2}"
        log.verbose(rstr)
        return report1, report2


class DatafeatureAffineLogisticProbe(Datablock):
    """Fit a logistic classifier on sample-level features for a single layer.

    Persists the fitted classifier's `coef_`, `intercept_`, and `classes_` arrays
    so that the separating hyperplane can be inspected after building.
    """

    VERSION = 1

    TOPICS = {
        'labels': 'labels.npz',
        'features': 'features.npy',
        'evaluation_report': 'evaluation_report.pkl',
        'coef': 'coef.npy',
        'intercept': 'intercept.npy',
        'classes': 'classes.npz',
    }

    @dataclass
    class VAR(Datablock.VAR):
        featuretable: DatafeatureTable | DatafeatureTab
        feature: tuple[str, str]
        label: tuple[str, str]
        fit_intercept: bool = True
        evaluation_fraction: float = 0.8
        aggregation: str = "mean"
        normalization: str | None = None  # None, 'l2', 'corner-l1', 'corner-l2', 'corner-linfty'

    def __post_init__(self):
        super().__post_init__()
        assert self.var.aggregation in ["mean"], f"Unknown aggregation: {self.var.aggregation}"
        assert self.var.normalization in NORMALIZATION_MODES, f"Unknown normalization mode: {self.var.normalization!r}"
        self._prober = DatafeatureAffineLogisticProber(log=self.log)

    def __build__(self):
        table = self.var.featuretable
        feat_spec = self.var.feature
        label_spec = self.var.label

        if isinstance(feat_spec, (list, tuple)):
            feat_slice = feat_spec[0]
            feat_col = feat_spec[1] if len(feat_spec) > 1 else feat_spec[0]
        else:
            feat_slice = str(feat_spec)
            feat_col = str(feat_spec)

        if isinstance(label_spec, (list, tuple)):
            label_slice = label_spec[0]
            label_col = label_spec[1] if len(label_spec) > 1 else label_spec[0]
        else:
            label_slice = str(label_spec)
            label_col = str(label_spec)

        self.log.verbose(f"Reading feature '{feat_slice}:{feat_col}' and label '{label_slice}:{label_col}'")

        data_dict = table.data(feat_slice, label_slice, concat=True)
        feat_data = data_dict[feat_slice]
        lbl_data = data_dict[label_slice]

        if isinstance(feat_data, dict):
            if feat_col in feat_data:
                raw_features = feat_data[feat_col]
            else:
                raw_features = next(iter(feat_data.values()))
        else:
            raw_features = feat_data

        if isinstance(lbl_data, dict):
            if label_col in lbl_data:
                raw_labels = lbl_data[label_col]
            else:
                raw_labels = next(iter(lbl_data.values()))
        else:
            raw_labels = lbl_data

        if torch is not None and isinstance(raw_features, torch.Tensor):
            feat_tensor = raw_features.float()
        else:
            feat_tensor = torch.from_numpy(np.array(raw_features)).float()

        feat_tensor = normalize_features(feat_tensor, self.var.normalization)

        if feat_tensor.dim() == 3:  # (N_tabs, N_tiles, D)
            if self.var.aggregation == "mean":
                sample_features = feat_tensor.mean(dim=1)
        elif feat_tensor.dim() == 2:  # (N_samples, D)
            sample_features = feat_tensor
        else:
            sample_features = feat_tensor.reshape(len(raw_labels), -1)

        labels = np.array(raw_labels)
        if labels.ndim > 1 and labels.shape[-1] == 1:
            labels = labels.squeeze(-1)

        assert len(labels) == len(sample_features), (
            f"len(labels) != len(sample_features): {len(labels)} != {len(sample_features)}"
        )

        write_npz(self.path('labels', ensure_dirpath=True), labels=labels)
        write_tensor(sample_features, self.path('features', ensure_dirpath=True))

        X = sample_features.numpy()
        y = labels
        N = len(y)
        ntrain = int(N * self.var.evaluation_fraction)
        perm = np.random.permutation(N)
        X_train, y_train = X[perm[:ntrain]], y[perm[:ntrain]]
        X_test, y_test = X[perm[ntrain:]], y[perm[ntrain:]]

        self.log.verbose(
            f"FITTING LogisticRegression "
            f"(fit_intercept={self.var.fit_intercept}, "
            f"feature={feat_spec!r}, "
            f"label={label_spec!r}, "
            f"normalization={self.var.normalization!r})"
        )
        clf = LogisticRegression(fit_intercept=self.var.fit_intercept)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred)
        self.log.verbose(f"Classification report:\n{report}")

        write_pickle(report, self.path('evaluation_report', ensure_dirpath=True))
        write_tensor(torch.from_numpy(clf.coef_), self.path('coef', ensure_dirpath=True))
        write_tensor(torch.from_numpy(clf.intercept_), self.path('intercept', ensure_dirpath=True))
        write_npz(self.path('classes', ensure_dirpath=True), classes=clf.classes_)

        return self

    def __read__(self, topic: str):
        if topic in ('labels', 'sample_labels', 'bag_labels'):
            result = read_npz(self.path('labels'), 'labels')['labels']
        elif topic in ('features', 'sample_features', 'bag_features'):
            result = read_tensor(self.path('features'))
        elif topic == 'evaluation_report':
            result = read_pickle(self.path('evaluation_report'))
        elif topic == 'coef':
            result = read_tensor(self.path('coef'))
        elif topic == 'intercept':
            result = read_tensor(self.path('intercept'))
        elif topic == 'classes':
            result = read_npz(self.path('classes'), 'classes')['classes']
        else:
            raise ValueError(f"Unknown topic: {topic}")
        return result

    def asphericity(self) -> dict[str, float]:
        """Per-class ratio ``||intercept|| / ||coef||``."""
        coef = self.read('coef')
        intercept = self.read('intercept')
        classes = self.read('classes')
        coef_norms = coef.norm(dim=1)
        ratios = intercept.abs() / coef_norms
        return {str(c): float(r) for c, r in zip(classes, ratios)}


class DatafeatureStatsProbe(Datablock):
    """Per-layer statistics for a DatafeatureTable.

    Computes overall feature/signal statistics and tab-granularity statistics
    for a DatafeatureTable or DatafeatureTab.
    """

    VERSION = 1

    TOPICS = {
        'feature_mean': 'feature_mean.npz',
        'feature_std': 'feature_std.npz',
        'feature_median': 'feature_median.npz',
        'feature_min': 'feature_min.npz',
        'feature_max': 'feature_max.npz',
        'feature_norms': 'feature_norms.npz',
        'tab_feature_mean': 'tab_feature_mean.npz',
        'tab_feature_std': 'tab_feature_std.npz',
        'tab_feature_median': 'tab_feature_median.npz',
        'tab_feature_min': 'tab_feature_min.npz',
        'tab_feature_max': 'tab_feature_max.npz',
        'tab_feature_norms': 'tab_feature_norms.npz',
        'signal_count': 'unique_signal_count.npz',
        'signal_mean': 'signal_mean.npz',
        'signal_std': 'signal_std.npz',
        'signal_median': 'signal_median.npz',
        'signal_min': 'signal_min.npz',
        'signal_max': 'signal_max.npz',
        'signal_norms': 'signal_norms.npz',
        'tab_signal_mean': 'tab_signal_mean.npz',
        'tab_signal_std': 'tab_signal_std.npz',
        'tab_signal_median': 'tab_signal_median.npz',
        'tab_signal_min': 'tab_signal_min.npz',
        'tab_signal_max': 'tab_signal_max.npz',
        'tab_signal_norms': 'tab_signal_norms.npz',
    }

    @dataclass
    class VAR(Datablock.VAR):
        featuretable: DatafeatureTable | DatafeatureTab
        feature: tuple[str, str]
        signal: tuple[str, str] | None = None
        label: tuple[str, str] | None = None
        normalization: str | None = None  # None, 'l2', 'corner-l1', 'corner-l2', 'corner-linfty'

    def __post_init__(self):
        super().__post_init__()
        assert self.var.normalization in NORMALIZATION_MODES, f"Unknown normalization mode: {self.var.normalization!r}"

    def __build__(self):
        table = self.var.featuretable
        feat_spec = self.var.feature
        sig_spec = self.var.signal

        if isinstance(feat_spec, (list, tuple)):
            feat_slice = feat_spec[0]
            feat_col = feat_spec[1] if len(feat_spec) > 1 else feat_spec[0]
        else:
            feat_slice = str(feat_spec)
            feat_col = str(feat_spec)

        self.log.verbose(f"COMPUTING stats for feature '{feat_slice}:{feat_col}' (normalization={self.var.normalization!r})")

        # Extract per-tab feature data
        if hasattr(table, 'n_tabs') and table.n_tabs > 0:
            tab_feat_list = []
            for i in range(table.n_tabs):
                tab_data = table.tab(i).data(feat_slice, concat=True)[feat_slice]
                if isinstance(tab_data, dict):
                    tab_data = tab_data.get(feat_col, next(iter(tab_data.values())))
                tab_feat_list.append(tab_data)
        else:
            tab_data = table.data(feat_slice, concat=True)[feat_slice]
            if isinstance(tab_data, dict):
                tab_data = tab_data.get(feat_col, next(iter(tab_data.values())))
            tab_feat_list = [tab_data]

        all_feats = []
        tab_f_means, tab_f_stds, tab_f_medians, tab_f_mins, tab_f_maxs, tab_f_norms = [], [], [], [], [], []

        for tab_f in tab_feat_list:
            if torch is not None and isinstance(tab_f, torch.Tensor):
                arr = tab_f.float()
            else:
                arr = torch.from_numpy(np.array(tab_f)).float()
            arr = normalize_features(arr, self.var.normalization).numpy()
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)

            all_feats.append(arr)
            tab_f_means.append(np.mean(arr, axis=0))
            tab_f_stds.append(np.std(arr, axis=0))
            tab_f_medians.append(np.median(arr, axis=0))
            tab_f_mins.append(np.min(arr, axis=0))
            tab_f_maxs.append(np.max(arr, axis=0))
            tab_f_norms.append(np.mean(np.linalg.norm(arr, axis=-1)))

        concat_feats = np.concatenate(all_feats, axis=0)

        # Overall feature stats
        write_npz(self.path('feature_mean', ensure_dirpath=True), feature_mean=np.mean(concat_feats, axis=0))
        write_npz(self.path('feature_std', ensure_dirpath=True), feature_std=np.std(concat_feats, axis=0))
        write_npz(self.path('feature_median', ensure_dirpath=True), feature_median=np.median(concat_feats, axis=0))
        write_npz(self.path('feature_min', ensure_dirpath=True), feature_min=np.min(concat_feats, axis=0))
        write_npz(self.path('feature_max', ensure_dirpath=True), feature_max=np.max(concat_feats, axis=0))
        write_npz(self.path('feature_norms', ensure_dirpath=True), feature_norms=np.linalg.norm(concat_feats, axis=-1))

        # Tab feature stats
        write_npz(self.path('tab_feature_mean', ensure_dirpath=True), tab_feature_mean=np.stack(tab_f_means))
        write_npz(self.path('tab_feature_std', ensure_dirpath=True), tab_feature_std=np.stack(tab_f_stds))
        write_npz(self.path('tab_feature_median', ensure_dirpath=True), tab_feature_median=np.stack(tab_f_medians))
        write_npz(self.path('tab_feature_min', ensure_dirpath=True), tab_feature_min=np.stack(tab_f_mins))
        write_npz(self.path('tab_feature_max', ensure_dirpath=True), tab_feature_max=np.stack(tab_f_maxs))
        write_npz(self.path('tab_feature_norms', ensure_dirpath=True), tab_feature_norms=np.array(tab_f_norms))

        # Extract per-tab signal data if signal is configured
        if sig_spec is not None:
            if isinstance(sig_spec, (list, tuple)):
                sig_slice = sig_spec[0]
                sig_col = sig_spec[1] if len(sig_spec) > 1 else sig_spec[0]
            else:
                sig_slice = str(sig_spec)
                sig_col = str(sig_spec)

            if hasattr(table, 'n_tabs') and table.n_tabs > 0:
                tab_sig_list = []
                for i in range(table.n_tabs):
                    s_data = table.tab(i).data(sig_slice, concat=True)[sig_slice]
                    if isinstance(s_data, dict):
                        s_data = s_data.get(sig_col, next(iter(s_data.values())))
                    tab_sig_list.append(s_data)
            else:
                s_data = table.data(sig_slice, concat=True)[sig_slice]
                if isinstance(s_data, dict):
                    s_data = s_data.get(sig_col, next(iter(s_data.values())))
                tab_sig_list = [s_data]

            all_sigs = []
            tab_s_means, tab_s_stds, tab_s_medians, tab_s_mins, tab_s_maxs, tab_s_norms = [], [], [], [], [], []

            for tab_s in tab_sig_list:
                s_arr = np.array(tab_s)
                if s_arr.ndim == 1:
                    s_arr_2d = s_arr.reshape(-1, 1)
                else:
                    s_arr_2d = s_arr
                all_sigs.append(s_arr)

                if np.issubdtype(s_arr_2d.dtype, np.number):
                    tab_s_means.append(np.mean(s_arr_2d, axis=0))
                    tab_s_stds.append(np.std(s_arr_2d, axis=0))
                    tab_s_medians.append(np.median(s_arr_2d, axis=0))
                    tab_s_mins.append(np.min(s_arr_2d, axis=0))
                    tab_s_maxs.append(np.max(s_arr_2d, axis=0))
                    tab_s_norms.append(np.mean(np.linalg.norm(s_arr_2d, axis=-1)))
                else:
                    tab_s_means.append(np.array([0.0]))
                    tab_s_stds.append(np.array([0.0]))
                    tab_s_medians.append(np.array([0.0]))
                    tab_s_mins.append(np.array([0.0]))
                    tab_s_maxs.append(np.array([0.0]))
                    tab_s_norms.append(0.0)

            try:
                concat_sigs = np.concatenate(all_sigs, axis=0)
            except Exception:
                concat_sigs = np.array(all_sigs, dtype=object)

            signal_count = np.array(len(np.unique(concat_sigs, axis=0)) if concat_sigs.ndim > 0 else len(concat_sigs))
            write_npz(self.path('signal_count', ensure_dirpath=True), signal_count=signal_count)

            if np.issubdtype(concat_sigs.dtype, np.number):
                write_npz(self.path('signal_mean', ensure_dirpath=True), signal_mean=np.mean(concat_sigs, axis=0))
                write_npz(self.path('signal_std', ensure_dirpath=True), signal_std=np.std(concat_sigs, axis=0))
                write_npz(self.path('signal_median', ensure_dirpath=True), signal_median=np.median(concat_sigs, axis=0))
                write_npz(self.path('signal_min', ensure_dirpath=True), signal_min=np.min(concat_sigs, axis=0))
                write_npz(self.path('signal_max', ensure_dirpath=True), signal_max=np.max(concat_sigs, axis=0))
                write_npz(self.path('signal_norms', ensure_dirpath=True), signal_norms=np.linalg.norm(concat_sigs, axis=-1))

            write_npz(self.path('tab_signal_mean', ensure_dirpath=True), tab_signal_mean=np.array(tab_s_means, dtype=object))
            write_npz(self.path('tab_signal_std', ensure_dirpath=True), tab_signal_std=np.array(tab_s_stds, dtype=object))
            write_npz(self.path('tab_signal_median', ensure_dirpath=True), tab_signal_median=np.array(tab_s_medians, dtype=object))
            write_npz(self.path('tab_signal_min', ensure_dirpath=True), tab_signal_min=np.array(tab_s_mins, dtype=object))
            write_npz(self.path('tab_signal_max', ensure_dirpath=True), tab_signal_max=np.array(tab_s_maxs, dtype=object))
            write_npz(self.path('tab_signal_norms', ensure_dirpath=True), tab_signal_norms=np.array(tab_s_norms))
        else:
            signal_count = np.array(len(concat_feats))
            write_npz(self.path('signal_count', ensure_dirpath=True), signal_count=signal_count)

        return self

    def __read__(self, topic: str):
        if topic in ('tile_count', 'distinct_tile_count'):
            return read_npz(self.path('signal_count'), 'signal_count')['signal_count']
        elif topic == 'tile_feature_mean':
            return read_npz(self.path('feature_mean'), 'feature_mean')['feature_mean']
        elif topic == 'tile_feature_std':
            return read_npz(self.path('feature_std'), 'feature_std')['feature_std']
        elif topic == 'tile_feature_median':
            return read_npz(self.path('feature_median'), 'feature_median')['feature_median']
        elif topic == 'tile_feature_min':
            return read_npz(self.path('feature_min'), 'feature_min')['feature_min']
        elif topic == 'tile_feature_max':
            return read_npz(self.path('feature_max'), 'feature_max')['feature_max']
        elif topic == 'tile_feature_norms':
            return read_npz(self.path('feature_norms'), 'feature_norms')['feature_norms']
        elif topic in ('bag_feature_mean', 'sample_feature_mean'):
            return read_npz(self.path('tab_feature_mean'), 'tab_feature_mean')['tab_feature_mean']
        elif topic in ('bag_feature_std', 'sample_feature_std'):
            return read_npz(self.path('tab_feature_std'), 'tab_feature_std')['tab_feature_std']

        return read_npz(self.path(topic), topic)[topic]

    @functools.cached_property
    def feature_mean(self):
        return self.read('feature_mean')

    @functools.cached_property
    def feature_std(self):
        return self.read('feature_std')

    @functools.cached_property
    def feature_median(self):
        return self.read('feature_median')

    @functools.cached_property
    def feature_min(self):
        return self.read('feature_min')

    @functools.cached_property
    def feature_max(self):
        return self.read('feature_max')

    @functools.cached_property
    def feature_norms(self):
        return self.read('feature_norms')

    @functools.cached_property
    def tab_feature_mean(self):
        return self.read('tab_feature_mean')

    @functools.cached_property
    def tab_feature_std(self):
        return self.read('tab_feature_std')

    @functools.cached_property
    def tab_feature_median(self):
        return self.read('tab_feature_median')

    @functools.cached_property
    def tab_feature_min(self):
        return self.read('tab_feature_min')

    @functools.cached_property
    def tab_feature_max(self):
        return self.read('tab_feature_max')

    @functools.cached_property
    def tab_feature_norms(self):
        return self.read('tab_feature_norms')

    @functools.cached_property
    def signal_count(self):
        return self.read('signal_count')
