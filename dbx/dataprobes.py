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
        regressor: tuple[str, str]
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
        reg_spec = self.var.regressor
        label_spec = self.var.label

        if isinstance(reg_spec, (list, tuple)):
            reg_slice = reg_spec[0]
            reg_col = reg_spec[1] if len(reg_spec) > 1 else reg_spec[0]
        else:
            reg_slice = str(reg_spec)
            reg_col = str(reg_spec)

        if isinstance(label_spec, (list, tuple)):
            label_slice = label_spec[0]
            label_col = label_spec[1] if len(label_spec) > 1 else label_spec[0]
        else:
            label_slice = str(label_spec)
            label_col = str(label_spec)

        self.log.verbose(f"Reading regressor '{reg_slice}:{reg_col}' and labels for '{label_slice}:{label_col}'")

        data_dict = table.data(reg_slice, label_slice, concat=True)
        feat_data = data_dict[reg_slice]
        lbl_data = data_dict[label_slice]

        if isinstance(feat_data, dict):
            if reg_col in feat_data:
                raw_features = feat_data[reg_col]
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
            f"regressor={reg_spec!r}, "
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

    Computes item-level and sample-level statistics — mean, std, median,
    min, max, L2 norms, and distinct counts — for a single layer.
    """

    VERSION = 1

    TOPICS = {
        'tile_count': 'tile_count.npz',
        'tile_feature_mean': 'tile_feature_mean.npz',
        'tile_feature_std': 'tile_feature_std.npz',
        'tile_feature_median': 'tile_feature_median.npz',
        'tile_feature_min': 'tile_feature_min.npz',
        'tile_feature_max': 'tile_feature_max.npz',
        'tile_feature_norms': 'tile_feature_norms.npz',
        'sample_feature_mean': 'sample_feature_mean.npz',
        'sample_feature_std': 'sample_feature_std.npz',
        'distinct_tile_count': 'distinct_tile_count.npz',
        'bag_feature_mean': 'sample_feature_mean.npz',
        'bag_feature_std': 'sample_feature_std.npz',
    }

    @dataclass
    class VAR(Datablock.VAR):
        featuretable: DatafeatureTable | DatafeatureTab
        feature: str = "features_final"
        normalization: str | None = None  # None, 'l2', 'corner-l1', 'corner-l2', 'corner-linfty'

    def __post_init__(self):
        super().__post_init__()
        assert self.var.normalization in NORMALIZATION_MODES, f"Unknown normalization mode: {self.var.normalization!r}"

    def __build__(self):
        table = self.var.featuretable
        feature = self.var.feature

        if hasattr(table, 'slices') and feature in table.slices:
            feat_slice = feature
        elif hasattr(table, 'slices') and f"features_{feature.replace('.', '_')}" in table.slices:
            feat_slice = f"features_{feature.replace('.', '_')}"
        elif hasattr(table, 'available_slices') and feature in table.available_slices:
            feat_slice = feature
        else:
            feat_slice = feature

        self.log.verbose(f"COMPUTING stats for feature '{feature}' (normalization={self.var.normalization!r}): BEGIN")

        raw_features = table.data(feat_slice, concat=True)[feat_slice]
        if torch is not None and isinstance(raw_features, torch.Tensor):
            feat_tensor = raw_features.float()
        else:
            feat_tensor = torch.from_numpy(np.array(raw_features)).float()

        feat_tensor = normalize_features(feat_tensor, self.var.normalization)

        if feat_tensor.dim() == 3:  # (N_tabs, N_tiles, D)
            all_tiles = feat_tensor.reshape(-1, feat_tensor.shape[-1]).numpy()
            sample_features = feat_tensor.mean(dim=1).numpy()
        elif feat_tensor.dim() == 2:  # (N_samples, D)
            all_tiles = feat_tensor.numpy()
            sample_features = feat_tensor.numpy()
        else:
            all_tiles = feat_tensor.reshape(-1, feat_tensor.shape[-1]).numpy()
            sample_features = all_tiles

        tile_count = np.array(len(all_tiles))
        tile_feature_mean = np.mean(all_tiles, axis=0)
        tile_feature_std = np.std(all_tiles, axis=0)
        tile_feature_median = np.median(all_tiles, axis=0)
        tile_feature_min = np.min(all_tiles, axis=0)
        tile_feature_max = np.max(all_tiles, axis=0)
        tile_feature_norms = np.linalg.norm(all_tiles, axis=1)

        write_npz(self.path('tile_count', ensure_dirpath=True), tile_count=tile_count)
        write_npz(self.path('tile_feature_mean', ensure_dirpath=True), tile_feature_mean=tile_feature_mean)
        write_npz(self.path('tile_feature_std', ensure_dirpath=True), tile_feature_std=tile_feature_std)
        write_npz(self.path('tile_feature_median', ensure_dirpath=True), tile_feature_median=tile_feature_median)
        write_npz(self.path('tile_feature_min', ensure_dirpath=True), tile_feature_min=tile_feature_min)
        write_npz(self.path('tile_feature_max', ensure_dirpath=True), tile_feature_max=tile_feature_max)
        write_npz(self.path('tile_feature_norms', ensure_dirpath=True), tile_feature_norms=tile_feature_norms)

        distinct_tile_count = np.array(np.unique(all_tiles, axis=0).shape[0])
        write_npz(self.path('distinct_tile_count', ensure_dirpath=True), distinct_tile_count=distinct_tile_count)

        sample_feature_mean = np.mean(sample_features, axis=0)
        sample_feature_std = np.std(sample_features, axis=0)
        write_npz(self.path('sample_feature_mean', ensure_dirpath=True), sample_feature_mean=sample_feature_mean)
        write_npz(self.path('sample_feature_std', ensure_dirpath=True), sample_feature_std=sample_feature_std)
        write_npz(self.path('bag_feature_mean', ensure_dirpath=True), bag_feature_mean=sample_feature_mean)
        write_npz(self.path('bag_feature_std', ensure_dirpath=True), bag_feature_std=sample_feature_std)

        return self

    def __read__(self, topic: str):
        if topic in ('bag_feature_mean', 'sample_feature_mean'):
            return read_npz(self.path('sample_feature_mean'), 'sample_feature_mean')['sample_feature_mean']
        elif topic in ('bag_feature_std', 'sample_feature_std'):
            return read_npz(self.path('sample_feature_std'), 'sample_feature_std')['sample_feature_std']
        return read_npz(self.path(topic), topic)[topic]

    @functools.cached_property
    def tile_count(self):
        return self.read('tile_count')

    @functools.cached_property
    def tile_feature_mean(self):
        return self.read('tile_feature_mean')

    @functools.cached_property
    def tile_feature_std(self):
        return self.read('tile_feature_std')

    @functools.cached_property
    def tile_feature_median(self):
        return self.read('tile_feature_median')

    @functools.cached_property
    def tile_feature_min(self):
        return self.read('tile_feature_min')

    @functools.cached_property
    def tile_feature_max(self):
        return self.read('tile_feature_max')

    @functools.cached_property
    def tile_feature_norms(self):
        return self.read('tile_feature_norms')
