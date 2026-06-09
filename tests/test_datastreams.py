"""Tests for :mod:`dbx.datastreams.ZipStreamingDataset`.

Covers zipping two and three streams, length validation, the
``zip_validator`` callback, ``None``-value filtering, and key merging.
"""
import os
os.environ.setdefault('DBX_DIRTY_REPO_OK', '1')

import pytest

from dbx.datastreams import ZipStreamingDataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')

# ---------------------------------------------------------------------------
# Helpers: lightweight dict-returning datasets (no streaming infra needed)
# ---------------------------------------------------------------------------

class DictDataset:
    """Minimal ``Dataset``-compatible object backed by a list of dicts."""

    def __init__(self, records: list[dict]):
        self._records = records

    def __len__(self):
        return len(self._records)

    def __getitem__(self, idx):
        return self._records[idx]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ds_alpha():
    """3-sample dataset with keys 'a' and 'id'."""
    return DictDataset([
        {"a": 10, "id": 0},
        {"a": 20, "id": 1},
        {"a": 30, "id": 2},
    ])


@pytest.fixture
def ds_beta():
    """3-sample dataset with keys 'b' and 'id'."""
    return DictDataset([
        {"b": 100, "id": 0},
        {"b": 200, "id": 1},
        {"b": 300, "id": 2},
    ])


@pytest.fixture
def ds_gamma():
    """3-sample dataset with key 'c'."""
    return DictDataset([
        {"c": 1.0},
        {"c": 2.0},
        {"c": 3.0},
    ])


# ---------------------------------------------------------------------------
# Zipping two streams
# ---------------------------------------------------------------------------

class TestZipTwo:

    def test_length_matches_input(self, ds_alpha, ds_beta):
        zipped = ZipStreamingDataset(ds_alpha, ds_beta)
        assert len(zipped) == 3

    def test_keys_are_merged(self, ds_alpha, ds_beta):
        zipped = ZipStreamingDataset(ds_alpha, ds_beta)
        sample = zipped[0]
        assert set(sample.keys()) == {"a", "b", "id"}

    def test_values_from_both_datasets(self, ds_alpha, ds_beta):
        zipped = ZipStreamingDataset(ds_alpha, ds_beta)
        assert zipped[0]["a"] == 10
        assert zipped[0]["b"] == 100

    def test_all_indices(self, ds_alpha, ds_beta):
        zipped = ZipStreamingDataset(ds_alpha, ds_beta)
        for i in range(3):
            sample = zipped[i]
            assert sample["a"] == (i + 1) * 10
            assert sample["b"] == (i + 1) * 100

    def test_later_dataset_wins_on_shared_key(self, ds_alpha, ds_beta):
        """When both datasets provide the same key, the later dataset's
        value should overwrite (dict merge order)."""
        zipped = ZipStreamingDataset(ds_alpha, ds_beta)
        # Both have 'id'; ds_beta comes second, so its value wins.
        assert zipped[0]["id"] == 0  # both are 0 here — same value
        # Build datasets with different 'id' to prove ordering:
        a = DictDataset([{"x": 1, "shared": "from_a"}])
        b = DictDataset([{"y": 2, "shared": "from_b"}])
        zipped = ZipStreamingDataset(a, b)
        assert zipped[0]["shared"] == "from_b"


# ---------------------------------------------------------------------------
# Zipping three streams
# ---------------------------------------------------------------------------

class TestZipThree:

    def test_length_matches_input(self, ds_alpha, ds_beta, ds_gamma):
        zipped = ZipStreamingDataset(ds_alpha, ds_beta, ds_gamma)
        assert len(zipped) == 3

    def test_keys_from_all_three(self, ds_alpha, ds_beta, ds_gamma):
        zipped = ZipStreamingDataset(ds_alpha, ds_beta, ds_gamma)
        sample = zipped[1]
        assert "a" in sample
        assert "b" in sample
        assert "c" in sample

    def test_values_from_all_three(self, ds_alpha, ds_beta, ds_gamma):
        zipped = ZipStreamingDataset(ds_alpha, ds_beta, ds_gamma)
        sample = zipped[2]
        assert sample["a"] == 30
        assert sample["b"] == 300
        assert sample["c"] == 3.0

    def test_all_indices(self, ds_alpha, ds_beta, ds_gamma):
        zipped = ZipStreamingDataset(ds_alpha, ds_beta, ds_gamma)
        for i in range(3):
            sample = zipped[i]
            assert sample["a"] == (i + 1) * 10
            assert sample["b"] == (i + 1) * 100
            assert sample["c"] == float(i + 1)


# ---------------------------------------------------------------------------
# Length-mismatch errors
# ---------------------------------------------------------------------------

class TestLengthMismatch:

    def test_two_datasets_different_lengths(self, ds_alpha):
        short = DictDataset([{"x": 1}])
        with pytest.raises(ValueError, match="equal length"):
            ZipStreamingDataset(ds_alpha, short)

    def test_three_datasets_one_short(self, ds_alpha, ds_beta):
        short = DictDataset([{"x": 1}, {"x": 2}])
        with pytest.raises(ValueError, match="equal length"):
            ZipStreamingDataset(ds_alpha, ds_beta, short)


# ---------------------------------------------------------------------------
# zip_validator callback
# ---------------------------------------------------------------------------

class TestZipValidator:

    def test_validator_called_with_idx_and_samples(self, ds_alpha, ds_beta):
        calls = []

        def recorder(idx, *samples):
            calls.append((idx, samples))

        zipped = ZipStreamingDataset(ds_alpha, ds_beta, zip_validator=recorder)
        _ = zipped[1]
        assert len(calls) == 1
        idx, samples = calls[0]
        assert idx == 1
        assert len(samples) == 2
        assert samples[0]["a"] == 20
        assert samples[1]["b"] == 200

    def test_validator_three_streams(self, ds_alpha, ds_beta, ds_gamma):
        calls = []

        def recorder(idx, *samples):
            calls.append((idx, samples))

        zipped = ZipStreamingDataset(
            ds_alpha, ds_beta, ds_gamma, zip_validator=recorder,
        )
        _ = zipped[0]
        assert len(calls) == 1
        _, samples = calls[0]
        assert len(samples) == 3

    def test_validator_can_raise(self, ds_alpha, ds_beta):
        def always_fail(idx, *samples):
            raise RuntimeError("mismatch!")

        zipped = ZipStreamingDataset(ds_alpha, ds_beta, zip_validator=always_fail)
        with pytest.raises(RuntimeError, match="mismatch"):
            _ = zipped[0]


# ---------------------------------------------------------------------------
# None-value filtering
# ---------------------------------------------------------------------------

class TestNoneFiltering:

    def test_none_values_are_excluded(self):
        a = DictDataset([{"x": 1, "y": None}])
        b = DictDataset([{"z": 3}])
        zipped = ZipStreamingDataset(a, b)
        sample = zipped[0]
        assert "y" not in sample
        assert sample["x"] == 1
        assert sample["z"] == 3

    def test_non_none_overwrites_none_from_earlier(self):
        """If dataset A has key='shared' → None but dataset B has
        key='shared' → 42, the merged result should have 42."""
        a = DictDataset([{"shared": None}])
        b = DictDataset([{"shared": 42}])
        zipped = ZipStreamingDataset(a, b)
        assert zipped[0]["shared"] == 42


# ---------------------------------------------------------------------------
# Single dataset (degenerate but valid)
# ---------------------------------------------------------------------------

class TestSingleDataset:

    def test_single_dataset_passthrough(self, ds_alpha):
        zipped = ZipStreamingDataset(ds_alpha)
        assert len(zipped) == 3
        assert zipped[0] == {"a": 10, "id": 0}
