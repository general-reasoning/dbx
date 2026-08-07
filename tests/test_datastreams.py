"""Tests for the :mod:`dbx.datastreams` zip datasets.

Covers zipping two and three streams, length validation, the
``zip_validator`` callback, ``None``-value filtering, and key merging --
for both ``ZipStreamingDataset`` (map-style) and
``ZipIterableStreamingDatasets`` (iterator-style), including the shard
alignment the latter needs before it may shuffle.
"""
import os
os.environ.setdefault('DBX_DIRTY_REPO_OK', '1')

import pytest

# torch and mosaicml-streaming are optional extras, and dbx.datastreams needs
# both at import time. importorskip skips this module when either is absent,
# rather than failing collection -- which takes the entire suite down before
# any test runs. Install them and these run as normal.
pytest.importorskip("torch", reason="torch is an optional dependency")
pytest.importorskip("streaming", reason="mosaicml-streaming is an optional dependency")

from dbx.datastreams import ZipIterableStreamingDatasets, ZipStreamingDataset


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


# ---------------------------------------------------------------------------
# Iterator-mode zipping
# ---------------------------------------------------------------------------

class TestZipIterable:
    """The same merge as ZipStreamingDataset, reached by iterating the
    sources in lockstep rather than indexing them."""

    def test_iterates_merged_samples(self, ds_alpha, ds_beta):
        zipped = ZipIterableStreamingDatasets(ds_alpha, ds_beta)
        assert list(zipped) == [
            {"a": 10, "b": 100, "id": 0},
            {"a": 20, "b": 200, "id": 1},
            {"a": 30, "b": 300, "id": 2},
        ]

    def test_agrees_with_map_mode(self, ds_alpha, ds_beta, ds_gamma):
        args = (ds_alpha, ds_beta, ds_gamma)
        mapped = ZipStreamingDataset(*args)
        assert list(ZipIterableStreamingDatasets(*args)) == \
            [mapped[i] for i in range(len(mapped))]

    def test_len_is_the_source_length(self, ds_alpha, ds_beta):
        assert len(ZipIterableStreamingDatasets(ds_alpha, ds_beta)) == 3

    def test_is_an_iterable_dataset(self, ds_alpha, ds_beta):
        from torch.utils.data import IterableDataset
        assert isinstance(ZipIterableStreamingDatasets(ds_alpha, ds_beta),
                          IterableDataset)

    def test_merge_policy_applies(self):
        a = DictDataset([{"x": 1, "both": "from_a"}])
        b = DictDataset([{"y": 2, "both": "from_b"}])
        with pytest.raises(KeyError, match="both"):
            list(ZipIterableStreamingDatasets(a, b, on_conflict='error'))

    def test_projection_applies(self, ds_alpha, ds_beta):
        zipped = ZipIterableStreamingDatasets(
            ds_alpha, ds_beta, columns=[['a'], ['b']],
        )
        assert next(iter(zipped)) == {"a": 10, "b": 100}

    def test_validator_gets_the_stream_position(self, ds_alpha, ds_beta):
        seen = []
        zipped = ZipIterableStreamingDatasets(
            ds_alpha, ds_beta, zip_validator=lambda idx, *s: seen.append(idx),
        )
        list(zipped)
        assert seen == [0, 1, 2]

    def test_shared_key_disagreement_is_caught(self):
        """The running safety net: iterator-mode alignment is not
        structural, so validate_shared is what notices when it breaks."""
        a = DictDataset([{"sample_id": 0, "x": 1}, {"sample_id": 1, "x": 2}])
        b = DictDataset([{"sample_id": 0, "y": 1}, {"sample_id": 9, "y": 2}])
        zipped = ZipIterableStreamingDatasets(
            a, b, shared={'sample_id'}, validate_shared=True,
        )
        with pytest.raises(ValueError, match="not aligned"):
            list(zipped)

    def test_diverging_streams_are_not_silently_truncated(self, ds_alpha):
        """Equal len() but unequal iteration -- which is what mismatched
        partitions look like -- must raise, not zip to the shorter."""

        class ShortIter(DictDataset):
            def __iter__(self):
                return iter(self._records[:2])

        short = ShortIter([{"b": 1}, {"b": 2}, {"b": 3}])
        zipped = ZipIterableStreamingDatasets(ds_alpha, short)
        with pytest.raises(ValueError):
            list(zipped)


# ---------------------------------------------------------------------------
# Iterator-mode shard alignment check
# ---------------------------------------------------------------------------

class FakeStream(DictDataset):
    """Just enough of a StreamingDataset for _check_shard_alignment()."""

    def __init__(self, records, samples_per_shard, shuffle=True,
                 shuffle_algo='py1e'):
        super().__init__(records)
        self.samples_per_shard = samples_per_shard
        self.shuffle = shuffle
        self.shuffle_algo = shuffle_algo


class TestShardAlignmentCheck:

    RECORDS = [{"v": i} for i in range(4)]

    def stream(self, **kwargs):
        return FakeStream(list(self.RECORDS), **kwargs)

    def test_matching_shard_boundaries_are_accepted(self):
        ZipIterableStreamingDatasets(
            self.stream(samples_per_shard=[2, 2]),
            self.stream(samples_per_shard=[2, 2]),
        )

    def test_differing_shard_boundaries_are_rejected(self):
        with pytest.raises(ValueError, match="shard boundaries"):
            ZipIterableStreamingDatasets(
                self.stream(samples_per_shard=[2, 2]),
                self.stream(samples_per_shard=[3, 1]),
            )

    def test_unshuffled_sources_need_no_alignment(self):
        """Without shuffle the order comes from the partition, which reads
        sample counts and not shard structure."""
        ZipIterableStreamingDatasets(
            self.stream(samples_per_shard=[2, 2], shuffle=False),
            self.stream(samples_per_shard=[3, 1], shuffle=False),
        )

    def test_naive_algo_needs_no_alignment(self):
        """'naive' permutes the total, so the boundaries do not matter."""
        ZipIterableStreamingDatasets(
            self.stream(samples_per_shard=[2, 2], shuffle_algo='naive'),
            self.stream(samples_per_shard=[3, 1], shuffle_algo='naive'),
        )

    def test_check_can_be_waived(self):
        ZipIterableStreamingDatasets(
            self.stream(samples_per_shard=[2, 2]),
            self.stream(samples_per_shard=[3, 1]),
            check_alignment=False,
        )

    def test_plain_datasets_are_skipped(self, ds_alpha, ds_beta):
        """Sources with no shard metadata carry nothing to check."""
        ZipIterableStreamingDatasets(ds_alpha, ds_beta)
