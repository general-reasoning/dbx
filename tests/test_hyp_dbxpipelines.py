"""HypDbxPipelines' corpus, from a fake datalake on local disk read by dbx itself."""

import re

import numpy as np
import pandas as pd
import pytest

from hyp import dbxpipelines
from hyp.engine import Engine
from hyp.store import Store

from test_hyp_engine import EchoSummarizer, quiet



def write_run(lake, anchor, key, i, spec, signature=None):
    """One pipeline journal entry, laid out as dbx's PipeJournal writes it."""
    tag = f"pkg.pipelines.{anchor.lower()}"
    cls_dir = lake / ".pipelines" / f"pkg.{anchor}" / tag / key / ".journal" / f"pkg.{anchor}"
    jdir = cls_dir / "journal" / f"h{key}"
    jdir.mkdir(parents=True, exist_ok=True)
    row = {
        "datetime": f"2026-10-{i + 1:02d}T12-00-00.000000",
        "datalake": str(lake),
        "anchor": f"pkg.{anchor}",
        "hash": f"h{key}",
        "keyby": "tag_version_shorthash",
        "key": f"{tag}/{key}",
        "anchorkeypath": f"{lake}/pkg.{anchor}/{tag}/{key}",
        "tag": tag,
        "version": "None",
        "id": f"e{i}",
        "event": "create",
        "note": "None",
        "entrypath": f"{lake}/pkg.Model/v{i}",
        "paths": "{'entrypoint': None}",
        "gitrepo": "/home/u/dbx",
        "wrkrepo": "/tmp/w/dbx",
    }
    for side, content in (("spec", spec), ("signature", signature), ("cite", f"cite {i}")):
        if content is not None:
            sdir = cls_dir / side / f"h{key}"
            sdir.mkdir(parents=True, exist_ok=True)
            path = sdir / f"{side}-{i}.{'yaml' if side == 'spec' else 'txt'}"
            path.write_text(content)
            row[side] = str(path)
    pd.DataFrame([row]).to_parquet(jdir / f"pkg.{anchor}-journal-h{key}-{i}.parquet")


def test_urls():
    assert dbxpipelines.normalize_url("ssh://wheelbarrow:/data/lake") == "ssh://wheelbarrow/data/lake"
    assert dbxpipelines.normalize_url("/data/lake") == "/data/lake"
    assert dbxpipelines.slug("ssh://wheelbarrow:/data/lake") == "wheelbarrow-data-lake"
    # Side file paths are as the datalake's host wrote them.
    assert dbxpipelines.rebase("/data/lake/x.yaml", "ssh://wheelbarrow/data/lake") == "ssh://wheelbarrow/data/lake/x.yaml"
    assert dbxpipelines.rebase("/data/lake/x.yaml", "/data/lake") == "/data/lake/x.yaml"


def write_build(lake, entrypath, i, event="build:end"):
    """An entry in the journal of the block at *entrypath*, as a Datablock writes its own."""
    jdir = lake / entrypath / ".journal" / "pkg.Model" / "journal" / "hm"
    jdir.mkdir(parents=True, exist_ok=True)
    row = {
        "datetime": f"2026-10-{i + 1:02d}T13-00-00.000000",
        "datalake": str(lake),
        "anchor": "pkg.Model",
        "hash": "hm",
        "key": entrypath.split("/", 1)[1],
        "anchorkeypath": f"{lake}/{entrypath}",
        "tag": "pkg.model",
        "event": event,
    }
    pd.DataFrame([row]).to_parquet(jdir / f"pkg.Model-journal-hm-{i}-{event.replace(':', '-')}.parquet")


def test_pipelines_aggregate_runs(tmp_path):
    write_run(tmp_path, "TrainPipeline", "aaaa", 0, "lr: 0.1\n", signature="Train(lr: float)")
    write_run(tmp_path, "TrainPipeline", "aaaa", 1, "lr: 0.1\n", signature="Train(lr: float)")
    write_run(tmp_path, "EvalPipeline", "bbbb", 2, "split: test\n")
    entries = dbxpipelines.pipelines(str(tmp_path), quiet)
    assert [e.summary for e in entries] == ["evalpipeline · 2026-10-03 12:00", "trainpipeline · 2026-10-02 12:00"]
    train = entries[1]
    assert train.extra["label"] == "trainpipeline"
    # Both runs, one point: every entrypath is shown, relative to the datalake.
    assert re.search(r"^entrypaths +pkg\.Model/v1\n +pkg\.Model/v0$", train.text, re.M)
    # Side files are shown by content, not by path.
    assert "spec-0.yaml" not in train.text and "spec-1.yaml" not in train.text
    # A spec is shown, and embedded, as the dict it is.
    assert "── spec ──\n{'lr': 0.1}" in train.text
    assert train.embed == "pkg.TrainPipeline\n{'lr': 0.1}\nTrain(lr: float)"
    # cite right below the table, which ends at paths; the repos and
    # entry_path below everything.
    # paths, as topics would be, one key per line.
    assert re.search(r"^paths +\{\n +'entrypoint': None\n +\}\n\n── cite ──\ncite 1\n", train.text, re.M)
    assert re.search(r"\n\ngitrepo +/home/u/dbx\nentry_path +\S+\nwrkrepo +/tmp/w/dbx$", train.text)
    assert "note" not in train.text  # "None" is dbx's missing value
    assert train.extra["anchor"] == "pkg.TrainPipeline"


class HashEmbedder:
    name = "fake:hash"

    def prepare(self, progress=print):
        pass

    def __call__(self, texts):
        return np.array([np.random.default_rng(abs(hash(t)) % 2**32).normal(size=8) for t in texts], dtype=np.float32)


def test_engine_updates_a_rerun_pipeline_in_place(tmp_path):
    lake, data = tmp_path / "lake", tmp_path / "data"
    for i, name in enumerate("ABCDEF"):
        write_run(lake, f"{name}Pipeline", f"k{i}", i, f"x: {i}\n")
    source = lambda progress: dbxpipelines.pipelines(str(lake), progress)  # noqa: E731
    Engine(Store(data), HashEmbedder(), EchoSummarizer(), source=source).start(quiet)
    assert len(Store(data)) == 6

    # A pipeline run again with a new spec stays one point, re-embedded.
    write_run(lake, "APipeline", "k0", 9, "x: 99\n")
    engine = Engine(Store(data), HashEmbedder(), EchoSummarizer(), source=source)
    before = engine.store.embeddings.copy()
    engine.start(quiet)
    s = engine.store
    assert len(s) == 6
    a = next(i for i, p in enumerate(s.points) if p["anchor"] == "pkg.APipeline")
    assert "{'x': 99}" in s.points[a]["text"]
    assert not np.allclose(s.embeddings[a], before[a])
    assert s.layout.shape == (6,)


def test_journal_entries_and_constructed_entrypoints(tmp_path):
    write_run(tmp_path, "TrainPipeline", "aaaa", 0, "lr: 0.1\n")
    write_run(tmp_path, "TrainPipeline", "aaaa", 1, "lr: 0.2\n")
    write_run(tmp_path, "EvalPipeline", "bbbb", 2, "split: test\n")
    # Of Train's two entrypoints, only v1 was built.
    write_build(tmp_path, "pkg.Model/v1", 3, "build:start")
    write_build(tmp_path, "pkg.Model/v1", 4, "build:end")

    entries = dbxpipelines.journal_entries(str(tmp_path))
    assert len(entries) == 3
    assert list(entries[entries["tag"] == "pkg.pipelines.trainpipeline"]["hash"]) == ["haaaa", "haaaa"]

    train = f"{tmp_path}/pkg.TrainPipeline/pkg.pipelines.trainpipeline/aaaa"
    assert dbxpipelines.constructed_entrypaths(str(tmp_path)) == {train: [f"{tmp_path}/pkg.Model/v1"]}

    row = {"datalake": str(tmp_path), "entry_path": str(tmp_path / "x.parquet")}
    text = dbxpipelines.entrypoint_text(f"{tmp_path}/pkg.Model/v1", row)
    assert text.startswith("── events ──\n2026-10-05 13:00:00  build:end\n2026-10-04 13:00:00  build:start\n")
    assert re.search(r"^event +build:end$", text, re.M)
