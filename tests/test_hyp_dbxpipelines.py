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


def test_pipelines_aggregate_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("DBX_DATALAKE", str(tmp_path))
    write_run(tmp_path, "TrainPipeline", "aaaa", 0, "lr: 0.1\n", signature="Train(lr: float)")
    write_run(tmp_path, "TrainPipeline", "aaaa", 1, "lr: 0.1\n", signature="Train(lr: float)")
    write_run(tmp_path, "EvalPipeline", "bbbb", 2, "split: test\n")
    entries = dbxpipelines.pipelines(quiet)
    assert [e.summary for e in entries] == ["evalpipeline · 2026-10-03 12:00", "trainpipeline · 2026-10-02 12:00"]
    train = entries[1]
    assert train.extra["label"] == "trainpipeline"
    # Both runs, one point: every entrypath is shown, relative to the datalake.
    assert re.search(r"^entrypaths +pkg\.Model/v1\n +pkg\.Model/v0$", train.text, re.M)
    # Side files are shown by content, not by path.
    assert "spec-0.yaml" not in train.text and "spec-1.yaml" not in train.text
    # A spec is shown, and embedded, as the dict it is.
    assert "── spec ──\n{\n    'lr': 0.1\n}" in train.text
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


def test_engine_updates_a_rerun_pipeline_in_place(tmp_path, monkeypatch):
    lake, data = tmp_path / "lake", tmp_path / "data"
    monkeypatch.setenv("DBX_DATALAKE", str(lake))
    for i, name in enumerate("ABCDEF"):
        write_run(lake, f"{name}Pipeline", f"k{i}", i, f"x: {i}\n")
    source = lambda progress: dbxpipelines.pipelines(progress)  # noqa: E731
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
    assert "\n    'x': 99\n" in s.points[a]["text"]
    assert not np.allclose(s.embeddings[a], before[a])
    assert s.layout.shape == (6,)


def test_journal_entries_and_constructed_entrypoints(tmp_path, monkeypatch):
    monkeypatch.setenv("DBX_DATALAKE", str(tmp_path))
    write_run(tmp_path, "TrainPipeline", "aaaa", 0, "lr: 0.1\n")
    write_run(tmp_path, "TrainPipeline", "aaaa", 1, "lr: 0.2\n")
    write_run(tmp_path, "EvalPipeline", "bbbb", 2, "split: test\n")
    # Of Train's two entrypoints, only v1 was built.
    write_build(tmp_path, "pkg.Model/v1", 3, "build:start")
    write_build(tmp_path, "pkg.Model/v1", 4, "build:end")

    entries = dbxpipelines.journal_entries()
    assert len(entries) == 3
    assert list(entries[entries["tag"] == "pkg.pipelines.trainpipeline"]["hash"]) == ["haaaa", "haaaa"]

    train = f"{tmp_path}/pkg.TrainPipeline/pkg.pipelines.trainpipeline/aaaa"
    assert dbxpipelines.constructed_entrypaths() == {train: [f"{tmp_path}/pkg.Model/v1"]}

    row = {"datalake": str(tmp_path), "entry_path": str(tmp_path / "x.parquet")}
    text, newest = dbxpipelines.entrypoint_text(f"{tmp_path}/pkg.Model/v1", row)
    assert newest["event"] == "build:end"
    assert text.startswith("── events ──\n2026-10-05 13:00:00  build:end\n2026-10-04 13:00:00  build:start\n")
    assert re.search(r"^event +build:end$", text, re.M)


def test_the_corpus_is_kept_in_the_datalake(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    monkeypatch.setenv("DBX_DATALAKE", str(lake))
    for i, name in enumerate("ABC"):
        write_run(lake, f"{name}Pipeline", f"k{i}", i, f"x: {i}\n")
    cls, args = dbxpipelines.HypDbxPipelines, None
    data = cls.__default_data__(args)
    assert data == f"{lake}/.hyp"
    source = lambda progress: dbxpipelines.pipelines(progress)  # noqa: E731
    Engine(Store(data, cls.__storage_options__(args)), HashEmbedder(), EchoSummarizer(), source=source).start(quiet)
    assert sorted(p.name for p in (lake / ".hyp").iterdir()) == ["embeddings.npy", "layout.npy", "meta.json", "points.jsonl"]
    assert len(Store(data)) == 3


def test_no_datalake_is_refused(monkeypatch):
    for var in ("DBX_DATALAKE", "DBX_ROOT", "DBX_URL"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(SystemExit, match="DBX_DATALAKE"):
        dbxpipelines.HypDbxPipelines.__default_data__(None)


def test_the_tag_menu_names_each_run_by_its_var_name(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("DBX_DATALAKE", str(tmp_path))
    write_run(tmp_path, "TrainPipeline", "aaaa", 0, "lr: 0.1\nname: gigapath clip\n")
    write_run(tmp_path, "TrainPipeline", "aaaa", 1, "lr: 0.2\nname: uni clip\n")
    write_run(tmp_path, "TrainPipeline", "aaaa", 2, "lr: 0.3\n")
    entries = dbxpipelines.journal_entries()
    assert sorted(entries["name"].fillna("-")) == ["-", "gigapath clip", "uni clip"]

    app = SimpleNamespace(journal=entries)
    items = [text for text, _ in dbxpipelines.HypDbxPipelines._datetimes_(app, "pkg.pipelines.trainpipeline")]
    assert items == [
        "2026-10-03 12:00:00   haaaa   create",
        "2026-10-02 12:00:00   uni clip   haaaa   create",
        "2026-10-01 12:00:00   gigapath clip   haaaa   create",
    ]


def test_a_point_carries_its_var_name(tmp_path, monkeypatch):
    monkeypatch.setenv("DBX_DATALAKE", str(tmp_path))
    write_run(tmp_path, "TrainPipeline", "aaaa", 0, "lr: 0.1\nname: gigapath clip\n")
    write_run(tmp_path, "EvalPipeline", "bbbb", 1, "split: test\n")
    by_anchor = {e.extra["anchor"]: e.extra["name"] for e in dbxpipelines.pipelines(quiet)}
    assert by_anchor == {"pkg.TrainPipeline": "gigapath clip", "pkg.EvalPipeline": None}


def test_has_output_only_where_a_capture_was_recorded():
    assert dbxpipelines.has_output({"log": "/lake/.journal/sessions/s/output/master.log"})
    assert not dbxpipelines.has_output({"log": "None"})
    assert not dbxpipelines.has_output({})
    assert not dbxpipelines.has_output(None)


def test_output_text_reads_the_capture_as_it_was_last_drawn(tmp_path):
    log = tmp_path / ".journal" / "sessions" / "s" / "output" / "master-h-1.log"
    log.parent.mkdir(parents=True)
    log.write_bytes(b"\x1b[32mstarting\x1b[0m\nepoch 1:  10%|#\repoch 1:  50%|#####\repoch 1: 100%|##########\r\ndone\n")
    row = {"log": str(log), "entry_path": str(tmp_path / "x.parquet")}
    text = dbxpipelines.output_text(row)
    assert text.splitlines()[0] == str(log)
    assert text.splitlines()[2:] == ["starting", "epoch 1: 100%|##########", "done"]


def test_output_text_shows_the_end_of_a_long_capture(tmp_path):
    log = tmp_path / "long.log"
    log.write_text("".join(f"line {i}\n" for i in range(1000)))
    text = dbxpipelines.output_text({"log": str(log), "entry_path": str(tmp_path / "x.parquet")}, limit=100)
    assert "bytes left out" in text.splitlines()[1]
    assert text.rstrip().endswith("line 999")
    assert "line 0\n" not in text


def test_output_text_of_a_capture_not_yet_uploaded(tmp_path):
    text = dbxpipelines.output_text({"log": str(tmp_path / "absent.log"), "entry_path": str(tmp_path / "x.parquet")})
    assert "may still be running" in text


def test_the_output_button_follows_the_entry_shown():
    from types import SimpleNamespace

    viewer = SimpleNamespace(row={"log": "None"})
    button = dbxpipelines.HypDbxPipelines._output_button_(SimpleNamespace(_open_output_=lambda row: None), viewer)
    assert not button.visible()
    viewer.row = {"log": "/lake/.journal/sessions/s/output/master.log"}
    assert button.visible()


def test_session_blocks_are_read_from_the_session_index(tmp_path):
    def entry_file(entrypath, i, event):
        write_build(tmp_path, entrypath, i, event)
        return next((tmp_path / entrypath).rglob(f"*-{i}-{event.replace(':', '-')}.parquet"))

    written = [
        ("2026-10-12T16-01-00.000000", entry_file("pkg.Model/v1", 1, "build:start")),
        ("2026-10-12T16-02-00.000000", entry_file("pkg.Data/d1", 2, "build:end")),
        ("2026-10-12T16-30-00.000000", entry_file("pkg.Model/v1", 3, "build:end")),
    ]
    # A pipeline's creation, under the same session: filed in the pipeline journal, and left out.
    write_run(tmp_path, "TrainPipeline", "aaaa", 4, "lr: 0.1\n")
    written.append(("2026-10-12T16-00-00.000000", next((tmp_path / ".pipelines").rglob("*-journal-*.parquet"))))
    index = tmp_path / ".journal" / "sessions" / "s1"
    index.mkdir(parents=True)
    for j, (dt, path) in enumerate(written):
        (index / f"m{j}.txt").write_text(f"{dt}\n{path}")

    row = {"session": "s1", "datalake": str(tmp_path), "entry_path": str(tmp_path / "x.parquet")}
    assert dbxpipelines.session_blocks(row) == [
        (f"{tmp_path}/pkg.Model/v1", "build:end"),
        (f"{tmp_path}/pkg.Data/d1", "build:end"),
    ]
    assert dbxpipelines.session_blocks({**row, "session": "not-a-command"}) == []
    assert dbxpipelines.session_blocks({**row, "session": "None"}) == []


def block_row(tmp_path):
    """A block's journal row as a window keeps one: strings, its topics and paths recorded as dbx records them."""
    akp = tmp_path / "pkg.Probe" / "aa20"
    (akp / "fit" / "labels").mkdir(parents=True)
    (akp / "fit" / "labels" / "labels.npz").write_bytes(b"x" * 10)
    (akp / "coef").mkdir()
    (akp / "coef" / "coef.npy").write_bytes(b"y" * 20)
    return {
        "datalake": str(tmp_path),
        "entry_path": str(tmp_path / "x.parquet"),
        "anchorkeypath": str(akp),
        "topics": "{'fit': {'labels': DATAFILE('labels.npz')}, 'coef': DATAFILE('coef.npy'), 'cache': SYNTHETIC}",
        "paths": str({"fit": {"labels": str(akp / "fit" / "labels" / "labels.npz")}, "coef": str(akp / "coef" / "coef.npy"), "cache": None}),
    }


def test_topics_are_the_recorded_leaves(tmp_path):
    assert dbxpipelines.topics(block_row(tmp_path)) == [(("fit", "labels"), True), (("coef",), True), (("cache",), False)]
    assert dbxpipelines.topics({"topics": "None"}) == []
    assert dbxpipelines.topics(None) == []


def test_ls_lists_the_recorded_path(tmp_path):
    text = dbxpipelines.ls_text(block_row(tmp_path), ("fit", "labels"))
    assert text.splitlines()[0].startswith("ls fit/labels of ")
    assert re.search(r"^\s+10  \S+/fit/labels/labels\.npz$", text, re.M)


class FakeRemote:
    """A value left on the worker: its repr only by name, as a dbx Remote proxy forwards it."""

    def __init__(self, value):
        self._value = value

    def __getattr__(self, name):
        return getattr(self._value, name)


FakeRemote.__name__ = "Remote"


@pytest.fixture
def remote_dbx(monkeypatch):
    """dbx as read_text uses it, with no Ray: an entry whose trueinst() is recorded, and an SshRay that starts nothing."""
    import dbx

    calls = {"trueinst": [], "session": []}

    class Block:
        def read(self, *topic):
            return {"classes": FakeRemote(["a", "b"]), "n": 3, "long": "x" * 50}[topic[0]]

    class Entry(dict):
        def trueinst(self, **kwargs):
            calls["trueinst"].append(kwargs)
            return Block()

    monkeypatch.setattr(dbx.DataJournal, "read_entries", lambda paths, **kw: [Entry(gitrepo="/home/me/dbx:/home/me/proj")])
    monkeypatch.setattr(dbx.DataJournalEntry, "column", staticmethod(lambda entry, name: entry.get(name)))
    monkeypatch.setattr(dbx.SshRay, "session", classmethod(lambda cls, host, python: calls["session"].append((host, python)) or SimpleNamespace(address="ray://localhost:9")))
    monkeypatch.setitem(dbxpipelines._REMOTE_, "key", None)
    return calls


from types import SimpleNamespace  # noqa: E402


def test_read_instantiates_on_the_datalakes_host_and_keeps_the_block(remote_dbx):
    row = {"entry_path": "ssh://wheelbarrow/data/lake/e.parquet", "id": "e1", "anchorkeypath": "/data/lake/pkg.Probe/aa20"}
    text = dbxpipelines.read_text(row, ("classes",), python="~/envs/x/bin/python")
    assert text.splitlines()[0].endswith("trueinst() on wheelbarrow")
    assert text.splitlines()[1:] == ["(a proxy to the value on wheelbarrow)", "", "['a', 'b']"]
    assert remote_dbx["session"] == [("wheelbarrow", "~/envs/x/bin/python")]
    assert remote_dbx["trueinst"] == [{"address": "ray://localhost:9", "gitrepo": "/home/me/dbx:/home/me/proj"}]
    # Read again: the same block, not another worker.
    assert dbxpipelines.read_text(row, ("n",)).splitlines()[1:] == ["builtins.int", "", "3"]
    assert len(remote_dbx["trueinst"]) == 1
    # Another entry: instantiated afresh.
    dbxpipelines.read_text({**row, "id": "e2"}, ("n",))
    assert len(remote_dbx["trueinst"]) == 2


def test_a_local_datalake_instantiates_here(remote_dbx):
    text = dbxpipelines.read_text({"entry_path": "/data/lake/e.parquet", "id": "e1"}, ("n",))
    assert text.splitlines()[0].endswith("trueinst() on here")
    assert remote_dbx["trueinst"] == [{}] and remote_dbx["session"] == []


def test_a_long_read_is_cut_short(remote_dbx):
    text = dbxpipelines.read_text({"entry_path": "/e.parquet", "id": "e1"}, ("long",), limit=10)
    assert "its first 10 of 52 characters" in text.splitlines()[1]


def test_a_failed_read_shows_its_traceback(remote_dbx):
    text = dbxpipelines.read_text({"entry_path": "/e.parquet", "id": "e1"}, ("nope",))
    assert "KeyError: 'nope'" in text
