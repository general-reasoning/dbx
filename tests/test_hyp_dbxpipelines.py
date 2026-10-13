"""HypDbxPipelines' corpus, from a fake datalake on local disk read by dbx itself."""

import re
import sys

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
    assert dbxpipelines.rebase("/data/lake/x.yaml", "ssh://host/data/lake") == "ssh://host/data/lake/x.yaml"
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
    assert train.embed == "trainpipeline\npkg.TrainPipeline\n{'lr': 0.1}\nTrain(lr: float)"
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

        def read(self, thing, raw=False):
            return "$pkg.Block()" if thing == "quote" else None

    monkeypatch.setattr(dbx.DataJournal, "read_entries", lambda paths, **kw: [Entry(gitrepo="/home/me/dbx:/home/me/proj")])
    monkeypatch.setattr(dbx.DataJournalEntry, "column", staticmethod(lambda entry, name: entry.get(name)))
    monkeypatch.setattr(dbx.SshRay, "session", classmethod(lambda cls, host, python: calls["session"].append((host, python)) or SimpleNamespace(address="ray://localhost:9")))
    monkeypatch.setitem(dbxpipelines._REMOTE_, "key", None)
    return calls


from types import SimpleNamespace  # noqa: E402


def test_read_instantiates_on_the_datalakes_host_and_keeps_the_block(remote_dbx):
    row = {"entry_path": "ssh://host/data/lake/e.parquet", "id": "e1", "anchorkeypath": "/data/lake/pkg.Probe/aa20"}
    text = dbxpipelines.read_text(row, ("classes",), python="~/envs/x/bin/python")
    assert text.splitlines()[0].endswith("trueinst() on host")
    assert text.splitlines()[1:] == ["(a proxy to the value on host)", "", "['a', 'b']"]
    assert remote_dbx["session"] == [("host", "~/envs/x/bin/python")]
    assert remote_dbx["trueinst"] == [{"address": "ray://localhost:9", "gitrepo": "/home/me/dbx:/home/me/proj"}]
    # Read again: the same block, not another worker.
    assert dbxpipelines.read_text(row, ("n",)).splitlines()[1:] == ["builtins.int", "", "3"]
    assert len(remote_dbx["trueinst"]) == 1
    # Another entry: instantiated afresh.
    dbxpipelines.read_text({**row, "id": "e2"}, ("n",))
    assert len(remote_dbx["trueinst"]) == 2


def test_the_head_runs_in_the_python_the_entrys_build_recorded(remote_dbx, monkeypatch):
    import dbx

    monkeypatch.setattr(dbx.DataJournal, "read_entries", lambda paths, **kw: [
        type("E", (dict,), {"trueinst": lambda self, **kw: object(), "read": lambda self, *a, **k: None})(
            python="/env/bin/python")])
    dbxpipelines.read_text({"entry_path": "ssh://host/lake/e.parquet", "id": "e1"}, ("n",))
    assert remote_dbx["session"] == [("host", "/env/bin/python")]


def test_an_override_beats_the_recorded_python_and_python3_stands_in_for_none():
    assert dbxpipelines.remote_python({"python": "/env/bin/python"}, "/other/bin/python") == "/other/bin/python"
    assert dbxpipelines.remote_python({"python": "/env/bin/python"}) == "/env/bin/python"
    assert dbxpipelines.remote_python({}) == dbxpipelines.FALLBACK_PYTHON


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


@pytest.fixture
def pipelines_app(monkeypatch):
    """A HypDbxPipelines with one point and no reads behind it."""
    import os

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from hyp.app import Worker

    monkeypatch.setattr(dbxpipelines.HypDbxPipelines, "run_async", lambda self, fn, then: None)
    app = dbxpipelines.HypDbxPipelines(engine=None, worker=Worker())
    row = {"anchorkeypath": "/lake/pkg.Train/k", "datalake": "/lake", "tag": "pkg.train"}
    app.snap = SimpleNamespace(points=[{"row": row}], summaries=["train"], texts=["t"])
    app.constructed = {"/lake/pkg.Train/k": ["/lake/pkg.Model/v1", "/lake/pkg.Model/v2"]}
    app.opened = []
    monkeypatch.setattr(app, "__open_point__", lambda i: app.opened.append(("pipeline", i)))
    monkeypatch.setattr(app, "_open_entrypoint_", lambda row, p: app.opened.append(("entrypoint", p)))
    return app


def test_a_left_click_on_a_pipeline_opens_it(pipelines_app):
    pipelines_app.__click_point__(0, 1.0)
    assert pipelines_app.opened == [("pipeline", 0)]


def test_a_right_click_on_a_pipeline_lists_its_entrypoints_there(pipelines_app):
    pipelines_app.__context_point__(0, (200, 150))
    popup = pipelines_app.popup
    assert popup.open and popup.button.topleft == (200, 150)
    assert popup.current() == [("pkg.Model/v1", "/lake/pkg.Model/v1"), ("pkg.Model/v2", "/lake/pkg.Model/v2")]
    popup.select("/lake/pkg.Model/v2")
    assert pipelines_app.opened == [("entrypoint", "/lake/pkg.Model/v2")], "no pipeline window under it"


def test_a_pipelines_version_is_its_recorded_one(pipelines_app):
    point = pipelines_app.snap.points[0]
    assert pipelines_app.__point_version__(0) is None
    point["row"]["version"] = "None"
    assert pipelines_app.__point_version__(0) is None
    point["row"]["version"] = "2"
    assert pipelines_app.__point_version__(0) == "2"
    assert "versions" in pipelines_app.settings and pipelines_app.settings["versions"]


def test_note_follows_slices_and_dfn_follows_cite_deslashed(tmp_path):
    dfn = r"""table: '$pkg.T(spec={\'n\': 1})'""" + "\n"
    for name, text in (("cite.txt", "$pkg.T(spec={})"), ("dfn.txt", dfn), ("spec.yaml", "n: 1\n"),
                       ("note.txt", "slices recorded:\nwhat can be read")):
        (tmp_path / name).write_text(text)
    row = {"anchor": "pkg.T", "tag": "pkg.t", "datetime": "2026-10-12T20-54-57.071615",
           "topics": "{'done': DATAFILE('done')}", "paths": "{'done': '/x/done'}",
           "slices": "{'features': None, 'labels': None}", "note": str(tmp_path / "note.txt"),
           "session": "s1", "cite": str(tmp_path / "cite.txt"), "dfn": str(tmp_path / "dfn.txt"),
           "spec": str(tmp_path / "spec.yaml"), "entry_path": str(tmp_path / "e.parquet")}
    lines = dbxpipelines.entry(row).text.split("\n")
    at = {ln.split()[0]: i for i, ln in enumerate(lines) if ln and not ln.startswith(" ")}
    assert at["slices"] < at["note"] < at["session"], "the note right after what can be read"
    assert lines[at["note"]].endswith("slices recorded:") and lines[at["note"] + 1].strip() == "what can be read"
    sections = [ln for ln in lines if ln.startswith("── ")]
    assert sections[:3] == ["── cite ──", "── dfn ──", "── spec ──"]
    dfn = lines[lines.index("── dfn ──") + 1 :]
    assert dfn[0] == "table: '$pkg.T(spec={'n': 1})'", "shown deslashed"


def test_reload_picks_up_new_pipelines_and_lays_out_afresh(tmp_path, monkeypatch):
    from hyp import layout

    lake, data = tmp_path / "lake", tmp_path / "data"
    monkeypatch.setenv("DBX_DATALAKE", str(lake))
    for i, name in enumerate("ABCDEF"):
        write_run(lake, f"{name}Pipeline", f"k{i}", i, f"x: {i}\n")
    engine = Engine(Store(data), HashEmbedder(), EchoSummarizer(), source=lambda progress: dbxpipelines.pipelines(progress))
    engine.start(quiet)
    assert len(engine.store) == 6
    write_run(lake, "GPipeline", "k6", 6, "x: 6\n")
    laid = []
    real = layout.compute
    monkeypatch.setattr(layout, "compute", lambda emb, **kw: laid.append(kw) or real(emb, **kw))
    engine.reload(quiet)
    assert len(engine.store) == 7 and engine.store.layout.shape == (7,)
    assert laid == [{"seed": 0}], "laid out afresh, not seeded from the old layout"
    assert len(Store(data)) == 7, "and saved"


def test_reload_rereads_what_pipeline_windows_keep(pipelines_app):
    reads = []
    pipelines_app.run_async = lambda fn, then: reads.append(fn)
    pipelines_app.sessions = {"s1": []}
    pipelines_app.__reload__()
    assert pipelines_app.journal is None and pipelines_app.constructed is None and pipelines_app.sessions == {}
    assert len(reads) == 2


class RestEmbedder:
    """A base embedder whose vectors are what the test says: by the text after the label."""
    name = "fake:rest"

    def __init__(self, vectors):
        self.vectors = vectors

    def prepare(self, progress=print):
        pass

    def __call__(self, texts):
        return np.array([self.vectors[t] for t in texts], dtype=np.float32)


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_a_labels_vector_is_its_own_and_near_its_neighbours():
    v = dbxpipelines.LabelFirstEmbedder.label_vector
    assert np.allclose(v("gigapath_probe_partition"), v("gigapath_probe_partition"))
    assert abs(np.linalg.norm(v("x")) - 1) < 1e-6
    assert cosine(v("gigapath_probe_partition"), v("gigapath_probe")) > cosine(v("gigapath_probe_partition"), v("pancan_tile_fold"))


def test_one_label_outweighs_any_difference_in_the_rest():
    rest = {"a": [1.0, 0.0], "b": [-1.0, 0.0], "c": [1.0, 0.0]}
    emb = dbxpipelines.LabelFirstEmbedder(RestEmbedder(rest))
    same_a, same_b, other = emb(["gigapath_probe_partition\na", "gigapath_probe_partition\nb", "pancan_tile_fold\nc"])
    assert cosine(same_a, same_b) >= 0.8 - 1e-6, "opposite rests, one label: still alike"
    assert cosine(same_a, same_b) > cosine(same_a, other), "identical rests, other labels: less alike"
    assert emb.name == "label-first(3):fake:rest"


def test_pipelines_sharing_a_label_lie_together():
    from hyp import layout

    rng = np.random.default_rng(0)
    labels = ["gigapath_probe_partition", "gigapath_deep_feature_clip", "pancan_tile_fold", "pancan_tile_clip"]
    texts = [f"{lab}\n{lab}-{i}" for lab in labels for i in range(12)]
    rest = {t.split("\n")[1]: rng.normal(size=32) for t in texts}
    positions = layout.compute(dbxpipelines.LabelFirstEmbedder(RestEmbedder(rest))(texts), seed=0)
    near = layout.nearest(np.stack([positions.real, positions.imag], axis=1), 3)
    agree = np.mean([texts[j].split("\n")[0] == texts[i].split("\n")[0] for i in range(len(texts)) for j in near[i]])
    assert agree > 0.9


def test_a_pipelines_embed_text_starts_with_its_label(tmp_path, monkeypatch):
    monkeypatch.setenv("DBX_DATALAKE", str(tmp_path))
    write_run(tmp_path, "TrainPipeline", "aaaa", 0, "lr: 0.1\n")
    (e,) = dbxpipelines.pipelines(quiet)
    assert e.embed.split("\n")[:2] == ["trainpipeline", "pkg.TrainPipeline"]
    assert isinstance(dbxpipelines.HypDbxPipelines.__embedder__(RestEmbedder({})), dbxpipelines.LabelFirstEmbedder)


class FakeTablet:
    """A tablet of FakeTable: *n* rows, a vector and a label each."""

    def __init__(self, i, n):
        self.i, self.n = i, n

    def n_rows(self, s):
        return self.n

    def data(self, *specs):
        vec = [np.full(4, self.i * 100 + r, dtype=np.int16) for r in range(self.n)]
        lab = [f"bag{self.i}"] * self.n
        return {"features": {"final": vec}, "bag_name": {"bag_name": lab}}


class FakeTable:
    n_tablets = 5

    def tablet(self, i):
        return FakeTablet(i, 10 + i)


@pytest.fixture
def fake_table(monkeypatch):
    import dbx

    monkeypatch.setattr(dbx, "eval", lambda quote: FakeTable(), raising=False)
    return FakeTable()


SLICED_ROW = {"entry_path": "ssh://host/lake/e.parquet", "id": "e1", "anchorkeypath": "/lake/pkg.T/k",
              "datalake": "/lake",
              "slices": "{'features': {'final': 'ndarray:int8'}, 'bag_name': {'bag_name': 'str'}, 'tiles': None}"}


def test_declared_columns_are_the_journals():
    assert dbxpipelines.declared_columns(SLICED_ROW) == {"features": {"final": "ndarray:int8"}, "bag_name": {"bag_name": "str"}}
    assert dbxpipelines.recorded_slices(SLICED_ROW) == ["features", "bag_name", "tiles"]
    assert dbxpipelines.declared_columns({"slices": "{'tiles': None}"}) == {}, "a bare DATASLICE declares nothing to view"
    assert dbxpipelines.declared_columns({"slices": "None"}) == {} and dbxpipelines.declared_columns(None) == {}


def test_describe_plans_by_the_first_tablet(fake_table):
    describe, _ = dbxpipelines._remote_fns_()
    assert describe("$q", ["features"]) == {"tablets": 5, "rows0": {"features": 10}}


@pytest.mark.parametrize("workers", [1, 2])
def test_sample_takes_rows_of_each_tablet(fake_table, workers):
    _, sample = dbxpipelines._remote_fns_()
    specs = [("features", ["final"]), ("bag_name", ["bag_name"])]
    got = sample("$q", specs, [1, 3], 1.0, 4, 0, workers, ("features", "final"), ("bag_name", "bag_name"))
    assert got["X"].shape == (8, 4) and got["X"].dtype == np.float32 and got["errors"] == []
    assert [t for t, _ in got["rows"]] == [1] * 4 + [3] * 4, "per_tablet rows of each, in tablet order"
    assert got["labels"] == ["bag1"] * 4 + ["bag3"] * 4
    assert all(got["X"][k][0] == t * 100 + r for k, (t, r) in enumerate(got["rows"])), "each vector its row's"
    half = sample("$q", specs[:1], [0, 4], 0.5, None, 0, workers, ("features", "final"), None)
    assert len(half["X"]) == 5 + 7 and half["labels"] is None


def test_slice_sample_runs_in_the_pinned_worker(fake_table, monkeypatch):
    class Handle:
        def run(self, fn, *args):
            return fn(*args)

    monkeypatch.setattr(dbxpipelines, "_quote_and_handle_", lambda row, python: ("$q", Handle()))
    X, L, rows, errors = dbxpipelines.slice_sample(SLICED_ROW, ("features", "final"), ("bag_name", "bag_name"),
                                                   tablets=[0, 2], workers=1, per_tablet=3)
    assert X.shape == (6, 4) and L == ["bag0"] * 3 + ["bag2"] * 3 and rows[0][0] == 0 and not errors


def test_the_window_samples_what_is_picked_and_shows_it(pipelines_app, monkeypatch):
    from types import SimpleNamespace as NS

    from hyp import layout

    app = pipelines_app
    app.run_async = lambda fn, then: then(fn())
    calls = []
    monkeypatch.setattr(dbxpipelines, "slice_plan", lambda row, python: {"tablets": 532, "rows0": {"features": 286}})
    monkeypatch.setattr(dbxpipelines, "slice_sample", lambda row, data, labels, fraction, **kw: calls.append((data, labels, fraction, kw))
                        or (np.eye(6, dtype=np.float32), ["a", "a", "b", "b", "c", "c"], [(0, r) for r in range(6)], []))
    monkeypatch.setattr(layout, "compute", lambda X, seed=0: np.linspace(0, 0.5, len(X)).astype(complex))
    window = NS(row=SLICED_ROW)
    menus = app._slice_menus_(window)
    data = window.slice_menus["data"]
    assert [v for _, v in data.current()] == [("features", "final", ()), ("bag_name", "bag_name", ())]
    data.select(("features", "final", ()))
    window.slice_menus["labels"].select(("bag_name", "bag_name", ()))
    count, fraction, per, expect = app._plan_(window)
    assert (count, per, expect) == (64, 63, 63 * 64), "auto: 64 tablets, 4,000 rows split among them"
    menus[-1].on_click()
    (data_, labels_, fraction_, kw), = calls
    assert data_ == ("features", "final", ()) and labels_ == ("bag_name", "bag_name", ())
    assert len(kw["tablets"]) == 64 and kw["per_tablet"] == 63 and kw["workers"] == dbxpipelines.DEFAULT_WORKERS
    disk = app.viewers[-1]
    assert len(disk.positions) == 6 and disk.note is None and [l for _, l, _ in disk.legend] == ["a", "b", "c"]


def test_undeclared_slices_can_not_be_viewed(pipelines_app):
    from types import SimpleNamespace as NS

    window = NS(row={**SLICED_ROW, "slices": "{'tiles': None, 'annotations': None}"})
    app = pipelines_app
    menus = app._slice_menus_(window)
    assert window.slice_menus["data"].visible() and window.slice_menus["data"].current() == []
    assert "declare none" in window.slice_menus["data"].empty
    assert not window.slice_menus["rows"].visible() and not menus[-1].visible()


DICT_ROW = {**SLICED_ROW, "slices": ("{'features': {'final': 'ndarray:int8'}, "
                                     "'annotations': {'annotations': {'cohort': 'str', 'gdc': {'stage': 'str', 'age': 'int'}}}}")}


def test_a_dict_column_is_offered_key_by_key():
    assert dbxpipelines.column_choices(DICT_ROW) == [
        ("features", "final", (), "ndarray:int8"),
        ("annotations", "annotations", ("cohort",), "str"),
        ("annotations", "annotations", ("gdc", "stage"), "str"),
        ("annotations", "annotations", ("gdc", "age"), "int"),
    ]


class FakeDictTable(FakeTable):
    """Its rows' annotations a dict, as data() reads a dict column narrowed to the keys asked: one key, the leaf; two, a pruned dict."""

    asked = []

    def tablet(self, i):
        table = self

        class Tablet(FakeTablet):
            def data(self, *specs):
                table.asked.append(specs)
                got = super().data()
                selections = dict(specs).get("annotations", [])
                rows = [{"cohort": f"c{self.i}", "gdc": {"stage": f"s{r % 2}"}} for r in range(self.n)]
                if len(selections) == 1 and isinstance(selections[0], tuple):
                    path = selections[0][1:]
                    leaf = rows
                    for k in path:
                        leaf = [v[k] for v in leaf]
                    got["annotations"] = {"annotations": leaf}
                else:
                    got["annotations"] = {"annotations": rows}
                return got

        return Tablet(i, 10 + i)


def test_a_key_of_a_dict_column_labels_the_points(monkeypatch):
    import dbx

    monkeypatch.setattr(dbx, "eval", lambda quote: FakeDictTable(), raising=False)

    class Handle:
        def run(self, fn, *args):
            return fn(*args)

    monkeypatch.setattr(dbxpipelines, "_quote_and_handle_", lambda row, python: ("$q", Handle()))
    FakeDictTable.asked = []
    X, L, rows, errors = dbxpipelines.slice_sample(DICT_ROW, ("features", "final", ()), ("annotations", "annotations", ("gdc", "stage")),
                                                   tablets=[0, 1], workers=1, per_tablet=4)
    assert FakeDictTable.asked[0] == (("features", ["final"]), ("annotations", [("annotations", "gdc", "stage")]))
    assert set(L) <= {"s0", "s1"} and len(L) == 8 and not errors
    # Two keys of one column: dbx hands back the dict pruned to both, and each pick is walked to its own.
    FakeDictTable.asked = []
    X, L, rows, errors = dbxpipelines.slice_sample(DICT_ROW, ("features", "final", ()), ("annotations", "annotations", ("cohort",)),
                                                   tablets=[1], workers=1, per_tablet=3)
    assert L == ["c1"] * 3


def test_a_dumps_blocks_are_shown_not_made(tmp_path, monkeypatch):
    """A dfn holds its upstream blocks: reading it for show must not instantiate them -- here, with whatever code this process has."""
    monkeypatch.setitem(sys.modules, "nowhere_pkg", None)  # not importable: making it would fail
    dump = tmp_path / "dfn.yaml"
    dump.write_text(
        "spec:\n"
        "  columns: !!python/tuple [annotations, case_id]\n"
        "  upstream: !!python/object:nowhere_pkg.Block\n"
        "    spec: {fold: 1}\n"
        "  fn: !!python/name:nowhere_pkg.fn ''\n"
    )
    data = dbxpipelines.read_dump(str(dump))
    assert data["spec"]["columns"] == ("annotations", "case_id")
    assert repr(data["spec"]["upstream"]) == "nowhere_pkg.Block({'spec': {'fold': 1}})"
    assert repr(data["spec"]["fn"]) == "nowhere_pkg.fn"
    shown, _ = dbxpipelines._side_file_("dfn", str(dump), "")
    assert "nowhere_pkg.Block({" in shown and "'fold': 1" in shown
