"""hyp --docs: a directory's markdowns and pdfs, found, read, laid out, and placed in windows on layers of their own."""

import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
import pytest

from hyp import docs, docview

pygame.init()


def _pdf_(path, pages=2):
    from PIL import Image

    images = [Image.new("RGB", (612, 792), "white") for _ in range(pages)]
    images[0].save(path, save_all=True, append_images=images[1:])


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "a.md").write_text("# Alpha\n\nSome *text*.\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.markdown").write_text("# Beta\n")
    _pdf_(tmp_path / "sub" / "c.pdf")
    (tmp_path / "notes.txt").write_text("not a document")
    for skipped in (".git", "node_modules", "__pycache__"):
        (tmp_path / skipped).mkdir()
        (tmp_path / skipped / "x.md").write_text("# skipped")
    (tmp_path / ".hidden.md").write_text("# hidden")
    return tmp_path


def test_markdowns_and_pdfs_are_found_and_the_rest_skipped(tree):
    fs, infos, cut = docs.find_docs(str(tree))
    assert [os.path.relpath(i["name"], tree) for i in infos] == ["a.md", "sub/b.markdown", "sub/c.pdf"]
    assert not cut


def test_past_the_limit_the_scan_stops_and_says_so(tree):
    _, infos, cut = docs.find_docs(str(tree), limit=2)
    assert len(infos) == 2 and cut


def test_a_document_is_labeled_by_its_name_and_embedded_by_its_text(tree):
    by_name = {d.summary: d for d in docs.documents(str(tree), progress=lambda m: None)}
    a = by_name["a.md"]
    assert a.key == str(tree / "a.md") and a.extra["kind"] == "markdown"
    assert a.embed.startswith("a.md\n\n# Alpha") and a.extra["mtime"] == pytest.approx(os.path.getmtime(tree / "a.md"))
    assert by_name["c.pdf"].extra["kind"] == "pdf"


def test_an_unchanged_document_is_not_read_again(tree, monkeypatch):
    cache, reads = {}, []
    real = docs.read_text
    monkeypatch.setattr(docs, "read_text", lambda fs, info, kind: reads.append(info["name"]) or real(fs, info, kind))
    docs.documents(str(tree), progress=lambda m: None, cache=cache)
    assert len(reads) == 3
    (tree / "a.md").write_text("# Alpha, longer now\n")
    os.utime(tree / "a.md", (time.time() + 10, time.time() + 10))
    again = {d.summary: d for d in docs.documents(str(tree), progress=lambda m: None, cache=cache)}
    assert len(reads) == 4 and "longer now" in again["a.md"].text


@pytest.mark.parametrize("info, expected", [
    ({"mtime": 1700000000.5}, 1700000000.5),
    ({"LastModified": "2023-11-14T22:13:20Z"}, 1700000000.0),
    ({"last_modified": __import__("datetime").datetime.fromtimestamp(1700000000, __import__("datetime").timezone.utc)}, 1700000000.0),
    ({}, None),
])
def test_modification_times_however_a_filesystem_names_them(info, expected):
    assert docs._mtime_(info) == expected


@pytest.mark.parametrize("url, top, name, expected", [
    ("ssh://host/data/docs", "/data/docs", "/data/docs/a.md", "ssh://host/data/docs/a.md"),
    ("sftp://me@host:22/data", "/data", "/data/b.pdf", "sftp://me@host:22/data/b.pdf"),
    ("gs://bucket/docs", "bucket/docs", "bucket/docs/a.md", "gs://bucket/docs/a.md"),
])
def test_a_remote_documents_url_keeps_its_host(url, top, name, expected):
    """fsspec's unstrip_protocol drops an ssh url's host: the url could not be opened again."""
    from types import SimpleNamespace as NS

    fs = NS(protocol=url.split("://")[0], hyp_prefix=docs._prefix_(url, top))
    assert docs.url_of(fs, name) == expected


def test_the_corpus_is_kept_apart_from_the_directory_scanned(tree):
    from types import SimpleNamespace as NS

    data = docs.HypDocs.__default_data__(NS(dir=str(tree)))
    assert data.startswith(os.path.join(os.path.expanduser("~"), ".hyp", "docs")) and str(tree) not in data


# -- markdown ------------------------------------------------------------------


MARKDOWN = """# Title

A **bold** and `code` paragraph.

- one
  - nested
1. first

> quoted

```python
x = 1
```

```mermaid
graph TD; A-->B;
```

| h1 | h2 |
|----|----|
| a  | b  |

---
"""


def test_markdown_is_parsed_into_its_blocks():
    blocks = docview.parse_markdown(MARKDOWN)
    assert [b.kind for b in blocks] == ["heading", "para", "para", "para", "para", "para", "code", "mermaid", "table", "rule"]
    one, nested, first, quoted = blocks[2:6]
    assert (one.prefix, one.indent) == ("•", 1) and (nested.prefix, nested.indent) == ("–", 2)
    assert first.prefix == "1." and quoted.quote == 1
    assert blocks[1].spans[1] == ("bold", (True, False, False, False)) and blocks[1].spans[3][1][2]
    assert blocks[8].rows[0][0][0][0] == "h1"


def test_a_diagram_without_mmdc_is_shown_as_its_source(tmp_path):
    doc = docview.MarkdownDoc(MARKDOWN, mermaid=docview.Mermaid(command="", cache=str(tmp_path)))
    laid = doc.layout(600, 1.0)
    assert laid.height > 200 and not doc.pending()


def test_a_diagram_is_rendered_by_mmdc_off_the_ui_thread(tmp_path):
    """A stand-in mmdc: writes a png to the -o it is given."""
    fake = tmp_path / "mmdc"
    fake.write_text(f"#!{sys.executable}\n"
                    "import sys, pygame\n"
                    "out = sys.argv[sys.argv.index('-o') + 1]\n"
                    "pygame.image.save(pygame.Surface((40, 20)), out)\n")
    fake.chmod(0o755)
    mermaid = docview.Mermaid(command=str(fake), cache=str(tmp_path / "cache"))
    doc = docview.MarkdownDoc("```mermaid\ngraph TD; A-->B;\n```\n", mermaid=mermaid)
    doc.layout(600, 1.0)
    deadline = time.monotonic() + 30
    while doc.version == 0:  # done() is called once the diagram's state is settled
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert mermaid.state("graph TD; A-->B;") == "done" and doc.pending(), "arrived, not yet laid out: keep redrawing"
    laid = doc.layout(600, 2.0)
    assert laid.width >= 40, "the image, at half its rendered size times the zoom"


def test_a_failed_diagram_says_why(tmp_path):
    fake = tmp_path / "mmdc"
    fake.write_text("#!/bin/sh\necho 'Parse error on line 1' >&2\nexit 1\n")
    fake.chmod(0o755)
    mermaid = docview.Mermaid(command=str(fake), cache=str(tmp_path / "cache"))
    assert mermaid.render("bad", lambda: None) == "pending"
    deadline = time.monotonic() + 30
    while mermaid.state("bad") == "pending":
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert mermaid.state("bad") == "error" and "Parse error" in mermaid.error("bad")


# -- pdf, and the window ---------------------------------------------------------


def test_a_pdf_is_laid_out_page_under_page_and_drawn(tmp_path):
    _pdf_(tmp_path / "p.pdf", pages=3)
    doc = docview.PdfDoc((tmp_path / "p.pdf").read_bytes())
    laid = doc.layout(400, 1.0)
    assert len(doc.tops) == 3 and laid.height >= 3 * (400 - 2 * doc.GAP) * 792 / 612, "as wide as the window, margins aside"
    surf = pygame.Surface((400, 300))
    doc.draw(surf, 0, 0, surf.get_rect())
    assert surf.get_at((200, 150))[:3] == (255, 255, 255), "the first page, white"


def test_the_window_scrolls_both_ways_and_zooms():
    window = docview.DocWindow("t")
    window.set_doc(docview.MarkdownDoc("```\n" + "x" * 400 + "\n```\n" + "line\n\n" * 200, mermaid=docview.Mermaid(command="")))
    surf = pygame.Surface((500, 400))
    font = pygame.font.SysFont(docview.SANS, 15)
    window.draw(surf, (0, 0, 500, 400), font, font)
    key = lambda k, mod=0: window.handle(pygame.event.Event(pygame.KEYDOWN, key=k, mod=mod))  # noqa: E731
    key(pygame.K_END)
    assert window.sy == window._size[1] - window.body.height > 0
    key(pygame.K_RIGHT)
    assert window.sx == docview.SCROLL_PX
    key(pygame.K_HOME)
    assert window.sx == window.sy == 0
    key(pygame.K_EQUALS)
    assert window.zoom == pytest.approx(docview.ZOOM_STEP)
    key(pygame.K_MINUS)
    key(pygame.K_MINUS)
    assert window.zoom == pytest.approx(1 / docview.ZOOM_STEP)
    key(pygame.K_0)
    assert window.zoom == 1.0
    assert key(pygame.K_ESCAPE) == "close"


# -- the app: placing, and layers ------------------------------------------------


@pytest.fixture
def app(tree):
    from types import SimpleNamespace as NS

    from hyp.app import Worker

    a = docs.HypDocs(engine=None, worker=Worker())
    a.run_async = lambda fn, then: then(fn())
    found = docs.documents(str(tree), progress=lambda m: None)
    a.snap = NS(positions=np.array([0.1 + 0.1j, -0.4 + 0.2j, 0.3 - 0.5j]), summaries=[d.summary for d in found],
                texts=[d.text for d in found], neighbors=np.array([[1], [0], [0]]), points=[dict(d.extra) for d in found])
    return a


def test_a_left_click_opens_a_document_in_the_center(app):
    app.__click_point__(0, 0.0)
    assert app.center is not None and app.center.title == "a.md" and isinstance(app.center.doc, docview.MarkdownDoc)
    assert app.layers[-1] == "center"
    app.draw(0)
    assert app.center.rect == app.slot_rect("center")


def test_a_right_click_offers_quarters_halves_and_the_center(app):
    app.__context_point__(2, (100, 100))
    labels = [t for t, _ in app.popup.current()]
    assert labels == [*docs.SLOTS, "center"]
    app.popup.select("bottom left")
    assert isinstance(app.tiles["bottom left"].doc, docview.PdfDoc)


def test_a_placed_window_replaces_those_it_overlaps(app):
    app.place(0, "top left")
    app.place(1, "bottom left")
    app.place(2, "top right")
    app.place(0, "left half")
    assert sorted(app.tiles) == ["left half", "top right"]
    app.place(1, "top half")
    assert sorted(app.tiles) == ["top half"]


def test_the_layers_are_rearranged(app):
    assert app.layers == ["disk", "tiles", "center"]
    app.send_to_back("center")
    assert app.layers == ["center", "disk", "tiles"]
    app.bring_to_front("disk")
    assert app.layers == ["center", "tiles", "disk"]
    app.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_TAB, mod=0), 0.0)
    assert app.layers == ["tiles", "disk", "center"], "Tab: the back layer to the front"
    assert ("disk: send to back", ("back", "disk")) in app._layers_items_()


def test_clicks_go_to_the_front_layer_under_the_mouse(app):
    app.place(0, "left half")
    app.draw(0)
    window = app.tiles["left half"]
    inside_disk = (window.rect.right - 30, app.screen.get_height() // 2)
    assert abs(app.to_disk(inside_disk)) < 1 and window.rect.collidepoint(inside_disk)
    assert app.layer_at(inside_disk) == "tiles"
    app.bring_to_front("disk")
    assert app.layer_at(inside_disk) == "disk", "the disk in front: its circle takes the clicks"
    corner = (window.rect.x + 20, window.rect.bottom - 20)
    assert abs(app.to_disk(corner)) > 1 and app.layer_at(corner) == "tiles", "outside the circle, the window still does"


def test_a_window_scrolls_where_the_mouse_is_and_the_disk_elsewhere(app, monkeypatch):
    app.place(0, "left half")
    app.draw(0)
    window = app.tiles["left half"]
    window._size = (window.body.width, window.body.height * 5)
    over = window.body.center
    monkeypatch.setattr(pygame.mouse, "get_pos", lambda: over)
    app.handle(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=-1, precise_x=0.0, precise_y=-1.0), 0.0)
    assert window.sy > 0 and app.view.k == 1.0
    monkeypatch.setattr(pygame.mouse, "get_pos", lambda: (app.screen.get_width() * 3 // 4, app.screen.get_height() // 2))
    app.handle(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=1, precise_x=0.0, precise_y=1.0), 0.0)
    assert app.view.k != 1.0


def test_esc_closes_the_window_last_clicked(app, monkeypatch):
    app.place(0, "top right")
    app.draw(0)
    window = app.tiles["top right"]
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=window.body.center), 0.0)
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=window.body.center), 0.0)
    monkeypatch.setattr(pygame.mouse, "get_pos", lambda: (5, app.screen.get_height() - 5))
    app.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0), 0.0)
    assert not app.tiles and app.focus is None


def test_docs_and_dir_are_checked():
    from hyp.app import main

    with pytest.raises(SystemExit):
        main(["--dir", "/somewhere"])
    with pytest.raises(SystemExit):
        main(["--docs", "--pipejournal"])


def test_a_click_on_a_layer_brings_it_to_the_front(app):
    app.place(0, "left half")
    app.place(1, "center")
    app.draw(0)
    assert app.layers[-1] == "center"
    w, h = app.screen.get_size()
    app.center = None  # the center layer, empty: the tile and the disk show
    tile = app.tiles["left half"]
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(tile.rect.x + 20, tile.rect.bottom - 20)), 0.0)
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=(tile.rect.x + 20, tile.rect.bottom - 20)), 0.0)
    assert app.layers[-1] == "tiles"
    on_disk = (w * 3 // 4, h // 2)
    assert app.layer_at(on_disk) == "disk"
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=on_disk), 0.0)
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=on_disk), 0.0)
    assert app.layers[-1] == "disk"
    inside_tile_and_disk = (tile.rect.right - 30, h // 2)
    assert app.layer_at(inside_tile_and_disk) == "disk", "now in front, the disk takes what it covers"
