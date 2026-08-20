"""Tests for dbx.valid() / datablocks.valid()."""
import os
import sys
import pytest

import dbx.datablocks as dbxmod
from dbx.datablocks import Datablock, valid


class SampleBlock(Datablock):
    TOPICS = {'data': 'data.txt'}

    def __build__(self):
        with open(self.path('data', ensure_dirpath=True), 'w') as f:
            f.write("hello world\n")


class SecondBlock(Datablock):
    TOPICS = {'output': 'output.txt'}

    def __build__(self):
        with open(self.path('output', ensure_dirpath=True), 'w') as f:
            f.write("result\n")


class TestValidProgrammatic:

    def test_valid_single_anchor(self, tmp_path, monkeypatch):
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        block = SampleBlock(url=str(tmp_path))
        block.build()

        anchor_str = block.anchor
        res = valid(SampleBlock, url=str(tmp_path))
        assert isinstance(res, dict)
        assert res[SampleBlock] is True

        res_str = valid(anchor_str, url=str(tmp_path))
        assert res_str[anchor_str] is True

    def test_valid_summary_mode(self, tmp_path, monkeypatch):
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        block = SampleBlock(url=str(tmp_path))
        block.build()

        assert valid(SampleBlock, summary=True, url=str(tmp_path)) is True

    def test_valid_invalid_data(self, tmp_path, monkeypatch):
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        block = SampleBlock(url=str(tmp_path))
        block.build()

        # Delete the built data file to make .valid() return False
        os.remove(block.path('data'))
        assert block.valid() is False

        res = valid(SampleBlock, url=str(tmp_path))
        assert res[SampleBlock] is False
        assert valid(SampleBlock, summary=True, url=str(tmp_path)) is False

    def test_valid_nonexistent_anchor(self, tmp_path, monkeypatch):
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        res = valid("nonexistent.AnchorKey", url=str(tmp_path))
        assert res["nonexistent.AnchorKey"] is False
        assert valid("nonexistent.AnchorKey", summary=True, url=str(tmp_path)) is False

    def test_valid_multiple_anchors(self, tmp_path, monkeypatch):
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        b1 = SampleBlock(url=str(tmp_path))
        b1.build()

        b2 = SecondBlock(url=str(tmp_path))
        b2.build()

        # Both valid
        res = valid(SampleBlock, SecondBlock, url=str(tmp_path))
        assert res[SampleBlock] is True
        assert res[SecondBlock] is True
        assert valid(SampleBlock, SecondBlock, summary=True, url=str(tmp_path)) is True

        # Corrupt b2
        os.remove(b2.path('output'))
        res = valid(SampleBlock, SecondBlock, url=str(tmp_path))
        assert res[SampleBlock] is True
        assert res[SecondBlock] is False
        assert valid(SampleBlock, SecondBlock, summary=True, url=str(tmp_path)) is False

    def test_valid_with_n_workers(self, tmp_path, monkeypatch):
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        b = SampleBlock(url=str(tmp_path))
        b.build()

        res = valid(SampleBlock, n_workers=2, url=str(tmp_path))
        assert res[SampleBlock] is True

    def test_valid_custom_events(self, tmp_path, monkeypatch):
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        b = SampleBlock(url=str(tmp_path))
        b.build()

        # Single event in list -> returns boolean for anchor
        res_single = valid(SampleBlock, events=['build:end'], url=str(tmp_path))
        assert res_single[SampleBlock] is True

        # Multiple events -> returns dict mapping each event to boolean
        res_multi = valid(SampleBlock, events=['build:end', 'nonexistent:event'], url=str(tmp_path))
        assert res_multi[SampleBlock] == {'build:end': True, 'nonexistent:event': False}

        # Summary mode with multiple events -> returns False because nonexistent:event is False
        assert valid(SampleBlock, events=['build:end', 'nonexistent:event'], summary=True, url=str(tmp_path)) is False


class TestValidCLI:

    def test_cli_invocation_dict(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        b = SampleBlock(url=str(tmp_path))
        b.build()

        anchor_str = b.anchor
        monkeypatch.setattr(sys, 'argv', ['dbx.valid', anchor_str, '--url', str(tmp_path)])
        res = valid()
        assert res == {anchor_str: True}
        captured = capsys.readouterr()
        assert anchor_str in captured.out
        assert "True" in captured.out

    def test_cli_invocation_summary(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        b = SampleBlock(url=str(tmp_path))
        b.build()

        anchor_str = b.anchor
        monkeypatch.setattr(sys, 'argv', ['dbx.valid', anchor_str, '--summary', '--url', str(tmp_path)])
        res = valid()
        assert res is True
        captured = capsys.readouterr()
        assert "True" in captured.out

    def test_cli_invocation_events(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        b = SampleBlock(url=str(tmp_path))
        b.build()

        anchor_str = b.anchor
        monkeypatch.setattr(sys, 'argv', ['dbx.valid', anchor_str, '--events', 'build:end', 'nonexistent:event', '--url', str(tmp_path)])
        res = valid()
        assert res == {anchor_str: {'build:end': True, 'nonexistent:event': False}}
        captured = capsys.readouterr()
        assert "build:end" in captured.out
        assert "nonexistent:event" in captured.out
