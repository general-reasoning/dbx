"""
Tests for VALID_CFG_SKIP: allows a Datablock to skip certain spec keys
when checking upstream validity in valid_cfg().
"""
import os
import pytest
from dataclasses import dataclass

from dbx.datablocks import Datablock, quote


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


# ---------------------------------------------------------------------------
# Sample Datablock subclasses
# ---------------------------------------------------------------------------

class Upstream(Datablock):
    """An upstream block that is never built (always invalid)."""
    TOPICFILE = 'data.txt'

    @dataclass
    class CONFIG(Datablock.CONFIG):
        pass

    def __build__(self):
        self.dirpath(ensure=True)
        with open(self.path(), 'w') as f:
            f.write('upstream')


class DownstreamNoSkip(Datablock):
    """Depends on an upstream block via spec — no VALID_CFG_SKIP."""
    TOPICFILE = 'output.txt'

    @dataclass
    class CONFIG(Datablock.CONFIG):
        src: str = "'invalid'"

    def __build__(self):
        self.dirpath(ensure=True)
        with open(self.path(), 'w') as f:
            f.write('downstream')


class DownstreamWithSkip(Datablock):
    """Depends on an upstream block via spec — VALID_CFG_SKIP skips 'src'."""
    TOPICFILE = 'output.txt'
    VALID_CFG_SKIP = ('src',)

    @dataclass
    class CONFIG(Datablock.CONFIG):
        src: str = "'invalid'"

    def __build__(self):
        self.dirpath(ensure=True)
        with open(self.path(), 'w') as f:
            f.write('downstream')


class TwoDeps(Datablock):
    """Two upstream deps: skip one, check the other."""
    TOPICFILE = 'output.txt'
    VALID_CFG_SKIP = ('optional',)

    @dataclass
    class CONFIG(Datablock.CONFIG):
        required: str = "'invalid'"
        optional: str = "'invalid'"

    def __build__(self):
        self.dirpath(ensure=True)
        with open(self.path(), 'w') as f:
            f.write('two-deps')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quote_upstream(tmp_path, name='up'):
    """Create a quoted spec value for an Upstream block at the given path."""
    return quote(Upstream, root=str(tmp_path / name))


def _make(cls, tmp_path, **extra_spec):
    spec = {**extra_spec} if extra_spec else None
    kwargs = {'root': str(tmp_path)}
    if spec:
        kwargs['spec'] = spec
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestValidCfgSkip:

    def test_no_skip_reports_invalid_upstream(self, tmp_path):
        """Without VALID_CFG_SKIP, an invalid upstream appears in valid_cfg()."""
        up_spec = _quote_upstream(tmp_path)

        down = DownstreamNoSkip(
            root=str(tmp_path / 'down'),
            spec={'src': up_spec},
        )
        result = down.valid_cfg()
        assert 'src' in result
        assert result['src'] is False

    def test_skip_omits_key_from_valid_cfg(self, tmp_path):
        """VALID_CFG_SKIP causes the key to be absent from valid_cfg() results."""
        up_spec = _quote_upstream(tmp_path)

        down = DownstreamWithSkip(
            root=str(tmp_path / 'down'),
            spec={'src': up_spec},
        )
        result = down.valid_cfg()
        assert 'src' not in result

    def test_skip_allows_build_with_invalid_upstream(self, tmp_path):
        """With VALID_CFG_SKIP, build() succeeds even if the skipped upstream is invalid."""
        up_spec = _quote_upstream(tmp_path)

        down = DownstreamWithSkip(
            root=str(tmp_path / 'down'),
            spec={'src': up_spec},
        )
        # Should NOT raise — 'src' is skipped in valid_cfg
        down.build()
        assert down.valid()

    def test_no_skip_build_raises_on_invalid_upstream(self, tmp_path):
        """Without VALID_CFG_SKIP, build() raises when upstream is invalid."""
        up_spec = _quote_upstream(tmp_path)

        down = DownstreamNoSkip(
            root=str(tmp_path / 'down'),
            spec={'src': up_spec},
        )
        with pytest.raises(ValueError, match="Not all upstream Datablocks"):
            down.build()

    def test_skip_partial_two_deps(self, tmp_path):
        """Only the skipped dep is omitted; the other still appears."""
        down = TwoDeps(
            root=str(tmp_path / 'down'),
            spec={
                'required': _quote_upstream(tmp_path, 'req'),
                'optional': _quote_upstream(tmp_path, 'opt'),
            },
        )
        result = down.valid_cfg()
        # 'optional' should be skipped
        assert 'optional' not in result
        # 'required' should still be checked
        assert 'required' in result
        assert result['required'] is False

    def test_skip_partial_build_still_checks_required(self, tmp_path):
        """Even with VALID_CFG_SKIP on 'optional', build fails if 'required' is invalid."""
        down = TwoDeps(
            root=str(tmp_path / 'down'),
            spec={
                'required': _quote_upstream(tmp_path, 'req'),
                'optional': _quote_upstream(tmp_path, 'opt'),
            },
        )
        with pytest.raises(ValueError, match="Not all upstream Datablocks"):
            down.build()

    def test_skip_partial_build_succeeds_when_required_valid(self, tmp_path):
        """If 'required' upstream is valid, build succeeds (skipped 'optional' is ignored)."""
        up_req = Upstream(root=str(tmp_path / 'req'))
        up_req.build()
        assert up_req.valid()

        down = TwoDeps(
            root=str(tmp_path / 'down'),
            spec={
                'required': quote(up_req),
                'optional': _quote_upstream(tmp_path, 'opt'),
            },
        )
        down.build()
        assert down.valid()

    def test_valid_cfg_reduce_with_skip(self, tmp_path):
        """valid_cfg(reduce=True) should ignore skipped keys."""
        up_spec = _quote_upstream(tmp_path)

        down = DownstreamWithSkip(
            root=str(tmp_path / 'down'),
            spec={'src': up_spec},
        )
        # No upstream keys remain after skipping → reduce over empty → True
        assert down.valid_cfg(reduce=True) is True

    def test_no_valid_cfg_skip_default(self, tmp_path):
        """Without VALID_CFG_SKIP attribute, all spec keys are checked (baseline)."""
        down = _make(DownstreamNoSkip, tmp_path)
        result = down.valid_cfg()
        # 'src' default is a string literal, not a Datablock → not in results
        assert result == {}

    def test_empty_valid_cfg_skip(self, tmp_path):
        """VALID_CFG_SKIP=() is equivalent to no skip."""
        class EmptySkip(Datablock):
            TOPICFILE = 'out.txt'
            VALID_CFG_SKIP = ()

            @dataclass
            class CONFIG(Datablock.CONFIG):
                src: str = "'x'"

        down = EmptySkip(root=str(tmp_path))
        result = down.valid_cfg()
        assert result == {}
