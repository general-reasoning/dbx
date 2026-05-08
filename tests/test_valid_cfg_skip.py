"""
Tests for validate_cfg=False: allows a Datablock to skip cfg validation
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
    """Depends on an upstream block via spec — validate_cfg=True (default)."""
    TOPICFILE = 'output.txt'

    @dataclass
    class CONFIG(Datablock.CONFIG):
        src: str = "'invalid'"

    def __build__(self):
        self.dirpath(ensure=True)
        with open(self.path(), 'w') as f:
            f.write('downstream')


class DownstreamWithSkip(Datablock):
    """Depends on an upstream block via spec — validate_cfg=False skips validation."""
    TOPICFILE = 'output.txt'

    @dataclass
    class CONFIG(Datablock.CONFIG):
        src: str = "'invalid'"

    def __build__(self):
        self.dirpath(ensure=True)
        with open(self.path(), 'w') as f:
            f.write('downstream')


class TwoDeps(Datablock):
    """Two upstream deps — when validate_cfg=False, all cfg validation is skipped."""
    TOPICFILE = 'output.txt'

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
    return quote(Upstream, url=str(tmp_path / name))


def _make(cls, tmp_path, **extra_spec):
    spec = {**extra_spec} if extra_spec else None
    kwargs = {'root': str(tmp_path)}
    if spec:
        kwargs['spec'] = spec
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestValidateCfg:

    def test_no_skip_reports_invalid_upstream(self, tmp_path):
        """Without validate_cfg=False, an invalid upstream appears in valid_cfg()."""
        up_spec = _quote_upstream(tmp_path)

        down = DownstreamNoSkip(
            url=str(tmp_path / 'down'),
            spec={'src': up_spec},
        )
        result = down.valid_cfg()
        assert 'src' in result
        assert result['src'] is False

    def test_skip_returns_empty_from_valid_cfg(self, tmp_path):
        """validate_cfg=False causes valid_cfg() to return empty results."""
        up_spec = _quote_upstream(tmp_path)

        down = DownstreamWithSkip(
            url=str(tmp_path / 'down'),
            spec={'src': up_spec},
            validate_cfg=False,
        )
        result = down.valid_cfg()
        assert result == {}

    def test_skip_allows_build_with_invalid_upstream(self, tmp_path):
        """With validate_cfg=False, build() succeeds even if the upstream is invalid."""
        up_spec = _quote_upstream(tmp_path)

        down = DownstreamWithSkip(
            url=str(tmp_path / 'down'),
            spec={'src': up_spec},
            validate_cfg=False,
        )
        # Should NOT raise — cfg validation is skipped
        down.build()
        assert down.valid()

    def test_no_skip_build_raises_on_invalid_upstream(self, tmp_path):
        """Without validate_cfg=False, build() raises when upstream is invalid."""
        up_spec = _quote_upstream(tmp_path)

        down = DownstreamNoSkip(
            url=str(tmp_path / 'down'),
            spec={'src': up_spec},
        )
        with pytest.raises(ValueError, match="Not all upstream Datablocks"):
            down.build()

    def test_skip_all_deps(self, tmp_path):
        """validate_cfg=False skips all deps."""
        down = TwoDeps(
            url=str(tmp_path / 'down'),
            spec={
                'required': _quote_upstream(tmp_path, 'req'),
                'optional': _quote_upstream(tmp_path, 'opt'),
            },
            validate_cfg=False,
        )
        result = down.valid_cfg()
        assert result == {}

    def test_skip_build_succeeds_with_invalid_deps(self, tmp_path):
        """With validate_cfg=False, build succeeds even if all deps are invalid."""
        down = TwoDeps(
            url=str(tmp_path / 'down'),
            spec={
                'required': _quote_upstream(tmp_path, 'req'),
                'optional': _quote_upstream(tmp_path, 'opt'),
            },
            validate_cfg=False,
        )
        down.build()
        assert down.valid()

    def test_default_build_raises_on_invalid_deps(self, tmp_path):
        """Default validate_cfg=True, build fails if deps are invalid."""
        down = TwoDeps(
            url=str(tmp_path / 'down'),
            spec={
                'required': _quote_upstream(tmp_path, 'req'),
                'optional': _quote_upstream(tmp_path, 'opt'),
            },
        )
        with pytest.raises(ValueError, match="Not all upstream Datablocks"):
            down.build()

    def test_valid_cfg_reduce_with_skip(self, tmp_path):
        """valid_cfg(reduce=True) should return True when validate_cfg=False."""
        up_spec = _quote_upstream(tmp_path)

        down = DownstreamWithSkip(
            url=str(tmp_path / 'down'),
            spec={'src': up_spec},
            validate_cfg=False,
        )
        assert down.valid_cfg(reduce=True) is True

    def test_default_validate_cfg(self, tmp_path):
        """Default validate_cfg=True: all spec keys are checked (baseline)."""
        down = _make(DownstreamNoSkip, tmp_path)
        result = down.valid_cfg()
        # 'src' default is a string literal, not a Datablock → not in results
        assert result == {}
