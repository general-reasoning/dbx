"""dbx.image: the layer of a venv's code, written deterministically so its hash is the code's.

Nothing is pushed: `build(dry_run=True)` stops before crane, and `write_layer` is checked directly.
"""
import json
import tarfile

import pytest

from dbx import image


@pytest.fixture
def venv(tmp_path):
    """A venv with one editable install (a package and a console script) and one plain package."""
    src = tmp_path / 'proj' / 'mypkg'
    src.mkdir(parents=True)
    (src / '__init__.py').write_text('X = 1\n')
    (src / '__pycache__').mkdir()
    (src / '__pycache__' / 'junk.pyc').write_bytes(b'x')
    v = tmp_path / 'venv'
    site = v / 'lib' / 'python3.11' / 'site-packages'
    site.mkdir(parents=True)
    (site / '__editable___mypkg_0_1_finder.py').write_text(f"MAPPING: dict[str, str] = {{'mypkg': '{src}'}}\n")
    (site / '__editable__.mypkg-0.1.pth').write_text('import x\n')
    info = site / 'mypkg-0.1.dist-info'
    info.mkdir()
    (info / 'direct_url.json').write_text(json.dumps({'dir_info': {'editable': True}}))
    (info / 'entry_points.txt').write_text('[console_scripts]\nmytool = mypkg:main\n')
    (site / 'timm').mkdir()
    (site / 'timm' / '__init__.py').write_text('')
    (site / 'pytest').mkdir()
    (site / 'zz_home_venv.pth').write_text('')
    (v / 'bin').mkdir()
    (v / 'bin' / 'mytool').write_text('#!/old/python\nimport mypkg\n')
    return v


def _names_(path):
    with tarfile.open(path) as tf:
        return {m.name: m for m in tf.getmembers()}


def test_the_layer_holds_the_editable_code_the_added_packages_and_the_scripts(venv, tmp_path):
    out = tmp_path / 'layer.tar'
    image.write_layer(out, str(venv), prefix='/opt/p', python='/home/ray/venv/bin/python', run_as=(1000, 100),
                      skip=('zz_home_venv.pth',), extra={str(tmp_path / 'proj'): 'checkout'})
    names = _names_(out)
    assert {'opt/p/src/mypkg/__init__.py', 'opt/p/site-packages/timm/__init__.py', 'opt/p/bin/mytool',
            'opt/p/checkout/mypkg/__init__.py', 'home/ray/venv/lib/python3.11/site-packages/aa_dbx_image.pth'} <= set(names)
    assert not any('pytest' in n or '__editable__' in n or 'zz_home_venv' in n or n.endswith('.pyc') for n in names)
    assert names['opt/p/bin/mytool'].mode == 0o755 and names['opt/p/src/mypkg/__init__.py'].uid == 1000
    with tarfile.open(out) as tf:
        assert tf.extractfile('opt/p/bin/mytool').read().decode().startswith('#!/home/ray/venv/bin/python\n')


def test_the_same_code_is_the_same_tag_and_a_change_is_another(venv, tmp_path):
    kw = dict(repo='reg.example.com/p', base='base:1', python='/usr/bin/python3', dry_run=True)
    first, again = image.build(str(venv), **kw), image.build(str(venv), **kw)
    assert first == again and first.startswith('reg.example.com/p:')
    (tmp_path / 'proj' / 'mypkg' / '__init__.py').write_text('X = 2\n')
    assert image.build(str(venv), **kw) != first


def test_anyscale_gives_the_base_the_interpreter_and_the_user(venv, monkeypatch):
    monkeypatch.setattr(image._anyscale_, 'IMAGE', 'anyscale-image:1')
    ref = image.build(str(venv), repo='reg.example.com/p', anyscale=True, dry_run=True)
    assert ref.startswith('reg.example.com/p:')
    with pytest.raises(ValueError, match='anyscale=True'):
        image.build(str(venv), repo='reg.example.com/p', dry_run=True)
