"""A container image for a cluster Job: a base image plus one layer holding a venv's code -- no Docker build.

    ref = dbx.image.build('/path/to/venv', repo='registry.example.com/team/proj', anyscale=True,
                          extra={'/path/to/checkout': 'checkout'}, login=dbx.image.acr_login)
    dbx.kube.submit('python -m my.thing', LAMBDA, image=ref)

The layer holds, under *prefix*:
- `src/`: every package *venv* has installed in editable mode, as it is on disk now;
- `site-packages/`: what *venv* has besides -- copied as installed, so the image runs the versions
  this venv does;
- `bin/`: the console scripts of the editable installs, on the image's *python*;
- each *extra* tree, at the path it maps to under *prefix*;
and a `.pth` in the image's site-packages putting `src/` and `site-packages/` first on `sys.path`.
Nothing is built: `crane append` adds the layer to *base* in the registry, and `crane mutate` sets
*env*, `PATH` and provenance labels -- the base, and each shipped git checkout's commit.

The tag is the first 12 hex digits of the layer's sha256: the layer is written deterministically
(sorted entries, fixed owner and mtime), so the same code gives the same tag, a rebuild of unchanged
code is a no-op push, and the image a Job ran -- recorded in its manifest -- names the code it ran.

*venv* must be one for the image's Python: a venv layered on the image, as Anyscale's workspaces
make (`anyscale=True`: its interpreter, its `ray` user, and the path hooks its venvs carry, left out).
"""
import ast
import configparser
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from . import anyscale as _anyscale_


__all__ = ['build', 'write_layer', 'editable_packages', 'editable_scripts', 'acr_login']


# What a venv's site-packages holds that is not a package to ship: the editable installs and path
# hooks (replaced by the layer's own .pth) and test tools.
SKIP_SITE = ('__editable__', '_virtualenv', 'distutils-precedence.pth', 'pip', '_pytest', 'pytest', 'py.py',
             'xdist', 'execnet', 'iniconfig', 'pluggy')
# Never shipped from any tree.
SKIP_NAMES = {'.git', '__pycache__', '.pytest_cache', '.ipynb_checkpoints'}


def build(venv: str, *, repo: str, base: str | None = None, prefix: str = '/opt/dbx', extra: dict | None = None,
          env: dict | None = None, python: str | None = None, run_as: tuple | None = None, skip: tuple = (),
          anyscale: bool = False, login=None, dry_run: bool = False) -> str:
    """Build the image of *venv*'s code on *base* and push it to *repo*; its reference, `repo:tag`.

    *extra* maps more trees to ship to their paths under *prefix*; *env* is set in the image (over
    *base*'s). *python* is the image's interpreter (the console scripts' shebang, and whose
    site-packages takes the `.pth`), *run_as* the uid and gid the layer's files are owned by, *skip*
    more site-packages entries to leave out. `anyscale=True` gives them Anyscale's
    (`dbx.anyscale.image_options`) and *base* the workspace's image. *login*, when given, is called
    with the registry's host before pushing. *dry_run* writes the layer and returns the reference;
    it pushes nothing.
    """
    if anyscale:
        a = _anyscale_.image_options(prefix)
        base, python, run_as = base or a['base'], python or a['python'], run_as or a['run_as']
        skip, env = (*skip, *a['skip']), {**a['env'], **(env or {})}
    if base is None or python is None:
        raise ValueError("dbx.image.build: give base= and python= -- the image and its interpreter -- or anyscale=True")
    with tempfile.TemporaryDirectory() as tmp:
        layer = Path(tmp) / 'layer.tar'
        digest = write_layer(layer, venv, prefix=prefix, extra=extra, python=python, run_as=run_as or (0, 0),
                             skip=skip)
        ref = f"{repo}:{digest[:12]}"
        print(f"layer {layer.stat().st_size / 2**20:.0f} MiB, sha256 {digest}", file=sys.stderr)
        if dry_run:
            return ref
        if login is not None:
            login(repo.split('/')[0])
        base_env = json.loads(_run_('crane', 'config', base))['config'].get('Env', [])
        path = next((e.split('=', 1)[1] for e in base_env if e.startswith('PATH=')), '/usr/bin:/bin')
        # append pushes the layer and tags the bare result; mutate then retags it with its config.
        _run_('crane', 'append', '-b', base, '-f', str(layer), '-t', ref)
        labels = _provenance_(base, venv, extra)
        _run_('crane', 'mutate', ref, '-t', ref, '--env', f"PATH={prefix}/bin:{path}",
              *[a for k, v in (env or {}).items() for a in ('--env', f"{k}={v}")],
              *[a for k, v in labels.items() for a in ('--label', f"{k}={v}")])
        return ref


def write_layer(out, venv: str, *, prefix: str = '/opt/dbx', extra: dict | None = None, python: str,
                run_as: tuple = (0, 0), skip: tuple = ()) -> str:
    """Write the layer tar of *venv*'s code to *out* (`build` for the rest); its sha256."""
    prefix = prefix.strip('/')
    site = _site_packages_(venv)
    image_site = f"{os.path.dirname(os.path.dirname(python.rstrip('/'))).strip('/')}/lib/{site.parent.name}/site-packages"

    def norm(ti: tarfile.TarInfo) -> tarfile.TarInfo:
        ti.uid, ti.gid, ti.uname, ti.gname, ti.mtime = run_as[0], run_as[1], '', '', 0
        return ti

    def add_file(tf, name: str, data: bytes, mode=0o644):
        ti = norm(tarfile.TarInfo(name))
        ti.size, ti.mode = len(data), mode
        tf.addfile(ti, io.BytesIO(data))

    def add_tree(tf, src: Path, dst: str):
        tf.add(src, dst, recursive=False, filter=norm)
        if src.is_dir() and not src.is_symlink():
            for child in sorted(src.iterdir()):
                if child.name not in SKIP_NAMES and not child.name.endswith('.pyc'):
                    add_tree(tf, child, f"{dst}/{child.name}")

    skip = (*SKIP_SITE, *skip)
    trees = ([(src, f"{prefix}/src/{name}") for name, src in editable_packages(venv).items()]
             + [(Path(src), f"{prefix}/{dst.strip('/')}") for src, dst in sorted((extra or {}).items())]
             + [(p, f"{prefix}/site-packages/{p.name}") for p in sorted(site.iterdir())
                if not p.name.startswith(skip) and p.name not in SKIP_NAMES])
    with tarfile.open(out, 'w', format=tarfile.PAX_FORMAT) as tf:
        dirs = sorted({'/'.join(d.split('/')[:i]) for d in (f"{prefix}/src", f"{prefix}/site-packages",
                                                           f"{prefix}/bin", f"{prefix}/raywd")
                       for i in range(1, d.count('/') + 2)})
        for d in dirs:
            ti = norm(tarfile.TarInfo(d))
            ti.type, ti.mode = tarfile.DIRTYPE, 0o755
            tf.addfile(ti)
        for src, dst in trees:
            add_tree(tf, src, dst)
        pth = f"import sys; sys.path[0:0] = ['/{prefix}/src', '/{prefix}/site-packages']\n"
        add_file(tf, f"{image_site}/aa_dbx_image.pth", pth.encode())
        for cmd in editable_scripts(venv):
            body = (Path(venv) / 'bin' / cmd).read_text().split('\n', 1)[1]
            add_file(tf, f"{prefix}/bin/{cmd}", f"#!{python}\n{body}".encode(), 0o755)
    h = hashlib.sha256()
    with open(out, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def editable_packages(venv: str) -> dict:
    """Top-level package name -> its source directory, for every editable install in *venv*
    (read from setuptools' `__editable___*_finder.py` mappings)."""
    out = {}
    for finder in sorted(_site_packages_(venv).glob('__editable___*_finder.py')):
        m = re.search(r"MAPPING: dict\[str, str\] = (\{.*?\})", finder.read_text(), re.S)
        if m:
            out.update({k: Path(v) for k, v in ast.literal_eval(m.group(1)).items()})
    return out


def editable_scripts(venv: str) -> list:
    """The console scripts of *venv*'s editable installs (their `entry_points.txt`)."""
    names = []
    for info in sorted(_site_packages_(venv).glob('*.dist-info')):
        direct = info / 'direct_url.json'
        if not direct.exists() or not json.loads(direct.read_text()).get('dir_info', {}).get('editable'):
            continue
        ep = configparser.ConfigParser()
        ep.read(info / 'entry_points.txt')
        names += list(ep['console_scripts']) if ep.has_section('console_scripts') else []
    return sorted(names)


def acr_login(registry: str) -> None:
    """crane's credentials for the Azure Container Registry *registry*, from the signed-in `az` CLI (AcrPush needed)."""
    token = json.loads(_run_('az', 'acr', 'login', '--name', registry.split('.')[0], '--expose-token', '-o', 'json'))
    subprocess.run(['crane', 'auth', 'login', registry, '-u', '00000000-0000-0000-0000-000000000000',
                    '--password-stdin'], input=token['accessToken'], text=True, check=True, capture_output=True)


# Helpers ------------------------------------------------------------------------------------------- #

def _site_packages_(venv: str) -> Path:
    found = sorted(Path(venv).glob('lib/python3*/site-packages'))
    if not found:
        raise FileNotFoundError(f"{venv} has no lib/python3.*/site-packages: not a venv")
    return found[0]


def _provenance_(base: str, venv: str, extra: dict | None) -> dict:
    """Labels recording what the image was made from: the base, and each shipped git checkout's commit."""
    labels = {'dbx.base': base}
    for src in [*editable_packages(venv).values(), *(extra or {})]:
        top = subprocess.run(['git', '-C', str(src), 'rev-parse', '--show-toplevel'], capture_output=True,
                             text=True).stdout.strip()
        if not top:
            continue
        sha = _run_('git', '-C', top, 'rev-parse', 'HEAD')
        dirty = _run_('git', '-C', top, 'status', '--porcelain', '--untracked-files=no')
        labels[f"dbx.git.{os.path.basename(top)}"] = f"{sha}{'+dirty' if dirty else ''}"
    return labels


def _run_(*cmd) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()
