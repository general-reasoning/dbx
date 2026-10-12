# hyp

`hyp` is an app, shipped with dbx as a package of its own: a text space on
the Poincaré disk. Texts are embedded (nomic-embed-text, through Ollama, by
default), laid out with UMAP's hyperbolic embedding
(`output_metric="hyperboloid"`), and drawn in a pygame window navigated
hyperbolically. Pointed at a datalake, its texts are the datalake's pipelines.

Nothing in dbx imports `hyp`; `hyp` imports dbx only to read a datalake.

## Setup

`hyp` shares dbx's env: `dbx.yml` installs `-e .[all]`, and `[all]` includes
the `[hyp]` extra. Into another env, `pip install -e '.[hyp]'`.

```bash
brew install ollama            # hyp starts `ollama serve` itself if it is not running
hyp                            # or: python -m hyp
```

On first run it pulls `nomic-embed-text` (embeddings) and `deepseek-r1:1.5b`
(one-line summaries) through Ollama. Each corpus is kept in a directory of its
own under `~/.hyp`.

## A datalake's pipelines

```bash
hyp --dbx.pipejournal --dbx.datalake ssh://wheelbarrow/home/t-9dkarp/datalake
hyp --dbx.pipejournal                    # the datalake dbx defaults to
```

runs `HypDbxPipelines`, the specialization of the app over a datalake's
pipelines. `dbx.pipejournal()` reads the pipeline journal and aggregates it --
one row per pipeline, every run of it, with every entrypoint it was created
to -- and each row is a point:

- embedded from the pipeline's fqcn (`anchor`), spec, and signature (or type)
- labeled with the last component of its `tag`, with its datetime below
- on right click, its row's columns and its side files' contents -- spec, dfn
  and kwargs as the dicts they hold, `topics` and `paths` one key per line --
  under two menus:
  - the tag: every journal entry with that tag, by datetime, to switch to
  - constructed entrypoints, from `dbx.pipejournal(construct_only=True)`:
    picking one opens that block's own journal over the pipeline's, its title
    a menu of the pipeline's other constructed blocks

The menus are read in the background once the points are up. Each launch adds
new pipelines and re-embeds the ones whose spec changed.

An `ssh://` datalake is read through fsspec's sshfs, on asyncssh, which applies
`~/.ssh/config`: `ssh://host/path` names the host as `ssh host` does, user and
all. `--dbx.datalake` also takes the scp-style `ssh://host:/path`.

## Controls

| Input | Effect |
|---|---|
| left click | slide the clicked point to the center along a geodesic; time ∝ distance |
| left drag | move the plane rigidly (a hyperbolic translation); the grabbed point stays under the cursor |
| scroll | hyperbolic zoom: a point at distance *d* is drawn at *k·d* |
| right click on a point | its stored text |
| right click elsewhere | type a text; Enter embeds, summarizes, stores and places it |
| hover | a one-line summary, and geodesics to the 5 nearest neighbors in embedding space |
| Home | back to the corpus center, zoom 1 |
| settings (top right) | datetimes on and off: under labels, and in tooltips |
| in a text window | scroll / Up Down / PgUp PgDn move; Left / Right step through its title menu; × or Esc closes it |

## Options

```
hyp --data DIR               # the corpus directory (default: one per source, under ~/.hyp)
hyp --reset                  # discard the corpus and start over
hyp --dbx.pipejournal        # a datalake's pipelines (HypDbxPipelines)
hyp --dbx.datalake URL       # ... of this datalake, rather than dbx's default
hyp --n-workers N            # threads reading the datalake (default 16)
hyp --embedder deepseek      # DeepSeek-R1-Distill-Qwen-1.5B's pooled hidden states,
                             # through transformers: `pip install -e '.[hyp-deepseek]'`
hyp --embed-model M          # another Ollama embedding model
hyp --summary-model M        # another Ollama model for the summaries
```

Ollama will not embed with a generative model such as DeepSeek-R1, hence the
transformers path for `--embedder deepseek`. Switching embedders re-embeds the
stored corpus.

## Specializing

`hyp.app.Hyp` is the app; a specialization overrides its hooks, and only
those: `__source__` and `__default_data__` (where the corpus comes from),
`__open_point__` (a right click on a point), and `__point_label__`,
`__point_datetime__`, `__point_tooltip__` (how a point is labeled).
`hyp.dbxpipelines.HypDbxPipelines` is the example.

## Tests

`tests/test_hyp_*.py`, run with the rest of the suite (see Running Tests in the
README).
