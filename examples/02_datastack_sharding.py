"""02_datastack_sharding.py — Datastack with parallel shard builds.

Demonstrates:
- Defining a Datastack subclass that produces N shards
- Each shard is a Datablock that processes a slice of work
- Building all shards with inline execution
"""

import math
import os
import tempfile
from dataclasses import dataclass

import pandas as pd

os.environ.pop('DBX_GIT_REPO', None)
os.environ.pop('DBXGITREPO', None)

from dbx import Datablock, Datastack, write_frame, read_frame


class ChunkBlock(Datablock):
    """A single shard that generates a slice of a sequence."""

    TOPICFILES = {'data': 'chunk.parquet'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        start: int = 0
        end: int = 10

    def __build__(self):
        df = pd.DataFrame({
            'value': range(self.cfg.start, self.cfg.end),
            'squared': [x ** 2 for x in range(self.cfg.start, self.cfg.end)],
        })
        write_frame(df, self.path('data', ensure_dirpath=True))
        return self

    def __read__(self, topic='data'):
        return read_frame(self.path(topic))


class SquaresStack(Datastack):
    """Stack that shards a range into chunks and computes squares."""

    TOPICFILES = {'manifest': 'manifest.parquet'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        total: int = 100
        chunk_size: int = 25

    @property
    def n_shards(self):
        return math.ceil(self.cfg.total / self.cfg.chunk_size)

    def __shard__(self, idx):
        start = idx * self.cfg.chunk_size
        end = min(start + self.cfg.chunk_size, self.cfg.total)
        return ChunkBlock(url=self.url, spec=dict(start=start, end=end))

    def __stack__(self):
        """Concatenate all shard outputs into a manifest."""
        frames = []
        for idx in range(self.n_shards):
            shard = self.shard(idx)
            frames.append(shard.read('data'))
        manifest = pd.concat(frames, ignore_index=True)
        write_frame(manifest, self.path('manifest', ensure_dirpath=True))
        self.log.info(f"Stacked {self.n_shards} shards → {len(manifest)} rows")
        return self

    def __read__(self, topic='manifest'):
        return read_frame(self.path(topic))


def main():
    root = tempfile.mkdtemp(prefix='dbx_stack_')
    print(f"Storage root: {root}")

    stack = SquaresStack(
        url=root,
        spec=dict(total=50, chunk_size=10),
    )

    print(f"Shards: {stack.n_shards}")
    print(f"Hash:   {stack.superhash}")

    stack.build()

    df = stack.read('manifest')
    print(f"\nManifest ({len(df)} rows):")
    print(df.head(10).to_string(index=False))
    print("...")


if __name__ == '__main__':
    main()
