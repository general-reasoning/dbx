"""02_datastack_sharding.py — Datastack with parallel block builds.

Demonstrates:
- Defining a Datastack subclass that produces N blocks
- Each block is a Datablock that processes a slice of work
- Building all blocks with inline execution
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
    """A single block that generates a slice of a sequence."""

    TOPICS = {'data': 'chunk.parquet'}

    @dataclass
    class VAR(Datablock.VAR):
        start: int = 0
        end: int = 10

    def __build__(self):
        df = pd.DataFrame({
            'value': range(self.var.start, self.var.end),
            'squared': [x ** 2 for x in range(self.var.start, self.var.end)],
        })
        write_frame(df, self.path('data', ensure_dirpath=True))
        return self

    def __read__(self, topic='data'):
        return read_frame(self.path(topic))


class SquaresStack(Datastack):
    """Stack that splits a range into chunks and computes squares."""

    TOPICS = {'manifest': 'manifest.parquet'}

    @dataclass
    class VAR(Datablock.VAR):
        total: int = 100
        chunk_size: int = 25

    @property
    def n_blocks(self):
        return math.ceil(self.var.total / self.var.chunk_size)

    def __block__(self, idx):
        start = idx * self.var.chunk_size
        end = min(start + self.var.chunk_size, self.var.total)
        return ChunkBlock(url=self.url, spec=dict(start=start, end=end))

    def __stack__(self):
        """Concatenate all block outputs into a manifest."""
        frames = []
        for idx in range(self.n_blocks):
            blk = self.block(idx)
            frames.append(blk.read('data'))
        manifest = pd.concat(frames, ignore_index=True)
        write_frame(manifest, self.path('manifest', ensure_dirpath=True))
        self.log.info(f"Stacked {self.n_blocks} blocks → {len(manifest)} rows")
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

    print(f"Blocks: {stack.n_blocks}")
    print(f"Hash:   {stack.superhash}")

    stack.build()

    df = stack.read('manifest')
    print(f"\nManifest ({len(df)} rows):")
    print(df.head(10).to_string(index=False))
    print("...")


if __name__ == '__main__':
    main()
