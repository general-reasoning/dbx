"""01_basic_datablock.py — Minimal Datablock example.

Demonstrates:
- Defining a Datablock subclass with TOPICS and CONFIG
- Building and reading a Datablock
- Inspecting the content hash and paths
"""

import os
import tempfile
from dataclasses import dataclass

import pandas as pd

# Ensure git check is skipped for this standalone example
os.environ.pop('DBX_GIT_REPO', None)
os.environ.pop('DBXGITREPO', None)

from dbx import Datablock, write_frame, read_frame


class WordCount(Datablock):
    """Count word frequencies in a text and store as Parquet."""

    TOPICS = {'counts': 'counts.parquet'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        text: str = "the quick brown fox jumps over the lazy dog"

    def __build__(self):
        words = self.cfg.text.lower().split()
        counts = pd.DataFrame({'word': words}).value_counts().reset_index()
        counts.columns = ['word', 'count']
        write_frame(counts, self.path('counts', ensure_dirpath=True))
        self.log.info(f"Built word counts: {len(counts)} unique words")
        return self

    def __read__(self, topic='counts'):
        return read_frame(self.path(topic))


def main():
    root = tempfile.mkdtemp(prefix='dbx_example_')
    print(f"Storage root: {root}")

    # Create and build
    block = WordCount(url=root, spec=dict(
        text="to be or not to be that is the question"
    ))

    print(f"Anchor:  {block.anchor}")
    print(f"Hash:    {block.superhash}")
    print(f"Key:     {block.key[:16]}...")
    print(f"Valid:   {block.valid(topic=None)}")

    block.build()

    print(f"Valid:   {block.valid(topic=None)}")
    print(f"Path:    {block.path('counts')}")

    # Read results
    df = block.read('counts')
    print(f"\nWord counts:")
    print(df.to_string(index=False))

    # Building again is a no-op (idempotent)
    print("\nRebuilding (should skip)...")
    block.build()
    print("Done.")


if __name__ == '__main__':
    main()
