"""
The figures of DATAPARTITION.md: one small table, partitioned by stratifyby, by partitionby, and by both.

Each figure is drawn from a real `DatatablePartition` of the table, so the cuts and the parts
are the code's, not the picture's. Run from the repo root:

    DBX_DIRTY_REPO_OK=1 python figures/datapartition/make_figures.py
"""
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.hashsalt'] = 'datapartition'      # the same SVG from every run
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from dbx.datatables import DATASLICE, Datacollator, Datatablet, Datatable
from dbx.dataparts import DatatablePartition

HERE = Path(__file__).parent

#: Each tablet's rows, (stratum, partition). Every stratum lies in one partition: s1, s3 in p1; s2, s4, s5, s6 in p2.
TABLETS = [
    [('s1', 'p1')] * 4,
    [('s2', 'p2'), ('s2', 'p2'), ('s3', 'p1'), ('s3', 'p1')],
    [('s1', 'p1')] * 3,
    [('s4', 'p2'), ('s4', 'p2'), ('s6', 'p2'), ('s6', 'p2')],
    [('s5', 'p2'), ('s3', 'p1'), ('s5', 'p2'), ('s5', 'p2')],
]
#: The seed of every figure's deal: one under which some tablet is cut and dealt to both parts.
SEED = 4
#: A tablet whose one stratum straddles two partitions: refused when both roles are set.
STRADDLING = [('s7', 'p1'), ('s7', 'p1'), ('s7', 'p2')]


class RowsTablet(Datatablet):
    TOPICS = {'meta': DATASLICE(stratum='str', partition='str')}

    @dataclass
    class VAR(Datatablet.VAR):
        rows: list = field(default_factory=list)

    def __build__(self):
        with self.slice_writers() as writers:
            for g, s in self.var.rows:
                writers['meta'].write({'stratum': g, 'partition': s})


class RowsTable(Datatable):
    TABLET = RowsTablet

    @dataclass
    class VAR(Datatable.VAR):
        tablets: list = None

    @property
    def n_tablets(self):
        return len(self.var.tablets)

    def __tablet__(self, idx, **kw):
        return super().__tablet__(idx, rows=[list(r) for r in self.var.tablets[idx]])


def partition(lake, tablets, roles, **spec):
    table = RowsTable(datalake=lake, spec={'tablets': [[list(r) for r in t] for t in tablets]}).build()
    columns = {r: ('meta', {'stratifyby': 'stratum', 'partitionby': 'partition'}[r]) for r in roles}
    p = DatatablePartition(datalake=lake, spec=dict(
        datatable=table, fractions=[0.5, 0.5], partition_slice='meta', seed=SEED,
        collator=Datacollator(spec=dict(columns=columns)), **spec)).build()
    return json.loads(p.fs.cat(p.path('tablets')))


# ---- drawing -------------------------------------------------------------------------------------

CW, H = 0.46, 0.34                                   # a row's two cells, g and s
W = 2 * CW
PART = ['#bfe3c8', '#e3cdf2']                        # each part's fill: its rows, its pieces, its box
PART_INK = ['#2e7d4f', '#7b4ea3']
EDGE, INK, FAINT, NONE = '#555555', '#222222', '#a8a8a8', '#f1f1f1'


def grid(ax, x, y, rows, roles, parts, title=None, idx=None, header=False):
    """A tablet or a piece: one row per line, top down, its stratum s and its partition p side by side.

    A column the partition reads is filled with the color of the part the row goes to; one it does
    not read is grey.
    """
    read = ('stratifyby' in roles, 'partitionby' in roles)
    for i, row in enumerate(rows):
        top = y - (i + 1) * H
        for c, value in enumerate(row):
            fill = PART[parts[i]] if read[c] and parts[i] is not None else NONE
            ax.add_patch(Rectangle((x + c * CW, top), CW, H, facecolor=fill, edgecolor=EDGE, lw=0.7))
            ax.text(x + (c + 0.5) * CW, top + H / 2, value, ha='center', va='center', fontsize=8,
                    color=INK if read[c] else FAINT)
        if idx is not None:
            ax.text(x - 0.06, top + H / 2, str(idx[i]), ha='right', va='center', fontsize=6, color=FAINT)
    if header:
        for c, name in enumerate('sp'):
            ax.text(x + (c + 0.5) * CW, y + 0.05, name, ha='center', va='bottom', fontsize=7, style='italic',
                    color=INK if read[c] else FAINT)
    if title:
        ax.text(x + W / 2, y + (0.3 if header else 0.1), title, ha='center', va='bottom', fontsize=8, color=INK)
    return y - len(rows) * H


def entry_rows(entry, roles):
    """An entry of tabs.json as (tablet, rows, source row indices, label)."""
    if isinstance(entry, int):
        return entry, TABLETS[entry], list(range(len(TABLETS[entry]))), f'tablet {entry}'
    t = entry['tab']
    want = tuple(entry[r] for r in roles)
    key = {'stratifyby': 0, 'partitionby': 1}
    idx = [i for i, row in enumerate(TABLETS[t]) if tuple(row[key[r]] for r in roles) == want]
    return t, [TABLETS[t][i] for i in idx], idx, f"tablet {t} #{'#'.join(want)}"


def dealt(tablets_json, roles):
    """What each part holds, as drawn: every whole tablet, piece and packed segment under its source tablet, and each part's tablets.

    Returns ``(items, contents)``: one item per thing dealt -- its source tablet, rows, their indices
    there, its part, and the packed tablet it went into, if any -- and per part ``(rows, title)`` per tablet.
    """
    items, contents = [], []
    for k, entries in enumerate(tablets_json):
        content = []
        for e in entries:
            pack = e.get('pack') if isinstance(e, dict) else None
            segments = e['segments'] if pack is not None else [[e, 0, None]]
            pooled = []
            for entry, offset, length in segments:
                t, rows, idx, label = entry_rows(entry, roles)
                end = None if length is None else offset + length
                rows, idx = rows[offset:end], idx[offset:end]
                name = 'whole' if isinstance(entry, int) else '#' + '#'.join(entry[r] for r in roles)
                items.append(dict(t=t, rows=rows, idx=idx, k=k, whole=isinstance(entry, int) and len(idx) == len(TABLETS[t]),
                                  name=name, pack=pack))
                pooled += rows
            content.append((pooled, f'pack {pack}') if pack is not None else (pooled, label))
        contents.append(content)
    return items, contents


def figure(tablets_json, roles, title, out):
    fig, ax = plt.subplots(figsize=(15, 8.4))
    ax.set_xlim(-0.6, 19.6)
    ax.set_ylim(-8.4, 2.2)
    ax.axis('off')
    ax.text(-0.5, 2.15, title, fontsize=11, weight='bold', va='top', color=INK)

    items, contents = dealt(tablets_json, roles)
    row_part = {(it['t'], i): it['k'] for it in items for i in it['idx']}

    # 1. the table, each row colored by the part it goes to
    ax.text(-0.5, 0.45, 'table', fontsize=9, style='italic', color=FAINT)
    tablet_x = [0.6 + 2.4 * t for t in range(len(TABLETS))]
    for t, rows in enumerate(TABLETS):
        parts = [row_part.get((t, i)) for i in range(len(rows))]
        split = len(set(parts)) > 1
        grid(ax, tablet_x[t], 0.2, rows, roles, parts, f'tablet {t}' + (' (split)' if split else ''), header=True)

    # 2. what is dealt: whole tablets, pieces and packed segments, under their source tablet
    by_tablet = {}
    for it in sorted(items, key=lambda it: (it['t'], it['name'], it['idx'])):
        by_tablet.setdefault(it['t'], []).append(it)
    ax.text(-0.5, -1.55, 'dealt', fontsize=9, style='italic', color=FAINT)
    y_item = -1.95
    for t, its in by_tablet.items():
        n = len(its)
        for j, it in enumerate(its):
            x = tablet_x[t] + (j - (n - 1) / 2) * 1.2
            k, rows = it['k'], it['rows']
            grid(ax, x, y_item, rows, roles, [k] * len(rows), idx=None if it['whole'] else it['idx'])
            to = f"→ pack {it['pack']}\nof part {k}" if it['pack'] is not None else f"→ part {k}"
            ax.text(x + W / 2, y_item - len(rows) * H - 0.12, f"{it['name']}\n{to}",
                    ha='center', va='top', fontsize=7, color=PART_INK[k])
            ax.add_patch(FancyArrowPatch((tablet_x[t] + W / 2, 0.2 - len(TABLETS[t]) * H - 0.05),
                                         (x + W / 2, y_item + 0.05), arrowstyle='-|>', mutation_scale=8,
                                         color=PART_INK[k], lw=0.9,
                                         linestyle='-' if it['whole'] and it['pack'] is None else (0, (3, 2))))

    # 3. the parts
    for k, entries in enumerate(tablets_json):
        x0 = 12.6 + 3.5 * k
        ax.add_patch(FancyBboxPatch((x0 - 0.15, -8.2), 3.3, 9.2, boxstyle='round,pad=0.05',
                                    facecolor='none', edgecolor=PART_INK[k], lw=1.4))
        ax.text(x0 + 1.5, 1.45, f'part {k}', ha='center', va='top', fontsize=10, weight='bold', color=PART_INK[k])
        x, y = x0, 0.35
        for rows, label in contents[k]:
            if y - len(rows) * H - 0.55 < -8.0:
                x, y = x + 1.6, 0.35
            grid(ax, x + 0.3, y - 0.25, rows, roles, [k] * len(rows), label)
            y = y - 0.25 - len(rows) * H - 0.35
        n = {s: sum(1 for rows, _ in contents[k] for _, p2 in rows if p2 == s) for s in ('p1', 'p2')}
        ax.text(x0 + 1.5, -8.1, f"{sum(n.values())} rows" + (f": p1 {n['p1']}, p2 {n['p2']}" if 'partitionby' in roles else ''),
                ha='center', va='bottom', fontsize=8, color=INK)

    # legend
    ly = -6.2
    for k in range(len(tablets_json)):
        ax.add_patch(Rectangle((0.4 + 1.8 * k, ly), CW * 0.6, H * 0.6, facecolor=PART[k], edgecolor=EDGE, lw=0.6))
        ax.text(0.4 + 1.8 * k + CW * 0.8, ly + H * 0.3, f'to part {k}', va='center', fontsize=8)
    ax.text(0.4, ly - 0.45, 's: the row\'s stratum (s1 … s6);  p: its partition (p1, p2);  grey: a column the partition '
            'does not read', fontsize=8, color=INK)
    ax.text(0.4, ly - 0.85, 'solid arrow: a whole tablet;  dashed: a piece, or rows packed;  small numbers: the source rows taken',
            fontsize=8, color=FAINT)
    fig.savefig(out, format='svg', bbox_inches='tight', facecolor='white', metadata={'Date': None})
    plt.close(fig)


def refused_figure(out, message):
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    ax.set_xlim(-0.2, 8.4)
    ax.set_ylim(-1.5, 0.9)
    ax.axis('off')
    both = ('stratifyby', 'partitionby')
    grid(ax, 0.3, 0.2, STRADDLING, both, [None] * 3, 'tablet 5', header=True)
    ax.add_patch(FancyArrowPatch((1.35, -0.35), (2.05, -0.35), arrowstyle='-|>', mutation_scale=8, color=EDGE, lw=0.8))
    grid(ax, 2.2, 0.2, STRADDLING[:2], both, [None] * 2, '#s7#p1', idx=[0, 1])
    grid(ax, 3.4, 0.2, STRADDLING[2:], both, [None], '#s7#p2', idx=[2])
    ax.text(4.7, -0.1, "s7's rows lie in partitions p1 and p2.\nA stratum goes to one part whole,\nso it cannot be dealt in two\npartitions: the partition raises", fontsize=8, va='center')
    ax.text(4.7, -1.0, 'ValueError', fontsize=9, weight='bold', color='#c0392b', va='center')
    fig.savefig(out, format='svg', bbox_inches='tight', facecolor='white', metadata={'Date': None})
    plt.close(fig)
    print(out.name, '<-', message[:140])


def main():
    os.environ.setdefault('DBX_DIRTY_REPO_OK', '1')
    cases = [(('stratifyby',), 'stratifyby: all the rows of a stratum go to one part', 'stratifyby.svg'),
             (('partitionby',), 'partitionby: the rows of each partition are divided between the parts', 'partitionby.svg'),
             (('stratifyby', 'partitionby'), 'stratifyby and partitionby: all the rows of a stratum go to one part, '
              'and the strata of each partition are divided between the parts', 'both.svg'),
             (('stratifyby',), "coalesce='whole': a cut tablet whose pieces all went to one part is that part's whole tablet "
              "again", 'whole.svg', {'coalesce': 'whole'}),
             (('stratifyby',), "coalesce='pack', min_tablet_rows=4: then every tablet or piece of fewer rows is pooled "
              "into packed tablets", 'pack.svg', {'coalesce': 'pack', 'min_tablet_rows': 4})]
    with tempfile.TemporaryDirectory() as lake:
        for roles, title, name, *spec in cases:
            tablets_json = partition(os.path.join(lake, name), TABLETS, roles, **(spec[0] if spec else {}))
            print(name, json.dumps(tablets_json))
            figure(tablets_json, roles, title, HERE / name)
        try:
            partition(os.path.join(lake, 'refused'), TABLETS + [STRADDLING], ('stratifyby', 'partitionby'))
        except ValueError as e:
            refused_figure(HERE / 'refused.svg', str(e))
        else:
            raise AssertionError('a stratum straddling two partitions was not refused')


if __name__ == '__main__':
    main()
