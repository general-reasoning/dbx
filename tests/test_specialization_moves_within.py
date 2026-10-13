"""A key renamed where it is -- within a field's dict, or a nested block's -- is a dotted move to a dotted name.

``redirect_vars={'collator.columns.stratifyby': 'collator.columns.groupby'}``
renders the current entry back under the name it had, in the same place.
Every entry is taken out before any is put back, so names that swap places --
stratifyby to groupby as partitionby goes to stratifyby -- each read what was
there before.
"""
import pytest
from dataclasses import dataclass, field

from dbx.datablocks import SAME, Datablock
from dbx.datatables import Datacollator

ANCHOR = 'Roles'


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class WithColumns(Datablock):
    TOPICS = {'out': 'out.txt'}

    @dataclass
    class VAR(Datablock.VAR):
        columns: dict = field(default_factory=dict)


class WithCollator(Datablock):
    TOPICS = {'out': 'out.txt'}

    @dataclass
    class VAR(Datablock.VAR):
        collator: Datacollator = None


SWAP = {'stratifyby': 'groupby', 'partitionby': 'stratifyby'}


def test_keys_that_swap_names_within_a_dict_field(tmp_path):
    old = WithColumns(datalake=str(tmp_path), anchor=ANCHOR,
                      spec={'columns': {'groupby': 'patient', 'stratifyby': 'cohort'}})
    sp = Datablock.Specialization(spec={}, topics=SAME,
                                  redirect_vars={f'columns.{k}': f'columns.{v}' for k, v in SWAP.items()})
    new = WithColumns(datalake=str(tmp_path), anchor=ANCHOR,
                      spec={'columns': {'stratifyby': 'patient', 'partitionby': 'cohort'}}, SPECIALIZATIONS=[sp])
    assert new.hash != old.hash
    assert new.specialization_hashes() == [old.hash]


def test_keys_that_swap_names_within_a_nested_blocks_dict(tmp_path):
    def holding(columns):
        return dict(collator=Datacollator(spec=dict(columns=columns)))
    old = WithCollator(datalake=str(tmp_path), anchor=ANCHOR,
                       spec=holding({'groupby': ('a', 'patient'), 'stratifyby': ('a', 'cohort')}))
    sp = Datablock.Specialization(spec={}, topics=SAME,
                                  redirect_vars={f'collator.columns.{k}': f'collator.columns.{v}'
                                                 for k, v in SWAP.items()})
    new = WithCollator(datalake=str(tmp_path), anchor=ANCHOR,
                       spec=holding({'stratifyby': ('a', 'patient'), 'partitionby': ('a', 'cohort')}),
                       SPECIALIZATIONS=[sp])
    assert new.hash != old.hash
    assert new.specialization_hashes() == [old.hash]


def test_a_dotted_name_with_no_dict_to_go_in_raises(tmp_path):
    sp = Datablock.Specialization(spec={}, topics=SAME, redirect_vars={'columns.a': 'nowhere.a'})
    block = WithColumns(datalake=str(tmp_path), anchor=ANCHOR, spec={'columns': {'a': 1}}, SPECIALIZATIONS=[sp])
    with pytest.raises(KeyError, match="nowhere.a"):
        block.specialization_hashes()



def test_a_field_holding_nothing_renames_nothing(tmp_path):
    """A partition with no collator renders as it is under a specialization renaming its collator's keys."""
    sp = Datablock.Specialization(spec={}, topics=SAME,
                                  redirect_vars={f'collator.columns.{k}': f'collator.columns.{v}' for k, v in SWAP.items()})
    block = WithCollator(datalake=str(tmp_path), anchor=ANCHOR, spec={'collator': None}, SPECIALIZATIONS=[sp])
    assert block.specialization_hashes() == [block.hash]
