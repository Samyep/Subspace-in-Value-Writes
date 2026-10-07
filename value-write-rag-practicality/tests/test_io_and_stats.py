from __future__ import annotations

from vw_rag.io_utils import select_shard, stable_int
from vw_rag.stats import bootstrap_mean


def test_shards_are_disjoint_and_restore_frozen_order():
    rows = [{"split_rank": index, "example_id": f"q{index}"} for index in range(1, 11)]
    shards = [select_shard(rows[::-1], index, 3) for index in range(3)]
    seen = [row["example_id"] for shard in shards for row in shard]
    assert len(seen) == len(set(seen)) == 10
    assert set(seen) == {f"q{index}" for index in range(1, 11)}
    assert [row["split_rank"] for row in shards[0]] == [1, 4, 7, 10]


def test_stable_int_is_namespace_sensitive_and_repeatable():
    assert stable_int(7, "a", "q") == stable_int(7, "a", "q")
    assert stable_int(7, "a", "q") != stable_int(7, "b", "q")


def test_bootstrap_mean_is_deterministic():
    first = bootstrap_mean([0.0, 1.0, 1.0], draws=1000, seed=9)
    second = bootstrap_mean([0.0, 1.0, 1.0], draws=1000, seed=9)
    assert first == second
    assert first["mean"] == 2 / 3
    assert first["n"] == 3
