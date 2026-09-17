from __future__ import annotations

from itertools import combinations
import pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config
from common import jsd


def load(seed, arena, word):
    path = config.PROFILE_ROOT / f"seed_{seed}" / f"{arena}__{word}.pkl"
    if not path.exists(): return None
    with path.open("rb") as handle: return pickle.load(handle)["GSP"]


def main():
    controls = pd.read_csv(config.TABLE_ROOT / "control_sets.csv")
    rows = []
    for seed in config.SEEDS:
        for row in controls.itertuples(index=False):
            p, q = load(seed, row.arena_a, row.target), load(seed, row.arena_b, row.target)
            if p is None or q is None: continue
            observed = jsd(p, q)
            selected = [x.strip() for x in str(row.selected_controls).split(";") if x.strip()]
            values = []
            for word in selected:
                # Controls use one fixed, reproducible reservoir (seed 11). Target
                # sentence sampling varies across seeds for the stability check.
                pc, qc = load(config.SEEDS[0], row.arena_a, word), load(config.SEEDS[0], row.arena_b, word)
                if pc is not None and qc is not None: values.append(jsd(pc, qc))
            n = len(values)
            scaled = float(np.mean(np.asarray(values) < observed)) if n else np.nan
            flag = "VERY_SMALL" if n < 10 else "SMALL" if n < config.SMALL_CONTROL_WARN else \
                   "LIMITED" if n < config.MAX_CONTROLS else "FULL"
            rows.append({"seed": seed, "word": row.target, "arena_a": row.arena_a,
                         "arena_b": row.arena_b, "comparison": f"{row.arena_a}–{row.arena_b}",
                         "jsd": observed, "scaled_jsd": scaled,
                         "n_matched_controls": row.n_matched_controls,
                         "n_selected_controls": row.n_selected_controls,
                         "n_valid_controls": n, "control_set_flag": flag})
    by_seed = pd.DataFrame(rows)
    by_seed.to_csv(config.TABLE_ROOT / "jsd_by_seed.csv", index=False)
    if by_seed.empty:
        print("No profiles found; run 03_build_profiles.py first."); return
    summary = (by_seed.groupby(["word", "comparison"])
               .agg(jsd_mean=("jsd", "mean"), jsd_sd=("jsd", "std"),
                    jsd_min=("jsd", "min"), jsd_max=("jsd", "max"),
                    scaled_jsd_mean=("scaled_jsd", "mean"),
                    valid_controls_min=("n_valid_controls", "min"),
                    valid_controls_max=("n_valid_controls", "max"))
               .reset_index())
    summary.to_csv(config.TABLE_ROOT / "jsd_stability_summary.csv", index=False)

    ranks = []
    for seed, part in by_seed.groupby("seed"):
        means = part.groupby("word").jsd.mean().sort_values()
        for rank, (word, value) in enumerate(means.items(), 1):
            ranks.append({"seed": seed, "word": word, "mean_jsd": value, "rank_low_to_high": rank,
                          "bridging_top3": rank <= 3, "faultline_top3": rank > len(means) - 3})
    rank_df = pd.DataFrame(ranks)
    rank_df.to_csv(config.TABLE_ROOT / "rank_by_seed.csv", index=False)
    stability = (rank_df.groupby("word")
                 .agg(mean_jsd=("mean_jsd", "mean"), sd_jsd=("mean_jsd", "std"),
                      mean_rank=("rank_low_to_high", "mean"), rank_min=("rank_low_to_high", "min"),
                      rank_max=("rank_low_to_high", "max"),
                      bridging_runs=("bridging_top3", "sum"), faultline_runs=("faultline_top3", "sum"))
                 .reset_index().sort_values("mean_jsd"))
    stability.to_csv(config.TABLE_ROOT / "extreme_terms_stability.csv", index=False)
    pivot = rank_df.pivot(index="word", columns="seed", values="rank_low_to_high")
    correlations = []
    for a, b in combinations(config.SEEDS, 2):
        pair = pivot[[a, b]].dropna()
        correlations.append({"seed_a": a, "seed_b": b, "spearman_rank_correlation":
                             spearmanr(pair[a], pair[b]).statistic if len(pair) > 1 else np.nan})
    pd.DataFrame(correlations).to_csv(config.TABLE_ROOT / "rank_correlations.csv", index=False)
    largest = (by_seed.groupby(["seed", "comparison"]).jsd.mean().reset_index()
               .sort_values(["seed", "jsd"]).groupby("seed").tail(1))
    largest.to_csv(config.TABLE_ROOT / "largest_arena_comparison_by_seed.csv", index=False)
    print("JSD, scaled controls, and stability tables written.")


if __name__ == "__main__": main()
