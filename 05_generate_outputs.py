from __future__ import annotations

import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

import config


def profile_table():
    rows = []
    seed = config.SEEDS[0]
    for word in config.TARGETS:
        for arena in config.ARENAS:
            path = config.PROFILE_ROOT / f"seed_{seed}" / f"{arena}__{word}.pkl"
            if not path.exists(): continue
            with path.open("rb") as handle: data = pickle.load(handle)
            for rank, (substitute, probability) in enumerate(data["GSP"].items(), 1):
                rows.append({"target": word, "arena": arena, "arena_label": config.ARENA_LABELS[arena],
                             "rank": rank, "substitute": substitute, "probability": probability,
                             "seed": seed, "eligible_sentences": data["eligible"],
                             "sampled_sentences": data["sampled"]})
    columns = ["target", "arena", "arena_label", "rank", "substitute", "probability",
               "seed", "eligible_sentences", "sampled_sentences"]
    profiles = pd.DataFrame(rows, columns=columns)
    profiles.to_csv(config.TABLE_ROOT / "substitution_profiles.csv", index=False)
    for target in ("intelligence", "algorithm"):
        part = profiles[(profiles.target == target) & (profiles["rank"] <= 10)]
        wide_rows = []
        for rank in range(1, 11):
            row = {"rank": rank}
            for arena in config.ARENAS:
                hit = part[(part.arena == arena) & (part["rank"] == rank)]
                row[f"{arena}_substitute"] = "" if hit.empty else hit.iloc[0].substitute
                row[f"{arena}_probability"] = "" if hit.empty else hit.iloc[0].probability
            wide_rows.append(row)
        pd.DataFrame(wide_rows).to_csv(config.TABLE_ROOT / f"top10_{target}.csv", index=False)


def comparison_tables():
    path = config.TABLE_ROOT / "jsd_by_seed.csv"
    if not path.exists(): return
    df = pd.read_csv(path)
    arena = (df.groupby("comparison")
             .agg(mean_jsd=("jsd", "mean"), sd_jsd=("jsd", "std"),
                  comparisons=("jsd", "size"),
                  mean_valid_controls=("n_valid_controls", "mean"),
                  minimum_valid_controls=("n_valid_controls", "min"))
             .reset_index().sort_values("mean_jsd", ascending=False))
    arena.to_csv(config.TABLE_ROOT / "arena_pair_summary.csv", index=False)
    corrected = df.groupby("word").jsd.mean().rename("corrected_mean_jsd").reset_index()
    legacy_path = config.PROJECT_ROOT / "outputs" / "tables" / "bridging_vs_faultlines.csv"
    if legacy_path.exists():
        legacy = pd.read_csv(legacy_path).rename(columns={"avg_jsd": "legacy_mean_jsd", "kind": "legacy_class"})
        change = legacy.merge(corrected, on="word", how="outer")
        change["absolute_change"] = change.corrected_mean_jsd - change.legacy_mean_jsd
        change["legacy_rank"] = change.legacy_mean_jsd.rank(method="min")
        change["corrected_rank"] = change.corrected_mean_jsd.rank(method="min")
        change["rank_change"] = change.corrected_rank - change.legacy_rank
        change.sort_values("corrected_mean_jsd").to_csv(
            config.TABLE_ROOT / "legacy_vs_corrected_jsd.csv", index=False)


def audit_summary():
    path = config.TABLE_ROOT / "target_sentence_coverage.csv"
    if not path.exists(): return
    coverage = pd.read_csv(path)
    coverage["removed_percent"] = 100 * coverage.removed_non_ai / coverage.extracted_sentences.replace(0, pd.NA)
    coverage["phrase_adjusted_percent_of_eligible"] = (
        100 * coverage.phrase_adjusted_sentences / coverage.eligible_sentences.replace(0, pd.NA))
    coverage.to_csv(config.TABLE_ROOT / "target_filter_audit_summary.csv", index=False)
    intelligence = coverage[coverage.target == "intelligence"].copy()
    intelligence.to_csv(config.TABLE_ROOT / "intelligence_audit_summary.csv", index=False)


def figures():
    path = config.TABLE_ROOT / "extreme_terms_stability.csv"
    if not path.exists(): return
    df = pd.read_csv(path).sort_values("mean_jsd")
    if df.empty: return
    fig, ax = plt.subplots(figsize=(10, max(7, len(df) * .36)))
    colors = ["#2ca02c" if r.bridging_runs >= 3 else "#d62728" if r.faultline_runs >= 3 else "#4c78a8"
              for r in df.itertuples()]
    ax.barh(df.word, df.mean_jsd, xerr=df.sd_jsd.fillna(0), color=colors, alpha=.85)
    ax.set_xlabel("Mean cross-arena Jensen–Shannon divergence (bits)")
    ax.set_title("Target-term divergence across five sentence-sampling seeds")
    ax.invert_yaxis(); fig.tight_layout()
    fig.savefig(config.FIGURE_ROOT / "figure_divergence_stability.png", dpi=300); plt.close(fig)


def two_dimensional_map():
    """Build corrected temporality/ontology map from five-seed target GSPs."""
    available = {}
    for seed in config.SEEDS:
        for arena in config.ARENAS:
            words = set()
            for target in config.TARGETS:
                if (config.PROFILE_ROOT / f"seed_{seed}" / f"{arena}__{target}.pkl").exists():
                    words.add(target)
            available[(seed, arena)] = words
    common_targets = set(config.TARGETS)
    for words in available.values():
        common_targets &= words
    common_targets = sorted(common_targets)
    if not common_targets:
        print("No common target profiles; corrected 2D map skipped.")
        return

    rows = []
    for seed in config.SEEDS:
        for arena in config.ARENAS:
            for target in common_targets:
                path = config.PROFILE_ROOT / f"seed_{seed}" / f"{arena}__{target}.pkl"
                with path.open("rb") as handle:
                    profile = pickle.load(handle)["GSP"]
                score = {name: sum(profile.get(word, 0.0) for word in anchors)
                         for name, anchors in config.ANCHORS.items()}
                rows.append({"seed": seed, "arena": arena,
                             "arena_label": config.ARENA_LABELS[arena], "target": target,
                             **{f"anchor_{k}": v for k, v in score.items()},
                             "temporality": score["future"] - score["past"],
                             "ontology": score["beyond"] - score["machine"]})
    coordinates = pd.DataFrame(rows)
    coordinates.to_csv(config.TABLE_ROOT / "two_dimensional_target_coordinates.csv", index=False)
    seed_centers = (coordinates.groupby(["seed", "arena", "arena_label"])
                    [["temporality", "ontology"]].mean().reset_index())
    seed_centers.to_csv(config.TABLE_ROOT / "two_dimensional_arena_centers_by_seed.csv", index=False)
    centers = (seed_centers.groupby(["arena", "arena_label"])
               .agg(temporality_mean=("temporality", "mean"),
                    temporality_sd=("temporality", "std"),
                    ontology_mean=("ontology", "mean"),
                    ontology_sd=("ontology", "std"),
                    seeds=("seed", "nunique")).reset_index())
    centers["common_targets"] = len(common_targets)
    centers.to_csv(config.TABLE_ROOT / "two_dimensional_arena_centers.csv", index=False)

    colors = {"NYT": "#d62728", "WSJ": "#1f77b4", "arxiv": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(9, 8))
    for row in centers.itertuples(index=False):
        ax.errorbar(row.temporality_mean, row.ontology_mean,
                    xerr=0 if pd.isna(row.temporality_sd) else row.temporality_sd,
                    yerr=0 if pd.isna(row.ontology_sd) else row.ontology_sd,
                    fmt="X", markersize=13, capsize=5, elinewidth=1.5,
                    markeredgecolor="black", color=colors[row.arena],
                    label=row.arena_label, zorder=5)
        ax.annotate(f"  {row.arena}", (row.temporality_mean, row.ontology_mean),
                    fontsize=11, fontweight="bold", color=colors[row.arena])
    ax.axhline(0, color="grey", linewidth=.8)
    ax.axvline(0, color="grey", linewidth=.8)
    ax.set_xlabel("Temporality: past-oriented  ←  0  →  future-oriented")
    ax.set_ylabel("Ontology: machine-side  ←  0  →  beyond-machine")
    ax.set_title("Corrected temporality–ontology positions by arena\n"
                 f"({len(common_targets)} common targets; mean ± SD across {len(config.SEEDS)} seeds)")
    ax.legend(loc="best", frameon=False)
    ax.margins(.25)
    fig.tight_layout()
    fig.savefig(config.FIGURE_ROOT / "figure_temporality_ontology_corrected.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def semantic_anchor_analysis():
    """Expanded-anchor MiniLM scores for every configured target/arena/seed."""
    model = SentenceTransformer(str(config.MINILM_MODEL_PATH))
    anchor_centers = {}
    for name, words in config.SEMANTIC_ANCHORS.items():
        center = model.encode(list(words), normalize_embeddings=True).mean(axis=0)
        anchor_centers[name] = center / np.linalg.norm(center)

    loaded, vocabulary = {}, set()
    for seed in config.SEEDS:
        for arena in config.ARENAS:
            for target in config.TARGETS:
                path = config.PROFILE_ROOT / f"seed_{seed}" / f"{arena}__{target}.pkl"
                if not path.exists(): continue
                with path.open("rb") as handle: data = pickle.load(handle)
                loaded[(seed, arena, target)] = data
                vocabulary.update(data["GSP"])
    words = sorted(vocabulary)
    matrix = model.encode(words, normalize_embeddings=True, show_progress_bar=False)
    word_vectors = dict(zip(words, matrix))

    rows = []
    for (seed, arena, target), data in loaded.items():
        profile = data["GSP"]
        pwords = list(profile)
        probabilities = np.asarray([profile[w] for w in pwords], dtype=float)
        probabilities /= probabilities.sum()
        embeddings = np.vstack([word_vectors[w] for w in pwords])
        scores = {name: float(probabilities @ (embeddings @ center))
                  for name, center in anchor_centers.items()}
        rows.append({"target": target, "arena": arena,
                     "arena_label": config.ARENA_LABELS[arena], "seed": seed,
                     "eligible_sentences": data["eligible"], "sampled_sentences": data["sampled"],
                     **{f"similarity_{name}": value for name, value in scores.items()},
                     "temporality_future_minus_past": scores["future"] - scores["past"],
                     "ontology_beyond_minus_machine": scores["beyond"] - scores["machine"]})
    detail = pd.DataFrame(rows)
    detail.to_csv(config.TABLE_ROOT / "semantic_anchor_scores_by_seed.csv", index=False)

    metrics = ["similarity_past", "similarity_future", "similarity_machine",
               "similarity_beyond", "temporality_future_minus_past",
               "ontology_beyond_minus_machine"]
    if detail.empty: return
    aggregations = {}
    for metric in metrics:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_sd"] = (metric, "std")
    summary = (detail.groupby(["target", "arena", "arena_label"])
               .agg(**aggregations, seeds=("seed", "nunique")).reset_index())

    coverage = pd.read_csv(config.TABLE_ROOT / "target_sentence_coverage.csv")
    grid = coverage[["target", "arena", "eligible_sentences", "threshold_met", "missing_reason"]]
    summary = grid.merge(summary, on=["target", "arena"], how="left")
    summary["arena_label"] = summary.arena.map(config.ARENA_LABELS)
    summary.to_csv(config.TABLE_ROOT / "semantic_anchor_scores_by_target_arena.csv", index=False)
    wide_parts = []
    for arena in config.ARENAS:
        part = summary[summary.arena == arena][[
            "target", "temporality_future_minus_past_mean", "temporality_future_minus_past_sd",
            "ontology_beyond_minus_machine_mean", "ontology_beyond_minus_machine_sd",
            "eligible_sentences", "missing_reason"]].copy()
        part = part.rename(columns={c: f"{arena}_{c}" for c in part.columns if c != "target"})
        wide_parts.append(part)
    wide = wide_parts[0]
    for part in wide_parts[1:]: wide = wide.merge(part, on="target", how="outer")
    wide.to_csv(config.TABLE_ROOT / "semantic_anchor_scores_wide.csv", index=False)

    valid_sets = {arena: set(summary[(summary.arena == arena) & summary.threshold_met].target)
                  for arena in config.ARENAS}
    common = set.intersection(*valid_sets.values())
    average_rows = []
    for arena in config.ARENAS:
        part = summary[(summary.arena == arena) & summary.threshold_met]
        for scope, scoped in (("all_available_targets", part),
                              ("common_targets_only", part[part.target.isin(common)])):
            average_rows.append({"arena": arena, "arena_label": config.ARENA_LABELS[arena],
                                 "scope": scope, "n_targets": len(scoped),
                                 "temporality_mean": scoped.temporality_future_minus_past_mean.mean(),
                                 "temporality_between_target_sd": scoped.temporality_future_minus_past_mean.std(),
                                 "ontology_mean": scoped.ontology_beyond_minus_machine_mean.mean(),
                                 "ontology_between_target_sd": scoped.ontology_beyond_minus_machine_mean.std()})
    pd.DataFrame(average_rows).to_csv(config.TABLE_ROOT / "semantic_anchor_arena_averages.csv", index=False)

    # Formal revised 2D map: common targets only, so every arena center has the
    # same lexical composition. Small points are target means across seeds; X
    # markers are arena means and error bars are SD of seed-level arena centers.
    common_detail = detail[detail.target.isin(common)].copy()
    seed_centers = (common_detail.groupby(["seed", "arena", "arena_label"])
                    .agg(temporality=("temporality_future_minus_past", "mean"),
                         ontology=("ontology_beyond_minus_machine", "mean"))
                    .reset_index())
    seed_centers.to_csv(config.TABLE_ROOT / "semantic_anchor_arena_centers_by_seed.csv", index=False)
    centers = (seed_centers.groupby(["arena", "arena_label"])
               .agg(temporality_mean=("temporality", "mean"),
                    temporality_sd=("temporality", "std"),
                    ontology_mean=("ontology", "mean"),
                    ontology_sd=("ontology", "std"), seeds=("seed", "nunique"))
               .reset_index())
    centers["common_targets"] = len(common)
    centers.to_csv(config.TABLE_ROOT / "semantic_anchor_2d_centers.csv", index=False)
    target_means = (common_detail.groupby(["target", "arena", "arena_label"])
                    .agg(temporality=("temporality_future_minus_past", "mean"),
                         ontology=("ontology_beyond_minus_machine", "mean"))
                    .reset_index())

    colors = {"NYT": "#d62728", "WSJ": "#1f77b4", "arxiv": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(10, 8))
    for arena in config.ARENAS:
        points = target_means[target_means.arena == arena]
        ax.scatter(points.temporality, points.ontology, s=45, alpha=.28,
                   color=colors[arena], edgecolors="none")
        row = centers[centers.arena == arena].iloc[0]
        ax.errorbar(row.temporality_mean, row.ontology_mean,
                    xerr=row.temporality_sd, yerr=row.ontology_sd,
                    fmt="X", markersize=15, capsize=5, elinewidth=1.8,
                    markeredgecolor="black", color=colors[arena],
                    label=row.arena_label, zorder=10)
        ax.annotate(f"  {arena}", (row.temporality_mean, row.ontology_mean),
                    color=colors[arena], fontsize=12, fontweight="bold")
    ax.axhline(0, color="grey", linewidth=.8)
    ax.axvline(0, color="grey", linewidth=.8)
    ax.set_xlabel("Temporality: past-oriented  ←  future − past  →  future-oriented")
    ax.set_ylabel("Ontology: machine-side  ←  beyond − machine  →  beyond-machine")
    ax.set_title("Semantic-anchor temporality–ontology map\n"
                 f"{len(common)} common targets; MiniLM similarity weighted by BERT GSP")
    ax.legend(frameon=False, loc="best")
    ax.margins(.12)
    fig.tight_layout()
    for filename in ("figure_temporality_ontology_semantic.png",
                     "figure_temporality_ontology_corrected.png"):
        fig.savefig(config.FIGURE_ROOT / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def notes():
    text = """# Stand-alone table and figure notes

## Corpus terminology

The academic corpus consists exclusively of arXiv papers. `arxiv` is the stable
machine identifier and “arXiv (academic)” is its display label. No dissertation
corpus is read by this pipeline.

## sentence_filtering_counts.csv

Each row is an arena–target combination. `extracted_sentences` counts sentences
containing the literal target. `removed_non_ai` counts sentences excluded because
they lack an independent AI-context cue or match an explicit non-AI sense.
`phrase_adjusted_sentences` are retained AI sentences in which a whole multiword
term (for example, “artificial intelligence”) was replaced by one mask to prevent
the remaining modifier from driving phrase completion.

## profile_coverage_by_seed.csv

Reports extracted, eligible, sampled, and model-used sentence counts for every
word, arena, and random seed. A profile requires at least 100 eligible sentences;
at most 500 are sampled. `missing_reason` explains every absent profile.

## substitution_profiles.csv

Filtered BERT (`bert-base-uncased`) mask substitutes for the first registered
seed. Special tokens, WordPiece continuations beginning `##`, nonalphabetic and
one-character tokens, the target itself, and the versioned stopword list are
excluded before the top 50 probabilities are renormalized.

## jsd_by_seed.csv and jsd_stability_summary.csv

JSD is base-2 Jensen–Shannon divergence between normalized top-50 substitute
profiles. Higher values mean greater cross-arena contextual divergence. Scaled
JSD is the fraction of valid frequency-matched controls with lower JSD than the
target. Matched, selected, and actually valid control counts are reported
separately. Fewer than 10 controls is `VERY_SMALL`, 10–29 `SMALL`, 30–99
`LIMITED`, and 100 `FULL`.

## arena_pair_summary.csv

Summarizes corrected JSD by arena pair over all available target terms and five
sampling seeds. `comparisons` is the number of valid target-seed observations;
terms failing the 100-eligible-sentence threshold in either arena are absent.

## legacy_vs_corrected_jsd.csv

Compares old and corrected mean JSD. Because filtering, phrase masking, token
preprocessing, and samples all changed, the difference is descriptive rather
than a single-factor causal estimate.

## top10_intelligence.csv and top10_algorithm.csv

Top ten filtered substitutes by arena for seed 11. Probabilities are normalized
over the retained top 50; empty cells indicate a missing threshold-qualified profile.

## figure_divergence_stability.png

Bars show each target's mean JSD across the three arena pairs and five random
sentence samples; error bars show the between-seed standard deviation. Green and
red identify terms appearing among the three lowest- or highest-divergence terms
in at least three of five runs. Blue terms do not meet that stability criterion.
Only profiles meeting the 100-eligible-sentence threshold enter the figure.

## figure_temporality_ontology_semantic.png / figure_temporality_ontology_corrected.png

Arena centers use only target terms with valid profiles in all three arenas and
all five seeds. MiniLM cosine similarity between every filtered top-50 GSP
substitute and expanded anchor centroids in `config.SEMANTIC_ANCHORS` is weighted
by BERT probabilities. Temporality equals future similarity minus past similarity;
ontology equals beyond-machine similarity minus machine similarity. Faint points
are target means across seeds. X markers are arena means; error bars are SD of
seed-level arena centers. Positive values indicate future-oriented and
beyond-machine language. Both filenames contain the same formal revised figure.

## semantic_anchor_scores_by_target_arena.csv

Expanded-anchor sensitivity analysis for the complete 23-target × 3-arena grid.
MiniLM cosine similarities between every filtered GSP substitute and normalized
anchor centroids are weighted by BERT probabilities. Temporality is future minus
past; ontology is beyond-machine minus machine. Means and SDs summarize five
sentence-sampling seeds. Missing combinations retain their eligibility count and
standard reason rather than being silently omitted.

## semantic_anchor_arena_averages.csv

Arena means are reported twice: `all_available_targets` uses every threshold-valid
target in that arena, while `common_targets_only` uses only targets valid in all
three arenas and is the appropriate direct arena comparison. Between-target SD
describes target heterogeneity, not sampling-seed uncertainty.
"""
    (config.RESULT_ROOT / "TABLE_FIGURE_NOTES.md").write_text(text, encoding="utf-8")


def result_summary():
    extreme_path = config.TABLE_ROOT / "extreme_terms_stability.csv"
    largest_path = config.TABLE_ROOT / "largest_arena_comparison_by_seed.csv"
    lines = ["# Corrected analysis summary", "",
             "This report is generated from AI-context-filtered, phrase-adjusted, stopword- and WordPiece-filtered profiles.", ""]
    if extreme_path.exists():
        df = pd.read_csv(extreme_path).sort_values("mean_jsd")
        if len(df):
            lines += [f"Lowest-divergence terms: {', '.join(df.head(3).word)}.",
                      f"Highest-divergence terms: {', '.join(df.tail(3).word)}.", ""]
    if largest_path.exists():
        df = pd.read_csv(largest_path)
        if len(df):
            counts = df.comparison.value_counts()
            lines += ["Largest mean arena comparison by seed: " +
                      "; ".join(f"{k}: {v}/{len(df)} runs" for k, v in counts.items()) + ".", ""]
    intel = config.TABLE_ROOT / "substitution_profiles.csv"
    if intel.exists():
        prof = pd.read_csv(intel)
        bad = prof[(prof.target == "intelligence") & prof.substitute.isin(["gravity", "light", "turf", "the"])]
        lines += [f"Intelligence artifact check: gravity/light/turf/the occur {len(bad)} times in the corrected top-50 reported profiles.", ""]
    jsd_path = config.TABLE_ROOT / "jsd_by_seed.csv"
    if jsd_path.exists():
        jsd_df = pd.read_csv(jsd_path)
        flags = jsd_df.control_set_flag.value_counts()
        lines += ["Scaled-JSD control warning: " + "; ".join(f"{k}={v}" for k, v in flags.items()) +
                  ". No scaled value based on a small set should be presented without its denominator.", ""]
    lines += ["The legacy and corrected runs are compared in tables/legacy_vs_corrected_jsd.csv. "
              "Legacy profiles did not preserve sampled sentence IDs, so changes can be quantified at the result level but not attributed through a paired legacy sentence audit."]
    (config.RESULT_ROOT / "RESULTS_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    profile_table(); comparison_tables(); audit_summary(); figures(); semantic_anchor_analysis(); notes(); result_summary()
    print("Final tables, figure, notes, and results summary written.")


if __name__ == "__main__": main()
