# Stand-alone table and figure notes

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
