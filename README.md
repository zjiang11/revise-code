# Revised AI-language analysis

This directory is self-contained: revised code, intermediate data, audit trails,
and corrected results stay below `revise_code/`. The original project files are
read-only inputs and are never overwritten.

## Confirmed corpus identity

The academic input is `../source/arxiv/<year>/*.txt`; it contains arXiv papers.
No ProQuest dissertation parser or dissertation input is used. The stable display
label throughout the revision is **arXiv (academic)**. NYT and WSJ are newspaper
records exported from ProQuest; “ProQuest” describes their delivery format, not
the academic corpus.

## Reproducible inclusion and masking rule

1. Extract every sentence containing a configured target as a whole word.
2. Retain it only when an independent AI cue occurs within 35 tokens of the
   target. The target itself never qualifies as its own cue.
3. Exclude explicit non-AI senses listed in `config.NON_AI_PATTERNS` (for example,
   “intelligence agency” or “real estate agent”). Every decision, cue, matching
   phrase, document ID, and sentence is retained in the compressed audit table.
4. For a known multiword AI term, replace the complete expression by one mask:
   `artificial intelligence -> [MASK]`, for example. This retains a relevant
   observation while preventing the word `artificial` from producing completions
   such as *gravity*, *light*, or *turf*. Counts are reported as
   `phrase_adjusted_sentences`; these are adjusted, not removed.
5. A target profile requires at least 100 eligible sentences. Sample at most 500
   using five registered seeds (11, 22, 33, 44, 55).
6. Before reporting BERT substitutes, remove all tokenizer special tokens,
   WordPiece continuations (`##...`), nonalphabetic tokens, one-character tokens,
   the target itself, and the explicit stopword list in `config.py`. Renormalize
   the retained top 50 substitutes.

The rule is intentionally conservative and inspectable. It should be validated
against human labels before making claims about recall; the audit CSV makes such
validation possible without re-extracting the corpus.

Control profiles use one fixed uniform reservoir sample (seed 11), while target
samples vary over five seeds. Holding the control baseline fixed isolates the
stability question asked of the target sentence sampling and avoids conflating it
with a second source of Monte Carlo variation.

## Run

Python 3.10 with `pandas`, `numpy`, `scipy`, `torch`, `transformers`, and
`matplotlib` is required. A CUDA GPU is used automatically.

`bert-base-uncased` must be downloadable or already cached. In an offline
environment, set `MODEL_NAME` in `config.py` to an absolute local snapshot.

```powershell
cd C:\dev\project\revise_code
.\run_pipeline.ps1
```

For a faster profile-generation smoke test that omits control profiles:

```powershell
py -3 03_build_profiles.py --targets-only --seeds 11
```

That shortcut cannot generate scaled-JSD or five-seed stability results.

## Outputs

- `results/audit/corpus_manifest.csv`: exact source provenance.
- `work/sentences/sentence_audit.csv.gz`: every target sentence and decision.
- `results/audit/intelligence_sentence_audit.csv`: readable intelligence audit.
- `results/audit/sentence_filtering_counts.csv`: removals and phrase adjustments.
- `results/tables/profile_coverage_by_seed.csv`: complete target/arena coverage.
- `results/tables/substitution_profiles.csv`: corrected reported profiles.
- `results/tables/jsd_by_seed.csv`: raw and scaled JSD plus all control counts.
- `results/tables/jsd_stability_summary.csv`: JSD range and SD over samples.
- `results/tables/extreme_terms_stability.csv`: bridging/fault-line recurrence.
- `results/tables/largest_arena_comparison_by_seed.csv`: direct NYT–arXiv check.
- `results/TABLE_FIGURE_NOTES.md`: stand-alone notes for every principal output.
- `results/RESULTS_SUMMARY.md`: automatically generated corrected summary.
- `results/validation_report.json`: automated consistency checks.

## Interpretation warning

The rule-based AI-context filter is transparent and reproducible, but its cue
lexicon is a methodological choice. Report its audit performance against a
manually labeled stratified sample before treating it as a validated classifier.
