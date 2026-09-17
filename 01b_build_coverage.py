"""Rebuild target coverage from the persisted sentence audit without re-extraction."""
import pandas as pd
import config


def main():
    df = pd.read_csv(config.SENTENCE_ROOT / "sentence_audit.csv.gz")
    df["eligible_phrase_adjusted"] = df["eligible"].astype(bool) & df["phrase_adjusted"].astype(bool)
    counts = (df.groupby(["target", "arena"])
              .agg(extracted_sentences=("sentence_id", "size"),
                   eligible_sentences=("eligible", "sum"),
                   phrase_adjusted_sentences=("eligible_phrase_adjusted", "sum"))
              .reset_index())
    counts["removed_non_ai"] = counts.extracted_sentences - counts.eligible_sentences
    grid = pd.MultiIndex.from_product([config.TARGETS, config.ARENAS],
                                      names=["target", "arena"]).to_frame(index=False)
    out = grid.merge(counts, on=["target", "arena"], how="left").fillna(0)
    for column in ("extracted_sentences", "eligible_sentences", "phrase_adjusted_sentences", "removed_non_ai"):
        out[column] = out[column].astype(int)
    out["minimum_required"] = config.MIN_ELIGIBLE_SENTENCES
    out["sample_cap"] = config.MAX_SENTENCES
    out["sampled_if_run"] = out.eligible_sentences.clip(upper=config.MAX_SENTENCES)
    out["threshold_met"] = out.eligible_sentences.ge(config.MIN_ELIGIBLE_SENTENCES)
    out["missing_reason"] = ""
    out.loc[out.extracted_sentences.eq(0), "missing_reason"] = "NO_OCCURRENCE"
    out.loc[out.extracted_sentences.gt(0) & out.eligible_sentences.eq(0), "missing_reason"] = "NO_AI_CONTEXT"
    out.loc[out.eligible_sentences.gt(0) & ~out.threshold_met, "missing_reason"] = "BELOW_MINIMUM"
    out.to_csv(config.TABLE_ROOT / "target_sentence_coverage.csv", index=False)
    out[["arena", "target", "extracted_sentences", "eligible_sentences",
         "phrase_adjusted_sentences", "removed_non_ai"]].to_csv(
             config.AUDIT_ROOT / "sentence_filtering_counts.csv", index=False)
    print(f"Wrote complete {len(out)}-row target/arena coverage table.")


if __name__ == "__main__": main()
