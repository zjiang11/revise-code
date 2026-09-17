from __future__ import annotations

import re
import csv
import pandas as pd

import config
from common import read_text, split_sentences, stable_id, tokens


def context_window(sentence: str, word: str) -> str:
    ts = tokens(sentence)
    positions = [i for i, token in enumerate(ts) if token == word]
    if not positions:
        return sentence.lower()
    i = positions[0]
    lo, hi = max(0, i - config.AI_WINDOW_TOKENS), i + config.AI_WINDOW_TOKENS + 1
    return " ".join(ts[lo:hi])


def cue_matches(window: str, target: str) -> list[str]:
    matches = []
    for cue in sorted(config.AI_CUES):
        # Do not allow a target to prove its own AI relevance.
        if cue == target:
            continue
        if re.search(rf"(?<![A-Za-z]){re.escape(cue)}(?![A-Za-z])", window, re.I):
            matches.append(cue)
    return matches


def non_ai_match(sentence: str, target: str) -> str:
    for pattern in config.NON_AI_PATTERNS.get(target, ()):
        found = re.search(pattern, sentence, re.I)
        if found:
            return found.group(0)
    return ""


def mask_text(sentence: str, target: str) -> tuple[str, bool, str]:
    for pattern in config.MASK_EXPANSIONS.get(target, ()):
        found = re.search(pattern, sentence, re.I)
        if found:
            return sentence[:found.start()] + "[MASK]" + sentence[found.end():], True, found.group(0)
    return re.sub(rf"\b{re.escape(target)}\b", "[MASK]", sentence, flags=re.I), False, ""


def classify(sentence: str, target: str):
    window = context_window(sentence, target)
    cues = cue_matches(window, target)
    veto = non_ai_match(sentence, target)
    # Explicit non-AI senses are excluded even if a broad cue such as "computer"
    # appears elsewhere. Otherwise at least one independent AI cue is required.
    if veto:
        return False, "EXPLICIT_NON_AI_SENSE", cues, veto
    if not cues:
        return False, "NO_AI_CONTEXT_CUE", [], ""
    return True, "ELIGIBLE_AI_CONTEXT", cues, ""


def main():
    rows = []
    for arena in config.ARENAS:
        print(f"[sentences] {arena}")
        for path in sorted((config.CLEAN_ROOT / arena).glob("*.txt")):
            for ordinal, sentence in enumerate(split_sentences(read_text(path)), 1):
                lower = sentence.lower()
                for target in config.TARGETS:
                    if not re.search(rf"\b{re.escape(target)}\b", lower):
                        continue
                    eligible, reason, cues, veto = classify(sentence, target)
                    masked, adjusted, phrase = mask_text(sentence, target)
                    rows.append({
                        "sentence_id": stable_id(arena, path.stem, str(ordinal), target),
                        "arena": arena, "arena_label": config.ARENA_LABELS[arena],
                        "document_id": path.stem, "sentence_ordinal": ordinal,
                        "target": target, "sentence": sentence, "masked_sentence": masked,
                        "eligible": eligible, "decision_reason": reason,
                        "ai_cues": "; ".join(cues), "non_ai_match": veto,
                        "phrase_adjusted": adjusted, "adjusted_phrase": phrase,
                    })
    df = pd.DataFrame(rows)
    all_path = config.SENTENCE_ROOT / "sentence_audit.csv.gz"
    # Some source exports contain malformed quote/control sequences. Quoting all
    # fields plus an explicit escape character keeps the audit lossless.
    df.to_csv(all_path, index=False, compression="gzip", quoting=csv.QUOTE_ALL,
              escapechar="\\")

    summary = (df.groupby(["arena", "target", "decision_reason"], dropna=False)
                 .size().rename("sentences").reset_index())
    summary.to_csv(config.AUDIT_ROOT / "filtering_summary_by_reason.csv", index=False)
    adjusted = (df.groupby(["arena", "target"])
                  .agg(extracted_sentences=("sentence_id", "size"),
                       eligible_sentences=("eligible", "sum"),
                       phrase_adjusted_sentences=("phrase_adjusted", "sum"))
                  .reset_index())
    adjusted["removed_non_ai"] = adjusted.extracted_sentences - adjusted.eligible_sentences
    adjusted.to_csv(config.AUDIT_ROOT / "sentence_filtering_counts.csv", index=False)
    # Complete target/arena grid, including zero rows and explicit reasons. This
    # exists before model inference so corpus eligibility remains auditable even
    # if model acquisition or GPU execution is interrupted.
    grid = pd.MultiIndex.from_product([config.TARGETS, config.ARENAS],
                                      names=["target", "arena"]).to_frame(index=False)
    coverage = grid.merge(adjusted, on=["target", "arena"], how="left").fillna(0)
    coverage["minimum_required"] = config.MIN_ELIGIBLE_SENTENCES
    coverage["sample_cap"] = config.MAX_SENTENCES
    coverage["sampled_if_run"] = coverage.eligible_sentences.clip(upper=config.MAX_SENTENCES).astype(int)
    coverage["threshold_met"] = coverage.eligible_sentences.ge(config.MIN_ELIGIBLE_SENTENCES)
    coverage["missing_reason"] = ""
    coverage.loc[coverage.extracted_sentences.eq(0), "missing_reason"] = "NO_OCCURRENCE"
    coverage.loc[coverage.extracted_sentences.gt(0) & coverage.eligible_sentences.eq(0),
                 "missing_reason"] = "NO_AI_CONTEXT"
    coverage.loc[coverage.eligible_sentences.gt(0) & ~coverage.threshold_met,
                 "missing_reason"] = "BELOW_MINIMUM"
    coverage.to_csv(config.TABLE_ROOT / "target_sentence_coverage.csv", index=False)
    intel = df[df.target.eq("intelligence")]
    intel.to_csv(config.AUDIT_ROOT / "intelligence_sentence_audit.csv", index=False,
                 quoting=csv.QUOTE_ALL, escapechar="\\")
    print(f"Wrote {all_path}: {len(df)} target-bearing sentences; {int(df.eligible.sum())} eligible.")


if __name__ == "__main__":
    main()
