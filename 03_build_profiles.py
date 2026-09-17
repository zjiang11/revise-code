from __future__ import annotations

import argparse
import json
import pickle
import random
import re
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

import config
from common import chunks, read_text, split_sentences, stable_id, tokens


def filtered_topk(vector, tokenizer, target):
    order = np.argsort(-vector)
    result, rejected = {}, []
    for idx in order:
        token = tokenizer.convert_ids_to_tokens(int(idx))
        normalized = token.lower().strip()
        reason = None
        if token in tokenizer.all_special_tokens: reason = "SPECIAL_TOKEN"
        elif token.startswith("##"): reason = "WORDPIECE_CONTINUATION"
        elif not normalized.isalpha(): reason = "NON_ALPHA"
        elif len(normalized) < 2: reason = "ONE_CHARACTER"
        elif normalized in config.STOPWORDS: reason = "STOPWORD"
        elif normalized == target: reason = "TARGET_ITSELF"
        if reason:
            if len(rejected) < 100:
                rejected.append({"token": token, "probability": float(vector[idx]), "reason": reason})
            continue
        result[normalized] = result.get(normalized, 0.0) + float(vector[idx])
        if len(result) >= config.TOP_K:
            break
    total = sum(result.values())
    return ({k: v / total for k, v in result.items()} if total else {}), rejected


@torch.inference_mode()
def profile(model, tokenizer, texts, device):
    total = np.zeros(len(tokenizer), dtype=np.float64); used = 0
    for batch in chunks(texts, config.BATCH_SIZE):
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=config.MAX_LENGTH,
                            return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        logits = model(**encoded).logits
        masks = encoded["input_ids"].eq(tokenizer.mask_token_id)
        for i in range(len(batch)):
            positions = masks[i].nonzero().flatten()
            if len(positions) != 1:
                continue
            total += torch.softmax(logits[i, positions[0]], -1).cpu().numpy(); used += 1
    return (total / used if used else total), used


def build_control_reservoirs(arena, words, seed):
    """One corpus pass, uniform reservoir of MAX_SENTENCES per control word."""
    wanted = set(words); reservoirs = {w: [] for w in words}; seen = {w: 0 for w in words}
    rngs = {w: random.Random(f"{seed}:{arena}:{w}") for w in words}
    for path in sorted((config.CLEAN_ROOT / arena).glob("*.txt")):
        for ordinal, sentence in enumerate(split_sentences(read_text(path)), 1):
            present = set(tokens(sentence)) & wanted
            for word in present:
                seen[word] += 1
                row = {"sentence_id": stable_id(arena, path.stem, str(ordinal), word),
                       "masked_sentence": re.sub(rf"\b{re.escape(word)}\b", "[MASK]", sentence,
                                                 flags=re.I)}
                bucket = reservoirs[word]
                if len(bucket) < config.MAX_SENTENCES:
                    bucket.append(row)
                else:
                    position = rngs[word].randrange(seen[word])
                    if position < config.MAX_SENTENCES: bucket[position] = row
    return reservoirs, seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="*", type=int, default=list(config.SEEDS))
    parser.add_argument("--targets-only", action="store_true",
                        help="Fast audit run; omit controls (scaled JSD will be unavailable).")
    parser.add_argument("--controls-only", action="store_true",
                        help="Build control profiles only; normally use seed 11 and hold controls fixed.")
    args = parser.parse_args()
    audit = pd.read_csv(config.SENTENCE_ROOT / "sentence_audit.csv.gz")
    if args.targets_only and args.controls_only:
        parser.error("Choose at most one of --targets-only and --controls-only")
    all_words = pd.read_csv(config.WORK_ROOT / "profile_words.csv").word.tolist()
    words = (list(config.TARGETS) if args.targets_only else
             [w for w in all_words if w not in config.TARGETS] if args.controls_only else all_words)
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        model = AutoModelForMaskedLM.from_pretrained(config.MODEL_NAME)
    except Exception as exc:
        failure = config.RESULT_ROOT / "MODEL_RUN_BLOCKED.txt"
        failure.write_text(
            "Masked-LM profile generation did not run. Required model: " + config.MODEL_NAME +
            "\nProvide a reachable Hugging Face endpoint or set MODEL_NAME in config.py "
            "to a local bert-base-uncased snapshot. A sentence-embedding MiniLM is not an "
            "equivalent substitute.\n\nError:\n" + repr(exc), encoding="utf-8")
        raise
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    manifest = []
    for arena in config.ARENAS:
        control_cache = {}
        control_seen = {}
        if any(w not in config.TARGETS for w in words):
            if len(args.seeds) != 1:
                raise ValueError("Control reservoir mode requires exactly one seed; use --seeds 11")
            control_cache, control_seen = build_control_reservoirs(
                arena, [w for w in words if w not in config.TARGETS], args.seeds[0])
        for word in words:
            if word in config.TARGETS:
                subset = audit[(audit.arena == arena) & (audit.target == word) & audit.eligible.astype(bool)]
                candidates = subset[["sentence_id", "masked_sentence"]].to_dict("records")
                extracted = int(((audit.arena == arena) & (audit.target == word)).sum())
            else:
                candidates = control_cache[word]; extracted = control_seen[word]
            threshold = len(candidates) >= config.MIN_ELIGIBLE_SENTENCES
            for seed in args.seeds:
                status = "OK" if threshold else ("NO_AI_CONTEXT" if word in config.TARGETS and extracted else
                                                  "NO_OCCURRENCE" if not extracted else "BELOW_MINIMUM")
                sampled = random.Random(seed).sample(candidates, config.MAX_SENTENCES) \
                    if len(candidates) > config.MAX_SENTENCES else candidates
                used = 0
                if threshold:
                    vector, used = profile(model, tokenizer, [x["masked_sentence"] for x in sampled], device)
                    gsp, rejected = filtered_topk(vector, tokenizer, word)
                    path = config.PROFILE_ROOT / f"seed_{seed}" / f"{arena}__{word}.pkl"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("wb") as handle:
                        pickle.dump({"arena": arena, "word": word, "seed": seed, "GSP": gsp,
                                     "eligible": len(candidates), "sampled": len(sampled), "used": used,
                                     "sentence_ids": [x["sentence_id"] for x in sampled],
                                     "rejected_tokens": rejected}, handle)
                manifest.append({"word": word, "is_target": word in config.TARGETS, "arena": arena,
                                 "extracted_sentences": extracted, "eligible_sentences": len(candidates),
                                 "seed": seed, "sampled_sentences": len(sampled) if threshold else 0,
                                 "minimum_required": config.MIN_ELIGIBLE_SENTENCES,
                                 "threshold_met": threshold, "profile_status": status,
                                 "missing_reason": "" if status == "OK" else status, "model_used": used})
    outfile = ("target_profile_coverage_by_seed.csv" if args.targets_only else
               "control_profile_coverage.csv" if args.controls_only else "profile_coverage_by_seed.csv")
    pd.DataFrame(manifest).to_csv(config.TABLE_ROOT / outfile, index=False)
    print("Profiles and full coverage manifest written.")


if __name__ == "__main__":
    main()
