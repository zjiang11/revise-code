from __future__ import annotations

from collections import Counter
from itertools import combinations
import random
import pandas as pd

import config
from common import read_text, tokens


def main():
    counts = {}
    frequency_rows = []
    for arena in config.ARENAS:
        counter = Counter()
        for path in sorted((config.CLEAN_ROOT / arena).glob("*.txt")):
            counter.update(tokens(read_text(path)))
        counts[arena] = counter
        frequency_rows.extend({"arena": arena, "word": word, "frequency": n}
                              for word, n in counter.items())
    pd.DataFrame(frequency_rows).to_csv(config.TABLE_ROOT / "full_frequencies.csv.gz",
                                         index=False, compression="gzip")

    targets = set(config.TARGETS)
    vocabulary = sorted(set.intersection(*(set(counts[a]) for a in config.ARENAS)) - targets)
    rows, required = [], set(targets)
    rng = random.Random(config.SEEDS[0])
    for target in config.TARGETS:
        for a, b in combinations(config.ARENAS, 2):
            fa, fb = counts[a][target], counts[b][target]
            matched = [word for word in vocabulary
                       if counts[a][word] and counts[b][word]
                       and abs(counts[a][word] - fa) / max(fa, 1) <= config.CONTROL_REL_TOLERANCE
                       and abs(counts[b][word] - fb) / max(fb, 1) <= config.CONTROL_REL_TOLERANCE]
            selected = rng.sample(matched, config.MAX_CONTROLS) if len(matched) > config.MAX_CONTROLS else matched
            required.update(selected)
            rows.append({"target": target, "arena_a": a, "arena_b": b,
                         "target_freq_a": fa, "target_freq_b": fb,
                         "n_matched_controls": len(matched), "n_selected_controls": len(selected),
                         "selected_controls": "; ".join(selected),
                         "pre_profile_flag": "VERY_SMALL" if len(selected) < 10 else
                                             "SMALL" if len(selected) < config.SMALL_CONTROL_WARN else
                                             "LIMITED" if len(selected) < config.MAX_CONTROLS else "FULL"})
    pd.DataFrame(rows).to_csv(config.TABLE_ROOT / "control_sets.csv", index=False)
    pd.DataFrame({"word": sorted(required), "is_target": [w in targets for w in sorted(required)]}) \
      .to_csv(config.WORK_ROOT / "profile_words.csv", index=False)
    print(f"Controls written; profiles required for {len(required)} words.")


if __name__ == "__main__":
    main()
