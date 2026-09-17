from __future__ import annotations

import json
import pandas as pd
import config
from common import write_json


def main():
    checks = []
    def check(name, passed, detail): checks.append({"check": name, "passed": bool(passed), "detail": detail})
    check("academic terminology", config.ARENA_LABELS["arxiv"] == "arXiv (academic)", config.ARENA_LABELS["arxiv"])
    coverage = config.TABLE_ROOT / "target_sentence_coverage.csv"
    check("target coverage table exists", coverage.exists(), str(coverage))
    profiles = config.TABLE_ROOT / "substitution_profiles.csv"
    if profiles.exists():
        df = pd.read_csv(profiles)
        check("reported profiles nonempty", not df.empty,
              f"rows={len(df)}" if not df.empty else
              "Run 03_build_profiles.py after bert-base-uncased is available")
        bad_stop = df.substitute.astype(str).str.lower().isin(config.STOPWORDS)
        bad_piece = df.substitute.astype(str).str.startswith("##")
        check("no stopwords in reported profiles", not bad_stop.any(), f"bad={int(bad_stop.sum())}")
        check("no partial WordPieces", not bad_piece.any(), f"bad={int(bad_piece.sum())}")
        check("only configured targets", set(df.target).issubset(config.TARGETS), str(sorted(set(df.target)-set(config.TARGETS))))
    else: check("reported profiles exist", False, str(profiles))
    controls = config.TABLE_ROOT / "jsd_by_seed.csv"
    if controls.exists():
        df = pd.read_csv(controls)
        check("control counts ordered", bool(((df.n_valid_controls <= df.n_selected_controls) &
                                               (df.n_selected_controls <= df.n_matched_controls)).all()), "valid <= selected <= matched")
    else: check("JSD output exists", False, str(controls))
    write_json(config.RESULT_ROOT / "validation_report.json", checks)
    failed = [x for x in checks if not x["passed"]]
    print(f"Validation: {len(checks)-len(failed)}/{len(checks)} checks passed")
    if failed: print("Incomplete/failed: " + ", ".join(x["check"] for x in failed))


if __name__ == "__main__": main()
