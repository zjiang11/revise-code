$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $Here
try {
    py -3 00_clean_corpus.py
    py -3 01_extract_and_filter_sentences.py
    py -3 01b_build_coverage.py
    py -3 02_build_controls.py
    py -3 03_build_profiles.py --targets-only
    py -3 03_build_profiles.py --controls-only --seeds 11
    py -3 04_analyze.py
    py -3 05_generate_outputs.py
    py -3 06_validate_outputs.py
} finally {
    Pop-Location
}
