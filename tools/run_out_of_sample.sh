#!/usr/bin/env bash
# THE SINGLE OUT-OF-SAMPLE RUN.
#
# HYPOTHESIS.md: "The OOS period is untouched during development and run
# exactly once at the end."  This script is that one run. It
#   1. downloads the out-of-sample data from Databento (this costs money),
#   2. rebuilds the data tables for the full sample,
#   3. reruns every variant, benchmark and diagnostic,
#   4. reruns versions A and B in backtrader and compares the two engines,
#   5. writes results/REPORT_full.md.
#
# It refuses to run without the flag below, and refuses to run a second time.
#
#     cd gqh-erc-strategy
#     bash tools/run_out_of_sample.sh --i-have-the-go-ahead
#
# Optional settings (environment variables):
#     PYTHON        the Python to use   (default: the starter kit's .venv)
#     STARTER_KIT   the starter kit dir (default: ../gqh-webull-backtrader-starter)

set -euo pipefail                       # stop at the first error

cd "$(dirname "$0")/.."                 # the repo folder
REPO="$(pwd)"
STARTER_KIT="${STARTER_KIT:-$REPO/../gqh-webull-backtrader-starter}"
PYTHON="${PYTHON:-$STARTER_KIT/.venv/bin/python}"
MARKER="$REPO/data/OOS_RUN_DATE.txt"

if [ "${1:-}" != "--i-have-the-go-ahead" ]; then
    echo "This is the one-time out-of-sample run. Read the top of this file first."
    exit 1
fi
if [ -f "$MARKER" ]; then
    echo "The out-of-sample run was already started on: $(cat "$MARKER")"
    echo "It is meant to be run exactly once. Refusing to run again."
    exit 1
fi
if [ -n "$(git status --porcelain)" ]; then
    echo "There are uncommitted changes. Commit first, so the code is frozen before the run."
    exit 1
fi

# Write the marker BEFORE touching any out-of-sample data.
echo "$(date '+%Y-%m-%d %H:%M:%S')  commit $(git rev-parse HEAD)" > "$MARKER"

"$PYTHON" data/download.py --final
"$PYTHON" data/build_processed.py --final
"$PYTHON" data/download_external.py --final
"$PYTHON" run_all.py --final
"$PYTHON" tools/export_backtrader_feeds.py --final

for VERSION in A B; do
    (cd "$STARTER_KIT" && STRATEGY_REPO="$REPO" BT_VERSION="$VERSION" \
        "$PYTHON" examples/backtest/main_futures.py --final)
done

"$PYTHON" tools/compare_engines.py --final
"$PYTHON" tools/make_report.py --final

echo "Done. The report is results/REPORT_full.md. Commit data/OOS_RUN_DATE.txt and data/DATA_FINGERPRINT.txt."
