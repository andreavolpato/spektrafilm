from __future__ import annotations

import sys
from pathlib import Path

# Allow direct script execution without requiring PYTHONPATH tweaks.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.regression_baselines import REGRESSION_CASES, compute_case_output, save_baseline


def main() -> None:
    print("Regenerating deterministic regression baselines...")
    for case in REGRESSION_CASES:
        output = compute_case_output(case)
        path = save_baseline(case.case_id, output)
        print(f"  wrote {path}")
    print("Done.")


if __name__ == "__main__":
    main()
