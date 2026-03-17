from __future__ import annotations

import sys
from pathlib import Path

# Allow direct script execution without requiring PYTHONPATH tweaks.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.profiles_creator.create_profile_regression_baselines import (
    CREATE_PROFILE_REGRESSION_CASES,
    compute_processed_profile,
    save_baseline,
)


def main() -> None:
    print('Regenerating create_profile regression baselines...')
    for case in CREATE_PROFILE_REGRESSION_CASES:
        profile = compute_processed_profile(case)
        path = save_baseline(case.case_id, profile)
        print(f'  wrote {path}')
    print('Done.')


if __name__ == '__main__':
    main()