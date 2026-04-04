from __future__ import annotations

import sys

from spektrafilm_profile_creator.profile_summary import main


DEFAULT_CSV_PATH = 'profile_summary.csv'


def _build_argv_with_default_csv(argv: list[str]) -> list[str]:
    if '--csv' in argv or any(argument.startswith('--csv=') for argument in argv):
        return argv
    return [*argv, '--csv', DEFAULT_CSV_PATH]


if __name__ == '__main__':
    raise SystemExit(main(_build_argv_with_default_csv(sys.argv[1:])))