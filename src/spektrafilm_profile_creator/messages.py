from __future__ import annotations

import contextlib
import io
import json

import numpy as np


PREFIX = '[profile_creator]'


def _format_value(value) -> str:
    if isinstance(value, dict):
        try:
            return json.dumps(value, indent=2, sort_keys=True, default=str)
        except TypeError:
            return str(value)
    if isinstance(value, np.ndarray):
        return np.array2string(value, precision=4, suppress_small=True, max_line_width=100)
    if isinstance(value, (list, tuple)):
        return np.array2string(np.asarray(value), precision=4, suppress_small=True, max_line_width=100)
    if isinstance(value, (np.floating, float)):
        return f'{float(value):.6g}'
    return str(value)


def log_event(title: str, **fields) -> None:
    print(f'{PREFIX} {title}')
    if not fields:
        return

    label_width = max(len(str(label)) for label in fields)
    for label, value in fields.items():
        formatted = _format_value(value)
        field_prefix = f'  {label:<{label_width}} : '
        if '\n' not in formatted:
            print(f'{field_prefix}{formatted}')
            continue
        print(field_prefix.rstrip())
        continuation_prefix = f'  {"":<{label_width}} : '
        for line in formatted.splitlines():
            print(f'{continuation_prefix}{line}')


def log_parameters(title: str, params) -> None:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        params.pretty_print()
    print(f'{PREFIX} {title}')
    for line in buffer.getvalue().splitlines():
        if line.strip():
            print(f'  {line}')