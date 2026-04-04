from __future__ import annotations

import contextlib
import copy
import io
import json

import numpy as np


_PREFIX_LABEL = 'profile_creator'
PREFIX = f'[{_PREFIX_LABEL}]'
_diagnostic_profile_snapshots: dict[str, list[dict[str, object]]] = {}
_diagnostic_snapshot_state = {'sequence': 0}


def _profile_stock(profile) -> str | None:
    info = getattr(profile, 'info', None)
    stock = getattr(info, 'stock', None)
    return str(stock) if stock else None


def _message_prefix(stock: str | None = None) -> str:
    if stock:
        return f'[{_PREFIX_LABEL} / {stock}]'
    return PREFIX


def _resolve_message_stock(source=None, fields=None) -> str | None:
    if fields:
        stock = fields.get('stock')
        if stock:
            return str(stock)

    stock = _profile_stock(source)
    if stock:
        return stock

    if source is None:
        return None

    for attr in ('film', 'profile'):
        stock = _profile_stock(getattr(source, attr, None))
        if stock:
            return stock
    return None


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


def _render_message_lines(title: str, fields, *, stock: str | None = None) -> list[str]:
    lines = [f'{_message_prefix(stock)} {title}']
    rendered_fields = fields
    if stock and fields.get('stock') == stock:
        rendered_fields = {label: value for label, value in fields.items() if label != 'stock'}

    if not rendered_fields:
        return lines

    label_width = max(len(str(label)) for label in rendered_fields)
    for label, value in rendered_fields.items():
        formatted = _format_value(value)
        field_prefix = f'  {label:<{label_width}} : '
        if '\n' not in formatted:
            lines.append(f'{field_prefix}{formatted}')
            continue
        lines.append(field_prefix.rstrip())
        continuation_prefix = f'  {"":<{label_width}} : '
        for line in formatted.splitlines():
            lines.append(f'{continuation_prefix}{line}')
    return lines


def _store_profile_snapshot(title: str, snapshot, output: str) -> None:
    _diagnostic_snapshot_state['sequence'] += 1
    _diagnostic_profile_snapshots.setdefault(title, []).append(
        {
            'sequence': _diagnostic_snapshot_state['sequence'],
            'stock': _profile_stock(snapshot),
            'profile': copy.deepcopy(snapshot),
            'output': output,
        }
    )


def get_diagnostic_profile_snapshots() -> dict[str, list[dict[str, object]]]:
    return copy.deepcopy(_diagnostic_profile_snapshots)


def clear_diagnostic_profile_snapshots() -> None:
    _diagnostic_profile_snapshots.clear()
    _diagnostic_snapshot_state['sequence'] = 0


def log_event(title: str, snapshot=None, /, **fields) -> None:
    stock = _resolve_message_stock(snapshot, fields)
    lines = _render_message_lines(title, fields, stock=stock)
    output = '\n'.join(lines)
    if snapshot is not None:
        _store_profile_snapshot(title, snapshot, output)
    print(output)


def log_parameters(title: str, params) -> None:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        params.pretty_print()
    lines = [f'{_message_prefix(_resolve_message_stock(params))} {title}']
    for line in buffer.getvalue().splitlines():
        if line.strip():
            lines.append(f'  {line}')
    print('\n'.join(lines))


__all__ = [
    'PREFIX',
    'clear_diagnostic_profile_snapshots',
    'get_diagnostic_profile_snapshots',
    'log_event',
    'log_parameters',
]