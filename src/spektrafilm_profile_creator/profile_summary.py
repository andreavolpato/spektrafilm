from __future__ import annotations

import argparse
import csv
import importlib.resources as pkg_resources
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import fields
from functools import lru_cache
from pathlib import Path

from spektrafilm.profiles.io import ProfileInfo
from spektrafilm_profile_creator.data.loader import load_raw_profile, load_raw_profile_manifest, load_stock_catalog
from spektrafilm_profile_creator.raw_profile import RawProfileRecipe


DONOR_FIELD_NAMES = (
    'log_sensitivity_donor',
    'density_curves_donor',
    'dye_density_cmy_donor',
    'dye_density_min_mid_donor',
)

INFO_FIELD_NAMES = tuple(field.name for field in fields(ProfileInfo))
RECIPE_FIELD_NAMES = tuple(field.name for field in fields(RawProfileRecipe))
DERIVED_FIELD_NAMES = (
    'package',
    'has_donors_section',
    'has_recipe_section',
    'donor_count',
    'self_referencing_donors',
    'unknown_donor_refs',
    'issues',
)

SUMMARY_COLUMNS = (
    'package',
    'has_donors_section',
    'has_recipe_section',
    *INFO_FIELD_NAMES,
    *RECIPE_FIELD_NAMES,
    'donor_count',
    'self_referencing_donors',
    'unknown_donor_refs',
    'issues',
)

DEFAULT_COLUMNS = (
    'stock',
    'support',
    'type',
    'stage',
    'use',
    'densitometer',
    'reference_illuminant',
    'viewing_illuminant',
    'target_film',
    'target_print',
    'data_trustability',
    'donor_count',
    'issues',
)

DEFAULT_SORT_COLUMNS = ('support', 'type', 'stage', 'use', 'stock')


def _dataclass_values(instance) -> dict[str, object]:
    return {field.name: getattr(instance, field.name) for field in fields(instance)}


def _mapping(payload: object) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        return {}
    return dict(payload)


def _normalize_columns(columns: Sequence[str] | None) -> tuple[str, ...]:
    if columns is None:
        return DEFAULT_COLUMNS

    normalized = []
    for column in columns:
        name = column.strip()
        if not name:
            continue
        if name not in SUMMARY_COLUMNS:
            raise ValueError(f'Unsupported summary column: {name!r}')
        normalized.append(name)

    if not normalized:
        raise ValueError('At least one summary column must be selected')
    return tuple(normalized)


def _sort_key(row: Mapping[str, object], sort_columns: Sequence[str]) -> tuple[tuple[int, object], ...]:
    key = []
    for column in sort_columns:
        value = row.get(column)
        if value is None:
            key.append((1, ''))
        elif isinstance(value, bool):
            key.append((0, int(value)))
        elif isinstance(value, (int, float)):
            key.append((0, value))
        else:
            key.append((0, str(value)))
    return tuple(key)


def _compact_bool(value: bool) -> str:
    return 'yes' if value else ''


def _stringify(value: object) -> str:
    if value is None:
        return ''
    if isinstance(value, bool):
        return _compact_bool(value)
    if isinstance(value, float):
        return f'{value:.6g}'
    if isinstance(value, (list, tuple, set)):
        return ', '.join(_stringify(item) for item in value if item not in (None, ''))
    return str(value)


@lru_cache(maxsize=None)
def _available_data_refs(data_package: str) -> tuple[str, ...]:
    package_root = pkg_resources.files(data_package)
    return tuple(sorted(entry.name for entry in package_root.iterdir() if entry.is_dir()))


def build_profile_summary_rows(stocks: Sequence[str] | None = None) -> list[dict[str, object]]:
    catalog = load_stock_catalog()
    selected_stocks = tuple(stocks) if stocks else tuple(catalog)

    rows: list[dict[str, object]] = []
    for stock in selected_stocks:
        data_package = catalog.get(stock)
        if data_package is None:
            raise FileNotFoundError(f'No raw profile manifest found for stock {stock!r}')

        manifest = load_raw_profile_manifest(stock)
        raw_profile = load_raw_profile(stock)
        donors_payload = _mapping(manifest.get('donors'))
        available_data_refs = set(_available_data_refs(data_package))
        info_values = _dataclass_values(raw_profile.info)
        recipe_values = _dataclass_values(raw_profile.recipe)

        donor_values = {name: recipe_values.get(name) for name in DONOR_FIELD_NAMES}
        donor_targets = tuple(value for value in donor_values.values() if value)
        self_referencing_donors = tuple(sorted({value for value in donor_targets if value == stock}))
        unknown_donor_refs = tuple(sorted({value for value in donor_targets if value not in available_data_refs}))

        issues = []
        if self_referencing_donors:
            issues.append('self_donor')
        if unknown_donor_refs:
            issues.append('unknown_donor')

        rows.append(
            {
                'package': data_package,
                'has_donors_section': 'donors' in manifest,
                'has_recipe_section': 'recipe' in manifest,
                **info_values,
                **recipe_values,
                'donor_count': len(donors_payload),
                'self_referencing_donors': self_referencing_donors,
                'unknown_donor_refs': unknown_donor_refs,
                'issues': tuple(issues),
            }
        )

    return sorted(rows, key=lambda row: _sort_key(row, DEFAULT_SORT_COLUMNS))


def format_profile_summary_table(
    rows: Sequence[Mapping[str, object]],
    *,
    columns: Sequence[str] | None = None,
) -> str:
    selected_columns = _normalize_columns(columns)
    if not rows:
        return 'No profile manifests found.'

    rendered_rows = []
    for row in rows:
        rendered_row = {column: _stringify(row.get(column)) for column in selected_columns}
        rendered_rows.append(rendered_row)

    widths = {
        column: max(len(column), *(len(rendered_row[column]) for rendered_row in rendered_rows))
        for column in selected_columns
    }

    header = '  '.join(f'{column:<{widths[column]}}' for column in selected_columns)
    separator = '  '.join('-' * widths[column] for column in selected_columns)
    lines = [header, separator]
    for rendered_row in rendered_rows:
        lines.append('  '.join(f'{rendered_row[column]:<{widths[column]}}' for column in selected_columns))
    return '\n'.join(lines)


def write_profile_summary_csv(
    destination: str | Path,
    rows: Sequence[Mapping[str, object]],
    *,
    columns: Sequence[str] | None = None,
) -> Path:
    selected_columns = _normalize_columns(columns or SUMMARY_COLUMNS)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    with destination_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=selected_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _stringify(row.get(column)) for column in selected_columns})
    return destination_path


def available_summary_columns() -> tuple[str, ...]:
    return SUMMARY_COLUMNS


def parse_csv_columns(raw_value: str | None) -> tuple[str, ...] | None:
    if raw_value is None:
        return None
    return tuple(part.strip() for part in raw_value.split(','))


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Inspect raw profile manifests as a compact parameter matrix.')
    parser.add_argument('stocks', nargs='*', help='Optional stock names to inspect. Defaults to all profile.yaml manifests.')
    parser.add_argument(
        '--columns',
        help='Comma-separated columns for terminal output, e.g. stock,target_print,data_trustability.',
    )
    parser.add_argument(
        '--sort-by',
        default=','.join(DEFAULT_SORT_COLUMNS),
        help='Comma-separated sort columns. Defaults to support,type,stage,use,stock.',
    )
    parser.add_argument(
        '--csv',
        help='Optional path to export the summary rows as CSV. Exports all columns unless --columns is provided.',
    )
    parser.add_argument(
        '--show-columns',
        action='store_true',
        help='List available summary columns and exit.',
    )
    parser.add_argument(
        '--all-columns',
        action='store_true',
        help='Show every available column in the terminal table.',
    )
    return parser


def _validate_sort_columns(columns: Iterable[str]) -> tuple[str, ...]:
    normalized = _normalize_columns(tuple(columns))
    return normalized


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    if args.show_columns:
        print('\n'.join(available_summary_columns()))
        return 0

    display_columns = SUMMARY_COLUMNS if args.all_columns else parse_csv_columns(args.columns)
    sort_columns = _validate_sort_columns(parse_csv_columns(args.sort_by) or DEFAULT_SORT_COLUMNS)
    rows = build_profile_summary_rows(args.stocks or None)
    rows = sorted(rows, key=lambda row: _sort_key(row, sort_columns))

    print(format_profile_summary_table(rows, columns=display_columns))

    if args.csv:
        csv_columns = display_columns if args.columns else SUMMARY_COLUMNS
        destination = write_profile_summary_csv(args.csv, rows, columns=csv_columns)
        print(f'\nCSV written to {destination}')

    return 0


__all__ = [
    'DEFAULT_COLUMNS',
    'DEFAULT_SORT_COLUMNS',
    'SUMMARY_COLUMNS',
    'available_summary_columns',
    'build_profile_summary_rows',
    'format_profile_summary_table',
    'main',
    'parse_csv_columns',
    'write_profile_summary_csv',
]