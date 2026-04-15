from __future__ import annotations

from pathlib import Path

import pytest

from spektrafilm.profiles.io import ProfileData, ProfileInfo
from spektrafilm_profile_creator.profile_summary import (
    INFO_FIELD_NAMES,
    RECIPE_FIELD_NAMES,
    SUMMARY_COLUMNS,
    build_profile_summary_rows,
    format_profile_summary_table,
    main,
    parse_csv_columns,
    write_profile_summary_csv,
)
from spektrafilm_profile_creator.raw_profile import RawProfile, RawProfileRecipe
import spektrafilm_profile_creator.profile_summary as profile_summary_module


def _make_raw_profile(
    *,
    stock: str,
    name: str | None = None,
    support: str = 'film',
    profile_type: str = 'negative',
    stage: str = 'filming',
    use: str = 'photo',
    target_print: str | None = None,
    densitometer: str = 'status_M',
    reference_illuminant: str = 'D55',
    viewing_illuminant: str = 'D50',
    log_sensitivity_density_over_min: float = 0.2,
    recipe: RawProfileRecipe | None = None,
) -> RawProfile:
    return RawProfile(
        info=ProfileInfo(
            stock=stock,
            name=name or stock,
            support=support,
            type=profile_type,
            stage=stage,
            use=use,
            target_print=target_print,
            densitometer=densitometer,
            reference_illuminant=reference_illuminant,
            viewing_illuminant=viewing_illuminant,
            log_sensitivity_density_over_min=log_sensitivity_density_over_min,
        ),
        data=ProfileData(),
        recipe=recipe or RawProfileRecipe(),
    )


def test_build_profile_summary_rows_extracts_effective_values_and_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    stock_catalog = {
        'stock_alpha': 'spektrafilm_profile_creator.data.film.negative',
        'stock_beta': 'spektrafilm_profile_creator.data.print.negative',
    }
    manifests = {
        'stock_alpha': {
            'name': 'Stock Alpha',
            'profile': {'stage': 'filming', 'use': 'photo', 'target_print': 'kodak_portra_endura'},
            'donors': {
                'log_sensitivity': 'stock_alpha',
                'density_curves': 'missing_stock',
            },
            'recipe': {'target_film': 'kodak_portra_400'},
        },
        'stock_beta': {
            'name': 'Stock Beta',
            'profile': {'stage': 'printing', 'use': 'cine'},
            'recipe': {'target_film': 'kodak_portra_400'},
        },
    }
    raw_profiles = {
        'stock_alpha': _make_raw_profile(
            stock='stock_alpha',
            name='Stock Alpha',
            target_print='kodak_portra_endura',
            recipe=RawProfileRecipe(
                log_sensitivity_donor='stock_alpha',
                density_curves_donor='missing_stock',
                target_film='kodak_portra_400',
                data_trustability=0.3,
                should_process=False,
            ),
        ),
        'stock_beta': _make_raw_profile(
            stock='stock_beta',
            name='Stock Beta',
            support='paper',
            stage='printing',
            use='cine',
            densitometer='status_A',
            reference_illuminant='D50',
            viewing_illuminant='D55',
            recipe=RawProfileRecipe(target_film='kodak_portra_400'),
        ),
    }

    monkeypatch.setattr(profile_summary_module, 'load_stock_catalog', lambda: stock_catalog)
    monkeypatch.setattr(profile_summary_module, 'load_raw_profile_manifest', lambda stock: manifests[stock])
    monkeypatch.setattr(profile_summary_module, 'load_raw_profile', lambda stock: raw_profiles[stock])
    monkeypatch.setattr(
        profile_summary_module,
        '_available_data_refs',
        lambda data_package: {
            'spektrafilm_profile_creator.data.film.negative': ('stock_alpha',),
            'spektrafilm_profile_creator.data.print.negative': ('stock_beta',),
        }[data_package],
    )

    rows = build_profile_summary_rows()

    assert [row['stock'] for row in rows] == ['stock_alpha', 'stock_beta']

    alpha_row = rows[0]
    assert alpha_row['package'] == 'spektrafilm_profile_creator.data.film.negative'
    assert alpha_row['has_donors_section'] is True
    assert alpha_row['has_recipe_section'] is True
    assert alpha_row['donor_count'] == 2
    assert alpha_row['target_film'] == 'kodak_portra_400'
    assert alpha_row['target_print'] == 'kodak_portra_endura'
    assert alpha_row['data_trustability'] == pytest.approx(0.3)
    assert alpha_row['should_process'] is False
    assert alpha_row['self_referencing_donors'] == ('stock_alpha',)
    assert alpha_row['unknown_donor_refs'] == ('missing_stock',)
    assert alpha_row['issues'] == ('self_donor', 'unknown_donor')

    beta_row = rows[1]
    assert beta_row['support'] == 'paper'
    assert beta_row['has_donors_section'] is False
    assert beta_row['has_recipe_section'] is True
    assert beta_row['donor_count'] == 0
    assert beta_row['target_film'] == 'kodak_portra_400'
    assert beta_row['issues'] == ()


def test_format_profile_summary_table_respects_selected_columns() -> None:
    table = format_profile_summary_table(
        [
            {'stock': 'stock_alpha', 'issues': ('self_donor',), 'target_film': 'kodak_portra_400', 'target_print': 'kodak_portra_endura'},
            {'stock': 'stock_beta', 'issues': (), 'target_film': None, 'target_print': None},
        ],
        columns=('stock', 'target_film', 'target_print', 'issues'),
    )

    lines = table.splitlines()
    assert lines[0].startswith('stock')
    assert 'target_film' in lines[0]
    assert 'target_print' in lines[0]
    assert 'issues' in lines[0]
    assert 'stock_alpha' in lines[2]
    assert 'kodak_portra_400' in lines[2]
    assert 'self_donor' in lines[2]
    assert 'stock_beta' in lines[3]


def test_write_profile_summary_csv_uses_selected_columns(tmp_path: Path) -> None:
    destination = tmp_path / 'profiles.csv'

    written_path = write_profile_summary_csv(
        destination,
        [{'stock': 'stock_alpha', 'target_film': 'kodak_portra_400', 'target_print': 'kodak_portra_endura', 'issues': ('unknown_donor',)}],
        columns=('stock', 'target_film', 'target_print', 'issues'),
    )

    assert written_path == destination
    assert destination.read_text(encoding='utf-8').splitlines() == [
        'stock,target_film,target_print,issues',
        'stock_alpha,kodak_portra_400,kodak_portra_endura,unknown_donor',
    ]


def test_parse_csv_columns_keeps_column_order() -> None:
    assert parse_csv_columns('stock, target_print , issues') == ('stock', 'target_print', 'issues')


def test_main_prints_table_and_writes_csv(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    destination = tmp_path / 'profile-summary.csv'
    rows = [
        {
            'stock': 'stock_alpha',
            'name': 'Stock Alpha',
            'package': 'fake.package',
            'support': 'film',
            'type': 'negative',
            'stage': 'filming',
            'use': 'photo',
            'channel_model': 'color',
            'densitometer': 'status_M',
            'log_sensitivity_density_over_min': 0.2,
            'reference_illuminant': 'D55',
            'viewing_illuminant': 'D50',
            'has_donors_section': True,
            'has_recipe_section': True,
            'log_sensitivity_donor': 'stock_beta',
            'density_curves_donor': None,
            'dye_density_cmy_donor': None,
            'dye_density_min_mid_donor': None,
            'donor_count': 1,
            'target_film': 'kodak_portra_400',
            'target_print': 'kodak_portra_endura',
            'data_trustability': 1.0,
            'stretch_curves': False,
            'should_process': True,
            'self_referencing_donors': (),
            'unknown_donor_refs': (),
            'issues': (),
        }
    ]

    monkeypatch.setattr(profile_summary_module, 'build_profile_summary_rows', lambda stocks=None: rows)

    exit_code = main(['--columns', 'stock,target_film,target_print,issues', '--csv', str(destination)])

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert 'stock_alpha' in stdout
    assert 'CSV written to' in stdout

    csv_lines = destination.read_text(encoding='utf-8').splitlines()
    assert csv_lines[0] == 'stock,target_film,target_print,issues'
    assert csv_lines[1] == 'stock_alpha,kodak_portra_400,kodak_portra_endura,'


def test_summary_columns_are_unique() -> None:
    assert len(SUMMARY_COLUMNS) == len(set(SUMMARY_COLUMNS))


def test_summary_columns_cover_info_and_recipe_dataclasses() -> None:
    assert set(INFO_FIELD_NAMES).issubset(SUMMARY_COLUMNS)
    assert set(RECIPE_FIELD_NAMES).issubset(SUMMARY_COLUMNS)