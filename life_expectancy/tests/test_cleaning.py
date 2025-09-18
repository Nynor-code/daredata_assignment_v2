"""Tests for the cleaning module (loading, cleaning, saving, CLI)."""

from __future__ import annotations

# Standard library
import argparse
from pathlib import Path
from unittest.mock import patch

# Third-party
import pandas as pd
import pytest

# Local
import life_expectancy.cleaning as cln
from life_expectancy.enums import Region
from life_expectancy.cleaning import (
    _map_geo_to_enum_value,
    clean_data,
    load_data,
    main,
    save_data,
)

# Paths
DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "life_expectancy"
    / "tests"
    / "fixtures"
)


# =============================================================================
# Core cleaning behavior
# =============================================================================


def test_clean_data(
    raw_life_expectancy_sample: pd.DataFrame,
    pt_life_expectancy_expected: pd.DataFrame,
) -> None:
    """Clean the sample and compare against the expected PT frame."""
    cleaned = clean_data(
        raw_life_expectancy_sample, export_country_code=Region.PT
    )
    pd.testing.assert_frame_equal(
        cleaned.reset_index(drop=True),
        pt_life_expectancy_expected.reset_index(drop=True),
    )


def test_clean_data_country_alias_param() -> None:
    """Resolve Region[country] via the 'country' alias."""
    df = pd.DataFrame(
        {"unit,sex,age,geo\\time": ["Y,M,Y_LT1,PT"], "2019": ["78.6"]}
    )
    out = clean_data(df, country="PT")
    assert set(out.columns) == {"unit", "sex", "age", "region", "year", "value"}
    assert (out["region"] == "PT").all()
    assert out.loc[0, "year"] == 2019
    assert out.loc[0, "value"] == 78.6


def test_clean_data_country_alias_invalid_raises() -> None:
    """Invalid 'country' alias should raise a KeyError on Region[country]."""
    df = pd.DataFrame(
        {"unit,sex,age,geo\\time": ["Y,M,Y_LT1,PT"], "2019": ["78.6"]}
    )
    with pytest.raises(KeyError):
        clean_data(df, country="NOPE")


def test_clean_data_from_normalized_input_to_region_geo() -> None:
    """Handle normalized DF (adapter output) and remap to region/year."""
    df_norm = pd.DataFrame(
        {
            "unit": ["Y"],
            "sex": ["M"],
            "age": ["Y_LT1"],
            "geo": ["PT"],
            "time": ["2019"],
            "value": [78.6],
        }
    )
    out = clean_data(df_norm, country="PT")
    assert set(out.columns) == {"unit", "sex", "age", "geo", "year", "value"}
    assert out.loc[0, "geo"] == "PT"
    assert out.loc[0, "year"] == 2019
    assert out.loc[0, "value"] == 78.6


def test_clean_data_from_normalized_multirow() -> None:
    """Normalized DF with multiple rows, filtered to one country."""
    df_norm = pd.DataFrame(
        {
            "unit": ["Y", "Y"],
            "sex": ["M", "F"],
            "age": ["Y_LT1", "Y_LT1"],
            "geo": ["PT", "ES"],
            "time": ["2019", "2019"],
            "value": [78.6, 84.3],
        }
    )
    out = clean_data(df_norm, country="PT")
    assert set(out.columns) == {"unit", "sex", "age", "geo", "year", "value"}
    assert (out["geo"] == "PT").all()
    assert (out["year"] == 2019).all()


def test_clean_data_step2_fallback_case_insensitive() -> None:
    """
    Step 2 fallback: the combined column exists but with different casing,
    so 'from_raw' is False and the fallback must find it and melt.
    """
    df = pd.DataFrame(
        {
            "UNIT,SEX,AGE,GEO\\TIME": ["Y,M,Y_LT1,PT", "Y,F,Y_LT1,ES"],
            "2019": ["78.6", ":"],
        }
    )
    out = clean_data(df, country="PT")
    assert set(out.columns) == {"unit", "sex", "age", "region", "year", "value"}
    assert len(out) == 1
    assert out.loc[out.index[0], "region"] == "PT"
    assert out.loc[out.index[0], "year"] == 2019
    assert out.loc[out.index[0], "value"] == 78.6


def test__map_geo_to_enum_value_none_enum() -> None:
    """Guard clause: return series as-is when enum_cls is None."""
    s = pd.Series(["PT", "ES", "XX"])
    out = _map_geo_to_enum_value(s, None)  # type: ignore[arg-type]
    pd.testing.assert_series_equal(out, s)


# =============================================================================
# IO: load/save helpers
# =============================================================================


def test_load_data() -> None:
    """load_data(path) returns a non-empty frame with normalized columns."""
    input_path = DATA_DIR / "eu_life_expectancy_raw.tsv"
    df = load_data(input_path)
    assert not df.empty
    expected_cols = {"unit", "sex", "age", "geo", "time", "value"}
    assert expected_cols.issubset(df.columns)


def test_load_data_fallback_to_data_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Non-existent absolute path: load_data should fall back to package data/<filename>.
    """
    fake_pkg_dir = tmp_path / "life_expectancy"
    data_dir = fake_pkg_dir / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "tiny.tsv").write_text(
        "unit,sex,age,geo\\time\t2019\nY,M,Y_LT1,PT\t78.6\n",
        encoding="utf-8",
    )

    # Point cleaning.py base_dir to fake package dir
    monkeypatch.setattr(cln, "__file__", str(fake_pkg_dir / "cleaning.py"))

    df = cln.load_data("/does/not/exist/tiny.tsv")
    assert set(df.columns) == {"unit", "sex", "age", "geo", "time", "value"}
    assert len(df) == 1
    assert df.loc[0, "geo"] == "PT"
    assert df.loc[0, "time"] == "2019"
    assert df.loc[0, "value"] == 78.6


def test_load_data_default_no_args(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Calling load_data() with no args should read <pkg>/data/eu_life_expectancy_raw.tsv."""
    fake_pkg_dir = tmp_path / "life_expectancy"
    data_dir = fake_pkg_dir / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "eu_life_expectancy_raw.tsv").write_text(
        "unit,sex,age,geo\\time\t2019\nY,M,Y_LT1,PT\t78.6\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(cln, "__file__", str(fake_pkg_dir / "cleaning.py"))

    df = cln.load_data()
    # Depending on implementation this may be raw or normalized; accept either.
    assert (
        "unit,sex,age,geo\\time" in df.columns
        or {"unit", "sex", "age", "geo", "time", "value"} <= set(df.columns)
    )
    assert len(df) == 1


@patch("pandas.DataFrame.to_csv")
def test_save_data_mocked(
    mock_to_csv: object, pt_life_expectancy_expected: pd.DataFrame
) -> None:
    """Ensure save_data calls to_csv without writing to disk."""
    save_data(pt_life_expectancy_expected, export_country_code=Region.PT)
    mock_to_csv.assert_called_once()
    _, kwargs = mock_to_csv.call_args
    assert kwargs.get("index") is False


# =============================================================================
# CLI orchestration (main)
# =============================================================================


@patch("pandas.DataFrame.to_csv")
def test_main_with_other_country(
    mock_to_csv: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run main() with a different country and mock saving."""
    monkeypatch.setattr(
        "argparse.ArgumentParser.parse_args",
        lambda self: argparse.Namespace(country="DE"),
    )
    df = main()
    assert not df.empty
    mock_to_csv.assert_called_once()


def test_main_invalid_country(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid country code should raise a user-friendly ValueError."""
    monkeypatch.setattr(
        "argparse.ArgumentParser.parse_args",
        lambda self: argparse.Namespace(country="XX"),
    )

    with pytest.raises(ValueError) as exc_info:
        main()

    msg = str(exc_info.value)
    assert "Invalid country code" in msg
    assert "Valid country codes are" in msg
    assert "XX" in msg
