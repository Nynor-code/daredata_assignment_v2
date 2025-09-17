"""Tests for the cleaning module"""
# Standard imports
import argparse
from pathlib import Path
from unittest.mock import patch
import pytest

# Third-party imports
import pandas as pd

# Local imports
from life_expectancy.enums import Region
from life_expectancy.cleaning import (
    clean_data,
    load_data,
    save_data,
    main
)

DATA_DIR = Path(__file__).resolve().parent.parent / "life_expectancy" / "tests" / "fixtures"


def test_clean_data(raw_life_expectancy_sample, pt_life_expectancy_expected):
    """
    Run the `clean_data` function and compare the output to the expected output
    """
    cleaned = clean_data(raw_life_expectancy_sample,
                         export_country_code=Region.PT
                         )
    pd.testing.assert_frame_equal(
        cleaned.reset_index(drop=True),
        pt_life_expectancy_expected.reset_index(drop=True)
    )


@patch("pandas.DataFrame.to_csv")
def test_save_data_mocked(mock_to_csv, pt_life_expectancy_expected):
    """
    Ensure save_data calls to_csv without writing to disk
    """
    save_data(pt_life_expectancy_expected, export_country_code=Region.PT)
    mock_to_csv.assert_called_once()
    _, kwargs = mock_to_csv.call_args
    assert kwargs.get("index") is False


@patch("pandas.DataFrame.to_csv")
def test_main_with_other_country(mock_to_csv, monkeypatch):
    """
    Test main() with mocked saving
    """
    monkeypatch.setattr(
        "argparse.ArgumentParser.parse_args",
        lambda self: argparse.Namespace(country="DE")
    )
    df = main()
    assert not df.empty
    mock_to_csv.assert_called_once()


def test_load_data():
    """
    Test that load_data returns a non-empty DataFrame with expected columns
    """
    input_path = DATA_DIR / "eu_life_expectancy_raw.tsv"
    df = load_data(input_path)
    assert not df.empty
    # assert "unit,sex,age,geo\\time" in df.columns
    expected_cols = {"unit", "sex", "age", "geo", "time", "value"}
    assert expected_cols.issubset(df.columns)

def test_main_invalid_country(monkeypatch):
    """
    Test that an invalid country code raises a ValueError with a helpful message.
    """
    monkeypatch.setattr(
        "argparse.ArgumentParser.parse_args",
        lambda self: argparse.Namespace(country="XX")  # Invalid country code
    )

    with pytest.raises(ValueError) as exc_info:
        main()

    assert "Invalid country code" in str(exc_info.value)
    assert "Valid country codes are" in str(exc_info.value)
    assert "XX" in str(exc_info.value)

def test_load_data_fallback_to_data_folder(monkeypatch, tmp_path):
    """
    Covers cleaning.py lines 25–26: when a non-existent path is given,
    load_data falls back to package data/<filename>.
    We simulate by passing just a filename that exists in tests' tmp dir,
    and monkeypatch the module's base_dir to that tmp for isolation.
    """
    # Prepare a tiny TSV in a fake data dir
    fake_pkg_dir = tmp_path / "life_expectancy"
    data_dir = fake_pkg_dir / "data"
    data_dir.mkdir(parents=True)
    tsv = data_dir / "tiny.tsv"
    tsv.write_text("unit,sex,age,geo\\time\t2019\nY,M,Y_LT1,PT\t78.6\n", encoding="utf-8")

    # Point cleaning.py base_dir to our fake package dir
    import life_expectancy.cleaning as cln
    monkeypatch.setattr(cln, "__file__", str(fake_pkg_dir / "cleaning.py"))

    # Pass a non-existent absolute path: function should fall back to data/<filename>
    df = cln.load_data("/does/not/exist/tiny.tsv")
    assert set(df.columns) == {"unit", "sex", "age", "geo", "time", "value"}
    assert len(df) == 1


def test_clean_data_country_alias_param():
    """
    Covers cleaning.py line 58: resolving Region[country] when using 'country' kwarg.
    """
    # Minimal raw-shape frame
    df = pd.DataFrame(
        {"unit,sex,age,geo\\time": ["Y,M,Y_LT1,PT"], "2019": ["78.6"]}
    )
    out = clean_data(df, country="PT")
    assert set(out.columns) == {"unit", "sex", "age", "region", "year", "value"}
    assert (out["region"] == "PT").all()


def test_clean_data_from_normalized_input_to_region_geo():
    """
    Covers cleaning.py line 102: handling already-normalized DF (adapter output)
    and remapping to region/year.
    """
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
    # normalized path returns geo in final output (per earlier behavior)
    assert set(out.columns) == {"unit", "sex", "age", "geo", "year", "value"}
    assert out.loc[0, "geo"] == "PT"
    assert out.loc[0, "year"] == 2019
    assert out.loc[0, "value"] == 78.6
