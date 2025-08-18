"""Tests for the cleaning module"""
# Standard imports
import argparse
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
    df = load_data()
    assert not df.empty
    assert "unit,sex,age,geo\\time" in df.columns


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
