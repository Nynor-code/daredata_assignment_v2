"""Reader adapter tests for CSV/TSV and Eurostat JSON."""

from __future__ import annotations

# Standard Lybrary
from pathlib import Path
import json

# Third-party
import numpy as np
import pandas as pd
import pytest

# Local
from life_expectancy.readers import (
    CSVReader,
    EurostatJSONAdapter,
    make_reader
)
from life_expectancy.cleaning import (
    load_data,
    clean_data
)


# =============================================================================
# CSVReader tests
# =============================================================================


def test_csv_reader_basic(tmp_path: Path) -> None:
    """Splits combined header, melts years to long format."""
    p = tmp_path / "sample.tsv"
    p.write_text(
        "unit,sex,age,geo\\time\t2019\t2020\n"
        "Y,M,Y_LT1,PT\t78.6\t79.1\n"
        "Y,F,Y_LT1,PT\t84.3\t84.7\n",
        encoding="utf-8",
    )

    df = CSVReader(p).read()
    assert set(df.columns) == {"unit", "sex", "age", "geo", "time", "value"}
    assert len(df) == 4
    assert df.loc[0, "geo"] == "PT"


def test_csv_reader_melts_and_cleans(tmp_path: Path) -> None:
    """
    Split first column, melt years, clean values.
    ':' should become NaN and be dropped.
    """
    p = tmp_path / "sample.tsv"
    p.write_text(
        "unit,sex,age,geo\\time\t2019\t2020\n"
        "Y,M,Y_LT1,PT\t78.6\t:\n"
        "Y,F,Y_LT1,PT\t84.3\t84.7\n",
        encoding="utf-8",
    )

    df = CSVReader(p).read()
    assert set(df.columns) == {"unit", "sex", "age", "geo", "time", "value"}
    # 2019(M), 2019(F), 2020(F) -> 3 rows after dropping 2020(M) ':'
    assert len(df) == 3
    assert df["time"].dtype == object
    assert pd.api.types.is_float_dtype(df["value"])


def test_csv_reader_header_with_spaces(tmp_path: Path) -> None:
    """Handles header with trailing spaces before the tab."""
    p = tmp_path / "hdr_spaces.tsv"
    p.write_text(
        "unit,sex,age,geo\\time  \t2019\n"  # two trailing spaces before \t
        "Y,M,Y_LT1,PT\t78.6\n",
        encoding="utf-8",
    )

    df = CSVReader(p).read()
    assert {"unit", "sex", "age", "geo", "time", "value"} <= set(df.columns)
    assert df.loc[0, "time"] == "2019"


def test_csv_reader_value_cleaning_colon_spaces_comma(tmp_path: Path) -> None:
    """
    Coerces non-parsable values (':', comma decimals like '78,6', spaced '84 , 3') 
    to NaN and drops them.
    Only strict decimal with dot survives.
    """
    p = tmp_path / "values_clean.tsv"
    p.write_text(
        "unit,sex,age,geo\\time\t2019\t2020\n"
        "Y,M,Y_LT1,PT\t : \t 78,6 \n"
        "Y,F,Y_LT1,PT\t84 , 3\t 84.7 \n",
        encoding="utf-8",
    )

    df = CSVReader(p).read()
    assert {"unit", "sex", "age", "geo", "time", "value"} <= set(df.columns)

    # Only the strictly numeric '84.7' survives -> 1 row
    assert len(df) == 1
    row = df.iloc[0]
    assert row["unit"] == "Y"
    assert row["sex"] == "F"
    assert row["age"] == "Y_LT1"
    assert row["geo"] == "PT"
    assert row["time"] == "2020"
    assert row["value"] == 84.7


def test_csv_reader_dropna_after_numeric_coercion(tmp_path: Path) -> None:
    """Drops rows where 'value' becomes NaN after numeric coercion."""
    p = tmp_path / "dropna.tsv"
    p.write_text(
        "unit,sex,age,geo\\time\t2019\n"
        "Y,M,Y_LT1,PT\tN/A\n"
        "Y,F,Y_LT1,PT\t84.3\n",
        encoding="utf-8",
    )

    df = CSVReader(p).read()
    assert len(df) == 1
    assert df.loc[0, "value"] == 84.3


def test_csv_reader_split_clean_dropna(tmp_path: Path) -> None:
    """
    Trim spaces around numbers, drop ':', and verify time stays strings.
    """
    p = tmp_path / "messy.tsv"
    p.write_text(
        "unit,sex,age,geo\\time\t2018\t2019\t2020\n"
        "Y,M,Y_LT1,PT\t 78.4 \t 78.6 \t : \n"
        "Y,F,Y_LT1,PT\t 84.0 \t 84.3 \t 84.7 \n",
        encoding="utf-8",
    )

    df = CSVReader(p).read()
    assert {"unit", "sex", "age", "geo", "time", "value"} <= set(df.columns)
    # valid: 2018(M), 2019(M), 2018(F), 2019(F), 2020(F) => 5 rows
    assert len(df) == 5
    assert df["value"].dtype.kind in "fi"
    assert set(df["time"].unique()) == {"2018", "2019", "2020"}


# =============================================================================
# EurostatJSONAdapter tests
# =============================================================================


def test_json_adapter_records_list(tmp_path: Path) -> None:
    """Records-list path returns normalized columns."""
    p = tmp_path / "sample.json"
    records = [
        {"unit": "Y", "sex": "M", "age": "Y_LT1", "geo": "PT", "time": 2019, "value": 78.6},
        {"unit": "Y", "sex": "F", "age": "Y_LT1", "geo": "PT", "time": 2019, "value": 84.3},
    ]
    p.write_text(json.dumps(records), encoding="utf-8")

    df = EurostatJSONAdapter(p).read()
    assert len(df) == 2
    assert df.loc[0, "geo"] == "PT"


def test_json_adapter_records_list_year_to_time(tmp_path: Path) -> None:
    """Records-list path maps 'year' to 'time' and drops 'year'."""
    recs = [
        {"unit": "Y", "sex": "M", "age": "Y_LT1", "geo": "PT", "year": 2019, "value": 78.6},
        {"unit": "Y", "sex": "F", "age": "Y_LT1", "geo": "PT", "time": 2019, "value": 84.3},
    ]
    p = tmp_path / "recs.json"
    p.write_text(json.dumps(recs), encoding="utf-8")

    df = EurostatJSONAdapter(p).read()
    assert set(df.columns) == {"unit", "sex", "age", "geo", "time", "value"}
    assert (df["time"] == "2019").all()
    assert len(df) == 2


def test_json_records_merge_year_into_time(tmp_path: Path) -> None:
    """If both 'year' and 'time' exist, fill missing 'time' values from 'year'."""
    recs = [
        {"unit": "Y",
         "sex": "M",
         "age": "Y_LT1",
         "geo": "PT",
         "year": 2018,
         "time": None,
         "value": 78.4
        },
        {"unit": "Y",
         "sex": "F",
         "age": "Y_LT1",
         "geo": "PT",
         "time": 2019,
         "value": 84.3
        },
    ]
    p = tmp_path / "recs_merge.json"
    p.write_text(json.dumps(recs), encoding="utf-8")

    df = EurostatJSONAdapter(p).read()
    assert set(df.columns) == {"unit", "sex", "age", "geo", "time", "value"}
    assert df["time"].tolist() == ["2018", "2019"]


def test_json_records_merge_year_when_time_is_nan(tmp_path: Path) -> None:
    """Fills NaN time from 'year' as well (not just None)."""
    recs = [
        {"unit": "Y",
         "sex": "M",
         "age": "Y_LT1",
         "geo": "PT",
         "year": 2019,
         "time": float("nan"),
         "value": 78.6
         },
        {"unit": "Y",
         "sex": "F",
         "age": "Y_LT1",
         "geo": "PT",
         "time": 2019,
         "value": 84.3
        },
    ]
    # ensure proper NaN (not 'nan' string)
    recs[0]["time"] = np.nan
    p = tmp_path / "merge_nan.json"
    p.write_text(json.dumps(recs), encoding="utf-8")

    df = EurostatJSONAdapter(p).read()
    assert df["time"].tolist() == ["2019", "2019"]


def test_json_adapter_records_missing_keys_raises(tmp_path: Path) -> None:
    """Missing required keys should raise a ValueError."""
    recs = [{"unit": "Y", "sex": "M"}]  # missing many keys
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(recs), encoding="utf-8")

    with pytest.raises(ValueError):
        EurostatJSONAdapter(p).read()


def test_json_adapter_compact_structure(tmp_path: Path) -> None:
    """Compact SDMX-like structure with a single linear index."""
    obj = {
        "dimension": {
            "unit": {"category": {"label": {"0": "Y"}}},
            "sex": {"category": {"label": {"0": "M"}}},
            "age": {"category": {"label": {"0": "Y_LT1"}}},
            "geo": {"category": {"label": {"0": "PT"}}},
            "time": {"category": {"label": {"0": "2019"}}},
        },
        "value": {"0": 78.6},  # linear index 0 -> (0,0,0,0,0)
    }
    p = tmp_path / "compact.json"
    p.write_text(json.dumps(obj), encoding="utf-8")

    df = EurostatJSONAdapter(p).read()
    assert len(df) == 1
    assert df.loc[0, "geo"] == "PT"
    assert df.loc[0, "time"] == "2019"
    assert df.loc[0, "value"] == 78.6


def test_json_compact_multiple_labels(tmp_path: Path) -> None:
    """Compact SDMX-like structure with multiple labels per dimension."""
    obj = {
        "dimension": {
            "unit": {"category": {"label": {"0": "Y"}}},
            "sex": {"category": {"label": {"0": "M", "1": "F"}}},
            "age": {"category": {"label": {"0": "Y_LT1"}}},
            "geo": {"category": {"label": {"0": "PT"}}},
            "time": {"category": {"label": {"0": "2018", "1": "2019"}}},
        },
        # indices map (sex,time): (0,0)->0, (1,1)->1 (depending on unravel order)
        "value": {"0": 78.4, "3": 84.3},
    }
    p = tmp_path / "compact_multi.json"
    p.write_text(json.dumps(obj), encoding="utf-8")

    df = EurostatJSONAdapter(p).read()
    assert {"unit", "sex", "age", "geo", "time", "value"} <= set(df.columns)
    assert set(df["time"]) == {"2018", "2019"}
    assert set(df["sex"]) == {"M", "F"}


def test_json_compact_unravel_string_keys(tmp_path: Path) -> None:
    """Compact SDMX-like structure where 'value' dict has string keys."""
    obj = {
        "dimension": {
            "unit": {"category": {"label": {"0": "Y"}}},
            "sex": {"category": {"label": {"0": "M", "1": "F"}}},
            "age": {"category": {"label": {"0": "Y_LT1"}}},
            "geo": {"category": {"label": {"0": "PT"}}},
            "time": {"category": {"label": {"0": "2018", "1": "2019"}}},
        },
        "value": {"0": 78.4, "3": 84.3},
    }
    p = tmp_path / "compact_string_keys.json"
    p.write_text(json.dumps(obj), encoding="utf-8")

    df = EurostatJSONAdapter(p).read()
    assert {"unit", "sex", "age", "geo", "time", "value"} <= set(df.columns)
    assert set(df["sex"]) == {"M", "F"}
    assert set(df["time"]) == {"2018", "2019"}


# =============================================================================
# Factory tests
# =============================================================================


def test_make_reader_selects_correct_reader(tmp_path: Path) -> None:
    """Factory returns correct adapter by file extension."""
    p_csv = tmp_path / "a.csv"
    p_csv.write_text("unit,sex,age,geo\\time,2019\n", encoding="utf-8")

    p_tsv = tmp_path / "a.tsv"
    p_tsv.write_text("unit,sex,age,geo\\time\t2019\n", encoding="utf-8")

    p_json = tmp_path / "a.json"
    p_json.write_text("[]", encoding="utf-8")

    assert isinstance(make_reader(p_csv), CSVReader)
    assert isinstance(make_reader(p_tsv), CSVReader)
    assert isinstance(make_reader(p_json), EurostatJSONAdapter)

    with pytest.raises(ValueError):
        make_reader(tmp_path / "a.parquet")


@pytest.mark.parametrize("ext", [".parquet", ".txt", ".foo"])
def test_make_reader_unsupported_many(tmp_path: Path, ext: str) -> None:
    """Factory raises ValueError on various unsupported extensions."""
    p = tmp_path / f"bad{ext}"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        make_reader(p)


# =============================================================================
# Simple pipeline smoke test (reader -> cleaner)
# =============================================================================


def test_cleaning_pipeline_filters_country(tmp_path: Path) -> None:
    """Read TSV via reader, then clean and filter to a single country."""
    # Import here to avoid unused import at module scope

    p = tmp_path / "sample.tsv"
    p.write_text(
        "unit,sex,age,geo\\time\t2019\n"
        "Y,M,Y_LT1,PT\t78.6\n"
        "Y,F,Y_LT1,ES\t84.3\n",
        encoding="utf-8",
    )

    df = load_data(p)
    out = clean_data(df, country="PT")
    assert (out["geo"] == "PT").all()
