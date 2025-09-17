from pathlib import Path
import json
import pandas as pd
import pytest
# local imports
from life_expectancy.readers import CSVReader, EurostatJSONAdapter, make_reader
from life_expectancy.cleaning import clean_data


def test_csv_reader_basic(tmp_path: Path):
    p = tmp_path / "sample.tsv"
    p.write_text("unit,sex,age,geo\t2019\t2020\nY,M,Y_LT1,PT\t78.6\t79.1\nY,F,Y_LT1,PT\t84.3\t84.7\n", encoding="utf-8")
    df = CSVReader(p).read()
    assert set(df.columns) == {"unit", "sex", "age", "geo", "time", "value"}
    assert len(df) == 4
    assert df.loc[0, "geo"] == "PT"


def test_json_adapter_records_list(tmp_path: Path):
    p = tmp_path / "sample.json"
    records = [
        {"unit": "Y", "sex": "M", "age": "Y_LT1", "geo": "PT", "time": 2019, "value": 78.6},
        {"unit": "Y", "sex": "F", "age": "Y_LT1", "geo": "PT", "time": 2019, "value": 84.3},
    ]
    p.write_text(json.dumps(records), encoding="utf-8")
    df = EurostatJSONAdapter(p).read()
    assert len(df) == 2
    assert df.loc[0, "geo"] == "PT"


def test_cleaning_pipeline_filters_country(tmp_path: Path):
    from life_expectancy.cleaning import load_data, clean_data

    # CSV path
    p = tmp_path / "sample.tsv"
    p.write_text("unit,sex,age,geo\t2019\nY,M,Y_LT1,PT\t78.6\nY,F,Y_LT1,ES\t84.3\n", encoding="utf-8")

    df = load_data(p)
    out = clean_data(df, country="PT")
    assert (out["geo"] == "PT").all()


def test_make_reader_selects_correct_reader(tmp_path):
    p_csv = tmp_path / "a.csv"; p_csv.write_text("unit,sex,age,geo\\time,2019\n", encoding="utf-8")
    p_tsv = tmp_path / "a.tsv"; p_tsv.write_text("unit,sex,age,geo\\time\t2019\n", encoding="utf-8")
    p_json = tmp_path / "a.json"; p_json.write_text("[]", encoding="utf-8")

    assert isinstance(make_reader(p_csv), CSVReader)
    assert isinstance(make_reader(p_tsv), CSVReader)
    assert isinstance(make_reader(p_json), EurostatJSONAdapter)

    with pytest.raises(ValueError):
        make_reader(tmp_path / "a.parquet")


def test_csv_reader_melts_and_cleans(tmp_path):
    """
    Hit CSVReader.read: split first column, melt years, clean values.
    Also checks ':' gets dropped (NaN -> dropna).
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
    # 2019 (M), 2019+2020 (F) -> 3 rows after dropping ':' in 2020(M)
    assert len(df) == 3
    assert df["time"].dtype == object
    assert pd.api.types.is_float_dtype(df["value"])


def test_json_adapter_records_list_year_to_time(tmp_path):
    """
    Hit EurostatJSONAdapter records-list path and 'year'->'time' mapping.
    """
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


def test_json_adapter_records_missing_keys_raises(tmp_path):
    """
    Ensure missing required keys triggers ValueError.
    """
    recs = [{"unit": "Y", "sex": "M"}]  # missing many keys
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(recs), encoding="utf-8")
    with pytest.raises(ValueError):
        EurostatJSONAdapter(p).read()


def test_json_adapter_compact_structure(tmp_path):
    """
    Cover the compact SDMX-like branch minimally.
    """
    obj = {
        "dimension": {
            "unit": {"category": {"label": {"0": "Y"}}},
            "sex": {"category": {"label": {"0": "M"}}},
            "age": {"category": {"label": {"0": "Y_LT1"}}},
            "geo": {"category": {"label": {"0": "PT"}}},
            "time": {"category": {"label": {"0": "2019"}}},
        },
        # linear index 0 -> (0,0,0,0,0)
        "value": {"0": 78.6},
    }
    p = tmp_path / "compact.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    df = EurostatJSONAdapter(p).read()
    assert len(df) == 1
    assert df.loc[0, "geo"] == "PT"
    assert df.loc[0, "time"] == "2019"
    assert df.loc[0, "value"] == 78.6

