"""Reader adapters for Eurostat CSV/TSV and Eurostat-like JSON."""

from __future__ import annotations

# standard imports
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Dict, Iterable, List

# third party imports
import pandas as pd



# =============================================================================
# CSV / TSV
# =============================================================================


@dataclass
class CSVReader:
    """Adapter for Eurostat CSV/TSV files (raw header + wide year columns)."""

    path: Path
    sep: str = "\t"

    def read(self) -> pd.DataFrame:
        """
        Read a raw Eurostat TSV/CSV, split the combined first column into
        (unit, sex, age, geo), melt year columns into long format, and clean values.
        """
        df = pd.read_csv(self.path, sep=self.sep)

        # Split first column e.g. "unit,sex,age,geo" into separate dimensions
        first = df.columns[0]
        dims = df[first].str.split(",", expand=True)
        dims.columns = ["unit", "sex", "age", "geo"][: dims.shape[1]]

        value_cols = [c for c in df.columns if c != first]

        out = pd.concat([dims, df[value_cols]], axis=1).melt(
            id_vars=["unit", "sex", "age", "geo"],
            var_name="time",
            value_name="value",
        )

        # normalize: keep time as string, coerce value to numeric and drop NaNs
        out["time"] = out["time"].astype(str)
        out["value"] = (
            out["value"]
            .astype(str)
            .str.replace(":", "", regex=False)  # Eurostat missing marker
            .str.replace(" ", "", regex=False)
        )
        out["value"] = pd.to_numeric(out["value"], errors="coerce")

        return out.dropna(subset=["value"]).reset_index(drop=True)


# =============================================================================
# Eurostat-like JSON
# =============================================================================

@dataclass
class EurostatJSONAdapter:
    """Adapter for Eurostat-like JSON files (records-list or compact SDMX-like)."""

    path: Path
    
    
    def read(self) -> pd.DataFrame:
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return self._read_records_list(data)

        if isinstance(data, dict) and "dimension" in data and "value" in data:
            return self._read_compact_dict(data)

        raise ValueError("Unsupported JSON shape.")


    @staticmethod
    def _read_records_list(records: Iterable[dict]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        # lower all columns for robust matching
        df.columns = [c.lower() for c in df.columns]


        # -------- alias resolution (geo/value/time) ----------
        def pick(cols, *candidates):
            for c in candidates:
                if c in cols:
                    return c
            return None

        cols = set(df.columns)

        # time can be: time | year | date
        time_col = pick(cols, "time", "year", "date")
        if time_col is None:
            # keep the original error behavior, but clearer
            raise ValueError("JSON missing a time-like key among: ['time','year','date']")

        # geo can be: geo | country | region | geo_code | geocode | nuts_code
        geo_col = pick(cols, "geo", "country", "region", "geo_code", "geocode", "nuts_code")
        # value can be: value | values | obs_value | obsvalue | life_expectancy | le | val
        value_col = pick(
            cols, "value", "values", "obs_value", "obsvalue", "life_expectancy", "lifeexpectancy", "le", "val"
        )

        # Standardize column names where present
        rename_map = {}
        if time_col != "time":
            rename_map[time_col] = "time"
        if geo_col and geo_col != "geo":
            rename_map[geo_col] = "geo"
        if value_col and value_col != "value":
            rename_map[value_col] = "value"

        df = df.rename(columns=rename_map)

        # If 'year' exists alongside 'time', fill NaNs in time from year (existing behavior)
        if "year" in df.columns and "time" in df.columns:
            df["time"] = df["time"].where(df["time"].notna(), df["year"])
            # Drop extra 'year' after merge
            df = df.drop(columns=["year"])

        # ---- required keys check (after aliasing) ----
        needed = {"unit", "sex", "age", "geo", "time", "value"}
        missing = needed - set(df.columns)
        if missing:
            # help the user by showing what IS present
            present = sorted(df.columns.tolist())
            raise ValueError(f"JSON missing keys: {sorted(missing)}. Present keys: {present}")


        # Normalize: 'time' → "YYYY" string, 'value' → numeric
        def _norm_time(x: object) -> str:
            try:
                v = float(x)
                if pd.notna(v):
                    return str(int(v))
            except Exception:
                pass
            return str(x)

        df["time"] = df["time"].map(_norm_time)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        return df.dropna(subset=["value"]).reset_index(drop=True)


# =============================================================================
# Factory
# =============================================================================


def make_reader(path: Path) -> CSVReader | EurostatJSONAdapter:
    """Factory: choose the appropriate reader by file extension."""
    ext = path.suffix.lower()
    if ext in {".tsv", ".csv"}:
        return CSVReader(path)
    if ext == ".json":
        return EurostatJSONAdapter(path)
    raise ValueError(f"Unsupported extension: {ext}")
