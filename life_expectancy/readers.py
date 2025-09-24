"""Reader adapters for Eurostat CSV/TSV and Eurostat-like JSON."""

from __future__ import annotations

# standard imports
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Dict, Iterable, List

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
    """
    Adapter for Eurostat-like JSON files.

    Supports two shapes:
      1) Records-list: list[dict] with keys similar to
         unit/sex/age/geo/(time|year|date)/value
         - Accepts common aliases (e.g., 'country' for geo, 'obs_value' for value).
      2) Compact object:
         {
           "dimension": { <per-dimension metadata> },
           "value": { "<linear_index>": <numeric> }
         }

    Returns a normalized long DataFrame with columns:
      ['unit', 'sex', 'age', 'geo', 'time', 'value'],
    where 'time' is a string (e.g. "2019") and 'value' is numeric.
    """

    path: Path

    def read(self) -> pd.DataFrame:
        """Route to the appropriate parser based on JSON shape."""
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return self._read_records_list(data)

        if isinstance(data, dict) and "dimension" in data and "value" in data:
            # Call staticmethod via class to avoid pylint no-member on instance.
            return EurostatJSONAdapter._read_compact_dict(data)

        raise ValueError("Unsupported JSON shape.")

    # ---------------------------------------------------------------------
    # Records-list shape
    # ---------------------------------------------------------------------

    @staticmethod
    def _read_records_list(records: Iterable[dict]) -> pd.DataFrame:
        """Read a list of record dicts and normalize columns."""
        df = pd.DataFrame(records)
        df.columns = [c.lower() for c in df.columns]

        def pick(cols: set[str], *candidates: str) -> str | None:
            """Return the first candidate present in `cols`, else None."""
            for c in candidates:
                if c in cols:
                    return c
            return None

        cols = set(df.columns)

        # Find time-like column
        time_col = pick(cols, "time", "year", "date")
        if time_col is None:
            raise ValueError(
                "JSON missing a time-like key among: ['time', 'year', 'date']"
            )

        # Aliases for geo and value
        geo_col = pick(
            cols, "geo", "country", "region", "geo_code", "geocode", "nuts_code"
        )
        value_col = pick(
            cols,
            "value",
            "values",
            "obs_value",
            "obsvalue",
            "life_expectancy",
            "lifeexpectancy",
            "le",
            "val",
        )

        # Standardize present columns
        rename_map: dict[str, str] = {}
        if time_col != "time":
            rename_map[time_col] = "time"
        if geo_col and geo_col != "geo":
            rename_map[geo_col] = "geo"
        if value_col and value_col != "value":
            rename_map[value_col] = "value"
        df = df.rename(columns=rename_map)

        # If both 'year' and 'time' exist, fill missing 'time' from 'year'
        if "year" in df.columns and "time" in df.columns:
            df["time"] = df["time"].where(df["time"].notna(), df["year"])
            df = df.drop(columns=["year"])

        needed = {"unit", "sex", "age", "geo", "time", "value"}
        missing = needed - set(df.columns)
        if missing:
            present = sorted(df.columns.tolist())
            raise ValueError(
                "JSON missing keys: "
                f"{sorted(missing)}. "
                f"Present keys: {present}"
            )

        # Normalize: time -> "YYYY" (string), value -> numeric
        def _norm_time(x: object) -> str:
            """Convert time-like value to string, e.g. 2019.0 -> '2019'."""
            # Only attempt float conversion for common scalar types
            if isinstance(x, (int, float, str)):
                try:
                    v = float(x)
                except (TypeError, ValueError):
                    return str(x)
                if pd.notna(v):
                    return str(int(v))
                return ""
            # Fallback for exotic types
            return str(x)

        df["time"] = df["time"].map(_norm_time)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        return df.dropna(subset=["value"]).reset_index(drop=True)

    # ---------------------------------------------------------------------
    # Compact SDMX-like shape (helpers to keep locals low)
    # ---------------------------------------------------------------------

    @staticmethod
    def _labels_from_dimension(dim: dict, key: str) -> Dict[int, str]:
        """Return index->label map for a given dimension key."""
        block = dim.get(key) or dim.get(key.upper())
        if not block:
            return {}
        cat = block.get("category", {})
        lab = cat.get("label", {})
        return {int(k): str(v) for k, v in lab.items()}

    @staticmethod
    def _dimension_order(dim: dict) -> List[str]:
        """Use explicit order; else default canonical order."""
        found = [k for k in ("unit", "sex", "age", "geo", "time") if k in dim]
        return found or ["unit", "sex", "age", "geo", "time"]

    @staticmethod
    def _dimension_sizes(dim: dict, order: List[str], maps: Dict[str, Dict[int, str]]
                         ) -> List[int]:
        """Return the cardinality of each dimension in order."""
        sizes: List[int] = []
        for k in order:
            block = dim.get(k) or dim.get(k.upper()) or {}
            idx = (block.get("category") or {}).get("index")
            sizes.append(len(idx) if idx else len(maps[k]))
        return sizes

    @staticmethod
    def _unravel(index: int, bases: List[int]) -> List[int]:
        """Convert linear index to multi-dimensional coordinates."""
        coords: List[int] = []
        for b in reversed(bases):
            coords.append(index % b)
            index //= b
        return list(reversed(coords))

    @staticmethod
    def _build_maps_and_bases(dim: dict) -> tuple[List[str], Dict[str, Dict[int, str]],
                                                  List[int]]:
        """Build dimension order, label maps, and base sizes."""
        order = EurostatJSONAdapter._dimension_order(dim)
        maps = {k: EurostatJSONAdapter._labels_from_dimension(dim, k) or {0: ""} for k in order}
        bases = EurostatJSONAdapter._dimension_sizes(dim, order, maps)
        return order, maps, bases

    @staticmethod
    def _rows_from_values(
        values: Dict[str, Any],
        order: List[str],
        maps: Dict[str, Dict[int, str]],
        bases: List[int],
    ) -> List[Dict[str, Any]]:
        """Materialize rows from value dict using dimension maps."""
        rows: List[Dict[str, Any]] = []
        for k, v in values.items():
            try:
                lin = int(k)
            except (TypeError, ValueError):
                continue
            coord = EurostatJSONAdapter._unravel(lin, bases)
            rec: Dict[str, Any] = {name: maps[name].get(c, str(c)) for name, c in zip(order, coord)}
            # value can be float/int/str; keep as-is, we'll coerce later with pd.to_numeric
            rec["value"] = v
            rows.append(rec)
        return rows

    @staticmethod
    def _read_compact_dict(data: dict) -> pd.DataFrame:
        """Read a compact Eurostat-like dict with 'dimension' and 'value'."""
        dim: dict = data["dimension"]
        values: Dict[str, float] = data["value"]

        order, maps, bases = EurostatJSONAdapter._build_maps_and_bases(dim)
        rows = EurostatJSONAdapter._rows_from_values(values, order, maps, bases)

        df = pd.DataFrame(rows)

        # Ensure all required columns exist
        for col in ["unit", "sex", "age", "geo", "time"]:
            if col not in df.columns:
                df[col] = ""

        df["time"] = df["time"].astype(str)
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
