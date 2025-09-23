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
        """
        Read either:
          1) A list of record dicts with keys like unit/sex/age/geo/(time|year)/value, or
          2) A compact SDMX-like object with "dimension" and "value" maps.

        Returns a normalized long DataFrame with columns:
        ['unit', 'sex', 'age', 'geo', 'time', 'value'] and numeric 'value'.
        """
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Case 1: list of record dicts
        if isinstance(data, list):
            return self._read_records_list(data)

        # Case 2: Eurostat compact structure
        if isinstance(data, dict) and "dimension" in data and "value" in data:
            return self._read_compact_dict(data)

        raise ValueError("Unsupported JSON shape.")

    # ----- helpers: records list -----

    @staticmethod
    def _read_records_list(records: Iterable[dict]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        df.columns = [c.lower() for c in df.columns]

        # Merge year/time -> standardize to 'time'
        if "year" in df.columns:
            if "time" in df.columns:
                # prefer 'time', but fill NaNs from 'year'
                df["time"] = df["time"].where(df["time"].notna(), df["year"])
            else:
                df["time"] = df["year"]
            df = df.drop(columns=["year"])

        needed = {"unit", "sex", "age", "geo", "time", "value"}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"JSON missing keys: {sorted(missing)}")

        # Normalize: 'time' -> "YYYY" string, 'value' -> numeric
        def _norm_time(x: object) -> str:
            try:
                v = float(x)  # handles ints/strings
                if pd.notna(v):
                    return str(int(v))
            except (TypeError, ValueError):
                # leave as string if it can't be parsed as float
                pass
            return str(x)

        df["time"] = df["time"].map(_norm_time)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        return df.dropna(subset=["value"]).reset_index(drop=True)

    # ----- helpers: compact dict -----

    @staticmethod
    def _labels_from_dimension(dim: dict, key: str) -> Dict[int, str]:
        """Extract {index:int -> label:str} for a dimension key (case-insensitive)."""
        block = dim.get(key) or dim.get(key.upper())
        if not block:
            return {}
        cat = block.get("category", {})
        lab = cat.get("label", {})
        return {int(k): str(v) for k, v in lab.items()}

    @staticmethod
    def _dimension_order(dim: dict) -> List[str]:
        """Prefer explicit dimension order, else default common order."""
        order = [k for k in ("unit", "sex", "age", "geo", "time") if k in dim]
        return order or ["unit", "sex", "age", "geo", "time"]

    @classmethod
    def _dimension_sizes(cls,
                         dim: dict,
                         order: List[str],
                         maps: Dict[str, Dict[int, str]]
                         ) -> List[int]:
        """Compute base sizes for each dimension (prefer category.index if present)."""
        sizes: List[int] = []
        for key in order:
            block = dim.get(key) or dim.get(key.upper()) or {}
            idx = (block.get("category") or {}).get("index")
            sizes.append(len(idx) if idx else len(maps[key]))
        return sizes

    @staticmethod
    def _unravel_index(i: int, base_sizes: List[int]) -> List[int]:
        """Unravel a linear index into coordinates for given base sizes."""
        coords: List[int] = []
        for b in reversed(base_sizes):
            coords.append(i % b)
            i //= b
        return list(reversed(coords))

    @classmethod
    def _read_compact_dict(cls, data: dict) -> pd.DataFrame:
        dim: dict = data["dimension"]
        values: Dict[str, float] = data["value"]

        order = cls._dimension_order(dim)

        maps: Dict[str, Dict[int, str]] = {}
        for k in order:
            maps[k] = cls._labels_from_dimension(dim, k) or {0: ""}

        bases = cls._dimension_sizes(dim, order, maps)

        rows: List[dict] = []
        for k, v in values.items():
            try:
                lin = int(k)
            except (TypeError, ValueError):
                # skip non-integer indices
                continue
            coord = cls._unravel_index(lin, bases)
            rec = {name: maps[name].get(c, str(c)) for name, c in zip(order, coord, strict=False)}
            rec["value"] = v
            rows.append(rec)

        df = pd.DataFrame(rows)
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
