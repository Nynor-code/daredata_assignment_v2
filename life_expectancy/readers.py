from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd


@dataclass
class CSVReader:
    """Adapter for Eurostat CSV/TSV files."""
    path: Path
    sep: str = "\t"

    def read(self) -> pd.DataFrame:
        df = pd.read_csv(self.path, sep=self.sep)
        first = df.columns[0]  # e.g., "unit,sex,age,geo"
        dims = df[first].str.split(",", expand=True)
        dims.columns = ["unit", "sex", "age", "geo"][: dims.shape[1]]
        value_cols = [c for c in df.columns if c != first]

        out = (
            pd.concat([dims, df[value_cols]], axis=1)
            .melt(id_vars=["unit", "sex", "age", "geo"], var_name="time", value_name="value")
        )
        # clean
        out["time"] = out["time"].astype(str)
        out["value"] = (
            out["value"].astype(str)
            .str.replace(":", "", regex=False)
            .str.replace(" ", "", regex=False)
        )
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        return out.dropna(subset=["value"]).reset_index(drop=True)


@dataclass
class EurostatJSONAdapter:
    """Adapter for Eurostat-like JSON files."""
    path: Path

    def read(self) -> pd.DataFrame:
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Case 1: list of record dicts
        if isinstance(data, list):
            df = pd.DataFrame(data)
            df.columns = [c.lower() for c in df.columns]
            if "year" in df.columns and "time" not in df.columns:
                df["time"] = df["year"]
            needed = {"unit", "sex", "age", "geo", "time", "value"}
            missing = needed - set(df.columns)
            if missing:
                raise ValueError(f"JSON missing keys: {sorted(missing)}")
            df["time"] = df["time"].astype(str)
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            return df.dropna(subset=["value"]).reset_index(drop=True)

        # Case 2: Eurostat compact structure
        if isinstance(data, dict) and "dimension" in data and "value" in data:
            dim = data["dimension"]
            values = data["value"]

            def labels(key: str):
                block = dim.get(key) or dim.get(key.upper())
                if not block:
                    return {}
                cat = block.get("category", {})
                lab = cat.get("label", {})
                return {int(k): str(v) for k, v in lab.items()}

            order = [k for k in ("unit", "sex", "age", "geo", "time") if k in dim] or \
                    ["unit", "sex", "age", "geo", "time"]

            maps = {k: labels(k) or {0: ""} for k in order}

            def size_of(k: str) -> int:
                block = dim.get(k) or dim.get(k.upper()) or {}
                idx = (block.get("category") or {}).get("index")
                return len(idx) if idx else len(maps[k])

            bases = [size_of(k) for k in order]

            def unravel(i: int, bases: list[int]) -> list[int]:
                coords = []
                for b in reversed(bases):
                    coords.append(i % b)
                    i //= b
                return list(reversed(coords))

            rows = []
            for k, v in values.items():
                try:
                    lin = int(k)
                except Exception:
                    continue
                coord = unravel(lin, bases)
                rec = {}
                for name, c in zip(order, coord):
                    rec[name] = maps[name].get(c, str(c))
                rec["value"] = v
                rows.append(rec)

            df = pd.DataFrame(rows)
            for col in ["unit", "sex", "age", "geo", "time"]:
                if col not in df.columns:
                    df[col] = ""
            df["time"] = df["time"].astype(str)
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            return df.dropna(subset=["value"]).reset_index(drop=True)

        raise ValueError("Unsupported JSON shape.")


def make_reader(path: Path):
    ext = path.suffix.lower()
    if ext in {".tsv", ".csv"}:
        return CSVReader(path)
    if ext == ".json":
        return EurostatJSONAdapter(path)
    raise ValueError(f"Unsupported extension: {ext}")
