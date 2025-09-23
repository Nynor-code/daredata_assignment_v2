"""Clean and export Eurostat life expectancy data for a selected country."""
from __future__ import annotations

# Standard library
import argparse
from pathlib import Path

# Third-party
import pandas as pd

# Local
from life_expectancy.readers import make_reader
from .enums import Region

DATA_DIR = Path(__file__).resolve().parent / "data"

__all__ = ["load_data", "save_data", "clean_data", "_map_geo_to_enum_value", "main"]


# =============================================================================
# IO helpers
# =============================================================================


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """
    Load raw life expectancy data.

    - If `path` is None, read the default package TSV: data/eu_life_expectancy_raw.tsv
    - If `path` does not exist, fall back to data/<basename>
    - Otherwise, use the reader factory (CSV/TSV/JSON) and return a normalized DataFrame
      (unit, sex, age, geo, time, value) for adapters, or the raw TSV shape when
      loading the default file for test parity.
    """
    base_dir = Path(__file__).resolve().parent

    if path is None:
        raw_path = base_dir / "data" / "eu_life_expectancy_raw.tsv"
        return pd.read_csv(raw_path, sep="\t")  # keep raw wide shape for tests

    p = Path(path)
    if not p.exists():
        p = base_dir / "data" / Path(path).name

    rd = make_reader(p)
    return rd.read()


def save_data(
    df: pd.DataFrame,
    export_country_code: Region,
    output_path: str | Path | None = None,
) -> Path:
    """
    Save cleaned data to CSV. Creates parent folders if needed.

    Default path (when `output_path` is None):
        data/cleaned/life_expectancy_<CC>.csv
    """
    if output_path is None:
        base_dir = Path(__file__).resolve().parent
        output_path = (
            base_dir
            / "data"
            / "cleaned"
            / f"life_expectancy_{export_country_code.value}.csv"
        )
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


# =============================================================================
# Cleaning
# =============================================================================


def _map_geo_to_enum_value(series: pd.Series, enum_cls: type[Region] | None) -> pd.Series:
    """Map free-text GEO values to Region enum .value; leave unmatched as-is."""
    if enum_cls is None:
        return series

    name_map = {e.name.upper(): str(e.value) for e in enum_cls}
    val_map = {str(e.value).upper(): str(e.value) for e in enum_cls}

    def _map_one(x: object) -> str | None:
        u = str(x).strip().upper()
        return val_map.get(u) or name_map.get(u) or None

    out = series.map(_map_one)
    return out.fillna(series)  # keep original where not matched


def clean_data(
    df: pd.DataFrame,
    export_country_code: Region | None = Region.PT,
    *,
    country: str | None = None,
) -> pd.DataFrame:
    """
    Clean the life expectancy dataset for a specific country.

    Returns a DataFrame of columns:
      - Raw TSV input -> ['unit','sex','age','region','year','value']
      - Normalized adapter input -> ['unit','sex','age','geo','year','value']

    NOTE: age 'Y_GE85' and 'Y_LT1' are kept as valid age groups.
    """
    # Resolve export_country_code from alias if provided
    if country is not None:
        export_country_code = Region[country]

    # Step 1: Strip column names to remove trailing spaces
    df.columns = df.columns.str.strip()

    # Step 2: Unpivot year columns to long format
    from_raw = "unit,sex,age,geo\\time" in df.columns

    if from_raw:
        df_long = df.melt(
            id_vars=["unit,sex,age,geo\\time"], var_name="year", value_name="value"
        )
    else:
        # Already normalized by adapters; Expect: unit, sex, age, geo, time, value
        norm = df.rename(columns=str.lower).copy()
        if {"unit", "sex", "age", "geo", "time", "value"}.issubset(norm.columns):
            norm = norm.rename(columns={"time": "year"})
            norm["region"] = norm["geo"]
            df_long = norm[["unit", "sex", "age", "region", "year", "value"]].copy()
        else:
            # Fallback safety: find the combined column case-insensitively
            combined_name = next(
                (c for c in df.columns if c.strip().lower() == "unit,sex,age,geo\\time"),
                None,
            )
            if combined_name is None:
                raise KeyError(
                    "Missing combined metadata column 'unit,sex,age,geo\\time' "
                    "and normalized columns; cannot melt."
                )
            df_long = df.melt(
                id_vars=[combined_name], var_name="year", value_name="value"
            )
            if combined_name != "unit,sex,age,geo\\time":
                df_long = df_long.rename(
                    columns={combined_name: "unit,sex,age,geo\\time"}
                )
            # Treat this path as raw from here on:
            from_raw = True

    # Step 3: Split the combined metadata column
    if "unit,sex,age,geo\\time" in df_long.columns:
        df_long[["unit", "sex", "age", "region"]] = (
            df_long["unit,sex,age,geo\\time"].str.split(",", expand=True)
        )
        df_long.drop(columns=["unit,sex,age,geo\\time"], inplace=True)

    # Step 4: Clean year and value columns
    df_long["year"] = pd.to_numeric(
        df_long["year"].astype(str).str.strip(), errors="coerce"
    )
    df_long["value"] = df_long["value"].astype(str).str.extract(
        r"(\d+\.?\d*)"
    )  # extract numeric part
    df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce")

    # Step 5: Drop rows with missing year or value
    df_clean = df_long.dropna(subset=["year", "value"]).copy()
    df_clean["year"] = df_clean["year"].astype(int)
    df_clean["value"] = df_clean["value"].astype(float)

    # Step 5.1: Normalize region to uppercase and map via Region enum (geo only)
    df_clean["region"] = df_clean["region"].astype(str).str.strip().str.upper()
    df_clean["region"] = _map_geo_to_enum_value(df_clean["region"], Region).str.upper()

    # Step 6: Filter for selected country only
    assert export_country_code is not None  # for type-checkers
    df_country = df_clean[df_clean["region"] == export_country_code.value]

    # Step 7: Ensure column order
    if from_raw:
        # raw TSV tests expect a 'region' column
        return df_country[["unit", "sex", "age", "region", "year", "value"]].copy()

    # readers/adapter tests expect a 'geo' column
    df_country = df_country[["unit", "sex", "age", "region", "year", "value"]].copy()
    df_country["geo"] = df_country["region"]
    return df_country[["unit", "sex", "age", "geo", "year", "value"]]


# =============================================================================
# CLI orchestration
# =============================================================================


def main() -> pd.DataFrame:
    """
    Orchestrate loading, cleaning, and saving.

    - Input file is looked up inside the package 'data/' folder.
    - Output is always saved to 'data/cleaned/life_expectancy_<CC>.csv'.
    """
    parser = argparse.ArgumentParser(
        description="Clean life expectancy dataset for a given country."
    )
    parser.add_argument(
        "--country",
        type=str,
        default="PT",
        help="Country code (e.g., PT, FR, DE)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="eu_life_expectancy_raw.tsv",
        help=(
            "File name inside the data/ folder (.tsv/.csv/.json). "
            "Defaults to eu_life_expectancy_raw.tsv"
        ),
    )
    args = parser.parse_args()

    # Validate / map country to Region enum
    try:
        country_enum = Region[args.country]

    except KeyError as exc:
        valid_values = [r.value for r in Region.actual_countries()]
        chunks = [
            ", ".join(valid_values[i : i + 10])
            for i in range(0, len(valid_values), 10)
        ]
        valid_codes_formatted = "\n  - " + "\n  - ".join(chunks)
        raise ValueError(
            f"\n\nInvalid country code: '{args.country}'\n\n"
            f"Valid country codes are:\n{valid_codes_formatted}"
        ) from exc

    # Build full paths
    base_dir = Path(__file__).resolve().parent
    input_name = getattr(args, "input", "eu_life_expectancy_raw.tsv")
    input_path = base_dir / "data" / input_name
    output_path = base_dir / "data" / "cleaned" / f"life_expectancy_{country_enum.value}.csv"

    # Run pipeline
    df_raw = load_data(input_path)
    df_cleaned = clean_data(df_raw, export_country_code=country_enum)
    save_data(df_cleaned, export_country_code=country_enum, output_path=output_path)

    print(f"Saved cleaned data to: {output_path}")
    return df_cleaned


if __name__ == "__main__":  # pragma: no cover
    main()
