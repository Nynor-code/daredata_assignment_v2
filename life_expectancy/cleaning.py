# File: life_expectancy/cleaning.py
"""
Cleans the life expectancy dataset for a specific country (default: Portugal).
"""
# standard imports
import argparse
from pathlib import Path
# third party imports
import pandas as pd
# local imports
from .enums import Region

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_data() -> pd.DataFrame:
    """
    Loads the raw life expectancy data.
    returns a DataFrame with the loaded raw data.
    """
    file_path = DATA_DIR / "eu_life_expectancy_raw.tsv"
    return pd.read_csv(file_path, sep='\t', encoding="utf-8")


def clean_data(df, export_country_code=['PT']) -> pd.DataFrame:
    """
    Cleans the life expectancy dataset for a specific country.
    returns a DataFrame with the cleaned data on the specific country.
    NOTE: age 'Y_GE85' and 'Y_LT1' were kept as they are valid age groups
    """
    # Step 1: Strip column names to remove trailing spaces
    df.columns = df.columns.str.strip()

    # Step 2: Unpivot year columns to long format
    df_long = df.melt(id_vars=['unit,sex,age,geo\\time'], var_name='year', value_name='value')

    # Step 3: Split the combined metadata column
    df_long[['unit', 'sex', 'age', 'region']] = \
        df_long['unit,sex,age,geo\\time'].str.split(',', expand=True)
    df_long.drop(columns=['unit,sex,age,geo\\time'], inplace=True)

    # Step 4: Clean year and value columns
    df_long['year'] = pd.to_numeric(df_long['year'].str.strip(), errors='coerce')
    df_long['value'] = df_long['value'].str.extract(r'(\d+\.?\d*)')  # extract numeric part
    df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce')

    # Step 5: Drop rows with missing year or value
    df_clean = df_long.dropna(subset=['year', 'value']).copy()
    df_clean['year'] = df_clean['year'].astype(int)
    df_clean['value'] = df_clean['value'].astype(float)

    # Step 6: Filter for selected country only
    # country_code is a Region enum, so we use its value
    df_country = df_clean[df_clean['region'] == export_country_code.value]

    # Step 7: Ensure column order
    df_country = df_country[['unit', 'sex', 'age', 'region', 'year', 'value']]

    return df_country


def save_data(df_country, export_country_code: Region = Region.PT) -> None:
    """
    Saves the cleaned dataset to a CSV file.
    The filename is based on the country code.
    """
    output_filename = f'{export_country_code.lower()}_life_expectancy.csv'
    output_path = DATA_DIR / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_country.to_csv(output_path, index=False)

def main() -> pd.DataFrame:
    """
    Main function to orchestrate the data loading, cleaning, and saving.
    """
    parser = argparse.ArgumentParser(
        description="Clean life expectancy dataset for a given country."
        )
    parser.add_argument('--country', type=str, default='PT', help='Country code (e.g., PT, FR, DE)')
    args = parser.parse_args()

    try:
        country_enum = Region[args.country]
    except KeyError as exc:
        # user friendly list presentation of valid country codes
        valid_values = [r.value for r in Region.actual_countries()]
        chunks = [", ".join(valid_values[i:i+10]) for i in range(0, len(valid_values), 10)]
        valid_codes_formatted = "\n  - " + "\n  - ".join(chunks)

        raise ValueError(
            f"\n\nInvalid country code: '{args.country}'\n\n"
            f"Valid country codes are:\n{valid_codes_formatted}"
        ) from exc

    df_raw = load_data()
    df_cleaned = clean_data(df_raw, export_country_code=country_enum)
    save_data(df_cleaned, export_country_code=country_enum)

    return df_cleaned

if __name__ == "__main__":  # pragma: no cover
    main()
