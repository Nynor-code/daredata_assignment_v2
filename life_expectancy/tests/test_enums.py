"""
Tests for the Region enum in the life_expectancy package.
Specifically, it checks that the actual_countries method excludes 
aggregate regions.
"""
# Local imports
from life_expectancy.enums import Region


def test_actual_countries_excludes_aggregates():
    """
    Test that the actual_countries method excludes aggregate regions.
    """
    aggregate_codes = {
        "DE_TOT",
        "EA18",
        "EA19",
        "EEA30_2007",
        "EEA31",
        "EFTA",
        "EU27_2007",
        "EU27_2020",
        "EU28"
    }

    actual_regions = Region.actual_countries()
    actual_codes = {r.value for r in actual_regions}

    for code in aggregate_codes:
        assert code not in actual_codes
    