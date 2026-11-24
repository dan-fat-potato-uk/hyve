# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration model of hyve-forecast-crps."""

import pandas as pd
import pytest
from pydantic import ValidationError

from hyve.tools.forecast_crps.config import Aggregation, ForecastCrpsConfig, load_config


def test_defaults_describe_the_reforecast_structure():
    config = ForecastCrpsConfig()
    assert config.structure == "reforecast"
    assert config.variable == "dis"
    assert config.spatial_dims == ["station"]
    assert config.climatology is None
    assert config.aggregation == Aggregation(by="valid_time", min_count=1)


def test_unknown_keys_are_rejected():
    with pytest.raises(ValidationError, match="not_a_key"):
        ForecastCrpsConfig(not_a_key=1)


def test_with_init_is_rejected_for_medium_range():
    with pytest.raises(ValidationError, match="with_init"):
        ForecastCrpsConfig(structure="medium_range", with_init=True)


def test_aggregation_is_rejected_for_reforecast():
    with pytest.raises(ValidationError, match="aggregation"):
        ForecastCrpsConfig(aggregation={"by": "start_time"})


@pytest.mark.parametrize(
    "field, value",
    [
        ("chunk_size", 0),
        ("threads", 0),
        ("coord_tolerance", -1.0),
        ("coord_tolerance", float("inf")),
        ("coord_tolerance", float("nan")),
        ("spatial_dims", []),
    ],
)
def test_invalid_values_are_rejected(field, value):
    with pytest.raises(ValidationError, match=field):
        ForecastCrpsConfig(**{field: value})


def test_min_count_must_be_positive():
    with pytest.raises(ValidationError, match="min_count"):
        Aggregation(min_count=0)


def test_load_config_reads_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "structure: medium_range\n"
        "spatial_dims: [point]\n"
        "chunk_size: 5000\n"
        "aggregation:\n"
        "  by: start_time\n"
    )
    config = load_config(path)
    assert config.structure == "medium_range"
    assert config.spatial_dims == ["point"]
    assert config.chunk_size == 5000
    assert config.aggregation.by == "start_time"


def test_overrides_win_over_the_file_and_none_is_ignored(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "structure: medium_range\nchunk_size: 5000\nthreads: 4\n"
        "aggregation:\n  by: start_time\n  min_count: 3\n"
    )
    config = load_config(
        path,
        chunk_size=10,
        threads=None,
        aggregation={"min_count": 5},
    )
    assert config.chunk_size == 10
    assert config.threads == 4
    assert config.aggregation.by == "start_time"
    assert config.aggregation.min_count == 5


def test_load_config_without_a_file_uses_defaults_and_overrides():
    config = load_config(None, structure="medium_range", spatial_dims=["point"])
    assert config.structure == "medium_range"
    assert config.spatial_dims == ["point"]


def test_lead_time_and_ensemble_dims_default_to_time_and_ensemble():
    config = ForecastCrpsConfig(structure="medium_range")
    assert config.lead_time_dim == "time"
    assert config.ensemble_dim == "ensemble"


def test_lead_time_dim_accepts_time_or_step_only():
    assert ForecastCrpsConfig(structure="medium_range", lead_time_dim="step")
    with pytest.raises(ValidationError, match="lead_time_dim"):
        ForecastCrpsConfig(structure="medium_range", lead_time_dim="valid_time")


@pytest.mark.parametrize(
    "field, value", [("lead_time_dim", "step"), ("ensemble_dim", "number")]
)
def test_forecast_layout_is_rejected_for_reforecast(field, value):
    with pytest.raises(ValidationError, match=field):
        ForecastCrpsConfig(**{field: value})


def test_aggregation_completeness_defaults():
    aggregation = Aggregation()
    assert aggregation.start_frequency == "1D"
    assert aggregation.start_step == pd.Timedelta(days=1)
    assert aggregation.allow_missing is False


@pytest.mark.parametrize("value", ["12h", "6h", "24h", "1D"])
def test_start_frequency_dividing_a_day_is_accepted(value):
    assert Aggregation(start_frequency=value).start_step <= pd.Timedelta(days=1)


@pytest.mark.parametrize("value", ["5h", "2D", "0h", "-6h", "daily"])
def test_start_frequency_must_divide_a_day(value):
    with pytest.raises(ValidationError, match="start_frequency"):
        Aggregation(start_frequency=value)
